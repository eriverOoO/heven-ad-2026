#include "ad_planner/local_planning/common/prediction_admission.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ad_planner
{

PredictionSnapshotHistory::PredictionSnapshotHistory(
  const std::size_t maximum_snapshots,
  const std::int64_t clock_rollback_threshold_ns)
: maximum_snapshots_(maximum_snapshots),
  clock_rollback_threshold_ns_(clock_rollback_threshold_ns)
{
  if (maximum_snapshots_ == 0U || clock_rollback_threshold_ns_ <= 0) {
    throw std::invalid_argument(
            "prediction snapshot history limits must be positive");
  }
  snapshots_.reserve(maximum_snapshots_);
}

PredictionSnapshotInsertResult PredictionSnapshotHistory::insert(
  PredictionSnapshot snapshot)
{
  if (snapshot.stamp_ns <= 0 ||
    !std::isfinite(snapshot.receipt_time_s) ||
    snapshot.receipt_time_s < 0.0)
  {
    throw std::invalid_argument("prediction snapshot metadata is invalid");
  }
  if (latest_arrival_stamp_ns_ > 0 &&
    snapshot.stamp_ns < latest_arrival_stamp_ns_ &&
    latest_arrival_stamp_ns_ - snapshot.stamp_ns >
    clock_rollback_threshold_ns_)
  {
    snapshots_.clear();
    latest_arrival_stamp_ns_ = snapshot.stamp_ns;
    return PredictionSnapshotInsertResult::kClockRollback;
  }
  latest_arrival_stamp_ns_ =
    std::max(latest_arrival_stamp_ns_, snapshot.stamp_ns);

  const auto position = std::lower_bound(
    snapshots_.begin(), snapshots_.end(), snapshot.stamp_ns,
    [](const PredictionSnapshot & candidate, const std::int64_t stamp_ns) {
      return candidate.stamp_ns < stamp_ns;
    });
  if (position != snapshots_.end() &&
    position->stamp_ns == snapshot.stamp_ns)
  {
    *position = std::move(snapshot);
    return PredictionSnapshotInsertResult::kReplaced;
  }
  snapshots_.insert(position, std::move(snapshot));
  if (snapshots_.size() > maximum_snapshots_) {
    snapshots_.erase(snapshots_.begin());
  }
  return PredictionSnapshotInsertResult::kInserted;
}

PredictionSnapshotSelection PredictionSnapshotHistory::select(
  const PredictionMode mode, const std::int64_t odometry_stamp_ns,
  const double current_receipt_time_s, const double timeout_s) const
{
  if (mode == PredictionMode::kDisabled) {
    return PredictionSnapshotSelection{
      true, false, 0.0, {}, "prediction explicitly disabled"};
  }
  if (odometry_stamp_ns <= 0 ||
    !std::isfinite(current_receipt_time_s) ||
    !std::isfinite(timeout_s) || timeout_s <= 0.0)
  {
    return PredictionSnapshotSelection{
      false, false, 0.0, {},
      "prediction snapshot selection configuration is invalid"};
  }
  const auto upper = std::upper_bound(
    snapshots_.begin(), snapshots_.end(), odometry_stamp_ns,
    [](const std::int64_t stamp_ns, const PredictionSnapshot & candidate) {
      return stamp_ns < candidate.stamp_ns;
    });
  if (upper == snapshots_.begin()) {
    return PredictionSnapshotSelection{
      false, false, 0.0, {},
      "required predicted-object snapshot at or before odometry is unavailable"};
  }
  const auto & snapshot = *(upper - 1);
  const double receipt_age_s =
    current_receipt_time_s - snapshot.receipt_time_s;
  const long double stamp_age_s =
    (static_cast<long double>(odometry_stamp_ns) -
    static_cast<long double>(snapshot.stamp_ns)) * 1.0e-9L;
  if (!std::isfinite(receipt_age_s) || receipt_age_s < 0.0 ||
    receipt_age_s > timeout_s || !std::isfinite(stamp_age_s) ||
    stamp_age_s < 0.0L ||
    stamp_age_s > static_cast<long double>(timeout_s) ||
    stamp_age_s > static_cast<long double>(
      std::numeric_limits<double>::max()))
  {
    return PredictionSnapshotSelection{
      false, false, 0.0, {},
      "required predicted-object snapshot is stale"};
  }
  return PredictionSnapshotSelection{
    true, true, static_cast<double>(stamp_age_s), snapshot,
    "prediction snapshot selected"};
}

void PredictionSnapshotHistory::clear() noexcept
{
  snapshots_.clear();
  latest_arrival_stamp_ns_ = 0;
}

std::size_t PredictionSnapshotHistory::size() const noexcept
{
  return snapshots_.size();
}

PredictionMode parse_prediction_mode(const std::string & value)
{
  if (value == "disabled") {
    return PredictionMode::kDisabled;
  }
  if (value == "required") {
    return PredictionMode::kRequired;
  }
  throw std::invalid_argument(
          "local_motion.prediction.mode must be disabled or required");
}

const char * prediction_mode_name(const PredictionMode mode) noexcept
{
  switch (mode) {
    case PredictionMode::kDisabled:
      return "disabled";
    case PredictionMode::kRequired:
      return "required";
  }
  return "invalid";
}

PredictionAdmission admit_predictions(
  const PredictionMode mode, const PredictionAdmissionInput & input)
{
  if (mode == PredictionMode::kDisabled) {
    return PredictionAdmission{true, false, "prediction explicitly disabled"};
  }
  if (!std::isfinite(input.timeout_s) || input.timeout_s <= 0.0 ||
    !std::isfinite(input.maximum_future_skew_s) ||
    input.maximum_future_skew_s < 0.0)
  {
    return PredictionAdmission{
      false, false, "prediction admission configuration is invalid"};
  }
  if (!input.received) {
    return PredictionAdmission{
      false, false, "required predicted-object input has not been received"};
  }
  if (!input.valid) {
    return PredictionAdmission{
      false, false, "required predicted-object input is malformed"};
  }
  if (!std::isfinite(input.receipt_age_s) ||
    input.receipt_age_s < 0.0 ||
    input.receipt_age_s > input.timeout_s)
  {
    return PredictionAdmission{
      false, false, "required predicted-object input is stale"};
  }
  if (!std::isfinite(input.stamp_age_s) ||
    input.stamp_age_s < -input.maximum_future_skew_s ||
    input.stamp_age_s > input.timeout_s)
  {
    return PredictionAdmission{
      false, false,
      "required predicted-object stamp is stale or too far in the future"};
  }
  return PredictionAdmission{true, true, "prediction input admitted"};
}

}  // namespace ad_planner
