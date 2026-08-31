#include "ad_planner/planning/roundabout_gap_response.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ad_planner
{
namespace
{

void require_nonnegative(const double value, const char * const name)
{
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(name) + " must be finite and nonnegative");
  }
}

void require_positive(const double value, const char * const name)
{
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) + " must be finite and positive");
  }
}

enum class UnsafeKind : std::uint8_t {kGap = 1U, kCoverage = 2U, kOverlap = 3U};

struct UnsafeObject
{
  const RoundaboutGapResponseObjectInput * object{nullptr};
  UnsafeKind kind{UnsafeKind::kGap};
};

bool is_more_limiting(const UnsafeObject & lhs, const UnsafeObject & rhs)
{
  if (lhs.kind != rhs.kind) {
    return static_cast<unsigned>(lhs.kind) > static_cast<unsigned>(rhs.kind);
  }
  if (lhs.kind == UnsafeKind::kGap &&
    lhs.object->minimum_temporal_gap_s != rhs.object->minimum_temporal_gap_s)
  {
    return lhs.object->minimum_temporal_gap_s < rhs.object->minimum_temporal_gap_s;
  }
  return lhs.object->object_id < rhs.object->object_id;
}

}  // namespace

RoundaboutGapResponseParameters RoundaboutGapResponseParameters::validated() const
{
  require_positive(minimum_release_gap_s, "minimum_release_gap_s");
  require_nonnegative(entry_standoff_m, "entry_standoff_m");
  require_positive(comfortable_deceleration_mps2, "comfortable_deceleration_mps2");
  require_nonnegative(stopped_speed_threshold_mps, "stopped_speed_threshold_mps");
  if (maximum_relevant_objects == 0U) {
    throw std::invalid_argument("maximum_relevant_objects must be positive");
  }
  return *this;
}

RoundaboutGapResponseResult compute_roundabout_gap_response(
  const RoundaboutGapResponseInput & input,
  const RoundaboutGapResponseParameters & parameters)
{
  const auto config = parameters.validated();
  const auto finite = [](const double value) {return std::isfinite(value);};
  if (!finite(input.ego_entry_time_s) || !finite(input.ego_exit_time_s) ||
    !finite(input.ego_route_distance_to_entry_m) ||
    !finite(input.ego_route_distance_to_exit_m) || !finite(input.ego_speed_mps))
  {
    throw std::invalid_argument("ego response facts must be finite");
  }
  if (input.ego_speed_mps < 0.0) {
    throw std::invalid_argument("ego speed must be nonnegative");
  }
  if (input.ego_entry_valid && input.ego_entry_time_s < 0.0) {
    throw std::invalid_argument("valid ego entry time must be nonnegative");
  }
  if (input.ego_exit_valid && input.ego_exit_time_s < 0.0) {
    throw std::invalid_argument("valid ego exit time must be nonnegative");
  }
  if (input.ego_entry_valid && input.ego_exit_valid &&
    input.ego_exit_time_s < input.ego_entry_time_s)
  {
    throw std::invalid_argument("ego exit must not precede ego entry");
  }

  RoundaboutGapResponseResult result;
  result.ego_speed_mps = std::max(0.0, input.ego_speed_mps);
  result.ego_route_distance_to_entry_m = input.ego_route_distance_to_entry_m;
  result.available_distance_m = std::max(
    0.0, input.ego_route_distance_to_entry_m - config.entry_standoff_m);
  result.comfortable_stop_distance_m = result.ego_speed_mps * result.ego_speed_mps /
    (2.0 * config.comfortable_deceleration_mps2);
  const std::size_t admitted = std::min(
    input.relevant_objects.size(), config.maximum_relevant_objects);
  result.relevant_object_count = admitted;
  result.rejected_over_budget = input.relevant_objects.size() - admitted;

  // Entry gating no longer applies after entry has begun or this traversal has
  // cleared the configured route exit. Never advise HOLD inside the conflict.
  if (input.ego_in_conflict_now || input.ego_route_distance_to_exit_m < 0.0) {
    return result;
  }

  result.active = true;
  result.complete_prediction_coverage = true;

  const bool timing_valid = input.ego_entry_valid && input.ego_exit_valid;
  const bool stopped = result.ego_speed_mps < config.stopped_speed_threshold_mps;
  std::vector<UnsafeObject> unsafe;
  unsafe.reserve(admitted);
  for (std::size_t index = 0U; index < admitted; ++index) {
    const auto & object = input.relevant_objects[index];
    if (!std::isfinite(object.minimum_temporal_gap_s) ||
      (object.minimum_temporal_gap_valid && object.minimum_temporal_gap_s < 0.0))
    {
      throw std::invalid_argument("object minimum temporal gap is malformed");
    }
    result.conflict_overlap_present =
      result.conflict_overlap_present || object.any_occupancy_overlap;
    result.complete_prediction_coverage =
      result.complete_prediction_coverage && object.prediction_covers_ego_exit;
    if (object.any_occupancy_overlap) {
      unsafe.push_back(UnsafeObject{&object, UnsafeKind::kOverlap});
    } else if (!object.prediction_covers_ego_exit ||
      !object.minimum_temporal_gap_valid)
    {
      unsafe.push_back(UnsafeObject{&object, UnsafeKind::kCoverage});
    } else if (object.minimum_temporal_gap_s < config.minimum_release_gap_s) {
      unsafe.push_back(UnsafeObject{&object, UnsafeKind::kGap});
    }
  }

  if (timing_valid && !stopped && result.rejected_over_budget == 0U && unsafe.empty()) {
    result.action = RoundaboutGapResponseAction::kRelease;
    result.reason = RoundaboutGapResponseReason::kClearGap;
    return result;
  }

  if (!unsafe.empty()) {
    const auto limiting = *std::min_element(
      unsafe.begin(), unsafe.end(),
      [](const UnsafeObject & lhs, const UnsafeObject & rhs) {
        return is_more_limiting(lhs, rhs);
      });
    result.has_source = true;
    result.source_object_id = limiting.object->object_id;
    result.limiting_gap_valid = limiting.object->minimum_temporal_gap_valid;
    result.limiting_gap_s = result.limiting_gap_valid ?
      limiting.object->minimum_temporal_gap_s : 0.0;
    if (limiting.kind == UnsafeKind::kOverlap) {
      result.reason = RoundaboutGapResponseReason::kOverlap;
    } else if (limiting.kind == UnsafeKind::kCoverage) {
      result.reason = RoundaboutGapResponseReason::kInsufficientPrediction;
    } else {
      result.reason = RoundaboutGapResponseReason::kGapTooSmall;
    }
  } else if (!timing_valid || stopped || result.rejected_over_budget > 0U) {
    result.reason = (!timing_valid || stopped) ? RoundaboutGapResponseReason::kInvalidEgoState :
      RoundaboutGapResponseReason::kInsufficientPrediction;
    result.complete_prediction_coverage = false;
  }

  // A stopped ego with no valid interval is held conservatively. Otherwise the
  // strict '>' boundary preserves the final comfortable stopping margin.
  result.action = (!stopped &&
    result.available_distance_m > result.comfortable_stop_distance_m) ?
    RoundaboutGapResponseAction::kYield : RoundaboutGapResponseAction::kHold;
  return result;
}

}  // namespace ad_planner
