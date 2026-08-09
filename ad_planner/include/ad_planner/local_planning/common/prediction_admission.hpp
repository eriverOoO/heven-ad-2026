#ifndef AD_PLANNER__LOCAL_PLANNING__PREDICTION_ADMISSION_HPP_
#define AD_PLANNER__LOCAL_PLANNING__PREDICTION_ADMISSION_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

enum class PredictionMode
{
  kDisabled,
  kRequired,
};

struct PredictionAdmissionInput
{
  bool received{false};
  bool valid{false};
  double receipt_age_s{0.0};
  double stamp_age_s{0.0};
  double timeout_s{0.0};
  double maximum_future_skew_s{0.0};
};

struct PredictionAdmission
{
  bool admitted{false};
  bool use_predictions{false};
  std::string reason;
};

struct PredictionSnapshot
{
  std::int64_t stamp_ns{0};
  double receipt_time_s{0.0};
  PredictedObjectSet objects;
};

enum class PredictionSnapshotInsertResult
{
  kInserted,
  kReplaced,
  kClockRollback,
};

struct PredictionSnapshotSelection
{
  bool admitted{false};
  bool use_predictions{false};
  double age_s{0.0};
  PredictionSnapshot snapshot;
  std::string reason;
};

class PredictionSnapshotHistory
{
public:
  PredictionSnapshotHistory(
    std::size_t maximum_snapshots,
    std::int64_t clock_rollback_threshold_ns);

  PredictionSnapshotInsertResult insert(PredictionSnapshot snapshot);
  PredictionSnapshotSelection select(
    PredictionMode mode, std::int64_t odometry_stamp_ns,
    double current_receipt_time_s, double timeout_s) const;
  void clear() noexcept;
  std::size_t size() const noexcept;

private:
  std::size_t maximum_snapshots_;
  std::int64_t clock_rollback_threshold_ns_;
  std::int64_t latest_arrival_stamp_ns_{0};
  std::vector<PredictionSnapshot> snapshots_;
};

PredictionMode parse_prediction_mode(const std::string & value);
const char * prediction_mode_name(PredictionMode mode) noexcept;
PredictionAdmission admit_predictions(
  PredictionMode mode, const PredictionAdmissionInput & input);

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__PREDICTION_ADMISSION_HPP_
