#ifndef AD_LIDAR_PERCEPTION__PREPROCESSING__MOTION_HISTORY_HPP_
#define AD_LIDAR_PERCEPTION__PREPROCESSING__MOTION_HISTORY_HPP_

#include <array>
#include <cstddef>
#include <deque>
#include <string>
#include <vector>

namespace ad_lidar_perception::preprocessing
{

enum class MotionHistoryUpdate
{
  kAccepted,
  kReplaced,
  kEpochReset,
};

struct MotionControl
{
  double longitudinal_velocity_mps{0.0};
  std::array<double, 3> angular_velocity_rps{0.0, 0.0, 0.0};
};

enum class MotionCoverageStatus
{
  kCovered,
  kAwaitingFuture,
  kMissingPast,
  kExcessiveGap,
};

struct MotionCoverageResult
{
  MotionCoverageStatus status{MotionCoverageStatus::kAwaitingFuture};
  std::string reason;
};

class MotionHistory
{
public:
  explicit MotionHistory(double maximum_age_sec, std::size_t maximum_samples = 4096U);

  MotionHistoryUpdate add_wheel_sample(double stamp_sec, double longitudinal_velocity_mps);
  MotionHistoryUpdate add_imu_sample(
    double stamp_sec, const std::array<double, 3> & angular_velocity_rps);

  void clear() noexcept;
  [[nodiscard]] std::size_t wheel_sample_count() const noexcept;
  [[nodiscard]] std::size_t imu_sample_count() const noexcept;
  [[nodiscard]] MotionCoverageResult coverage(
    double start_sec, double end_sec, double maximum_wheel_gap_sec,
    double maximum_imu_gap_sec) const;
  [[nodiscard]] bool covers(
    double start_sec, double end_sec, double maximum_wheel_gap_sec,
    double maximum_imu_gap_sec, std::string * reason = nullptr) const;
  [[nodiscard]] MotionControl interpolate(double stamp_sec) const;
  [[nodiscard]] std::vector<double> knots(double start_sec, double end_sec) const;

private:
  struct WheelSample
  {
    double stamp_sec;
    double value;
  };
  struct ImuSample
  {
    double stamp_sec;
    std::array<double, 3> value;
  };

  double maximum_age_sec_;
  std::size_t maximum_samples_;
  std::deque<WheelSample> wheel_samples_;
  std::deque<ImuSample> imu_samples_;
};

}  // namespace ad_lidar_perception::preprocessing

#endif  // AD_LIDAR_PERCEPTION__PREPROCESSING__MOTION_HISTORY_HPP_
