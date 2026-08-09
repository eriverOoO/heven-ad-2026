#include "ad_lidar_perception/preprocessing/motion_history.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace ad_lidar_perception::preprocessing
{
namespace
{

constexpr double kComparisonToleranceSec = 1.0e-8;

void require_finite_nonnegative(const double value, const char * label)
{
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(label) + " must be finite and nonnegative");
  }
}

template<typename Sample>
void prune(
  std::deque<Sample> & samples, const double maximum_age_sec,
  const std::size_t maximum_samples)
{
  const auto newest = samples.back().stamp_sec;
  while (
    samples.size() > 1U &&
    newest - samples.front().stamp_sec > maximum_age_sec + kComparisonToleranceSec)
  {
    samples.pop_front();
  }
  while (samples.size() > maximum_samples) {
    samples.pop_front();
  }
}

template<typename Sample>
MotionCoverageResult series_coverage(
  const std::deque<Sample> & samples, const double start_sec, const double end_sec,
  const double maximum_gap_sec, const char * label)
{
  if (samples.empty()) {
    return {
      MotionCoverageStatus::kAwaitingFuture,
      std::string(label) + " history is awaiting samples",
    };
  }
  if (samples.front().stamp_sec > start_sec) {
    return {
      MotionCoverageStatus::kMissingPast,
      std::string(label) + " history is missing the scan-start bracket",
    };
  }
  for (std::size_t index = 1U; index < samples.size(); ++index) {
    const auto left = samples[index - 1U].stamp_sec;
    const auto right = samples[index].stamp_sec;
    if (right <= start_sec || left >= end_sec) {
      continue;
    }
    if (right - left > maximum_gap_sec + kComparisonToleranceSec) {
      return {
        MotionCoverageStatus::kExcessiveGap,
        std::string(label) + " interpolation gap exceeds configured maximum",
      };
    }
  }
  if (samples.back().stamp_sec < end_sec) {
    return {
      MotionCoverageStatus::kAwaitingFuture,
      std::string(label) + " history is awaiting the scan-end bracket",
    };
  }
  return {MotionCoverageStatus::kCovered, ""};
}

template<typename Sample, typename Value, typename Interpolate>
Value interpolate_series(
  const std::deque<Sample> & samples, const double stamp_sec,
  const char * label, Interpolate interpolate)
{
  const auto right = std::lower_bound(
    samples.begin(), samples.end(), stamp_sec,
    [](const Sample & sample, const double stamp) {
      return sample.stamp_sec < stamp;
    });
  if (right != samples.end() && right->stamp_sec == stamp_sec) {
    return right->value;
  }
  if (right == samples.begin() || right == samples.end()) {
    throw std::out_of_range(std::string(label) + " interpolation would extrapolate");
  }
  const auto & before = *std::prev(right);
  const auto ratio =
    (stamp_sec - before.stamp_sec) / (right->stamp_sec - before.stamp_sec);
  return interpolate(before.value, right->value, ratio);
}

}  // namespace

MotionHistory::MotionHistory(
  const double maximum_age_sec, const std::size_t maximum_samples)
: maximum_age_sec_(maximum_age_sec), maximum_samples_(maximum_samples)
{
  if (!std::isfinite(maximum_age_sec_) || maximum_age_sec_ <= 0.0) {
    throw std::invalid_argument("maximum history age must be finite and positive");
  }
  if (maximum_samples_ == 0U) {
    throw std::invalid_argument("maximum history samples must be positive");
  }
}

MotionHistoryUpdate MotionHistory::add_wheel_sample(
  const double stamp_sec, const double longitudinal_velocity_mps)
{
  require_finite_nonnegative(stamp_sec, "wheel timestamp");
  if (!std::isfinite(longitudinal_velocity_mps)) {
    throw std::invalid_argument("wheel velocity must be finite");
  }
  if (!wheel_samples_.empty() && stamp_sec < wheel_samples_.back().stamp_sec) {
    clear();
    wheel_samples_.push_back({stamp_sec, longitudinal_velocity_mps});
    return MotionHistoryUpdate::kEpochReset;
  }
  if (!wheel_samples_.empty() && stamp_sec == wheel_samples_.back().stamp_sec) {
    wheel_samples_.back().value = longitudinal_velocity_mps;
    return MotionHistoryUpdate::kReplaced;
  }
  wheel_samples_.push_back({stamp_sec, longitudinal_velocity_mps});
  prune(wheel_samples_, maximum_age_sec_, maximum_samples_);
  return MotionHistoryUpdate::kAccepted;
}

MotionHistoryUpdate MotionHistory::add_imu_sample(
  const double stamp_sec, const std::array<double, 3> & angular_velocity_rps)
{
  require_finite_nonnegative(stamp_sec, "IMU timestamp");
  if (!std::all_of(
      angular_velocity_rps.begin(), angular_velocity_rps.end(),
      [](const double value) {return std::isfinite(value);} ))
  {
    throw std::invalid_argument("IMU angular velocity must be finite");
  }
  if (!imu_samples_.empty() && stamp_sec < imu_samples_.back().stamp_sec) {
    clear();
    imu_samples_.push_back({stamp_sec, angular_velocity_rps});
    return MotionHistoryUpdate::kEpochReset;
  }
  if (!imu_samples_.empty() && stamp_sec == imu_samples_.back().stamp_sec) {
    imu_samples_.back().value = angular_velocity_rps;
    return MotionHistoryUpdate::kReplaced;
  }
  imu_samples_.push_back({stamp_sec, angular_velocity_rps});
  prune(imu_samples_, maximum_age_sec_, maximum_samples_);
  return MotionHistoryUpdate::kAccepted;
}

void MotionHistory::clear() noexcept
{
  wheel_samples_.clear();
  imu_samples_.clear();
}

std::size_t MotionHistory::wheel_sample_count() const noexcept
{
  return wheel_samples_.size();
}

std::size_t MotionHistory::imu_sample_count() const noexcept
{
  return imu_samples_.size();
}

bool MotionHistory::covers(
  const double start_sec, const double end_sec,
  const double maximum_wheel_gap_sec, const double maximum_imu_gap_sec,
  std::string * reason) const
{
  const auto result = coverage(
    start_sec, end_sec, maximum_wheel_gap_sec, maximum_imu_gap_sec);
  if (reason != nullptr) {
    *reason = result.reason;
  }
  return result.status == MotionCoverageStatus::kCovered;
}

MotionCoverageResult MotionHistory::coverage(
  const double start_sec, const double end_sec,
  const double maximum_wheel_gap_sec, const double maximum_imu_gap_sec) const
{
  require_finite_nonnegative(start_sec, "scan start");
  require_finite_nonnegative(end_sec, "scan end");
  if (end_sec < start_sec) {
    throw std::invalid_argument("scan end must not precede scan start");
  }
  if (!std::isfinite(maximum_wheel_gap_sec) || maximum_wheel_gap_sec <= 0.0 ||
    !std::isfinite(maximum_imu_gap_sec) || maximum_imu_gap_sec <= 0.0)
  {
    throw std::invalid_argument("motion interpolation gaps must be finite and positive");
  }
  const auto wheel = series_coverage(
    wheel_samples_, start_sec, end_sec, maximum_wheel_gap_sec, "wheel");
  const auto imu = series_coverage(
    imu_samples_, start_sec, end_sec, maximum_imu_gap_sec, "IMU");
  const auto permanent = [](const MotionCoverageStatus status) {
      return status == MotionCoverageStatus::kMissingPast ||
             status == MotionCoverageStatus::kExcessiveGap;
    };
  if (permanent(wheel.status)) {
    return wheel;
  }
  if (permanent(imu.status)) {
    return imu;
  }
  if (wheel.status != MotionCoverageStatus::kCovered) {
    return wheel;
  }
  return imu;
}

MotionControl MotionHistory::interpolate(const double stamp_sec) const
{
  require_finite_nonnegative(stamp_sec, "interpolation timestamp");
  MotionControl result;
  result.longitudinal_velocity_mps = interpolate_series<WheelSample, double>(
    wheel_samples_, stamp_sec, "wheel",
    [](const double before, const double after, const double ratio) {
      return before + ratio * (after - before);
    });
  result.angular_velocity_rps =
    interpolate_series<ImuSample, std::array<double, 3>>(
    imu_samples_, stamp_sec, "IMU",
    [](const std::array<double, 3> & before, const std::array<double, 3> & after,
      const double ratio)
    {
      return std::array<double, 3>{
        before[0] + ratio * (after[0] - before[0]),
        before[1] + ratio * (after[1] - before[1]),
        before[2] + ratio * (after[2] - before[2]),
      };
    });
  return result;
}

std::vector<double> MotionHistory::knots(
  const double start_sec, const double end_sec) const
{
  std::vector<double> result{start_sec, end_sec};
  for (const auto & sample : wheel_samples_) {
    if (sample.stamp_sec > start_sec && sample.stamp_sec < end_sec) {
      result.push_back(sample.stamp_sec);
    }
  }
  for (const auto & sample : imu_samples_) {
    if (sample.stamp_sec > start_sec && sample.stamp_sec < end_sec) {
      result.push_back(sample.stamp_sec);
    }
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

}  // namespace ad_lidar_perception::preprocessing
