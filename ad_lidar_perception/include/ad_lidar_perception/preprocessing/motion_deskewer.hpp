#ifndef AD_LIDAR_PERCEPTION__PREPROCESSING__MOTION_DESKEWER_HPP_
#define AD_LIDAR_PERCEPTION__PREPROCESSING__MOTION_DESKEWER_HPP_

#include "ad_lidar_perception/preprocessing/motion_history.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>

#include <builtin_interfaces/msg/time.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace ad_lidar_perception::preprocessing
{

struct RigidTransform3d
{
  std::array<double, 9> rotation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::array<double, 3> translation{0.0, 0.0, 0.0};
};

std::int64_t ros_stamp_nanoseconds(const builtin_interfaces::msg::Time & stamp);
double ros_stamp_seconds(const builtin_interfaces::msg::Time & stamp);

RigidTransform3d rigid_transform_from_quaternion(
  const std::array<double, 3> & translation,
  const std::array<double, 4> & quaternion_xyzw);

enum class DeskewMode
{
  kThreeDimensional,
  kTwoDimensional,
};

struct MotionDeskewOptions
{
  DeskewMode mode{DeskewMode::kThreeDimensional};
  double maximum_scan_duration_sec{0.20};
  std::size_t maximum_point_count{300000U};
  double maximum_imu_gap_sec{0.12};
  double maximum_wheel_gap_sec{0.20};
  double maximum_integration_step_sec{0.005};
};

enum class MotionDeskewRetryability
{
  kNotApplicable,
  kRetryable,
  kPermanent,
};

struct MotionDeskewResult
{
  std::optional<sensor_msgs::msg::PointCloud2> cloud;
  MotionDeskewRetryability retryability{MotionDeskewRetryability::kPermanent};
  std::string error;
};

enum class PendingDeskewAction
{
  kPublish,
  kRetry,
  kDrop,
};

PendingDeskewAction pending_deskew_action(const MotionDeskewResult & result) noexcept;

MotionDeskewResult deskew_xyzirt_cloud(
  const sensor_msgs::msg::PointCloud2 & input, const MotionHistory & history,
  const RigidTransform3d & base_from_lidar,
  const MotionDeskewOptions & options = MotionDeskewOptions{});

}  // namespace ad_lidar_perception::preprocessing

#endif  // AD_LIDAR_PERCEPTION__PREPROCESSING__MOTION_DESKEWER_HPP_
