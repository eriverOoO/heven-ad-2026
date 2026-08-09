#ifndef AD_LOCALIZATION__GNSS_IMU__GNSS_IMU_LOCALIZER_HPP_
#define AD_LOCALIZATION__GNSS_IMU__GNSS_IMU_LOCALIZER_HPP_

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"

namespace ad_localization
{

struct GnssImuLocalizerConfig
{
  std::array<double, 3> gnss_lever_arm_m{0.0, 0.0, 0.0};
  geometry_msgs::msg::Quaternion base_to_imu_orientation = [] {
      geometry_msgs::msg::Quaternion orientation;
      orientation.w = 1.0;
      return orientation;
    }();
  double synchronization_tolerance_sec{0.1};
  double gnss_xy_variance_m2{0.25};
  double gnss_z_variance_m2{0.25};
  double imu_orientation_variance_rad2{0.0001};
  double unobserved_twist_variance{1.0e6};
  double world_yaw_offset_rad{0.0};
  std::string reference_frame{"odom"};
  std::string base_frame{"base_link"};
  std::string imu_frame{"imu_link"};
};

class GnssImuLocalizer
{
public:
  explicit GnssImuLocalizer(GnssImuLocalizerConfig config);

  void reset();
  bool observe_imu(const sensor_msgs::msg::Imu & imu);
  bool observe_wheel_speed(
    const geometry_msgs::msg::TwistWithCovarianceStamped & wheel_speed);
  std::optional<nav_msgs::msg::Odometry> observe_gnss(
    const geometry_msgs::msg::PoseStamped & antenna_pose);

private:
  GnssImuLocalizerConfig config_;
  std::optional<sensor_msgs::msg::Imu> latest_imu_;
  std::optional<geometry_msgs::msg::TwistWithCovarianceStamped> latest_wheel_speed_;
  std::optional<std::int64_t> last_imu_stamp_ns_;
  std::optional<std::int64_t> last_wheel_stamp_ns_;
  std::optional<std::int64_t> last_output_stamp_ns_;
};

}  // namespace ad_localization

#endif  // AD_LOCALIZATION__GNSS_IMU__GNSS_IMU_LOCALIZER_HPP_
