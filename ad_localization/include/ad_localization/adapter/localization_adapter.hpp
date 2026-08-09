#ifndef AD_LOCALIZATION__ADAPTER__LOCALIZATION_ADAPTER_HPP_
#define AD_LOCALIZATION__ADAPTER__LOCALIZATION_ADAPTER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "ad_morai_interfaces/msg/ego_vehicle_status.hpp"
#include "ad_morai_interfaces/msg/gps_rmc.hpp"
#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"

namespace ad_localization
{

struct AdapterConfig
{
  int utm_epsg{32652};
  double easting_offset_m{302595.0};
  double northing_offset_m{4124145.0};
  double altitude_offset_m{0.0};
  double maximum_map_radius_m{100000.0};
  std::array<double, 3> gnss_lever_arm_m{0.0, 0.0, 1.5685};
  double wheel_standstill_threshold_mps{0.05};
  double wheel_speed_variance{0.04};
  double wheel_lateral_speed_variance{0.04};
  bool wheel_use_device_timestamp{false};
  bool gps_course_enabled{false};
  double gps_course_maximum_age_sec{0.25};
  double gps_course_yaw_offset_rad{0.0};
  double maximum_abs_sideslip_rad{0.35};
  double initialization_sync_tolerance_sec{0.25};
  std::string initial_orientation_source{"imu"};
  double initial_orientation_yaw_offset_rad{0.0};
  int initial_position_sample_count{1};
  std::optional<std::array<double, 2>> initial_position_override_xy_m;
  std::string reference_frame{"odom"};
  std::string base_frame{"base_link"};
};

std::optional<sensor_msgs::msg::Imu> apply_world_yaw_correction(
  const sensor_msgs::msg::Imu & imu, double yaw_offset_rad);

class LocalizationAdapter
{
public:
  explicit LocalizationAdapter(AdapterConfig config);
  ~LocalizationAdapter();

  void reset();
  LocalizationAdapter(const LocalizationAdapter &) = delete;
  LocalizationAdapter & operator=(const LocalizationAdapter &) = delete;
  LocalizationAdapter(LocalizationAdapter &&) = delete;
  LocalizationAdapter & operator=(LocalizationAdapter &&) = delete;

  std::optional<geometry_msgs::msg::PoseStamped> convert_gnss(
    const sensor_msgs::msg::NavSatFix & fix);
  std::optional<geometry_msgs::msg::TwistWithCovarianceStamped> convert_wheel_speed(
    const ad_morai_interfaces::msg::EgoVehicleStatus & status);
  bool observe_gps_rmc(const ad_morai_interfaces::msg::GpsRmc & rmc);
  std::optional<geometry_msgs::msg::PoseStamped> observe_gnss_for_initialization(
    const geometry_msgs::msg::PoseStamped & gnss_pose);
  std::optional<geometry_msgs::msg::PoseStamped> observe_imu_for_initialization(
    const sensor_msgs::msg::Imu & imu);
  std::optional<geometry_msgs::msg::PoseStamped> observe_vehicle_status_for_initialization(
    const ad_morai_interfaces::msg::EgoVehicleStatus & status);
  std::optional<geometry_msgs::msg::PoseStamped> pending_initial_pose() const;
  void acknowledge_filtered_odometry();
  std::optional<geometry_msgs::msg::Pose2D> convert_filtered_odometry(
    const nav_msgs::msg::Odometry & odometry);

private:
  struct Projection;

  std::optional<geometry_msgs::msg::PoseStamped> try_build_initial_pose();

  AdapterConfig config_;
  std::unique_ptr<Projection> projection_;
  std::optional<std::int64_t> last_gnss_stamp_ns_;
  std::optional<std::int64_t> last_wheel_stamp_ns_;
  std::optional<std::int64_t> last_gps_rmc_stamp_ns_;
  std::optional<std::int64_t> last_odometry_stamp_ns_;
  std::optional<geometry_msgs::msg::PoseStamped> latest_gnss_pose_;
  std::array<double, 3> initial_position_sum_{0.0, 0.0, 0.0};
  std::size_t initial_position_samples_{0};
  std::optional<sensor_msgs::msg::Imu> latest_imu_;
  std::optional<ad_morai_interfaces::msg::EgoVehicleStatus> latest_vehicle_status_;
  std::optional<ad_morai_interfaces::msg::GpsRmc> latest_gps_rmc_;
  std::optional<geometry_msgs::msg::PoseStamped> initial_pose_;
  bool initialization_acknowledged_{false};
};

}  // namespace ad_localization

#endif  // AD_LOCALIZATION__ADAPTER__LOCALIZATION_ADAPTER_HPP_
