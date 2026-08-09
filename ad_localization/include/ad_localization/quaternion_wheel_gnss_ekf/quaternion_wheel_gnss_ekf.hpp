#ifndef AD_LOCALIZATION__QUATERNION_WHEEL_GNSS_EKF__QUATERNION_WHEEL_GNSS_EKF_HPP_
#define AD_LOCALIZATION__QUATERNION_WHEEL_GNSS_EKF__QUATERNION_WHEEL_GNSS_EKF_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"

namespace ad_localization
{

struct QuaternionWheelGnssEkfConfig
{
  std::string reference_frame{"odom"};
  std::string base_frame{"base_link"};
  std::string imu_frame{"imu_link"};
  std::array<double, 3> gnss_lever_arm_m{0.0, 0.0, 0.0};
  geometry_msgs::msg::Quaternion base_to_imu_orientation = [] {
      geometry_msgs::msg::Quaternion orientation;
      orientation.w = 1.0;
      return orientation;
    }();
  double world_yaw_offset_rad{0.0};
  double maximum_imu_age_sec{0.1};
  double maximum_prediction_dt_sec{0.5};
  int initialization_sample_count{20};
  double initial_position_variance_m2{1.0};
  double initial_wheel_bias_mps{0.0};
  double initial_wheel_bias_variance_m2ps2{0.25};
  double wheel_speed_variance_floor_m2ps2{0.001};
  double wheel_bias_random_walk_variance_m2ps3{0.01};
  double gnss_variance_m2{9.0};
  double gnss_mahalanobis_threshold{9.21};
  double teleport_distance_m{8.0};
  int teleport_confirmation_samples{3};
  double teleport_candidate_radius_m{4.0};
  double teleport_max_interval_sec{0.5};
  double fixed_output_z_m{0.0};
  double unobserved_variance{1.0e6};
  double orientation_variance_rad2{0.0001};
};

struct QuaternionWheelGnssEkfState
{
  bool initialized{false};
  std::array<double, 3> value{0.0, 0.0, 0.0};
  std::array<double, 9> covariance{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

class QuaternionWheelGnssEkf
{
public:
  explicit QuaternionWheelGnssEkf(QuaternionWheelGnssEkfConfig config);

  void reset() noexcept;
  bool observe_imu(const sensor_msgs::msg::Imu & imu);
  std::optional<nav_msgs::msg::Odometry> observe_wheel_speed(
    const geometry_msgs::msg::TwistWithCovarianceStamped & wheel);
  std::optional<nav_msgs::msg::Odometry> observe_gnss(
    const geometry_msgs::msg::PoseStamped & antenna_pose);
  QuaternionWheelGnssEkfState state() const noexcept;

private:
  struct OrientationSample
  {
    std::int64_t stamp_ns{};
    geometry_msgs::msg::Quaternion orientation;
  };

  struct WheelControl
  {
    std::int64_t stamp_ns{};
    double longitudinal_speed_mps{};
    double variance_m2ps2{};
  };

  std::optional<OrientationSample> causal_orientation_at(
    std::int64_t target_stamp_ns) const;
  bool predict_to(
    std::int64_t target_stamp_ns,
    const geometry_msgs::msg::Quaternion & orientation,
    const WheelControl & wheel_control);
  std::optional<nav_msgs::msg::Odometry> make_output(
    std::int64_t stamp_ns,
    const geometry_msgs::msg::Quaternion & orientation);
  void initialize_state(double x, double y) noexcept;
  void clear_initialization() noexcept;
  void clear_teleport_candidate() noexcept;
  std::optional<nav_msgs::msg::Odometry> consider_teleport(
    std::int64_t stamp_ns, double body_x, double body_y,
    const geometry_msgs::msg::Quaternion & orientation);

  QuaternionWheelGnssEkfConfig config_;
  QuaternionWheelGnssEkfState state_;
  std::deque<OrientationSample> orientation_history_;
  std::optional<std::int64_t> latest_imu_stamp_ns_;
  std::optional<std::int64_t> latest_wheel_stamp_ns_;
  std::optional<std::int64_t> latest_gnss_stamp_ns_;
  std::optional<std::int64_t> prediction_stamp_ns_;
  std::optional<std::int64_t> last_output_stamp_ns_;
  std::optional<WheelControl> held_wheel_control_;
  std::array<double, 2> initialization_sum_{0.0, 0.0};
  std::size_t initialization_samples_{0U};
  std::optional<std::int64_t> teleport_candidate_stamp_ns_;
  std::array<double, 2> teleport_candidate_anchor_{0.0, 0.0};
  std::array<double, 2> teleport_candidate_sum_{0.0, 0.0};
  std::size_t teleport_candidate_samples_{0U};
};

}  // namespace ad_localization

#endif  // AD_LOCALIZATION__QUATERNION_WHEEL_GNSS_EKF__QUATERNION_WHEEL_GNSS_EKF_HPP_
