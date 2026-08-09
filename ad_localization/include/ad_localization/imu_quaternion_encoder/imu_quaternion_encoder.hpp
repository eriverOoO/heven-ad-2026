#ifndef AD_LOCALIZATION__IMU_QUATERNION_ENCODER__IMU_QUATERNION_ENCODER_HPP_
#define AD_LOCALIZATION__IMU_QUATERNION_ENCODER__IMU_QUATERNION_ENCODER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>

#include "ad_morai_interfaces/msg/ego_vehicle_status.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/quaternion.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"

namespace ad_localization
{

enum class ImuQuaternionEncoderMode
{
  kStatusPose,
  kDeadReckoning,
};

ImuQuaternionEncoderMode parse_imu_quaternion_encoder_mode(
  const std::string & value);

struct ImuQuaternionEncoderConfig
{
  ImuQuaternionEncoderMode mode{ImuQuaternionEncoderMode::kStatusPose};
  std::string status_frame{"map"};
  std::string reference_frame{"odom"};
  std::string base_frame{"base_link"};
  std::string imu_frame{"imu_link"};
  std::array<double, 3> status_origin_to_base_m{0.0, 0.0, 0.0};
  std::array<double, 3> gnss_lever_arm_m{0.0, 0.0, 0.0};
  bool reject_zero_status_position{true};
  geometry_msgs::msg::Quaternion base_to_imu_orientation = [] {
      geometry_msgs::msg::Quaternion orientation;
      orientation.w = 1.0;
      return orientation;
    }();
  double world_yaw_offset_rad{0.0};
  double maximum_imu_age_sec{0.1};
  double maximum_integration_dt_sec{0.5};
  int initial_seed_sample_count{1};
  bool automatic_reseed_enabled{false};
  double automatic_reseed_distance_m{8.0};
  int automatic_reseed_confirmation_samples{3};
  double automatic_reseed_candidate_radius_m{4.0};
  double automatic_reseed_max_interval_sec{0.5};
  double position_variance_m2{0.01};
  double orientation_variance_rad2{0.0001};
  double speed_variance_m2ps2{0.04};
};

class ImuQuaternionEncoder
{
public:
  explicit ImuQuaternionEncoder(ImuQuaternionEncoderConfig config);

  void reset() noexcept;
  bool observe_imu(const sensor_msgs::msg::Imu & imu);
  bool observe_gnss_seed(const geometry_msgs::msg::PoseStamped & seed);
  std::optional<nav_msgs::msg::Odometry> observe_status(
    const ad_morai_interfaces::msg::EgoVehicleStatus & status);

private:
  std::optional<sensor_msgs::msg::Imu> causal_imu_at(
    std::int64_t target_stamp_ns) const;
  std::optional<geometry_msgs::msg::Quaternion> world_base_orientation(
    const sensor_msgs::msg::Imu & imu) const;
  bool consider_automatic_reseed(
    const geometry_msgs::msg::PoseStamped & seed,
    std::int64_t seed_stamp_ns);
  bool accumulate_initial_seed(
    const geometry_msgs::msg::PoseStamped & seed);
  void clear_initial_seed_accumulator() noexcept;
  void clear_automatic_reseed_candidate() noexcept;
  void try_initialize_pending_seed();

  ImuQuaternionEncoderConfig config_;
  std::deque<sensor_msgs::msg::Imu> imu_history_;
  std::optional<std::int64_t> latest_imu_stamp_ns_;
  std::optional<std::int64_t> latest_seed_stamp_ns_;
  std::optional<std::int64_t> last_status_stamp_ns_;
  std::optional<std::int64_t> integration_stamp_ns_;
  std::optional<geometry_msgs::msg::PoseStamped> pending_seed_;
  std::optional<std::array<double, 3>> dead_reckoning_position_;
  std::array<double, 3> initial_seed_sum_{0.0, 0.0, 0.0};
  std::size_t initial_seed_samples_{0U};
  std::optional<geometry_msgs::msg::PoseStamped> automatic_reseed_candidate_;
  std::array<double, 3> automatic_reseed_candidate_sum_{0.0, 0.0, 0.0};
  std::size_t automatic_reseed_candidate_samples_{0U};
};

}  // namespace ad_localization

#endif  // AD_LOCALIZATION__IMU_QUATERNION_ENCODER__IMU_QUATERNION_ENCODER_HPP_
