#include "ad_localization/adapter/localization_adapter.hpp"

#include <proj.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "sensor_msgs/msg/nav_sat_status.hpp"

namespace ad_localization
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1000000000LL;
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

std::optional<std::int64_t> valid_stamp_ns(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond ||
    (stamp.sec == 0 && stamp.nanosec == 0))
  {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond + stamp.nanosec;
}

bool valid_frame(const std::string & frame)
{
  return !frame.empty() && frame.front() != '/' && frame.find_first_of(" \t\r\n") == std::string::npos;
}

bool finite_quaternion(const geometry_msgs::msg::Quaternion & quaternion)
{
  return std::isfinite(quaternion.x) && std::isfinite(quaternion.y) &&
         std::isfinite(quaternion.z) && std::isfinite(quaternion.w);
}

std::optional<geometry_msgs::msg::Quaternion> normalized_quaternion(
  const geometry_msgs::msg::Quaternion & quaternion)
{
  if (!finite_quaternion(quaternion)) {
    return std::nullopt;
  }
  const double norm_squared = quaternion.x * quaternion.x + quaternion.y * quaternion.y +
    quaternion.z * quaternion.z + quaternion.w * quaternion.w;
  if (!std::isfinite(norm_squared) || norm_squared < 1.0e-12) {
    return std::nullopt;
  }
  const double inverse_norm = 1.0 / std::sqrt(norm_squared);
  geometry_msgs::msg::Quaternion result;
  result.x = quaternion.x * inverse_norm;
  result.y = quaternion.y * inverse_norm;
  result.z = quaternion.z * inverse_norm;
  result.w = quaternion.w * inverse_norm;
  return result;
}

std::optional<geometry_msgs::msg::Quaternion> quaternion_from_rpy(
  double roll, double pitch, double yaw)
{
  if (!std::isfinite(roll) || !std::isfinite(pitch) || !std::isfinite(yaw)) {
    return std::nullopt;
  }

  const double half_roll = 0.5 * roll;
  const double half_pitch = 0.5 * pitch;
  const double half_yaw = 0.5 * yaw;
  const double cr = std::cos(half_roll);
  const double sr = std::sin(half_roll);
  const double cp = std::cos(half_pitch);
  const double sp = std::sin(half_pitch);
  const double cy = std::cos(half_yaw);
  const double sy = std::sin(half_yaw);

  geometry_msgs::msg::Quaternion quaternion;
  quaternion.w = cr * cp * cy + sr * sp * sy;
  quaternion.x = sr * cp * cy - cr * sp * sy;
  quaternion.y = cr * sp * cy + sr * cp * sy;
  quaternion.z = cr * cp * sy - sr * sp * cy;
  return normalized_quaternion(quaternion);
}

std::optional<geometry_msgs::msg::Quaternion> apply_world_yaw_offset(
  const geometry_msgs::msg::Quaternion & quaternion, double yaw_offset_rad)
{
  const auto normalized = normalized_quaternion(quaternion);
  if (!normalized || !std::isfinite(yaw_offset_rad)) {
    return std::nullopt;
  }
  const double half_offset = 0.5 * yaw_offset_rad;
  const double c = std::cos(half_offset);
  const double s = std::sin(half_offset);
  geometry_msgs::msg::Quaternion result;
  result.x = c * normalized->x - s * normalized->y;
  result.y = c * normalized->y + s * normalized->x;
  result.z = c * normalized->z + s * normalized->w;
  result.w = c * normalized->w - s * normalized->z;
  return normalized_quaternion(result);
}

std::array<double, 3> rotate_vector(
  const geometry_msgs::msg::Quaternion & quaternion,
  const std::array<double, 3> & vector)
{
  const double qx = quaternion.x;
  const double qy = quaternion.y;
  const double qz = quaternion.z;
  const double qw = quaternion.w;
  const double vx = vector[0];
  const double vy = vector[1];
  const double vz = vector[2];

  const double tx = 2.0 * (qy * vz - qz * vy);
  const double ty = 2.0 * (qz * vx - qx * vz);
  const double tz = 2.0 * (qx * vy - qy * vx);
  return {
    vx + qw * tx + (qy * tz - qz * ty),
    vy + qw * ty + (qz * tx - qx * tz),
    vz + qw * tz + (qx * ty - qy * tx),
  };
}

double normalized_yaw(double yaw)
{
  double wrapped = std::fmod(yaw + kPi, kTwoPi);
  if (wrapped < 0.0) {
    wrapped += kTwoPi;
  }
  return wrapped - kPi;
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & quaternion)
{
  const double sin_yaw = 2.0 *
    (quaternion.w * quaternion.z + quaternion.x * quaternion.y);
  const double cos_yaw = 1.0 - 2.0 *
    (quaternion.y * quaternion.y + quaternion.z * quaternion.z);
  return normalized_yaw(std::atan2(sin_yaw, cos_yaw));
}

void validate_config(const AdapterConfig & config)
{
  if (config.utm_epsg != 32652) {
    throw std::invalid_argument("utm_epsg must be 32652");
  }
  const bool finite_values =
    std::isfinite(config.easting_offset_m) && std::isfinite(config.northing_offset_m) &&
    std::isfinite(config.altitude_offset_m) && std::isfinite(config.maximum_map_radius_m) &&
    std::all_of(
      config.gnss_lever_arm_m.begin(), config.gnss_lever_arm_m.end(),
      [](double value) {return std::isfinite(value);}) &&
    std::isfinite(config.wheel_standstill_threshold_mps) &&
    std::isfinite(config.wheel_speed_variance) &&
    std::isfinite(config.wheel_lateral_speed_variance) &&
    std::isfinite(config.gps_course_maximum_age_sec) &&
    std::isfinite(config.gps_course_yaw_offset_rad) &&
    std::isfinite(config.maximum_abs_sideslip_rad) &&
    std::isfinite(config.initialization_sync_tolerance_sec) &&
    std::isfinite(config.initial_orientation_yaw_offset_rad) &&
    (!config.initial_position_override_xy_m ||
    std::all_of(
      config.initial_position_override_xy_m->begin(),
      config.initial_position_override_xy_m->end(),
      [](double value) {return std::isfinite(value);}));
  if (!finite_values) {
    throw std::invalid_argument("adapter configuration values must be finite");
  }
  if (config.maximum_map_radius_m <= 0.0) {
    throw std::invalid_argument("maximum map radius must be positive");
  }
  if (config.wheel_standstill_threshold_mps < 0.0) {
    throw std::invalid_argument("wheel standstill threshold must be nonnegative");
  }
  if (config.wheel_speed_variance <= 0.0) {
    throw std::invalid_argument("wheel speed variance must be positive");
  }
  if (config.wheel_lateral_speed_variance <= 0.0) {
    throw std::invalid_argument("wheel lateral speed variance must be positive");
  }
  if (config.gps_course_maximum_age_sec <= 0.0) {
    throw std::invalid_argument("GPS course maximum age must be positive");
  }
  if (config.maximum_abs_sideslip_rad <= 0.0 ||
    config.maximum_abs_sideslip_rad > kPi / 2.0)
  {
    throw std::invalid_argument("maximum absolute sideslip must be in (0, pi/2]");
  }
  if (config.initialization_sync_tolerance_sec < 0.0) {
    throw std::invalid_argument("initialization sync tolerance must be nonnegative");
  }
  if (config.initial_orientation_source != "imu" &&
    config.initial_orientation_source != "vehicle_status")
  {
    throw std::invalid_argument(
            "initial_orientation_source must be exactly imu or vehicle_status");
  }
  if (config.initial_position_sample_count <= 0) {
    throw std::invalid_argument("initial_position_sample_count must be positive");
  }
  if (!valid_frame(config.reference_frame) || !valid_frame(config.base_frame) ||
    config.reference_frame == config.base_frame)
  {
    throw std::invalid_argument("reference and base frames must be distinct relative frames");
  }
}

}  // namespace

std::optional<sensor_msgs::msg::Imu> apply_world_yaw_correction(
  const sensor_msgs::msg::Imu & imu, double yaw_offset_rad)
{
  const auto orientation = apply_world_yaw_offset(imu.orientation, yaw_offset_rad);
  if (!orientation) {
    return std::nullopt;
  }
  sensor_msgs::msg::Imu corrected = imu;
  corrected.orientation = *orientation;
  return corrected;
}

struct LocalizationAdapter::Projection
{
  struct ContextDeleter
  {
    void operator()(PJ_CONTEXT * context) const {proj_context_destroy(context);}
  };
  struct TransformDeleter
  {
    void operator()(PJ * transform) const {proj_destroy(transform);}
  };

  Projection()
  : context(proj_context_create())
  {
    if (!context) {
      throw std::runtime_error("failed to create PROJ context");
    }
    std::unique_ptr<PJ, TransformDeleter> raw_transform(
      proj_create_crs_to_crs(context.get(), "EPSG:4326", "EPSG:32652", nullptr));
    if (!raw_transform) {
      throw std::runtime_error("failed to create WGS84 to EPSG:32652 transform");
    }
    transform.reset(proj_normalize_for_visualization(context.get(), raw_transform.get()));
    if (!transform) {
      throw std::runtime_error("failed to normalize EPSG:32652 transform");
    }
  }

  std::unique_ptr<PJ_CONTEXT, ContextDeleter> context;
  std::unique_ptr<PJ, TransformDeleter> transform;
};

LocalizationAdapter::LocalizationAdapter(AdapterConfig config)
: config_(std::move(config))
{
  validate_config(config_);
  projection_ = std::make_unique<Projection>();
}

LocalizationAdapter::~LocalizationAdapter() = default;

void LocalizationAdapter::reset()
{
  last_gnss_stamp_ns_.reset();
  last_wheel_stamp_ns_.reset();
  last_gps_rmc_stamp_ns_.reset();
  last_odometry_stamp_ns_.reset();
  latest_gnss_pose_.reset();
  initial_position_sum_ = {0.0, 0.0, 0.0};
  initial_position_samples_ = 0;
  latest_imu_.reset();
  latest_vehicle_status_.reset();
  latest_gps_rmc_.reset();
  initial_pose_.reset();
  initialization_acknowledged_ = false;
}

std::optional<geometry_msgs::msg::PoseStamped> LocalizationAdapter::convert_gnss(
  const sensor_msgs::msg::NavSatFix & fix)
{
  const auto stamp_ns = valid_stamp_ns(fix.header.stamp);
  if (!stamp_ns || fix.status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX ||
    !std::isfinite(fix.latitude) || fix.latitude < -90.0 || fix.latitude > 90.0 ||
    !std::isfinite(fix.longitude) || fix.longitude < -180.0 || fix.longitude > 180.0 ||
    !std::isfinite(fix.altitude) ||
    (fix.latitude == 0.0 && fix.longitude == 0.0) ||
    (last_gnss_stamp_ns_ && *stamp_ns <= *last_gnss_stamp_ns_))
  {
    return std::nullopt;
  }

  proj_errno_reset(projection_->transform.get());
  const PJ_COORD input = proj_coord(fix.longitude, fix.latitude, fix.altitude, 0.0);
  const PJ_COORD output = proj_trans(projection_->transform.get(), PJ_FWD, input);
  if (proj_errno(projection_->transform.get()) != 0 ||
    !std::isfinite(output.xyz.x) || !std::isfinite(output.xyz.y) ||
    !std::isfinite(output.xyz.z))
  {
    return std::nullopt;
  }
  const double local_x = output.xyz.x - config_.easting_offset_m;
  const double local_y = output.xyz.y - config_.northing_offset_m;
  if (std::hypot(local_x, local_y) > config_.maximum_map_radius_m) {
    return std::nullopt;
  }

  geometry_msgs::msg::PoseStamped pose;
  pose.header.stamp = fix.header.stamp;
  pose.header.frame_id = config_.reference_frame;
  pose.pose.position.x = local_x;
  pose.pose.position.y = local_y;
  pose.pose.position.z = output.xyz.z - config_.altitude_offset_m;
  pose.pose.orientation.w = 1.0;
  last_gnss_stamp_ns_ = stamp_ns;
  return pose;
}

std::optional<geometry_msgs::msg::TwistWithCovarianceStamped>
LocalizationAdapter::convert_wheel_speed(
  const ad_morai_interfaces::msg::EgoVehicleStatus & status)
{
  const auto arrival_stamp_ns = valid_stamp_ns(status.header.stamp);
  const auto wheel_stamp_ns = config_.wheel_use_device_timestamp ?
    (status.has_device_stamp ? valid_stamp_ns(status.device_stamp) : std::nullopt) :
    arrival_stamp_ns;
  if (!arrival_stamp_ns || !wheel_stamp_ns || !std::isfinite(status.velocity.x) ||
    (last_wheel_stamp_ns_ && *wheel_stamp_ns <= *last_wheel_stamp_ns_))
  {
    return std::nullopt;
  }

  const double magnitude = std::abs(static_cast<double>(status.velocity.x));
  double signed_speed = 0.0;
  double lateral_speed = 0.0;
  if (magnitude > config_.wheel_standstill_threshold_mps) {
    bool reverse = false;
    if (status.gear == 2) {
      signed_speed = -magnitude;
      reverse = true;
    } else if (status.gear == 4 || status.gear == 5) {
      signed_speed = magnitude;
    } else {
      return std::nullopt;
    }

    if (config_.gps_course_enabled) {
      if (!latest_gps_rmc_ || !last_gps_rmc_stamp_ns_ || !std::isfinite(status.rpy.z)) {
        return std::nullopt;
      }
      const std::int64_t age_ns = *arrival_stamp_ns - *last_gps_rmc_stamp_ns_;
      const double age_sec =
        static_cast<double>(age_ns) / static_cast<double>(kNanosecondsPerSecond);
      if (age_ns < 0 || !std::isfinite(age_sec) ||
        age_sec > config_.gps_course_maximum_age_sec)
      {
        return std::nullopt;
      }

      const double track_rad = latest_gps_rmc_->track_degrees * kPi / 180.0;
      const double course_yaw = normalized_yaw(
        kPi / 2.0 - track_rad + config_.gps_course_yaw_offset_rad);
      const double vehicle_yaw = normalized_yaw(status.rpy.z);
      const double travel_axis_yaw = normalized_yaw(vehicle_yaw + (reverse ? kPi : 0.0));
      const double sideslip = normalized_yaw(course_yaw - travel_axis_yaw);
      if (std::abs(sideslip) > config_.maximum_abs_sideslip_rad) {
        return std::nullopt;
      }
      lateral_speed = magnitude * std::sin(normalized_yaw(course_yaw - vehicle_yaw));
    }
  }

  geometry_msgs::msg::TwistWithCovarianceStamped wheel;
  wheel.header.stamp = config_.wheel_use_device_timestamp ?
    status.device_stamp : status.header.stamp;
  wheel.header.frame_id = config_.base_frame;
  wheel.twist.twist.linear.x = signed_speed;
  wheel.twist.twist.linear.y = lateral_speed;
  wheel.twist.covariance.fill(0.0);
  wheel.twist.covariance[0] = config_.wheel_speed_variance;
  if (config_.gps_course_enabled) {
    wheel.twist.covariance[7] = config_.wheel_lateral_speed_variance;
  }
  last_wheel_stamp_ns_ = wheel_stamp_ns;
  return wheel;
}

bool LocalizationAdapter::observe_gps_rmc(const ad_morai_interfaces::msg::GpsRmc & rmc)
{
  const auto stamp_ns = valid_stamp_ns(rmc.header.stamp);
  if (!stamp_ns || !rmc.valid || !rmc.has_track ||
    !std::isfinite(rmc.track_degrees) || rmc.track_degrees < 0.0 ||
    rmc.track_degrees >= 360.0 ||
    (last_gps_rmc_stamp_ns_ && *stamp_ns <= *last_gps_rmc_stamp_ns_))
  {
    return false;
  }
  latest_gps_rmc_ = rmc;
  last_gps_rmc_stamp_ns_ = stamp_ns;
  return true;
}

std::optional<geometry_msgs::msg::PoseStamped>
LocalizationAdapter::observe_gnss_for_initialization(
  const geometry_msgs::msg::PoseStamped & gnss_pose)
{
  if (initial_pose_ || initialization_acknowledged_ ||
    !valid_stamp_ns(gnss_pose.header.stamp) ||
    gnss_pose.header.frame_id != config_.reference_frame ||
    !std::isfinite(gnss_pose.pose.position.x) ||
    !std::isfinite(gnss_pose.pose.position.y) ||
    !std::isfinite(gnss_pose.pose.position.z))
  {
    return std::nullopt;
  }
  const std::array<double, 3> updated_sum{
    initial_position_sum_[0] + gnss_pose.pose.position.x,
    initial_position_sum_[1] + gnss_pose.pose.position.y,
    initial_position_sum_[2] + gnss_pose.pose.position.z,
  };
  if (!std::all_of(
      updated_sum.begin(), updated_sum.end(),
      [](double value) {return std::isfinite(value);}))
  {
    return std::nullopt;
  }
  initial_position_sum_ = updated_sum;
  ++initial_position_samples_;
  latest_gnss_pose_ = gnss_pose;
  return try_build_initial_pose();
}

std::optional<geometry_msgs::msg::PoseStamped>
LocalizationAdapter::observe_imu_for_initialization(const sensor_msgs::msg::Imu & imu)
{
  if (initial_pose_ || initialization_acknowledged_ || !valid_stamp_ns(imu.header.stamp) ||
    !normalized_quaternion(imu.orientation))
  {
    return std::nullopt;
  }
  latest_imu_ = imu;
  return try_build_initial_pose();
}

std::optional<geometry_msgs::msg::PoseStamped>
LocalizationAdapter::observe_vehicle_status_for_initialization(
  const ad_morai_interfaces::msg::EgoVehicleStatus & status)
{
  if (config_.initial_orientation_source != "vehicle_status" || initial_pose_ ||
    initialization_acknowledged_ || !valid_stamp_ns(status.header.stamp) ||
    !quaternion_from_rpy(status.rpy.x, status.rpy.y, status.rpy.z))
  {
    return std::nullopt;
  }
  latest_vehicle_status_ = status;
  return try_build_initial_pose();
}

std::optional<geometry_msgs::msg::PoseStamped> LocalizationAdapter::try_build_initial_pose()
{
  if (!latest_gnss_pose_ ||
    initial_position_samples_ < static_cast<std::size_t>(config_.initial_position_sample_count) ||
    initial_pose_ || initialization_acknowledged_)
  {
    return std::nullopt;
  }
  const auto gnss_stamp_ns = valid_stamp_ns(latest_gnss_pose_->header.stamp);
  std::optional<std::int64_t> orientation_stamp_ns;
  std::optional<geometry_msgs::msg::Quaternion> orientation;
  if (config_.initial_orientation_source == "imu") {
    if (!latest_imu_) {
      return std::nullopt;
    }
    orientation_stamp_ns = valid_stamp_ns(latest_imu_->header.stamp);
    orientation = normalized_quaternion(latest_imu_->orientation);
  } else {
    if (!latest_vehicle_status_) {
      return std::nullopt;
    }
    orientation_stamp_ns = valid_stamp_ns(latest_vehicle_status_->header.stamp);
    orientation = quaternion_from_rpy(
      latest_vehicle_status_->rpy.x,
      latest_vehicle_status_->rpy.y,
      latest_vehicle_status_->rpy.z);
  }
  if (!gnss_stamp_ns || !orientation_stamp_ns || !orientation) {
    return std::nullopt;
  }
  orientation = apply_world_yaw_offset(
    *orientation, config_.initial_orientation_yaw_offset_rad);
  if (!orientation) {
    return std::nullopt;
  }
  const double difference_seconds =
    std::abs(static_cast<double>(*gnss_stamp_ns - *orientation_stamp_ns)) /
    static_cast<double>(kNanosecondsPerSecond);
  if (difference_seconds > config_.initialization_sync_tolerance_sec) {
    return std::nullopt;
  }
  const auto rotated_lever_arm = rotate_vector(*orientation, config_.gnss_lever_arm_m);

  geometry_msgs::msg::PoseStamped body_pose;
  body_pose.header = latest_gnss_pose_->header;
  const double sample_count = static_cast<double>(initial_position_samples_);
  body_pose.pose.position.x = initial_position_sum_[0] / sample_count - rotated_lever_arm[0];
  body_pose.pose.position.y = initial_position_sum_[1] / sample_count - rotated_lever_arm[1];
  body_pose.pose.position.z = initial_position_sum_[2] / sample_count - rotated_lever_arm[2];
  if (config_.initial_position_override_xy_m) {
    body_pose.pose.position.x = (*config_.initial_position_override_xy_m)[0];
    body_pose.pose.position.y = (*config_.initial_position_override_xy_m)[1];
  }
  body_pose.pose.orientation = *orientation;
  initial_pose_ = body_pose;
  return body_pose;
}

std::optional<geometry_msgs::msg::PoseStamped> LocalizationAdapter::pending_initial_pose() const
{
  if (initialization_acknowledged_) {
    return std::nullopt;
  }
  return initial_pose_;
}

void LocalizationAdapter::acknowledge_filtered_odometry()
{
  initialization_acknowledged_ = true;
  initial_pose_.reset();
  latest_gnss_pose_.reset();
  latest_imu_.reset();
  latest_vehicle_status_.reset();
  initial_position_sum_ = {0.0, 0.0, 0.0};
  initial_position_samples_ = 0;
}

std::optional<geometry_msgs::msg::Pose2D> LocalizationAdapter::convert_filtered_odometry(
  const nav_msgs::msg::Odometry & odometry)
{
  const auto stamp_ns = valid_stamp_ns(odometry.header.stamp);
  const auto orientation = normalized_quaternion(odometry.pose.pose.orientation);
  if (!stamp_ns || (last_odometry_stamp_ns_ && *stamp_ns <= *last_odometry_stamp_ns_) ||
    odometry.header.frame_id != config_.reference_frame ||
    odometry.child_frame_id != config_.base_frame ||
    !std::isfinite(odometry.pose.pose.position.x) ||
    !std::isfinite(odometry.pose.pose.position.y) ||
    !std::isfinite(odometry.pose.pose.position.z) || !orientation)
  {
    return std::nullopt;
  }

  geometry_msgs::msg::Pose2D pose;
  pose.x = odometry.pose.pose.position.x;
  pose.y = odometry.pose.pose.position.y;
  pose.theta = yaw_from_quaternion(*orientation);
  last_odometry_stamp_ns_ = stamp_ns;
  return pose;
}

}  // namespace ad_localization
