#include "ad_localization/quaternion_wheel_gnss_ekf/quaternion_wheel_gnss_ekf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ad_localization
{
namespace
{

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
constexpr std::size_t kMaximumOrientationHistorySamples = 512U;

bool finite(double value)
{
  return std::isfinite(value);
}

bool valid_frame(const std::string & frame)
{
  return !frame.empty() && frame.front() != '/' &&
         frame.find(' ') == std::string::npos;
}

std::optional<std::int64_t> valid_stamp_ns(
  const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= kNanosecondsPerSecond) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(stamp.sec) * kNanosecondsPerSecond +
         static_cast<std::int64_t>(stamp.nanosec);
}

builtin_interfaces::msg::Time stamp_from_ns(std::int64_t stamp_ns)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<std::int32_t>(stamp_ns / kNanosecondsPerSecond);
  stamp.nanosec = static_cast<std::uint32_t>(stamp_ns % kNanosecondsPerSecond);
  return stamp;
}

bool normalize_quaternion(geometry_msgs::msg::Quaternion & quaternion)
{
  const std::array<double, 4> values{
    quaternion.x, quaternion.y, quaternion.z, quaternion.w};
  if (!std::all_of(values.begin(), values.end(), finite)) {
    return false;
  }
  const double scale = *std::max_element(
    values.begin(), values.end(), [](double lhs, double rhs) {
      return std::abs(lhs) < std::abs(rhs);
    });
  const double magnitude = std::abs(scale);
  if (!finite(magnitude) || magnitude < 1.0e-12) {
    return false;
  }
  const double x = quaternion.x / magnitude;
  const double y = quaternion.y / magnitude;
  const double z = quaternion.z / magnitude;
  const double w = quaternion.w / magnitude;
  const double scaled_norm = std::hypot(std::hypot(x, y), std::hypot(z, w));
  if (!finite(scaled_norm) || scaled_norm < 1.0e-12) {
    return false;
  }
  quaternion.x = x / scaled_norm;
  quaternion.y = y / scaled_norm;
  quaternion.z = z / scaled_norm;
  quaternion.w = w / scaled_norm;
  return true;
}

geometry_msgs::msg::Quaternion inverse(
  const geometry_msgs::msg::Quaternion & quaternion)
{
  geometry_msgs::msg::Quaternion result;
  result.x = -quaternion.x;
  result.y = -quaternion.y;
  result.z = -quaternion.z;
  result.w = quaternion.w;
  return result;
}

geometry_msgs::msg::Quaternion multiply(
  const geometry_msgs::msg::Quaternion & lhs,
  const geometry_msgs::msg::Quaternion & rhs)
{
  geometry_msgs::msg::Quaternion result;
  result.x = lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y;
  result.y = lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x;
  result.z = lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w;
  result.w = lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z;
  return result;
}

geometry_msgs::msg::Quaternion yaw_quaternion(double yaw)
{
  geometry_msgs::msg::Quaternion result;
  result.z = std::sin(yaw * 0.5);
  result.w = std::cos(yaw * 0.5);
  return result;
}

std::array<double, 3> rotate(
  const geometry_msgs::msg::Quaternion & quaternion,
  const std::array<double, 3> & vector)
{
  const std::array<double, 3> axis{
    quaternion.x, quaternion.y, quaternion.z};
  const double scalar = quaternion.w;
  const double dot =
    axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2];
  const std::array<double, 3> cross{
    axis[1] * vector[2] - axis[2] * vector[1],
    axis[2] * vector[0] - axis[0] * vector[2],
    axis[0] * vector[1] - axis[1] * vector[0]};
  const double squared_axis =
    axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
  return {
    2.0 * dot * axis[0] + (scalar * scalar - squared_axis) * vector[0] +
    2.0 * scalar * cross[0],
    2.0 * dot * axis[1] + (scalar * scalar - squared_axis) * vector[1] +
    2.0 * scalar * cross[1],
    2.0 * dot * axis[2] + (scalar * scalar - squared_axis) * vector[2] +
    2.0 * scalar * cross[2]};
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & quaternion)
{
  return std::atan2(
    2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
    1.0 - 2.0 *
    (quaternion.y * quaternion.y + quaternion.z * quaternion.z));
}

template<typename ContainerT>
bool all_finite(const ContainerT & values)
{
  return std::all_of(values.begin(), values.end(), finite);
}

bool valid_imu_payload(const sensor_msgs::msg::Imu & imu)
{
  const std::array<double, 10> values{
    imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w,
    imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,
    imu.linear_acceleration.x, imu.linear_acceleration.y,
    imu.linear_acceleration.z};
  return all_finite(values) && all_finite(imu.orientation_covariance) &&
         all_finite(imu.angular_velocity_covariance) &&
         all_finite(imu.linear_acceleration_covariance);
}

bool valid_wheel_payload(
  const geometry_msgs::msg::TwistWithCovarianceStamped & wheel)
{
  const auto & linear = wheel.twist.twist.linear;
  const auto & angular = wheel.twist.twist.angular;
  const std::array<double, 6> values{
    linear.x, linear.y, linear.z, angular.x, angular.y, angular.z};
  return all_finite(values) && all_finite(wheel.twist.covariance) &&
         wheel.twist.covariance[0] > 0.0;
}

bool valid_gnss_payload(const geometry_msgs::msg::PoseStamped & pose)
{
  const auto & position = pose.pose.position;
  const auto & orientation = pose.pose.orientation;
  const std::array<double, 7> values{
    position.x, position.y, position.z, orientation.x, orientation.y,
    orientation.z, orientation.w};
  return all_finite(values);
}

std::array<double, 9> multiply_3x3(
  const std::array<double, 9> & lhs,
  const std::array<double, 9> & rhs)
{
  std::array<double, 9> result{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      for (std::size_t inner = 0; inner < 3; ++inner) {
        result[row * 3 + column] +=
          lhs[row * 3 + inner] * rhs[inner * 3 + column];
      }
    }
  }
  return result;
}

std::array<double, 9> transpose_3x3(const std::array<double, 9> & matrix)
{
  return {
    matrix[0], matrix[3], matrix[6],
    matrix[1], matrix[4], matrix[7],
    matrix[2], matrix[5], matrix[8]};
}

bool stable_covariance(std::array<double, 9> & covariance)
{
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = row + 1; column < 3; ++column) {
      const double symmetric = 0.5 *
        (covariance[row * 3 + column] + covariance[column * 3 + row]);
      covariance[row * 3 + column] = symmetric;
      covariance[column * 3 + row] = symmetric;
    }
  }
  return all_finite(covariance) && covariance[0] > 0.0 &&
         covariance[4] > 0.0 && covariance[8] > 0.0;
}

}  // namespace

QuaternionWheelGnssEkf::QuaternionWheelGnssEkf(
  QuaternionWheelGnssEkfConfig config)
: config_(std::move(config))
{
  const std::array<double, 20> numeric_values{
    config_.gnss_lever_arm_m[0], config_.gnss_lever_arm_m[1],
    config_.gnss_lever_arm_m[2], config_.world_yaw_offset_rad,
    config_.maximum_imu_age_sec, config_.maximum_prediction_dt_sec,
    config_.initial_position_variance_m2, config_.initial_wheel_bias_mps,
    config_.initial_wheel_bias_variance_m2ps2,
    config_.wheel_speed_variance_floor_m2ps2,
    config_.wheel_bias_random_walk_variance_m2ps3,
    config_.gnss_variance_m2, config_.gnss_mahalanobis_threshold,
    config_.teleport_distance_m, config_.teleport_candidate_radius_m,
    config_.teleport_max_interval_sec, config_.fixed_output_z_m,
    config_.unobserved_variance, config_.orientation_variance_rad2,
    0.0};
  if (!valid_frame(config_.reference_frame) || !valid_frame(config_.base_frame) ||
    !valid_frame(config_.imu_frame) || config_.reference_frame == config_.base_frame ||
    !all_finite(numeric_values) || config_.maximum_imu_age_sec < 0.0 ||
    config_.maximum_prediction_dt_sec <= 0.0 ||
    config_.initialization_sample_count <= 0 ||
    config_.initial_position_variance_m2 <= 0.0 ||
    config_.initial_wheel_bias_variance_m2ps2 <= 0.0 ||
    config_.wheel_speed_variance_floor_m2ps2 <= 0.0 ||
    config_.wheel_bias_random_walk_variance_m2ps3 < 0.0 ||
    config_.gnss_variance_m2 <= 0.0 ||
    config_.gnss_mahalanobis_threshold <= 0.0 ||
    config_.teleport_distance_m <= 0.0 ||
    config_.teleport_confirmation_samples <= 0 ||
    config_.teleport_candidate_radius_m < 0.0 ||
    config_.teleport_max_interval_sec <= 0.0 ||
    config_.unobserved_variance <= 0.0 ||
    config_.orientation_variance_rad2 <= 0.0 ||
    !normalize_quaternion(config_.base_to_imu_orientation))
  {
    throw std::invalid_argument(
            "quaternion wheel GNSS EKF configuration is invalid");
  }
}

void QuaternionWheelGnssEkf::reset() noexcept
{
  state_ = {};
  orientation_history_.clear();
  latest_imu_stamp_ns_.reset();
  latest_wheel_stamp_ns_.reset();
  latest_gnss_stamp_ns_.reset();
  prediction_stamp_ns_.reset();
  last_output_stamp_ns_.reset();
  held_wheel_control_.reset();
  clear_initialization();
  clear_teleport_candidate();
}

QuaternionWheelGnssEkfState QuaternionWheelGnssEkf::state() const noexcept
{
  return state_;
}

void QuaternionWheelGnssEkf::clear_initialization() noexcept
{
  initialization_sum_ = {0.0, 0.0};
  initialization_samples_ = 0U;
}

void QuaternionWheelGnssEkf::clear_teleport_candidate() noexcept
{
  teleport_candidate_stamp_ns_.reset();
  teleport_candidate_anchor_ = {0.0, 0.0};
  teleport_candidate_sum_ = {0.0, 0.0};
  teleport_candidate_samples_ = 0U;
}

void QuaternionWheelGnssEkf::initialize_state(double x, double y) noexcept
{
  state_.initialized = true;
  state_.value = {x, y, config_.initial_wheel_bias_mps};
  state_.covariance = {
    config_.initial_position_variance_m2, 0.0, 0.0,
    0.0, config_.initial_position_variance_m2, 0.0,
    0.0, 0.0, config_.initial_wheel_bias_variance_m2ps2};
  prediction_stamp_ns_.reset();
  held_wheel_control_.reset();
  clear_initialization();
  clear_teleport_candidate();
}

bool QuaternionWheelGnssEkf::observe_imu(const sensor_msgs::msg::Imu & imu)
{
  const auto stamp_ns = valid_stamp_ns(imu.header.stamp);
  if (imu.header.frame_id != config_.imu_frame || !stamp_ns ||
    !valid_imu_payload(imu))
  {
    return false;
  }
  if (latest_imu_stamp_ns_ && *stamp_ns == *latest_imu_stamp_ns_) {
    return false;
  }
  if (latest_imu_stamp_ns_ && *stamp_ns < *latest_imu_stamp_ns_) {
    reset();
  }

  auto raw_orientation = imu.orientation;
  if (!normalize_quaternion(raw_orientation)) {
    return false;
  }
  auto corrected = multiply(
    yaw_quaternion(config_.world_yaw_offset_rad),
    multiply(raw_orientation, inverse(config_.base_to_imu_orientation)));
  if (!normalize_quaternion(corrected)) {
    return false;
  }
  orientation_history_.push_back({*stamp_ns, corrected});
  if (orientation_history_.size() > kMaximumOrientationHistorySamples) {
    orientation_history_.pop_front();
  }
  latest_imu_stamp_ns_ = *stamp_ns;
  return true;
}

std::optional<QuaternionWheelGnssEkf::OrientationSample>
QuaternionWheelGnssEkf::causal_orientation_at(
  std::int64_t target_stamp_ns) const
{
  for (auto iterator = orientation_history_.rbegin();
    iterator != orientation_history_.rend(); ++iterator)
  {
    if (iterator->stamp_ns > target_stamp_ns) {
      continue;
    }
    const double age_sec = static_cast<double>(
      target_stamp_ns - iterator->stamp_ns) * 1.0e-9;
    if (finite(age_sec) && age_sec <= config_.maximum_imu_age_sec) {
      return *iterator;
    }
    break;
  }
  return std::nullopt;
}

std::optional<nav_msgs::msg::Odometry> QuaternionWheelGnssEkf::make_output(
  std::int64_t stamp_ns,
  const geometry_msgs::msg::Quaternion & orientation)
{
  if (!state_.initialized ||
    (last_output_stamp_ns_ && stamp_ns <= *last_output_stamp_ns_))
  {
    return std::nullopt;
  }
  nav_msgs::msg::Odometry output;
  output.header.stamp = stamp_from_ns(stamp_ns);
  output.header.frame_id = config_.reference_frame;
  output.child_frame_id = config_.base_frame;
  output.pose.pose.position.x = state_.value[0];
  output.pose.pose.position.y = state_.value[1];
  output.pose.pose.position.z = config_.fixed_output_z_m;
  output.pose.pose.orientation = orientation;
  output.twist.twist.linear.x = 0.0;
  output.twist.twist.linear.y = 0.0;
  output.twist.twist.linear.z = 0.0;
  output.twist.twist.angular.x = 0.0;
  output.twist.twist.angular.y = 0.0;
  output.twist.twist.angular.z = 0.0;
  output.pose.covariance.fill(0.0);
  output.twist.covariance.fill(0.0);
  output.pose.covariance[0] = state_.covariance[0];
  output.pose.covariance[1] = state_.covariance[1];
  output.pose.covariance[6] = state_.covariance[3];
  output.pose.covariance[7] = state_.covariance[4];
  output.pose.covariance[14] = config_.unobserved_variance;
  output.pose.covariance[21] = config_.orientation_variance_rad2;
  output.pose.covariance[28] = config_.orientation_variance_rad2;
  output.pose.covariance[35] = config_.orientation_variance_rad2;
  output.twist.covariance[0] = config_.unobserved_variance;
  if (held_wheel_control_) {
    const double age_sec = static_cast<double>(
      stamp_ns - held_wheel_control_->stamp_ns) * 1.0e-9;
    const double corrected_speed =
      held_wheel_control_->longitudinal_speed_mps - state_.value[2];
    const double corrected_variance =
      held_wheel_control_->variance_m2ps2 + state_.covariance[8];
    if (finite(age_sec) && age_sec >= 0.0 &&
      age_sec <= config_.maximum_prediction_dt_sec &&
      finite(corrected_speed) && finite(corrected_variance))
    {
      output.twist.twist.linear.x = corrected_speed;
      output.twist.covariance[0] = std::max(
        corrected_variance, std::numeric_limits<double>::epsilon());
    }
  }
  output.twist.covariance[7] = config_.unobserved_variance;
  output.twist.covariance[14] = config_.unobserved_variance;
  output.twist.covariance[21] = config_.unobserved_variance;
  output.twist.covariance[28] = config_.unobserved_variance;
  output.twist.covariance[35] = config_.unobserved_variance;
  last_output_stamp_ns_ = stamp_ns;
  return output;
}

bool QuaternionWheelGnssEkf::predict_to(
  std::int64_t target_stamp_ns,
  const geometry_msgs::msg::Quaternion & orientation,
  const WheelControl & wheel_control)
{
  if (!prediction_stamp_ns_) {
    return false;
  }
  const double dt = static_cast<double>(
    target_stamp_ns - *prediction_stamp_ns_) * 1.0e-9;
  const double control_age_sec = static_cast<double>(
    target_stamp_ns - wheel_control.stamp_ns) * 1.0e-9;
  if (!finite(dt) || !finite(control_age_sec) || dt < 0.0 ||
    control_age_sec < 0.0 || dt > config_.maximum_prediction_dt_sec ||
    control_age_sec > config_.maximum_prediction_dt_sec)
  {
    reset();
    return false;
  }
  if (dt == 0.0) {
    return true;
  }

  const double yaw = yaw_from_quaternion(orientation);
  const double cosine = std::cos(yaw);
  const double sine = std::sin(yaw);
  const double corrected_speed =
    wheel_control.longitudinal_speed_mps - state_.value[2];
  auto next_value = state_.value;
  next_value[0] += corrected_speed * cosine * dt;
  next_value[1] += corrected_speed * sine * dt;

  const std::array<double, 9> transition{
    1.0, 0.0, -cosine * dt,
    0.0, 1.0, -sine * dt,
    0.0, 0.0, 1.0};
  auto next_covariance = multiply_3x3(
    multiply_3x3(transition, state_.covariance),
    transpose_3x3(transition));
  const double distance_variance = wheel_control.variance_m2ps2 * dt * dt;
  next_covariance[0] += cosine * cosine * distance_variance;
  next_covariance[1] += cosine * sine * distance_variance;
  next_covariance[3] += cosine * sine * distance_variance;
  next_covariance[4] += sine * sine * distance_variance;
  next_covariance[8] +=
    config_.wheel_bias_random_walk_variance_m2ps3 * dt;
  if (!all_finite(next_value) || !stable_covariance(next_covariance)) {
    reset();
    return false;
  }
  state_.value = next_value;
  state_.covariance = next_covariance;
  prediction_stamp_ns_ = target_stamp_ns;
  return true;
}

std::optional<nav_msgs::msg::Odometry>
QuaternionWheelGnssEkf::observe_wheel_speed(
  const geometry_msgs::msg::TwistWithCovarianceStamped & wheel)
{
  const auto stamp_ns = valid_stamp_ns(wheel.header.stamp);
  if (wheel.header.frame_id != config_.base_frame || !stamp_ns ||
    !valid_wheel_payload(wheel))
  {
    return std::nullopt;
  }
  if (latest_wheel_stamp_ns_ && *stamp_ns == *latest_wheel_stamp_ns_) {
    return std::nullopt;
  }
  if (latest_wheel_stamp_ns_ && *stamp_ns < *latest_wheel_stamp_ns_) {
    reset();
    return std::nullopt;
  }
  if (!state_.initialized) {
    return std::nullopt;
  }
  const auto orientation = causal_orientation_at(*stamp_ns);
  if (!orientation) {
    return std::nullopt;
  }
  if (last_output_stamp_ns_ && *stamp_ns <= *last_output_stamp_ns_) {
    return std::nullopt;
  }
  const WheelControl wheel_control{
    *stamp_ns, wheel.twist.twist.linear.x,
    std::max(
      wheel.twist.covariance[0],
      config_.wheel_speed_variance_floor_m2ps2)};
  if (!prediction_stamp_ns_) {
    prediction_stamp_ns_ = *stamp_ns;
    held_wheel_control_ = wheel_control;
    latest_wheel_stamp_ns_ = *stamp_ns;
    return std::nullopt;
  }
  if (!predict_to(*stamp_ns, orientation->orientation, wheel_control)) {
    return std::nullopt;
  }
  held_wheel_control_ = wheel_control;
  latest_wheel_stamp_ns_ = *stamp_ns;
  return make_output(*stamp_ns, orientation->orientation);
}

std::optional<nav_msgs::msg::Odometry>
QuaternionWheelGnssEkf::consider_teleport(
  std::int64_t stamp_ns, double body_x, double body_y,
  const geometry_msgs::msg::Quaternion & orientation)
{
  const double predicted_distance = std::hypot(
    body_x - state_.value[0], body_y - state_.value[1]);
  if (!finite(predicted_distance) || predicted_distance < config_.teleport_distance_m) {
    clear_teleport_candidate();
    return std::nullopt;
  }

  bool continues = false;
  if (teleport_candidate_stamp_ns_) {
    const double interval_sec = static_cast<double>(
      stamp_ns - *teleport_candidate_stamp_ns_) * 1.0e-9;
    const double candidate_distance = std::hypot(
      body_x - teleport_candidate_anchor_[0],
      body_y - teleport_candidate_anchor_[1]);
    continues = interval_sec > 0.0 &&
      interval_sec <= config_.teleport_max_interval_sec &&
      candidate_distance <= config_.teleport_candidate_radius_m;
  }
  if (continues) {
    teleport_candidate_sum_[0] += body_x;
    teleport_candidate_sum_[1] += body_y;
    ++teleport_candidate_samples_;
  } else {
    teleport_candidate_sum_ = {body_x, body_y};
    teleport_candidate_samples_ = 1U;
    teleport_candidate_anchor_ = {body_x, body_y};
  }
  teleport_candidate_stamp_ns_ = stamp_ns;
  if (teleport_candidate_samples_ < static_cast<std::size_t>(
      config_.teleport_confirmation_samples))
  {
    return std::nullopt;
  }
  const double sample_count = static_cast<double>(teleport_candidate_samples_);
  initialize_state(
    teleport_candidate_sum_[0] / sample_count,
    teleport_candidate_sum_[1] / sample_count);
  return make_output(stamp_ns, orientation);
}

std::optional<nav_msgs::msg::Odometry> QuaternionWheelGnssEkf::observe_gnss(
  const geometry_msgs::msg::PoseStamped & antenna_pose)
{
  const auto stamp_ns = valid_stamp_ns(antenna_pose.header.stamp);
  if (antenna_pose.header.frame_id != config_.reference_frame || !stamp_ns ||
    !valid_gnss_payload(antenna_pose))
  {
    return std::nullopt;
  }
  if (latest_gnss_stamp_ns_ && *stamp_ns == *latest_gnss_stamp_ns_) {
    return std::nullopt;
  }
  if (latest_gnss_stamp_ns_ && *stamp_ns < *latest_gnss_stamp_ns_) {
    reset();
    return std::nullopt;
  }
  const auto orientation = causal_orientation_at(*stamp_ns);
  if (!orientation) {
    return std::nullopt;
  }
  if (last_output_stamp_ns_ && *stamp_ns <= *last_output_stamp_ns_) {
    return std::nullopt;
  }
  latest_gnss_stamp_ns_ = *stamp_ns;
  const auto world_lever_arm = rotate(
    orientation->orientation, config_.gnss_lever_arm_m);
  const double body_x = antenna_pose.pose.position.x - world_lever_arm[0];
  const double body_y = antenna_pose.pose.position.y - world_lever_arm[1];
  if (!finite(body_x) || !finite(body_y)) {
    return std::nullopt;
  }

  if (!state_.initialized) {
    initialization_sum_[0] += body_x;
    initialization_sum_[1] += body_y;
    ++initialization_samples_;
    if (initialization_samples_ < static_cast<std::size_t>(
        config_.initialization_sample_count))
    {
      return std::nullopt;
    }
    const double sample_count = static_cast<double>(initialization_samples_);
    initialize_state(
      initialization_sum_[0] / sample_count,
      initialization_sum_[1] / sample_count);
    return make_output(*stamp_ns, orientation->orientation);
  }

  if (prediction_stamp_ns_) {
    if (!held_wheel_control_ ||
      !predict_to(
        *stamp_ns, orientation->orientation, *held_wheel_control_))
    {
      return std::nullopt;
    }
  }

  const double innovation_x = body_x - state_.value[0];
  const double innovation_y = body_y - state_.value[1];
  const double s00 = state_.covariance[0] + config_.gnss_variance_m2;
  const double s01 = state_.covariance[1];
  const double s10 = state_.covariance[3];
  const double s11 = state_.covariance[4] + config_.gnss_variance_m2;
  const double determinant = s00 * s11 - s01 * s10;
  if (!finite(determinant) || determinant <= 0.0) {
    reset();
    return std::nullopt;
  }
  const double inverse00 = s11 / determinant;
  const double inverse01 = -s01 / determinant;
  const double inverse10 = -s10 / determinant;
  const double inverse11 = s00 / determinant;
  const double mahalanobis =
    innovation_x * (inverse00 * innovation_x + inverse01 * innovation_y) +
    innovation_y * (inverse10 * innovation_x + inverse11 * innovation_y);
  if (!finite(mahalanobis) ||
    mahalanobis > config_.gnss_mahalanobis_threshold)
  {
    return consider_teleport(
      *stamp_ns, body_x, body_y, orientation->orientation);
  }
  clear_teleport_candidate();

  std::array<double, 6> gain{};
  for (std::size_t row = 0; row < 3; ++row) {
    const double p0 = state_.covariance[row * 3];
    const double p1 = state_.covariance[row * 3 + 1];
    gain[row * 2] = p0 * inverse00 + p1 * inverse10;
    gain[row * 2 + 1] = p0 * inverse01 + p1 * inverse11;
  }
  auto next_value = state_.value;
  for (std::size_t row = 0; row < 3; ++row) {
    next_value[row] += gain[row * 2] * innovation_x +
      gain[row * 2 + 1] * innovation_y;
  }

  std::array<double, 9> joseph_left{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0};
  for (std::size_t row = 0; row < 3; ++row) {
    joseph_left[row * 3] -= gain[row * 2];
    joseph_left[row * 3 + 1] -= gain[row * 2 + 1];
  }
  auto next_covariance = multiply_3x3(
    multiply_3x3(joseph_left, state_.covariance),
    transpose_3x3(joseph_left));
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      next_covariance[row * 3 + column] += config_.gnss_variance_m2 *
        (gain[row * 2] * gain[column * 2] +
        gain[row * 2 + 1] * gain[column * 2 + 1]);
    }
  }
  if (!all_finite(next_value) || !stable_covariance(next_covariance)) {
    reset();
    return std::nullopt;
  }
  state_.value = next_value;
  state_.covariance = next_covariance;
  return make_output(*stamp_ns, orientation->orientation);
}

}  // namespace ad_localization
