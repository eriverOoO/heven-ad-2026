#include "ad_localization/gnss_shadow/localization_handoff.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ad_localization
{
namespace
{

constexpr double kPi = 3.14159265358979323846;
constexpr double kQuaternionEpsilon = 1.0e-9;

bool finite(double value)
{
  return std::isfinite(value);
}

double square(double value)
{
  return value * value;
}

double wrap_angle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

std::int64_t stamp_ns(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= 1000000000U) {
    return -1;
  }
  return static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

double quaternion_norm(const geometry_msgs::msg::Quaternion & quaternion)
{
  return std::sqrt(
    square(quaternion.x) + square(quaternion.y) +
    square(quaternion.z) + square(quaternion.w));
}

bool valid_quaternion(const geometry_msgs::msg::Quaternion & quaternion)
{
  return finite(quaternion.x) && finite(quaternion.y) &&
         finite(quaternion.z) && finite(quaternion.w) &&
         quaternion_norm(quaternion) > kQuaternionEpsilon;
}

geometry_msgs::msg::Quaternion normalized(
  const geometry_msgs::msg::Quaternion & quaternion)
{
  const double norm = quaternion_norm(quaternion);
  geometry_msgs::msg::Quaternion result;
  result.x = quaternion.x / norm;
  result.y = quaternion.y / norm;
  result.z = quaternion.z / norm;
  result.w = quaternion.w / norm;
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
  return normalized(result);
}

geometry_msgs::msg::Quaternion yaw_quaternion(double yaw)
{
  geometry_msgs::msg::Quaternion result;
  result.z = std::sin(yaw * 0.5);
  result.w = std::cos(yaw * 0.5);
  return result;
}

double yaw(const geometry_msgs::msg::Quaternion & quaternion)
{
  const auto unit = normalized(quaternion);
  return std::atan2(
    2.0 * (unit.w * unit.z + unit.x * unit.y),
    1.0 - 2.0 * (unit.y * unit.y + unit.z * unit.z));
}

bool finite_pose(const geometry_msgs::msg::Pose & pose)
{
  return finite(pose.position.x) && finite(pose.position.y) &&
         finite(pose.position.z) && valid_quaternion(pose.orientation);
}

bool finite_twist(const geometry_msgs::msg::Twist & twist)
{
  return finite(twist.linear.x) && finite(twist.linear.y) &&
         finite(twist.linear.z) && finite(twist.angular.x) &&
         finite(twist.angular.y) && finite(twist.angular.z);
}

bool finite_covariance(const std::array<double, 36> & covariance)
{
  return std::all_of(covariance.begin(), covariance.end(), finite);
}

}  // namespace

LocalizationHandoff::LocalizationHandoff(HandoffConfig config)
: config_(std::move(config))
{
  const bool finite_points =
    finite(config_.entry_xy.x) && finite(config_.entry_xy.y) &&
    finite(config_.exit_xy.x) && finite(config_.exit_xy.y);
  const bool positive_values =
    finite(config_.prewarm_radius_m) && config_.prewarm_radius_m > 0.0 &&
    finite(config_.entry_switch_radius_m) && config_.entry_switch_radius_m > 0.0 &&
    finite(config_.exit_switch_radius_m) && config_.exit_switch_radius_m > 0.0 &&
    finite(config_.source_timeout_sec) && config_.source_timeout_sec > 0.0 &&
    finite(config_.maximum_position_disagreement_m) &&
    config_.maximum_position_disagreement_m > 0.0 &&
    finite(config_.maximum_yaw_disagreement_rad) &&
    config_.maximum_yaw_disagreement_rad > 0.0 &&
    config_.maximum_yaw_disagreement_rad <= kPi &&
    finite(config_.blend_duration_sec) && config_.blend_duration_sec > 0.0;
  if (!finite_points || !positive_values || config_.reference_frame.empty() ||
    config_.base_frame.empty())
  {
    throw std::invalid_argument("invalid localization handoff configuration");
  }
  if (config_.entry_switch_radius_m > config_.prewarm_radius_m) {
    throw std::invalid_argument(
            "entry_switch_radius_m must not exceed prewarm_radius_m");
  }
}

HandoffUpdate LocalizationHandoff::observe(
  Backend backend, const nav_msgs::msg::Odometry & candidate,
  double receipt_time_sec)
{
  auto update = current_update();
  SourceState & candidate_source = source(backend);
  if (!validate_candidate(candidate, receipt_time_sec, candidate_source)) {
    return update;
  }

  candidate_source.odometry = candidate;
  candidate_source.odometry->pose.pose.orientation =
    normalized(candidate.pose.pose.orientation);
  candidate_source.stamp_ns = stamp_ns(candidate.header.stamp);
  candidate_source.receipt_time_sec = receipt_time_sec;
  const auto & normalized_candidate = *candidate_source.odometry;

  if (backend == Backend::kGnssImu &&
    active_backend_ == Backend::kGnssImu &&
    phase_ != HandoffPhase::kGnssFinish &&
    near(normalized_candidate, config_.entry_xy, config_.prewarm_radius_m))
  {
    geometry_msgs::msg::PoseStamped initial_pose;
    initial_pose.header = normalized_candidate.header;
    initial_pose.pose = normalized_candidate.pose.pose;
    update.fastlio_initial_pose = initial_pose;
    if (!initial_pose_sent_) {
      initial_pose_sent_ = true;
      phase_ = HandoffPhase::kFastlioReady;
    }
  }

  if (active_backend_ == Backend::kGnssImu) {
    if (backend == Backend::kGnssImu) {
      update.canonical_odometry =
        make_canonical(normalized_candidate, receipt_time_sec);
    } else if (
      phase_ == HandoffPhase::kFastlioReady &&
      gnss_.odometry.has_value() &&
      fresh(gnss_, receipt_time_sec) &&
      near(*gnss_.odometry, config_.entry_xy, config_.entry_switch_radius_m) &&
      poses_agree(*gnss_.odometry, normalized_candidate))
    {
      begin_switch(
        Backend::kFastlio, HandoffPhase::kFastlioActive,
        normalized_candidate, receipt_time_sec);
      update.canonical_odometry =
        make_canonical(normalized_candidate, receipt_time_sec);
    }
  } else {
    if (backend == Backend::kFastlio) {
      if (near(normalized_candidate, config_.exit_xy, config_.exit_switch_radius_m)) {
        phase_ = HandoffPhase::kGnssRecovery;
      }
      update.canonical_odometry =
        make_canonical(normalized_candidate, receipt_time_sec);
    } else if (
      (phase_ == HandoffPhase::kFastlioActive ||
      phase_ == HandoffPhase::kGnssRecovery) &&
      fastlio_.odometry.has_value() &&
      fresh(fastlio_, receipt_time_sec) &&
      near(*fastlio_.odometry, config_.exit_xy, config_.exit_switch_radius_m) &&
      poses_agree(*fastlio_.odometry, normalized_candidate))
    {
      begin_switch(
        Backend::kGnssImu, HandoffPhase::kGnssFinish,
        normalized_candidate, receipt_time_sec);
      update.canonical_odometry =
        make_canonical(normalized_candidate, receipt_time_sec);
    }
  }

  update.active_backend = active_backend_;
  update.phase = phase_;
  update.switch_count = switch_count_;
  return update;
}

Backend LocalizationHandoff::active_backend() const noexcept
{
  return active_backend_;
}

HandoffPhase LocalizationHandoff::phase() const noexcept
{
  return phase_;
}

std::size_t LocalizationHandoff::switch_count() const noexcept
{
  return switch_count_;
}

HandoffUpdate LocalizationHandoff::current_update() const
{
  HandoffUpdate update;
  update.active_backend = active_backend_;
  update.phase = phase_;
  update.switch_count = switch_count_;
  return update;
}

bool LocalizationHandoff::validate_candidate(
  const nav_msgs::msg::Odometry & candidate, double receipt_time_sec,
  const SourceState & state) const
{
  const auto candidate_stamp_ns = stamp_ns(candidate.header.stamp);
  return finite(receipt_time_sec) && receipt_time_sec >= 0.0 &&
         receipt_time_sec > state.receipt_time_sec &&
         candidate_stamp_ns >= 0 && candidate_stamp_ns > state.stamp_ns &&
         candidate.header.frame_id == config_.reference_frame &&
         candidate.child_frame_id == config_.base_frame &&
         finite_pose(candidate.pose.pose) && finite_twist(candidate.twist.twist) &&
         finite_covariance(candidate.pose.covariance) &&
         finite_covariance(candidate.twist.covariance);
}

bool LocalizationHandoff::fresh(
  const SourceState & state, double now_sec) const
{
  const double age = now_sec - state.receipt_time_sec;
  return state.odometry.has_value() && finite(age) && age >= 0.0 &&
         age <= config_.source_timeout_sec;
}

bool LocalizationHandoff::poses_agree(
  const nav_msgs::msg::Odometry & lhs,
  const nav_msgs::msg::Odometry & rhs) const
{
  const double dx = lhs.pose.pose.position.x - rhs.pose.pose.position.x;
  const double dy = lhs.pose.pose.position.y - rhs.pose.pose.position.y;
  const double position_difference = std::hypot(dx, dy);
  const double yaw_difference = std::abs(wrap_angle(
      yaw(lhs.pose.pose.orientation) - yaw(rhs.pose.pose.orientation)));
  return position_difference <= config_.maximum_position_disagreement_m &&
         yaw_difference <= config_.maximum_yaw_disagreement_rad;
}

bool LocalizationHandoff::near(
  const nav_msgs::msg::Odometry & candidate, const Point2 & point,
  double radius_m) const
{
  return std::hypot(
    candidate.pose.pose.position.x - point.x,
    candidate.pose.pose.position.y - point.y) <= radius_m;
}

void LocalizationHandoff::begin_switch(
  Backend backend, HandoffPhase phase,
  const nav_msgs::msg::Odometry & first_candidate, double receipt_time_sec)
{
  correction_ = Correction{};
  if (last_canonical_.has_value()) {
    correction_.x_m =
      last_canonical_->pose.pose.position.x - first_candidate.pose.pose.position.x;
    correction_.y_m =
      last_canonical_->pose.pose.position.y - first_candidate.pose.pose.position.y;
    correction_.yaw_rad = wrap_angle(
      yaw(last_canonical_->pose.pose.orientation) -
      yaw(first_candidate.pose.pose.orientation));
    correction_.start_time_sec = receipt_time_sec;
    correction_.active = true;
  }
  active_backend_ = backend;
  phase_ = phase;
  ++switch_count_;
}

std::optional<nav_msgs::msg::Odometry> LocalizationHandoff::make_canonical(
  const nav_msgs::msg::Odometry & candidate, double receipt_time_sec)
{
  const auto candidate_stamp_ns = stamp_ns(candidate.header.stamp);
  if (candidate_stamp_ns <= last_canonical_stamp_ns_) {
    return std::nullopt;
  }

  nav_msgs::msg::Odometry result = candidate;
  if (correction_.active) {
    const double elapsed = std::max(0.0, receipt_time_sec - correction_.start_time_sec);
    const double weight = std::clamp(
      1.0 - elapsed / config_.blend_duration_sec, 0.0, 1.0);
    result.pose.pose.position.x += weight * correction_.x_m;
    result.pose.pose.position.y += weight * correction_.y_m;
    result.pose.pose.orientation = multiply(
      yaw_quaternion(weight * correction_.yaw_rad),
      result.pose.pose.orientation);
    result.pose.covariance[0] += square(weight * correction_.x_m);
    result.pose.covariance[7] += square(weight * correction_.y_m);
    result.pose.covariance[35] += square(weight * correction_.yaw_rad);
    correction_.active = weight > 0.0;
  }

  last_canonical_ = result;
  last_canonical_stamp_ns_ = candidate_stamp_ns;
  return result;
}

LocalizationHandoff::SourceState & LocalizationHandoff::source(Backend backend)
{
  return backend == Backend::kGnssImu ? gnss_ : fastlio_;
}

const LocalizationHandoff::SourceState & LocalizationHandoff::source(
  Backend backend) const
{
  return backend == Backend::kGnssImu ? gnss_ : fastlio_;
}

const char * to_string(Backend backend) noexcept
{
  switch (backend) {
    case Backend::kGnssImu:
      return "gnss_imu";
    case Backend::kFastlio:
      return "fastlio";
  }
  return "unknown";
}

const char * to_string(HandoffPhase phase) noexcept
{
  switch (phase) {
    case HandoffPhase::kGnssApproach:
      return "gnss_approach";
    case HandoffPhase::kFastlioReady:
      return "fastlio_ready";
    case HandoffPhase::kFastlioActive:
      return "fastlio_active";
    case HandoffPhase::kGnssRecovery:
      return "gnss_recovery";
    case HandoffPhase::kGnssFinish:
      return "gnss_finish";
  }
  return "unknown";
}

}  // namespace ad_localization
