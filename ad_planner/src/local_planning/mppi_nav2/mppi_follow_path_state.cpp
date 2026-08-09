#include "ad_planner/local_planning/mppi_nav2/mppi_follow_path_state.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ad_planner
{
namespace
{

constexpr double kQuaternionNormTolerance = 1.0e-6;

MppiFollowPathAdmission invalid_admission(const std::string & reason)
{
  MppiFollowPathAdmission result;
  result.reason = reason;
  return result;
}

bool finite_point(const MppiFollowPathPoint & point)
{
  return std::isfinite(point.position_x) &&
         std::isfinite(point.position_y) &&
         std::isfinite(point.position_z) &&
         std::isfinite(point.orientation_x) &&
         std::isfinite(point.orientation_y) &&
         std::isfinite(point.orientation_z) &&
         std::isfinite(point.orientation_w);
}

bool normalized_quaternion(const MppiFollowPathPoint & point)
{
  const double norm = std::hypot(
    std::hypot(point.orientation_x, point.orientation_y),
    std::hypot(point.orientation_z, point.orientation_w));
  return std::isfinite(norm) && norm > 0.0 &&
         std::abs(norm - 1.0) <= kQuaternionNormTolerance;
}

bool valid_canonical_path(const MppiFollowPathCanonicalPath & path)
{
  if (path.frame_id.empty() || path.points.empty()) {
    return false;
  }
  for (const auto & point : path.points) {
    if (!finite_point(point) || !normalized_quaternion(point)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool operator==(
  const MppiFollowPathPoint & lhs,
  const MppiFollowPathPoint & rhs) noexcept
{
  return lhs.position_x == rhs.position_x &&
         lhs.position_y == rhs.position_y &&
         lhs.position_z == rhs.position_z &&
         lhs.orientation_x == rhs.orientation_x &&
         lhs.orientation_y == rhs.orientation_y &&
         lhs.orientation_z == rhs.orientation_z &&
         lhs.orientation_w == rhs.orientation_w;
}

bool operator!=(
  const MppiFollowPathPoint & lhs,
  const MppiFollowPathPoint & rhs) noexcept
{
  return !(lhs == rhs);
}

bool operator==(
  const MppiFollowPathCanonicalPath & lhs,
  const MppiFollowPathCanonicalPath & rhs) noexcept
{
  return lhs.frame_id == rhs.frame_id && lhs.points == rhs.points;
}

bool operator!=(
  const MppiFollowPathCanonicalPath & lhs,
  const MppiFollowPathCanonicalPath & rhs) noexcept
{
  return !(lhs == rhs);
}

MppiFollowPathAdmission canonicalize_mppi_follow_path(
  const nav_msgs::msg::Path & path, const std::size_t maximum_pose_count)
{
  if (maximum_pose_count == 0U) {
    return invalid_admission("maximum path pose count is zero");
  }
  if (path.header.frame_id.empty()) {
    return invalid_admission("path frame is empty");
  }
  if (path.poses.empty()) {
    return invalid_admission("path is empty");
  }
  if (path.poses.size() > maximum_pose_count) {
    return invalid_admission("path exceeds maximum pose count");
  }

  MppiFollowPathCanonicalPath canonical;
  canonical.frame_id = path.header.frame_id;
  canonical.points.reserve(path.poses.size());
  for (const auto & pose : path.poses) {
    if (!pose.header.frame_id.empty() &&
      pose.header.frame_id != path.header.frame_id)
    {
      return invalid_admission("path pose frame does not match path frame");
    }
    MppiFollowPathPoint point{
      pose.pose.position.x,
      pose.pose.position.y,
      pose.pose.position.z,
      pose.pose.orientation.x,
      pose.pose.orientation.y,
      pose.pose.orientation.z,
      pose.pose.orientation.w};
    if (!finite_point(point)) {
      return invalid_admission("path pose is not finite");
    }
    if (!normalized_quaternion(point)) {
      return invalid_admission("path pose quaternion is not normalized");
    }
    canonical.points.push_back(point);
  }

  MppiFollowPathAdmission result;
  result.valid = true;
  result.reason = "ok";
  result.path = std::move(canonical);
  return result;
}

MppiFollowPathState::MppiFollowPathState(MppiFollowPathStateConfig config)
: config_(config)
{
  if (config_.path_timeout_ns <= 0) {
    throw std::invalid_argument(
            "FollowPath state timeout must be positive");
  }
}

MppiFollowPathEffects MppiFollowPathState::observe_path(
  const MppiFollowPathCanonicalPath & path,
  const std::int64_t receipt_steady_ns)
{
  if (receipt_steady_ns <= 0 || !valid_canonical_path(path)) {
    return deactivate();
  }
  if (current_path_ && receipt_steady_ns < receipt_steady_ns_) {
    return MppiFollowPathEffects{};
  }
  if (current_path_ && *current_path_ == path) {
    receipt_steady_ns_ = receipt_steady_ns;
    return MppiFollowPathEffects{};
  }
  if (next_generation_ == std::numeric_limits<std::uint64_t>::max()) {
    return deactivate();
  }

  MppiFollowPathCanonicalPath copied_path = path;
  const std::uint64_t generation = next_generation_ + 1U;
  current_path_ = std::move(copied_path);
  receipt_steady_ns_ = receipt_steady_ns;
  current_generation_ = generation;
  waiting_generation_ = generation;
  next_generation_ = generation;

  MppiFollowPathEffects effects;
  effects.new_path_generation = generation;
  return effects;
}

MppiFollowPathEffects MppiFollowPathState::poll(
  const std::int64_t now_steady_ns, const bool action_server_ready)
{
  if (!current_path_ || !current_generation_) {
    return MppiFollowPathEffects{};
  }
  if (now_steady_ns <= 0 || now_steady_ns < receipt_steady_ns_) {
    return deactivate();
  }
  const std::int64_t age_ns = now_steady_ns - receipt_steady_ns_;
  if (age_ns > config_.path_timeout_ns) {
    return deactivate();
  }
  if (!action_server_ready || !pending_goal_responses_.empty() ||
    !waiting_generation_ ||
    *waiting_generation_ != *current_generation_)
  {
    return MppiFollowPathEffects{};
  }

  MppiFollowPathEffects effects;
  effects.actions.push_back(MppiFollowPathAction{
    MppiFollowPathActionType::kSendGoal, *waiting_generation_});
  pending_goal_responses_.insert(*waiting_generation_);
  waiting_generation_.reset();
  return effects;
}

MppiFollowPathEffects MppiFollowPathState::on_goal_response(
  const std::uint64_t generation, const bool accepted)
{
  const auto pending = pending_goal_responses_.find(generation);
  if (pending == pending_goal_responses_.end()) {
    return MppiFollowPathEffects{};
  }

  MppiFollowPathEffects effects;
  effects.goal_response_was_pending = true;
  if (!accepted) {
    pending_goal_responses_.erase(pending);
    if (current_generation_ && *current_generation_ == generation) {
      effects = deactivate();
      effects.goal_response_was_pending = true;
    }
    return effects;
  }

  if (!current_generation_ || *current_generation_ != generation) {
    effects.actions.push_back(MppiFollowPathAction{
      MppiFollowPathActionType::kCancelGoal, generation});
    pending_goal_responses_.erase(pending);
    return effects;
  }

  pending_goal_responses_.erase(pending);
  accepted_generation_ = generation;
  return effects;
}

MppiFollowPathEffects MppiFollowPathState::on_goal_result(
  const std::uint64_t generation, const MppiFollowPathResultCode result)
{
  static_cast<void>(result);
  if (!current_generation_ || *current_generation_ != generation) {
    return MppiFollowPathEffects{};
  }
  clear_current_without_cancel();
  return MppiFollowPathEffects{};
}

MppiFollowPathEffects MppiFollowPathState::deactivate()
{
  MppiFollowPathEffects effects;
  if (accepted_generation_) {
    effects.actions.push_back(MppiFollowPathAction{
      MppiFollowPathActionType::kCancelGoal, *accepted_generation_});
  }
  clear_current_without_cancel();
  return effects;
}

MppiFollowPathStateSnapshot MppiFollowPathState::snapshot() const
{
  return MppiFollowPathStateSnapshot{
    current_path_.has_value(),
    current_generation_,
    accepted_generation_,
    pending_goal_responses_.size(),
    receipt_steady_ns_};
}

void MppiFollowPathState::clear_current_without_cancel() noexcept
{
  current_path_.reset();
  receipt_steady_ns_ = 0;
  current_generation_.reset();
  waiting_generation_.reset();
  accepted_generation_.reset();
}

}  // namespace ad_planner
