#include "mppi_follow_path_proxy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ad_planner
{
namespace
{

constexpr long double kNanosecondsPerSecond = 1'000'000'000.0L;
constexpr std::int64_t kDefaultMaximumPoseCount = 10'000;

std::int64_t checked_timeout_ns(const double timeout_s)
{
  const long double nanoseconds =
    static_cast<long double>(timeout_s) * kNanosecondsPerSecond;
  if (!std::isfinite(timeout_s) || timeout_s <= 0.0 ||
    !std::isfinite(nanoseconds) || nanoseconds < 1.0L ||
    nanoseconds >
    static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
  {
    throw std::invalid_argument(
            "path_timeout_s must be finite, positive, and representable");
  }
  return static_cast<std::int64_t>(nanoseconds);
}

std::size_t checked_pose_count(const std::int64_t count)
{
  if (count <= 0 ||
    static_cast<std::uint64_t>(count) >
    static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
  {
    throw std::invalid_argument(
            "maximum_pose_count must be positive and representable");
  }
  return static_cast<std::size_t>(count);
}

MppiFollowPathResultCode result_code(
  const rclcpp_action::ResultCode code)
{
  switch (code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      return MppiFollowPathResultCode::kSucceeded;
    case rclcpp_action::ResultCode::ABORTED:
      return MppiFollowPathResultCode::kAborted;
    case rclcpp_action::ResultCode::CANCELED:
      return MppiFollowPathResultCode::kCanceled;
    default:
      return MppiFollowPathResultCode::kAborted;
  }
}

}  // namespace

MppiFollowPathProxy::MppiFollowPathProxy(
  const rclcpp::NodeOptions & options)
: Node("ad_planner_mppi_follow_path_proxy", options),
  steady_clock_(RCL_STEADY_TIME),
  state_(MppiFollowPathStateConfig{
    checked_timeout_ns(
      declare_parameter<double>("path_timeout_s", 0.5))}),
  maximum_pose_count_(
    checked_pose_count(
      declare_parameter<std::int64_t>(
        "maximum_pose_count", kDefaultMaximumPoseCount)))
{
  const std::string path_topic = declare_parameter<std::string>(
    "path_topic", "/ad/planner/path");
  const std::string action_name = declare_parameter<std::string>(
    "action_name", "follow_path");
  if (path_topic.empty() || action_name.empty()) {
    throw std::invalid_argument("path topic and action name must not be empty");
  }

  action_client_ =
    rclcpp_action::create_client<FollowPath>(this, action_name);
  path_subscription_ = create_subscription<nav_msgs::msg::Path>(
    path_topic,
    rclcpp::QoS(rclcpp::KeepLast(1))
    .reliable().transient_local(),
    [this](nav_msgs::msg::Path::ConstSharedPtr message) {
      on_path(std::move(message));
    });
  timer_ = create_wall_timer(
    std::chrono::milliseconds(50),
    [this]() {on_timer();});
}

void MppiFollowPathProxy::on_path(
  const nav_msgs::msg::Path::ConstSharedPtr message)
{
  const std::int64_t receipt_steady_ns =
    steady_clock_.now().nanoseconds();
  MppiFollowPathAdmission admission;
  try {
    admission = canonicalize_mppi_follow_path(
      *message, maximum_pose_count_);
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      get_logger(), "failed to canonicalize FollowPath route: %s",
      error.what());
  }

  std::vector<ActionWork> work;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    MppiFollowPathEffects effects;
    if (!admission.valid || receipt_steady_ns <= 0) {
      effects = state_.deactivate();
      pending_paths_.clear();
    } else {
      effects = state_.observe_path(
        admission.path, receipt_steady_ns);
      if (effects.new_path_generation) {
        pending_paths_.clear();
        pending_paths_.emplace(
          *effects.new_path_generation, *message);
      }
    }
    work = collect_work_locked(effects);
  }
  if (!admission.valid) {
    RCLCPP_WARN(
      get_logger(), "rejecting FollowPath route: %s",
      admission.reason.c_str());
  }
  execute_work(std::move(work));
}

void MppiFollowPathProxy::on_timer()
{
  const std::int64_t now_steady_ns =
    steady_clock_.now().nanoseconds();
  bool server_ready = false;
  try {
    server_ready = action_client_->action_server_is_ready();
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      get_logger(), "FollowPath server readiness check failed: %s",
      error.what());
    deactivate();
    return;
  } catch (...) {
    RCLCPP_ERROR(
      get_logger(), "FollowPath server readiness check failed");
    deactivate();
    return;
  }

  std::vector<ActionWork> work;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    const auto effects = state_.poll(now_steady_ns, server_ready);
    if (!state_.snapshot().has_path) {
      pending_paths_.clear();
    }
    work = collect_work_locked(effects);
  }
  execute_work(std::move(work));
}

void MppiFollowPathProxy::on_goal_response(
  const std::uint64_t generation,
  const GoalHandle::SharedPtr & goal_handle)
{
  std::vector<ActionWork> work;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    const auto effects =
      state_.on_goal_response(generation, goal_handle != nullptr);
    const auto snapshot = state_.snapshot();
    if (goal_handle) {
      const bool cancel_response = std::any_of(
        effects.actions.begin(), effects.actions.end(),
        [generation](const MppiFollowPathAction & action) {
          return action.type == MppiFollowPathActionType::kCancelGoal &&
                 action.generation == generation;
        });
      goal_handles_.record_accepted_response(
        generation, goal_handle, effects.goal_response_was_pending,
        cancel_response, snapshot.accepted_generation);
    }
    if (!snapshot.has_path) {
      pending_paths_.clear();
    }
    work = collect_work_locked(effects);
  }
  execute_work(std::move(work));
}

void MppiFollowPathProxy::on_goal_result(
  const std::uint64_t generation,
  const GoalHandle::WrappedResult & result)
{
  std::vector<ActionWork> work;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    goal_handles_.erase_result(generation);
    const auto effects =
      state_.on_goal_result(generation, result_code(result.code));
    if (!state_.snapshot().has_path) {
      pending_paths_.clear();
    }
    work = collect_work_locked(effects);
  }
  execute_work(std::move(work));
}

void MppiFollowPathProxy::deactivate()
{
  std::vector<ActionWork> work;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    pending_paths_.clear();
    work = collect_work_locked(state_.deactivate());
  }
  execute_work(std::move(work));
}

std::vector<MppiFollowPathProxy::ActionWork>
MppiFollowPathProxy::collect_work_locked(
  const MppiFollowPathEffects & effects)
{
  std::vector<ActionWork> work;
  work.reserve(effects.actions.size());
  for (const auto & action : effects.actions) {
    ActionWork item;
    item.type = action.type;
    item.generation = action.generation;
    if (action.type == MppiFollowPathActionType::kSendGoal) {
      const auto path = pending_paths_.find(action.generation);
      if (path == pending_paths_.end()) {
        continue;
      }
      item.path = std::move(path->second);
      pending_paths_.erase(path);
    } else {
      auto handle = goal_handles_.take_for_cancel(action.generation);
      if (!handle) {
        continue;
      }
      item.goal_handle = std::move(*handle);
    }
    work.push_back(std::move(item));
  }
  return work;
}

void MppiFollowPathProxy::execute_work(std::vector<ActionWork> work)
{
  for (auto & item : work) {
    if (item.type == MppiFollowPathActionType::kSendGoal) {
      send_goal(std::move(item));
    } else {
      cancel_goal(item);
    }
  }
}

void MppiFollowPathProxy::send_goal(ActionWork work)
{
  FollowPath::Goal goal;
  goal.path = std::move(work.path);
  goal.controller_id = "FollowPath";
  goal.goal_checker_id.clear();

  rclcpp_action::Client<FollowPath>::SendGoalOptions options;
  const std::uint64_t generation = work.generation;
  const auto self =
    std::static_pointer_cast<MppiFollowPathProxy>(shared_from_this());
  const std::weak_ptr<MppiFollowPathProxy> weak_self(self);
  options.goal_response_callback =
    [weak_self, generation](const GoalHandle::SharedPtr goal_handle) {
      if (const auto locked = weak_self.lock()) {
        locked->on_goal_response(generation, goal_handle);
      }
    };
  options.result_callback =
    [weak_self, generation](const GoalHandle::WrappedResult & result) {
      if (const auto locked = weak_self.lock()) {
        locked->on_goal_result(generation, result);
      }
    };

  try {
    static_cast<void>(action_client_->async_send_goal(goal, options));
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      get_logger(), "failed to send FollowPath goal: %s",
      error.what());
    on_goal_response(generation, nullptr);
  }
}

void MppiFollowPathProxy::cancel_goal(const ActionWork & work)
{
  if (!work.goal_handle) {
    return;
  }
  try {
    static_cast<void>(
      action_client_->async_cancel_goal(work.goal_handle));
  } catch (const std::exception & error) {
    RCLCPP_WARN(
      get_logger(), "best-effort FollowPath cancellation failed: %s",
      error.what());
  }
}

}  // namespace ad_planner
