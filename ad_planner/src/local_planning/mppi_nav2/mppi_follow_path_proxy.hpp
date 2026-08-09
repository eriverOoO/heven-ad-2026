#ifndef AD_PLANNER__SRC__LOCAL_PLANNING__MPPI_FOLLOW_PATH_PROXY_HPP_
#define AD_PLANNER__SRC__LOCAL_PLANNING__MPPI_FOLLOW_PATH_PROXY_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <nav2_msgs/action/follow_path.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "ad_planner/local_planning/mppi_nav2/mppi_follow_path_goal_handle_registry.hpp"
#include "ad_planner/local_planning/mppi_nav2/mppi_follow_path_state.hpp"

namespace ad_planner
{

class MppiFollowPathProxy final : public rclcpp::Node
{
public:
  explicit MppiFollowPathProxy(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  void deactivate();

private:
  using FollowPath = nav2_msgs::action::FollowPath;
  using GoalHandle = rclcpp_action::ClientGoalHandle<FollowPath>;

  struct ActionWork
  {
    MppiFollowPathActionType type{MppiFollowPathActionType::kSendGoal};
    std::uint64_t generation{0U};
    nav_msgs::msg::Path path;
    GoalHandle::SharedPtr goal_handle;
  };

  void on_path(const nav_msgs::msg::Path::ConstSharedPtr message);
  void on_timer();
  void on_goal_response(
    std::uint64_t generation, const GoalHandle::SharedPtr & goal_handle);
  void on_goal_result(
    std::uint64_t generation, const GoalHandle::WrappedResult & result);

  std::vector<ActionWork> collect_work_locked(
    const MppiFollowPathEffects & effects);
  void execute_work(std::vector<ActionWork> work);
  void send_goal(ActionWork work);
  void cancel_goal(const ActionWork & work);

  rclcpp::Clock steady_clock_;
  MppiFollowPathState state_;
  std::size_t maximum_pose_count_{0U};
  std::mutex state_mutex_;
  std::unordered_map<std::uint64_t, nav_msgs::msg::Path> pending_paths_;
  MppiFollowPathGoalHandleRegistry<GoalHandle::SharedPtr> goal_handles_;
  rclcpp_action::Client<FollowPath>::SharedPtr action_client_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__SRC__LOCAL_PLANNING__MPPI_FOLLOW_PATH_PROXY_HPP_
