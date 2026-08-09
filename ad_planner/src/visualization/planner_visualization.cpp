#include "ad_planner/visualization/planner_visualization.hpp"

#include <cmath>
#include <cstdint>
#include <utility>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include "ad_planner/local_planning/common/local_motion_validation.hpp"

namespace ad_planner {
namespace {

visualization_msgs::msg::Marker delete_all_marker(const std::string &frame_id,
                                                  const rclcpp::Time &stamp) {
  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = stamp;
  marker.action = visualization_msgs::msg::Marker::DELETEALL;
  return marker;
}

} // namespace

nav_msgs::msg::Path make_global_path_message(const Route &route,
                                             const std::string &frame_id,
                                             const rclcpp::Time &stamp) {
  nav_msgs::msg::Path message;
  message.header.stamp = stamp;
  message.header.frame_id = frame_id;
  message.poses.reserve(route.points.size());
  for (const auto &point : route.points) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = message.header;
    pose.pose.position.x = point.x;
    pose.pose.position.y = point.y;
    pose.pose.position.z = point.z;
    pose.pose.orientation.w = 1.0;
    message.poses.push_back(std::move(pose));
  }
  return message;
}

LocalMotionVisualization
make_local_motion_visualization(const LocalPlanningResult *result,
                                const std::string &frame_id,
                                const rclcpp::Time &stamp) {
  LocalMotionVisualization messages;
  messages.selected_path.header.stamp = stamp;
  messages.selected_path.header.frame_id = frame_id;
  messages.candidates.markers.push_back(delete_all_marker(frame_id, stamp));

  if (!result) {
    return messages;
  }
  if (valid_timed_trajectory(result->trajectory, frame_id)) {
    messages.selected_path.poses.reserve(result->trajectory.points.size());
    for (const auto &point : result->trajectory.points) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = messages.selected_path.header;
      pose.pose.position.x = point.pose.x;
      pose.pose.position.y = point.pose.y;
      pose.pose.orientation.z = std::sin(point.pose.yaw_rad * 0.5);
      pose.pose.orientation.w = std::cos(point.pose.yaw_rad * 0.5);
      messages.selected_path.poses.push_back(std::move(pose));
    }
  }

  for (std::size_t index = 0U; index < result->candidate_trajectories.size();
       ++index) {
    const auto &candidate = result->candidate_trajectories[index];
    if (!valid_timed_trajectory(candidate, frame_id)) {
      continue;
    }
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = frame_id;
    marker.ns = "ad_planner_candidates";
    marker.id = static_cast<std::int32_t>(index);
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.05;
    marker.color.a = 0.75;
    marker.color.r = 0.2;
    marker.color.g = 0.7;
    marker.color.b = 1.0;
    marker.points.reserve(candidate.points.size());
    for (const auto &point : candidate.points) {
      geometry_msgs::msg::Point marker_point;
      marker_point.x = point.pose.x;
      marker_point.y = point.pose.y;
      marker.points.push_back(std::move(marker_point));
    }
    messages.candidates.markers.push_back(std::move(marker));
  }
  return messages;
}

ControllerVisualization
make_controller_visualization(const ControllerResult &result,
                              const std::string &frame_id,
                              const rclcpp::Time &stamp) {
  ControllerVisualization messages;
  if (result.target_speed_mps) {
    std_msgs::msg::Float32 target_speed;
    target_speed.data = static_cast<float>(*result.target_speed_mps);
    messages.target_speed = target_speed;
  }
  if (result.local_trajectory) {
    nav_msgs::msg::Path path;
    path.header.stamp = stamp;
    path.header.frame_id = frame_id;
    path.poses.reserve(result.local_trajectory->poses.size());
    for (const auto &point : result.local_trajectory->poses) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = point.x;
      pose.pose.position.y = point.y;
      pose.pose.orientation.w = 1.0;
      path.poses.push_back(std::move(pose));
    }
    messages.local_path = std::move(path);
  }
  if (result.target) {
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = stamp;
    marker.header.frame_id = frame_id;
    marker.ns = "ad_planner";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = result.target->x;
    marker.pose.position.y = result.target->y;
    marker.pose.position.z = result.target->z;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = marker.scale.y = marker.scale.z = 0.6;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    messages.target = std::move(marker);
  }
  return messages;
}

PlannerVisualization::PlannerVisualization(rclcpp::Node &node,
                                           PlannerVisualizationTopics topics) {
  const auto reliable_qos = rclcpp::QoS(1).reliable();
  path_publisher_ = node.create_publisher<nav_msgs::msg::Path>(
      topics.global_path, rclcpp::QoS(1).reliable().transient_local());
  local_path_publisher_ = node.create_publisher<nav_msgs::msg::Path>(
      topics.local_path, reliable_qos);
  candidate_paths_publisher_ =
      node.create_publisher<visualization_msgs::msg::MarkerArray>(
          topics.candidate_paths, reliable_qos);
  path_tracking_publisher_ =
      node.create_publisher<visualization_msgs::msg::MarkerArray>(
          topics.path_tracking, rclcpp::QoS(1).reliable().transient_local());
  occupancy_relevance_publisher_ =
      node.create_publisher<visualization_msgs::msg::MarkerArray>(
          topics.occupancy_relevance, reliable_qos);
  planner_relevant_objects_publisher_ =
      node.create_publisher<visualization_msgs::msg::MarkerArray>(
          topics.planner_relevant_objects, reliable_qos);
  target_publisher_ = node.create_publisher<visualization_msgs::msg::Marker>(
      topics.target, reliable_qos);
  target_speed_publisher_ = node.create_publisher<std_msgs::msg::Float32>(
      topics.target_speed, reliable_qos);
}

void PlannerVisualization::publish_global_path(const Route &route,
                                               const std::string &frame_id,
                                               const rclcpp::Time &stamp) {
  path_publisher_->publish(make_global_path_message(route, frame_id, stamp));
}

void PlannerVisualization::publish_local_motion(
    const LocalPlanningResult *result, const std::string &frame_id,
    const rclcpp::Time &stamp) {
  const auto messages =
      make_local_motion_visualization(result, frame_id, stamp);
  local_path_publisher_->publish(messages.selected_path);
  candidate_paths_publisher_->publish(messages.candidates);
}

void PlannerVisualization::publish_controller(const ControllerResult &result,
                                              const std::string &frame_id,
                                              const rclcpp::Time &stamp) {
  const auto messages = make_controller_visualization(result, frame_id, stamp);
  if (messages.target_speed) {
    target_speed_publisher_->publish(*messages.target_speed);
  }
  if (messages.local_path) {
    local_path_publisher_->publish(*messages.local_path);
  }
  if (messages.target) {
    target_publisher_->publish(*messages.target);
  }
}

void PlannerVisualization::publish_path_tracking(
    visualization_msgs::msg::MarkerArray route_profile,
    const rclcpp::Time &stamp) {
  for (auto &marker : route_profile.markers) {
    marker.header.stamp = stamp;
  }
  path_tracking_publisher_->publish(route_profile);
}

void PlannerVisualization::publish_route_profile(
    visualization_msgs::msg::MarkerArray markers, const rclcpp::Time &stamp) {
  for (auto &marker : markers.markers) {
    marker.header.stamp = stamp;
  }
  path_tracking_publisher_->publish(markers);
}

void PlannerVisualization::publish_occupancy_relevance(
    const visualization_msgs::msg::MarkerArray &markers) {
  occupancy_relevance_publisher_->publish(markers);
}

void PlannerVisualization::publish_planner_relevant_objects(
    const visualization_msgs::msg::MarkerArray &markers) {
  planner_relevant_objects_publisher_->publish(markers);
}

} // namespace ad_planner
