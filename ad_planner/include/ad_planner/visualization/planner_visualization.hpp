#ifndef AD_PLANNER__VISUALIZATION__PLANNER_VISUALIZATION_HPP_
#define AD_PLANNER__VISUALIZATION__PLANNER_VISUALIZATION_HPP_

#include <optional>
#include <string>

#include <nav_msgs/msg/path.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/time.hpp>
#include <std_msgs/msg/float32.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "ad_planner/common/types.hpp"
#include "ad_planner/local_planning/common/local_motion.hpp"
#include "ad_planner/visualization/path_tracking_markers.hpp"

namespace ad_planner {

struct PlannerVisualizationTopics {
  std::string global_path;
  std::string local_path;
  std::string candidate_paths;
  std::string path_tracking;
  std::string occupancy_relevance;
  std::string planner_relevant_objects;
  std::string target;
  std::string target_speed;
};

struct LocalMotionVisualization {
  nav_msgs::msg::Path selected_path;
  visualization_msgs::msg::MarkerArray candidates;
};

struct ControllerVisualization {
  std::optional<std_msgs::msg::Float32> target_speed;
  std::optional<nav_msgs::msg::Path> local_path;
  std::optional<visualization_msgs::msg::Marker> target;
};

nav_msgs::msg::Path make_global_path_message(const Route &route,
                                             const std::string &frame_id,
                                             const rclcpp::Time &stamp);

LocalMotionVisualization
make_local_motion_visualization(const LocalPlanningResult *result,
                                const std::string &frame_id,
                                const rclcpp::Time &stamp);

ControllerVisualization
make_controller_visualization(const ControllerResult &result,
                              const std::string &frame_id,
                              const rclcpp::Time &stamp);

class PlannerVisualization {
public:
  PlannerVisualization(rclcpp::Node &node, PlannerVisualizationTopics topics);

  void publish_global_path(const Route &route, const std::string &frame_id,
                           const rclcpp::Time &stamp);

  void publish_local_motion(const LocalPlanningResult *result,
                            const std::string &frame_id,
                            const rclcpp::Time &stamp);

  void publish_controller(const ControllerResult &result,
                          const std::string &frame_id,
                          const rclcpp::Time &stamp);

  void publish_path_tracking(visualization_msgs::msg::MarkerArray route_profile,
                             const rclcpp::Time &stamp);

  void publish_route_profile(visualization_msgs::msg::MarkerArray markers,
                             const rclcpp::Time &stamp);

  void publish_occupancy_relevance(
      const visualization_msgs::msg::MarkerArray &markers);

  void publish_planner_relevant_objects(
      const visualization_msgs::msg::MarkerArray &markers);

private:
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      candidate_paths_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      path_tracking_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      occupancy_relevance_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      planner_relevant_objects_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      target_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr target_speed_publisher_;
};

} // namespace ad_planner

#endif // AD_PLANNER__VISUALIZATION__PLANNER_VISUALIZATION_HPP_
