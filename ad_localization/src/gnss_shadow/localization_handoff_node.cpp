#include "ad_localization/gnss_shadow/localization_handoff_node.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ad_localization/gnss_shadow/localization_handoff.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace ad_localization
{
namespace
{

diagnostic_msgs::msg::KeyValue diagnostic_value(
  std::string key, std::string value)
{
  diagnostic_msgs::msg::KeyValue result;
  result.key = std::move(key);
  result.value = std::move(value);
  return result;
}

class LocalizationHandoffNode final : public rclcpp::Node
{
public:
  explicit LocalizationHandoffNode(const rclcpp::NodeOptions & options)
  : rclcpp::Node("localization_handoff", options)
  {
    declare_parameter<std::string>(
      "gnss_odometry_topic", "/ad/localization/backends/gnss_imu/odometry");
    declare_parameter<std::string>(
      "fastlio_odometry_topic", "/ad/localization/backends/fastlio/odometry");
    declare_parameter<std::string>(
      "canonical_odometry_topic", "/ad/localization/odometry");
    declare_parameter<std::string>(
      "fastlio_initial_pose_topic", "/ad/localization/input/initial_pose");
    declare_parameter<std::string>("diagnostics_topic", "/diagnostics");
    declare_parameter<std::string>("reference_frame", "odom");
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<std::vector<double>>(
      "entry_xy_m", {38.868875371112615, -480.68740975673563});
    declare_parameter<std::vector<double>>(
      "exit_xy_m", {-81.83284234744308, -547.3316347631321});
    declare_parameter<double>("prewarm_radius_m", 20.0);
    declare_parameter<double>("entry_switch_radius_m", 8.0);
    declare_parameter<double>("exit_switch_radius_m", 20.0);
    declare_parameter<double>("source_timeout_sec", 0.5);
    declare_parameter<double>("maximum_position_disagreement_m", 2.0);
    declare_parameter<double>("maximum_yaw_disagreement_rad", 0.20);
    declare_parameter<double>("blend_duration_sec", 2.0);
    declare_parameter<double>("diagnostics_period_sec", 1.0);

    const auto entry = get_parameter("entry_xy_m").as_double_array();
    const auto exit = get_parameter("exit_xy_m").as_double_array();
    if (entry.size() != 2U || exit.size() != 2U) {
      throw std::invalid_argument("entry_xy_m and exit_xy_m must each contain two values");
    }

    HandoffConfig config;
    config.entry_xy = {entry[0], entry[1]};
    config.exit_xy = {exit[0], exit[1]};
    config.prewarm_radius_m = get_parameter("prewarm_radius_m").as_double();
    config.entry_switch_radius_m =
      get_parameter("entry_switch_radius_m").as_double();
    config.exit_switch_radius_m = get_parameter("exit_switch_radius_m").as_double();
    config.source_timeout_sec = get_parameter("source_timeout_sec").as_double();
    config.maximum_position_disagreement_m =
      get_parameter("maximum_position_disagreement_m").as_double();
    config.maximum_yaw_disagreement_rad =
      get_parameter("maximum_yaw_disagreement_rad").as_double();
    config.blend_duration_sec = get_parameter("blend_duration_sec").as_double();
    config.reference_frame = get_parameter("reference_frame").as_string();
    config.base_frame = get_parameter("base_frame").as_string();
    source_timeout_sec_ = config.source_timeout_sec;
    handoff_ = std::make_unique<LocalizationHandoff>(std::move(config));

    diagnostics_period_sec_ = get_parameter("diagnostics_period_sec").as_double();
    if (!std::isfinite(diagnostics_period_sec_) || diagnostics_period_sec_ <= 0.0) {
      throw std::invalid_argument("diagnostics_period_sec must be positive");
    }

    const auto reliable_qos =
      rclcpp::QoS(rclcpp::KeepLast(20)).reliable().durability_volatile();
    canonical_publisher_ = create_publisher<nav_msgs::msg::Odometry>(
      get_parameter("canonical_odometry_topic").as_string(), reliable_qos);
    initial_pose_publisher_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      get_parameter("fastlio_initial_pose_topic").as_string(),
      rclcpp::QoS(1).reliable().transient_local());
    diagnostics_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
      get_parameter("diagnostics_topic").as_string(), rclcpp::QoS(10).reliable());
    transform_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    gnss_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      get_parameter("gnss_odometry_topic").as_string(), reliable_qos,
      [this](nav_msgs::msg::Odometry::SharedPtr message) {
        process(Backend::kGnssImu, *message);
      });
    fastlio_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      get_parameter("fastlio_odometry_topic").as_string(), reliable_qos,
      [this](nav_msgs::msg::Odometry::SharedPtr message) {
        process(Backend::kFastlio, *message);
      });
    diagnostics_timer_ = create_wall_timer(
      std::chrono::duration<double>(diagnostics_period_sec_),
      std::bind(&LocalizationHandoffNode::publish_diagnostics, this));
  }

private:
  static double steady_time_sec()
  {
    return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void process(Backend backend, const nav_msgs::msg::Odometry & candidate)
  {
    const double receipt_time_sec = steady_time_sec();
    const auto update = handoff_->observe(backend, candidate, receipt_time_sec);
    if (update.fastlio_initial_pose) {
      initial_pose_publisher_->publish(*update.fastlio_initial_pose);
      RCLCPP_INFO_ONCE(
        get_logger(), "FastLIO prewarm initialized near the fixed-map entrance");
    }
    if (!update.canonical_odometry) {
      return;
    }

    canonical_publisher_->publish(*update.canonical_odometry);
    geometry_msgs::msg::TransformStamped transform;
    transform.header = update.canonical_odometry->header;
    transform.child_frame_id = update.canonical_odometry->child_frame_id;
    transform.transform.translation.x =
      update.canonical_odometry->pose.pose.position.x;
    transform.transform.translation.y =
      update.canonical_odometry->pose.pose.position.y;
    transform.transform.translation.z =
      update.canonical_odometry->pose.pose.position.z;
    transform.transform.rotation =
      update.canonical_odometry->pose.pose.orientation;
    transform_broadcaster_->sendTransform(transform);
    last_canonical_receipt_sec_ = receipt_time_sec;

    if (update.switch_count != last_reported_switch_count_) {
      last_reported_switch_count_ = update.switch_count;
      RCLCPP_INFO(
        get_logger(), "localization handoff selected %s in phase %s",
        to_string(update.active_backend), to_string(update.phase));
    }
  }

  void publish_diagnostics()
  {
    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "ad_localization/handoff";
    status.hardware_id = "morai_cp14_cp15_fixed_map";

    const double age = last_canonical_receipt_sec_ ?
      steady_time_sec() - *last_canonical_receipt_sec_ :
      std::numeric_limits<double>::infinity();
    if (!std::isfinite(age) || age > source_timeout_sec_) {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
      status.message = "selected localization output is stale";
    } else {
      status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      status.message = "selected localization output is fresh";
    }
    status.values = {
      diagnostic_value("active_backend", to_string(handoff_->active_backend())),
      diagnostic_value("phase", to_string(handoff_->phase())),
      diagnostic_value("switch_count", std::to_string(handoff_->switch_count())),
    };
    array.status.push_back(std::move(status));
    diagnostics_publisher_->publish(array);
  }

  std::unique_ptr<LocalizationHandoff> handoff_;
  double source_timeout_sec_{0.5};
  double diagnostics_period_sec_{1.0};
  std::size_t last_reported_switch_count_{0U};
  std::optional<double> last_canonical_receipt_sec_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr canonical_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr
    initial_pose_publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    diagnostics_publisher_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gnss_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr fastlio_subscription_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> transform_broadcaster_;
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;
};

}  // namespace

std::shared_ptr<rclcpp::Node> make_localization_handoff_node(
  const rclcpp::NodeOptions & options)
{
  return std::make_shared<LocalizationHandoffNode>(options);
}

}  // namespace ad_localization
