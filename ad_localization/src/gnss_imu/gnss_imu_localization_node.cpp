#include "ad_localization/gnss_imu/gnss_imu_localization_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ad_localization/gnss_imu/gnss_imu_localizer.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace ad_localization
{
namespace
{

geometry_msgs::msg::Quaternion quaternion_from_rpy(double roll, double pitch, double yaw)
{
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);
  geometry_msgs::msg::Quaternion result;
  result.x = sr * cp * cy - cr * sp * sy;
  result.y = cr * sp * cy + sr * cp * sy;
  result.z = cr * cp * sy - sr * sp * cy;
  result.w = cr * cp * cy + sr * sp * sy;
  return result;
}

class GnssImuLocalizationNode final : public rclcpp::Node
{
public:
  explicit GnssImuLocalizationNode(const rclcpp::NodeOptions & options)
  : rclcpp::Node("gnss_imu_localization", options)
  {
    declare_parameter<std::string>("gnss_pose_topic", "/ad/localization/input/gnss_pose");
    declare_parameter<std::string>("imu_topic", "/ad/sensors/imu/data");
    declare_parameter<std::string>("wheel_speed_topic", "/ad/localization/input/wheel_speed");
    declare_parameter<std::string>("output_odometry_topic", "/ad/localization/odometry");
    declare_parameter<std::string>("reference_frame", "odom");
    declare_parameter<std::string>("base_frame", "base_link");
    declare_parameter<std::string>("imu_frame", "imu_link");
    declare_parameter<std::vector<double>>("gnss_lever_arm_m", {0.0, 0.0, 0.0});
    declare_parameter<std::vector<double>>("imu_mount_rpy_rad", {0.0, 0.0, 0.0});
    declare_parameter<double>("synchronization_tolerance_sec", 0.1);
    declare_parameter<double>("gnss_xy_variance_m2", 0.25);
    declare_parameter<double>("gnss_z_variance_m2", 0.25);
    declare_parameter<double>("imu_orientation_variance_rad2", 0.0001);
    declare_parameter<double>("unobserved_twist_variance", 1.0e6);
    declare_parameter<double>("world_yaw_offset_rad", 0.0);
    declare_parameter<bool>("publish_tf", true);

    GnssImuLocalizerConfig config;
    config.reference_frame = get_parameter("reference_frame").as_string();
    config.base_frame = get_parameter("base_frame").as_string();
    config.imu_frame = get_parameter("imu_frame").as_string();
    config.synchronization_tolerance_sec =
      get_parameter("synchronization_tolerance_sec").as_double();
    config.gnss_xy_variance_m2 = get_parameter("gnss_xy_variance_m2").as_double();
    config.gnss_z_variance_m2 = get_parameter("gnss_z_variance_m2").as_double();
    config.imu_orientation_variance_rad2 =
      get_parameter("imu_orientation_variance_rad2").as_double();
    config.unobserved_twist_variance =
      get_parameter("unobserved_twist_variance").as_double();
    config.world_yaw_offset_rad =
      get_parameter("world_yaw_offset_rad").as_double();
    const auto lever_arm = get_parameter("gnss_lever_arm_m").as_double_array();
    const auto imu_mount_rpy = get_parameter("imu_mount_rpy_rad").as_double_array();
    if (lever_arm.size() != 3 || imu_mount_rpy.size() != 3) {
      throw std::invalid_argument(
              "gnss_lever_arm_m and imu_mount_rpy_rad must contain exactly three values");
    }
    if (!std::all_of(
        imu_mount_rpy.begin(), imu_mount_rpy.end(), [](double value) {
          return std::isfinite(value);
        }))
    {
      throw std::invalid_argument("imu_mount_rpy_rad values must be finite");
    }
    std::copy(lever_arm.begin(), lever_arm.end(), config.gnss_lever_arm_m.begin());
    config.base_to_imu_orientation = quaternion_from_rpy(
      imu_mount_rpy[0], imu_mount_rpy[1], imu_mount_rpy[2]);
    localizer_ = std::make_unique<GnssImuLocalizer>(std::move(config));

    const auto reliable_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    odometry_publisher_ = create_publisher<nav_msgs::msg::Odometry>(
      get_parameter("output_odometry_topic").as_string(), reliable_qos);
    publish_tf_ = get_parameter("publish_tf").as_bool();
    if (publish_tf_) {
      transform_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }
    imu_subscription_ = create_subscription<sensor_msgs::msg::Imu>(
      get_parameter("imu_topic").as_string(), rclcpp::SensorDataQoS(),
      std::bind(&GnssImuLocalizationNode::imu_callback, this, std::placeholders::_1));
    wheel_subscription_ = create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
      get_parameter("wheel_speed_topic").as_string(), reliable_qos,
      std::bind(&GnssImuLocalizationNode::wheel_callback, this, std::placeholders::_1));
    gnss_subscription_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      get_parameter("gnss_pose_topic").as_string(), reliable_qos,
      std::bind(&GnssImuLocalizationNode::gnss_callback, this, std::placeholders::_1));
    reset_service_ = create_service<std_srvs::srv::Trigger>(
      "/ad/localization/reset_gnss_imu",
      [this](
        const std_srvs::srv::Trigger::Request::SharedPtr,
        std_srvs::srv::Trigger::Response::SharedPtr response)
      {
        localizer_->reset();
        response->success = true;
        response->message = "GNSS/IMU localizer state reset";
      });
  }

private:
  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu)
  {
    if (!localizer_->observe_imu(*imu)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "dropping invalid direct-localizer IMU sample");
    }
  }

  void wheel_callback(const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr wheel)
  {
    if (!localizer_->observe_wheel_speed(*wheel)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "dropping invalid direct-localizer wheel sample");
    }
  }

  void gnss_callback(const geometry_msgs::msg::PoseStamped::SharedPtr antenna_pose)
  {
    const auto odometry = localizer_->observe_gnss(*antenna_pose);
    if (!odometry) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "direct-localizer GNSS sample lacks a synchronized IMU orientation");
      return;
    }
    odometry_publisher_->publish(*odometry);

    geometry_msgs::msg::TransformStamped transform;
    transform.header = odometry->header;
    transform.child_frame_id = odometry->child_frame_id;
    transform.transform.translation.x = odometry->pose.pose.position.x;
    transform.transform.translation.y = odometry->pose.pose.position.y;
    transform.transform.translation.z = odometry->pose.pose.position.z;
    transform.transform.rotation = odometry->pose.pose.orientation;
    if (publish_tf_ && transform_broadcaster_) {
      transform_broadcaster_->sendTransform(transform);
    }
  }

  std::unique_ptr<GnssImuLocalizer> localizer_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr
    wheel_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr gnss_subscription_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> transform_broadcaster_;
  bool publish_tf_{true};
};

}  // namespace

std::shared_ptr<rclcpp::Node> make_gnss_imu_localization_node(
  const rclcpp::NodeOptions & options)
{
  return std::make_shared<GnssImuLocalizationNode>(options);
}

}  // namespace ad_localization
