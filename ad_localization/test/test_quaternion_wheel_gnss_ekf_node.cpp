#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>

#include "ad_localization/quaternion_wheel_gnss_ekf/quaternion_wheel_gnss_ekf_node.hpp"
#include "rclcpp/rclcpp.hpp"

namespace
{

class QuaternionWheelGnssEkfNodeTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    rclcpp::init(0, nullptr);
  }

  static void TearDownTestSuite()
  {
    rclcpp::shutdown();
  }
};

TEST_F(QuaternionWheelGnssEkfNodeTest, RejectsTfPublicationOverride)
{
  rclcpp::NodeOptions options;
  options.append_parameter_override("publish_tf", true);

  EXPECT_THROW(
    ad_localization::make_quaternion_wheel_gnss_ekf_node(options),
    std::invalid_argument);
}

TEST_F(QuaternionWheelGnssEkfNodeTest, UsesExactTopicsTypesQosAndResetService)
{
  auto node = ad_localization::make_quaternion_wheel_gnss_ekf_node();
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    executor.spin_some();
    if (!node->get_subscriptions_info_by_topic(
        "/ad/localization/input/wheel_speed").empty() &&
      !node->get_publishers_info_by_topic(
        "/ad/localization/backends/quaternion_wheel_gnss_ekf/odometry").empty())
    {
      break;
    }
  }

  const auto imu = node->get_subscriptions_info_by_topic(
    "/ad/sensors/imu/data");
  const auto wheel = node->get_subscriptions_info_by_topic(
    "/ad/localization/input/wheel_speed");
  const auto gnss = node->get_subscriptions_info_by_topic(
    "/ad/localization/input/gnss_pose");
  const auto output = node->get_publishers_info_by_topic(
    "/ad/localization/backends/quaternion_wheel_gnss_ekf/odometry");
  ASSERT_EQ(imu.size(), 1U);
  ASSERT_EQ(wheel.size(), 1U);
  ASSERT_EQ(gnss.size(), 1U);
  ASSERT_EQ(output.size(), 1U);
  EXPECT_EQ(imu.front().topic_type(), "sensor_msgs/msg/Imu");
  EXPECT_EQ(
    wheel.front().topic_type(),
    "geometry_msgs/msg/TwistWithCovarianceStamped");
  EXPECT_EQ(gnss.front().topic_type(), "geometry_msgs/msg/PoseStamped");
  EXPECT_EQ(output.front().topic_type(), "nav_msgs/msg/Odometry");
  EXPECT_EQ(
    imu.front().qos_profile().reliability(),
    rclcpp::ReliabilityPolicy::BestEffort);
  EXPECT_EQ(
    wheel.front().qos_profile().reliability(),
    rclcpp::ReliabilityPolicy::Reliable);
  EXPECT_EQ(
    gnss.front().qos_profile().reliability(),
    rclcpp::ReliabilityPolicy::Reliable);
  EXPECT_EQ(
    output.front().qos_profile().reliability(),
    rclcpp::ReliabilityPolicy::Reliable);

  const auto services = node->get_service_names_and_types();
  EXPECT_EQ(
    services.at("/ad/localization/reset_quaternion_wheel_gnss_ekf"),
    (std::vector<std::string>{"std_srvs/srv/Trigger"}));
  EXPECT_EQ(node->count_publishers("/tf"), 0U);
  EXPECT_EQ(node->count_publishers("/tf_static"), 0U);
  EXPECT_EQ(node->count_publishers("/ad/localization/odometry"), 0U);
}

}  // namespace
