#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "ad_localization/gnss_imu/gnss_imu_localizer.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "gtest/gtest.h"
#include "sensor_msgs/msg/imu.hpp"

namespace
{

constexpr double kPi = 3.14159265358979323846;

builtin_interfaces::msg::Time stamp(double seconds)
{
  builtin_interfaces::msg::Time result;
  result.sec = static_cast<std::int32_t>(std::floor(seconds));
  result.nanosec = static_cast<std::uint32_t>(
    std::llround((seconds - static_cast<double>(result.sec)) * 1.0e9));
  return result;
}

geometry_msgs::msg::Quaternion yaw_quaternion(double yaw, double scale = 1.0)
{
  geometry_msgs::msg::Quaternion result;
  result.z = scale * std::sin(yaw * 0.5);
  result.w = scale * std::cos(yaw * 0.5);
  return result;
}

geometry_msgs::msg::Quaternion rpy_quaternion(double roll, double pitch, double yaw)
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

sensor_msgs::msg::Imu imu_sample(
  double seconds, double world_imu_yaw, const std::string & frame = "imu_link")
{
  sensor_msgs::msg::Imu result;
  result.header.stamp = stamp(seconds);
  result.header.frame_id = frame;
  result.orientation = yaw_quaternion(world_imu_yaw, 4.0);
  return result;
}

geometry_msgs::msg::TwistWithCovarianceStamped wheel_sample(
  double seconds, double signed_speed, const std::string & frame = "base_link")
{
  geometry_msgs::msg::TwistWithCovarianceStamped result;
  result.header.stamp = stamp(seconds);
  result.header.frame_id = frame;
  result.twist.twist.linear.x = signed_speed;
  result.twist.covariance[0] = 0.16;
  return result;
}

geometry_msgs::msg::PoseStamped gnss_sample(
  double seconds, double x = 10.0, double y = 20.0, double z = 1.0,
  const std::string & frame = "odom")
{
  geometry_msgs::msg::PoseStamped result;
  result.header.stamp = stamp(seconds);
  result.header.frame_id = frame;
  result.pose.position.x = x;
  result.pose.position.y = y;
  result.pose.position.z = z;
  result.pose.orientation.w = 1.0;
  return result;
}

ad_localization::GnssImuLocalizerConfig default_config()
{
  ad_localization::GnssImuLocalizerConfig result;
  result.reference_frame = "odom";
  result.base_frame = "base_link";
  result.imu_frame = "imu_link";
  result.synchronization_tolerance_sec = 0.1;
  return result;
}

double quaternion_norm(const geometry_msgs::msg::Quaternion & quaternion)
{
  return std::sqrt(
    quaternion.x * quaternion.x + quaternion.y * quaternion.y +
    quaternion.z * quaternion.z + quaternion.w * quaternion.w);
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & quaternion)
{
  return std::atan2(
    2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
    1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z));
}

TEST(GnssImuLocalizerTest, ConvertsNormalizedImuOrientationToBaseAndCompensatesLeverArmOnce)
{
  auto config = default_config();
  config.gnss_lever_arm_m = {2.0, 0.0, 1.0};
  config.base_to_imu_orientation = yaw_quaternion(kPi / 6.0);
  ad_localization::GnssImuLocalizer localizer(config);

  EXPECT_TRUE(localizer.observe_imu(imu_sample(10.02, 2.0 * kPi / 3.0)));
  const auto odometry = localizer.observe_gnss(gnss_sample(10.0));

  ASSERT_TRUE(odometry.has_value());
  EXPECT_EQ(odometry->header.stamp, stamp(10.0));
  EXPECT_EQ(odometry->header.frame_id, "odom");
  EXPECT_EQ(odometry->child_frame_id, "base_link");
  EXPECT_NEAR(quaternion_norm(odometry->pose.pose.orientation), 1.0, 1.0e-12);
  EXPECT_NEAR(yaw_from_quaternion(odometry->pose.pose.orientation), kPi / 2.0, 1.0e-12);
  EXPECT_NEAR(odometry->pose.pose.position.x, 10.0, 1.0e-12);
  EXPECT_NEAR(odometry->pose.pose.position.y, 18.0, 1.0e-12);
  EXPECT_NEAR(odometry->pose.pose.position.z, 0.0, 1.0e-12);
}

TEST(GnssImuLocalizerTest, ComposesNoncommutativeImuMountRotationInTheCorrectOrder)
{
  auto config = default_config();
  config.base_to_imu_orientation = rpy_quaternion(0.3, -0.2, 0.1);
  const auto expected_world_base = rpy_quaternion(-0.25, 0.4, 0.8);
  const auto world_imu = multiply(expected_world_base, config.base_to_imu_orientation);
  ad_localization::GnssImuLocalizer localizer(config);

  auto imu = imu_sample(15.0, 0.0);
  imu.orientation = world_imu;
  ASSERT_TRUE(localizer.observe_imu(imu));
  const auto odometry = localizer.observe_gnss(gnss_sample(15.0));

  ASSERT_TRUE(odometry.has_value());
  const auto & actual = odometry->pose.pose.orientation;
  const double absolute_dot = std::abs(
    actual.x * expected_world_base.x + actual.y * expected_world_base.y +
    actual.z * expected_world_base.z + actual.w * expected_world_base.w);
  EXPECT_NEAR(absolute_dot, 1.0, 1.0e-12);
}

TEST(GnssImuLocalizerTest, AppliesWorldYawOffsetToOrientationAndLeverArm)
{
  auto config = default_config();
  config.world_yaw_offset_rad = -0.2;
  config.gnss_lever_arm_m = {2.0, 0.0, 0.0};
  ad_localization::GnssImuLocalizer localizer(config);

  ASSERT_TRUE(localizer.observe_imu(imu_sample(16.0, 0.7)));
  const auto odometry = localizer.observe_gnss(gnss_sample(16.0));

  ASSERT_TRUE(odometry.has_value());
  EXPECT_NEAR(yaw_from_quaternion(odometry->pose.pose.orientation), 0.5, 1.0e-12);
  EXPECT_NEAR(odometry->pose.pose.position.x, 10.0 - 2.0 * std::cos(0.5), 1.0e-12);
  EXPECT_NEAR(odometry->pose.pose.position.y, 20.0 - 2.0 * std::sin(0.5), 1.0e-12);
}

TEST(GnssImuLocalizerTest, PublishesWithoutWheelAndUsesItOnlyWhenSynchronized)
{
  ad_localization::GnssImuLocalizer localizer(default_config());
  ASSERT_TRUE(localizer.observe_imu(imu_sample(20.0, 0.0)));

  const auto without_wheel = localizer.observe_gnss(gnss_sample(20.0));
  ASSERT_TRUE(without_wheel.has_value());
  EXPECT_DOUBLE_EQ(without_wheel->twist.twist.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(without_wheel->twist.covariance[0], 1.0e6);

  ASSERT_TRUE(localizer.observe_imu(imu_sample(20.2, 0.0)));
  ASSERT_TRUE(localizer.observe_wheel_speed(wheel_sample(19.7, -4.25)));
  const auto stale_wheel = localizer.observe_gnss(gnss_sample(20.2));
  ASSERT_TRUE(stale_wheel.has_value());
  EXPECT_DOUBLE_EQ(stale_wheel->twist.twist.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(stale_wheel->twist.covariance[0], 1.0e6);

  ASSERT_TRUE(localizer.observe_imu(imu_sample(20.4, 0.0)));
  ASSERT_TRUE(localizer.observe_wheel_speed(wheel_sample(20.35, -4.25)));

  const auto odometry = localizer.observe_gnss(gnss_sample(20.4));
  ASSERT_TRUE(odometry.has_value());
  EXPECT_DOUBLE_EQ(odometry->twist.twist.linear.x, -4.25);
  EXPECT_DOUBLE_EQ(odometry->twist.twist.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(odometry->twist.twist.linear.z, 0.0);
  EXPECT_DOUBLE_EQ(odometry->twist.twist.angular.x, 0.0);
  EXPECT_DOUBLE_EQ(odometry->twist.twist.angular.y, 0.0);
  EXPECT_DOUBLE_EQ(odometry->twist.twist.angular.z, 0.0);
  EXPECT_DOUBLE_EQ(odometry->twist.covariance[0], 0.16);
}

TEST(GnssImuLocalizerTest, PopulatesEveryPoseAndTwistCovarianceDiagonal)
{
  ad_localization::GnssImuLocalizer localizer(default_config());
  ASSERT_TRUE(localizer.observe_imu(imu_sample(25.0, 0.0)));

  const auto without_wheel = localizer.observe_gnss(gnss_sample(25.0));
  ASSERT_TRUE(without_wheel.has_value());
  const std::array<double, 6> expected_pose_diagonal{
    0.25, 0.25, 0.25, 0.0001, 0.0001, 0.0001};
  for (std::size_t row = 0; row < 6; ++row) {
    for (std::size_t column = 0; column < 6; ++column) {
      const std::size_t index = row * 6 + column;
      EXPECT_DOUBLE_EQ(
        without_wheel->pose.covariance[index],
        row == column ? expected_pose_diagonal[row] : 0.0);
      EXPECT_DOUBLE_EQ(
        without_wheel->twist.covariance[index], row == column ? 1.0e6 : 0.0);
    }
  }

  ASSERT_TRUE(localizer.observe_imu(imu_sample(25.2, 0.0)));
  ASSERT_TRUE(localizer.observe_wheel_speed(wheel_sample(25.2, 3.0)));
  const auto with_wheel = localizer.observe_gnss(gnss_sample(25.2));
  ASSERT_TRUE(with_wheel.has_value());
  for (std::size_t row = 0; row < 6; ++row) {
    EXPECT_DOUBLE_EQ(
      with_wheel->twist.covariance[row * 6 + row], row == 0 ? 0.16 : 1.0e6);
  }
}

TEST(GnssImuLocalizerTest, UsesConfiguredPoseAndUnobservedTwistVariances)
{
  auto config = default_config();
  config.gnss_xy_variance_m2 = 0.36;
  config.gnss_z_variance_m2 = 0.49;
  config.imu_orientation_variance_rad2 = 0.0004;
  config.unobserved_twist_variance = 9000.0;
  ad_localization::GnssImuLocalizer localizer(config);
  ASSERT_TRUE(localizer.observe_imu(imu_sample(26.0, 0.0)));

  const auto odometry = localizer.observe_gnss(gnss_sample(26.0));
  ASSERT_TRUE(odometry.has_value());
  const std::array<double, 6> expected_pose_diagonal{
    0.36, 0.36, 0.49, 0.0004, 0.0004, 0.0004};
  for (std::size_t axis = 0; axis < 6; ++axis) {
    EXPECT_DOUBLE_EQ(
      odometry->pose.covariance[axis * 6 + axis], expected_pose_diagonal[axis]);
    EXPECT_DOUBLE_EQ(odometry->twist.covariance[axis * 6 + axis], 9000.0);
  }
}

TEST(GnssImuLocalizerTest, EmitsAtMostOnceForEachMonotonicGnssPose)
{
  ad_localization::GnssImuLocalizer localizer(default_config());
  ASSERT_TRUE(localizer.observe_imu(imu_sample(30.0, 0.0)));
  ASSERT_TRUE(localizer.observe_gnss(gnss_sample(30.0)).has_value());

  EXPECT_FALSE(localizer.observe_gnss(gnss_sample(30.0)).has_value());
  EXPECT_FALSE(localizer.observe_gnss(gnss_sample(29.9)).has_value());

  ASSERT_TRUE(localizer.observe_imu(imu_sample(30.2, 0.0)));
  EXPECT_TRUE(localizer.observe_gnss(gnss_sample(30.2)).has_value());
}

TEST(GnssImuLocalizerTest, ResetAcceptsARegressedSimulatorTimestamp)
{
  ad_localization::GnssImuLocalizer localizer(default_config());
  ASSERT_TRUE(localizer.observe_imu(imu_sample(30.0, 0.0)));
  ASSERT_TRUE(localizer.observe_gnss(gnss_sample(30.0)).has_value());

  localizer.reset();

  ASSERT_TRUE(localizer.observe_imu(imu_sample(1.0, 0.0)));
  EXPECT_TRUE(localizer.observe_gnss(gnss_sample(1.0)).has_value());
}

TEST(GnssImuLocalizerTest, RejectsInvalidFramesStampsAndFiniteValues)
{
  ad_localization::GnssImuLocalizer localizer(default_config());

  EXPECT_FALSE(localizer.observe_imu(imu_sample(40.0, 0.0, "wrong_imu")));
  auto invalid_imu = imu_sample(40.0, 0.0);
  invalid_imu.orientation.w = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(localizer.observe_imu(invalid_imu));
  EXPECT_FALSE(localizer.observe_wheel_speed(wheel_sample(40.0, 1.0, "odom")));
  auto invalid_wheel = wheel_sample(40.0, std::numeric_limits<double>::infinity());
  EXPECT_FALSE(localizer.observe_wheel_speed(invalid_wheel));
  auto zero_variance_wheel = wheel_sample(40.0, 1.0);
  zero_variance_wheel.twist.covariance[0] = 0.0;
  EXPECT_FALSE(localizer.observe_wheel_speed(zero_variance_wheel));
  auto negative_variance_wheel = wheel_sample(40.0, 1.0);
  negative_variance_wheel.twist.covariance[0] = -0.1;
  EXPECT_FALSE(localizer.observe_wheel_speed(negative_variance_wheel));

  ASSERT_TRUE(localizer.observe_imu(imu_sample(40.0, 0.0)));
  ASSERT_TRUE(localizer.observe_wheel_speed(wheel_sample(40.0, 1.0)));
  EXPECT_FALSE(localizer.observe_gnss(gnss_sample(40.0, 1.0, 2.0, 3.0, "map")).has_value());
  auto invalid_stamp = gnss_sample(40.0);
  invalid_stamp.header.stamp = builtin_interfaces::msg::Time{};
  EXPECT_FALSE(localizer.observe_gnss(invalid_stamp).has_value());
  auto invalid_position = gnss_sample(40.0);
  invalid_position.pose.position.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(localizer.observe_gnss(invalid_position).has_value());
}

TEST(GnssImuLocalizerTest, RejectsNonPositiveOrNonFiniteConfiguredVariances)
{
  const std::array<double ad_localization::GnssImuLocalizerConfig::*, 4> fields{
    &ad_localization::GnssImuLocalizerConfig::gnss_xy_variance_m2,
    &ad_localization::GnssImuLocalizerConfig::gnss_z_variance_m2,
    &ad_localization::GnssImuLocalizerConfig::imu_orientation_variance_rad2,
    &ad_localization::GnssImuLocalizerConfig::unobserved_twist_variance,
  };
  for (const auto field : fields) {
    auto zero = default_config();
    zero.*field = 0.0;
    EXPECT_THROW(ad_localization::GnssImuLocalizer{zero}, std::invalid_argument);

    auto negative = default_config();
    negative.*field = -0.1;
    EXPECT_THROW(ad_localization::GnssImuLocalizer{negative}, std::invalid_argument);

    auto non_finite = default_config();
    non_finite.*field = std::numeric_limits<double>::infinity();
    EXPECT_THROW(ad_localization::GnssImuLocalizer{non_finite}, std::invalid_argument);
  }
}

TEST(GnssImuLocalizerTest, RejectsNonFiniteWorldYawOffset)
{
  auto config = default_config();
  config.world_yaw_offset_rad = std::numeric_limits<double>::infinity();
  EXPECT_THROW(ad_localization::GnssImuLocalizer{config}, std::invalid_argument);
}

}  // namespace
