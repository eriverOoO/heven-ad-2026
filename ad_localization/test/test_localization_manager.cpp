#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "ad_localization/manager/localization_manager.hpp"

namespace
{

nav_msgs::msg::Odometry sample(std::int32_t sec = 10, std::uint32_t nanosec = 0)
{
  nav_msgs::msg::Odometry message;
  message.header.stamp.sec = sec;
  message.header.stamp.nanosec = nanosec;
  message.header.frame_id = "odom";
  message.child_frame_id = "base_link";
  message.pose.pose.position.x = 3.0;
  message.pose.pose.position.y = -2.0;
  message.pose.pose.position.z = 0.5;
  message.pose.pose.orientation.w = 1.0;
  message.twist.twist.linear.x = 4.0;
  message.pose.covariance[0] = 0.25;
  message.twist.covariance[0] = 0.04;
  return message;
}

TEST(LocalizationManager, PreservesAValidBackendSampleAndBuildsDynamicTf)
{
  ad_localization::LocalizationManager manager({"map", "odom", "base_link"});
  const auto input = sample();

  const auto output = manager.accept(input);

  ASSERT_TRUE(output.has_value());
  EXPECT_EQ(output->header, input.header);
  EXPECT_EQ(output->child_frame_id, input.child_frame_id);
  EXPECT_DOUBLE_EQ(output->pose.pose.position.x, 3.0);
  EXPECT_DOUBLE_EQ(output->twist.twist.linear.x, 4.0);
  EXPECT_DOUBLE_EQ(output->pose.covariance[0], 0.25);
  const auto transform = ad_localization::odometry_transform(*output);
  EXPECT_EQ(transform.header, output->header);
  EXPECT_EQ(transform.child_frame_id, "base_link");
  EXPECT_DOUBLE_EQ(transform.transform.translation.y, -2.0);
}

TEST(LocalizationManager, RejectsDuplicateRegressionWrongFramesAndNonFinitePose)
{
  ad_localization::LocalizationManager manager({"map", "odom", "base_link"});
  ASSERT_TRUE(manager.accept(sample(10)).has_value());
  EXPECT_FALSE(manager.accept(sample(10)).has_value());
  EXPECT_FALSE(manager.accept(sample(9)).has_value());

  auto wrong_parent = sample(11);
  wrong_parent.header.frame_id = "map";
  EXPECT_FALSE(manager.accept(wrong_parent).has_value());
  auto wrong_child = sample(11);
  wrong_child.child_frame_id = "rear_axle";
  EXPECT_FALSE(manager.accept(wrong_child).has_value());
  auto nonfinite = sample(11);
  nonfinite.pose.pose.position.z = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(manager.accept(nonfinite).has_value());
}

TEST(LocalizationManager, RejectsStructurallyInvalidRosTimestamps)
{
  ad_localization::LocalizationManager manager({"map", "odom", "base_link"});

  auto negative_seconds = sample();
  negative_seconds.header.stamp.sec = -1;
  negative_seconds.header.stamp.nanosec = 999'999'999U;
  EXPECT_FALSE(manager.accept(negative_seconds).has_value());

  auto overflowing_nanoseconds = sample();
  overflowing_nanoseconds.header.stamp.nanosec = 1'000'000'000U;
  EXPECT_FALSE(manager.accept(overflowing_nanoseconds).has_value());

  EXPECT_TRUE(manager.accept(sample(0, 0)).has_value());
}

TEST(LocalizationManager, RejectsNonFinitePoseAndTwistCovarianceEntries)
{
  ad_localization::LocalizationManager manager({"map", "odom", "base_link"});

  auto invalid_pose_covariance = sample();
  invalid_pose_covariance.pose.covariance[17] =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(manager.accept(invalid_pose_covariance).has_value());

  auto invalid_twist_covariance = sample();
  invalid_twist_covariance.twist.covariance[35] =
    std::numeric_limits<double>::infinity();
  EXPECT_FALSE(manager.accept(invalid_twist_covariance).has_value());
}

TEST(LocalizationManager, RejectsInvalidQuaternionAndResetStartsANewEpoch)
{
  ad_localization::LocalizationManager manager({"map", "odom", "base_link"});
  auto zero = sample();
  zero.pose.pose.orientation.w = 0.0;
  EXPECT_FALSE(manager.accept(zero).has_value());

  auto not_unit = sample();
  not_unit.pose.pose.orientation.w = 2.0;
  EXPECT_FALSE(manager.accept(not_unit).has_value());

  ASSERT_TRUE(manager.accept(sample(20)).has_value());
  manager.reset();
  EXPECT_TRUE(manager.accept(sample(1)).has_value());
}

TEST(LocalizationManager, CreatesOnlyTheStaticMapToOdomEdge)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = 42;
  const auto transform = ad_localization::map_to_odom_transform(
    {"map", "odom", "base_link"}, stamp);
  EXPECT_EQ(transform.header.stamp, stamp);
  EXPECT_EQ(transform.header.frame_id, "map");
  EXPECT_EQ(transform.child_frame_id, "odom");
  EXPECT_DOUBLE_EQ(transform.transform.rotation.w, 1.0);
}

}  // namespace
