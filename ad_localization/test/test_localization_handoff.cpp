#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "ad_localization/gnss_shadow/localization_handoff.hpp"
#include "gtest/gtest.h"

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

geometry_msgs::msg::Quaternion yaw_quaternion(double yaw)
{
  geometry_msgs::msg::Quaternion result;
  result.z = std::sin(yaw * 0.5);
  result.w = std::cos(yaw * 0.5);
  return result;
}

double yaw(const geometry_msgs::msg::Quaternion & quaternion)
{
  return std::atan2(
    2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
    1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z));
}

nav_msgs::msg::Odometry odometry(
  double x, double y, double heading, double seconds,
  const std::string & frame = "odom", const std::string & child_frame = "base_link")
{
  nav_msgs::msg::Odometry result;
  result.header.stamp = stamp(seconds);
  result.header.frame_id = frame;
  result.child_frame_id = child_frame;
  result.pose.pose.position.x = x;
  result.pose.pose.position.y = y;
  result.pose.pose.position.z = 1.0;
  result.pose.pose.orientation = yaw_quaternion(heading);
  result.pose.covariance[0] = 0.25;
  result.pose.covariance[7] = 0.25;
  result.pose.covariance[35] = 0.01;
  return result;
}

ad_localization::HandoffConfig default_config()
{
  ad_localization::HandoffConfig result;
  result.entry_xy = {10.0, 0.0};
  result.exit_xy = {100.0, 0.0};
  result.prewarm_radius_m = 20.0;
  result.entry_switch_radius_m = 8.0;
  result.exit_switch_radius_m = 20.0;
  result.source_timeout_sec = 0.5;
  result.maximum_position_disagreement_m = 2.0;
  result.maximum_yaw_disagreement_rad = 0.20;
  result.blend_duration_sec = 2.0;
  result.reference_frame = "odom";
  result.base_frame = "base_link";
  return result;
}

ad_localization::LocalizationHandoff fastlio_active_handoff()
{
  ad_localization::LocalizationHandoff handoff(default_config());
  const auto approach = handoff.observe(
    ad_localization::Backend::kGnssImu, odometry(5.0, 0.0, 0.0, 1.0), 1.0);
  EXPECT_TRUE(approach.fastlio_initial_pose.has_value());
  const auto selected = handoff.observe(
    ad_localization::Backend::kFastlio, odometry(5.1, 0.0, 0.01, 1.1), 1.1);
  EXPECT_EQ(selected.active_backend, ad_localization::Backend::kFastlio);
  return handoff;
}

TEST(LocalizationHandoffTest, RefreshesPrewarmPoseUntilFastlioIsSelected)
{
  ad_localization::LocalizationHandoff handoff(default_config());

  const auto far = handoff.observe(
    ad_localization::Backend::kGnssImu, odometry(-20.0, 0.0, 0.0, 1.0), 1.0);
  EXPECT_FALSE(far.fastlio_initial_pose.has_value());
  ASSERT_TRUE(far.canonical_odometry.has_value());

  const auto prewarm = handoff.observe(
    ad_localization::Backend::kGnssImu, odometry(0.0, 0.0, 0.0, 2.0), 2.0);
  ASSERT_TRUE(prewarm.fastlio_initial_pose.has_value());
  EXPECT_EQ(prewarm.phase, ad_localization::HandoffPhase::kFastlioReady);
  EXPECT_DOUBLE_EQ(prewarm.fastlio_initial_pose->pose.position.x, 0.0);

  const auto repeated = handoff.observe(
    ad_localization::Backend::kGnssImu, odometry(4.0, 0.0, 0.0, 2.1), 2.1);
  ASSERT_TRUE(repeated.fastlio_initial_pose.has_value());
  EXPECT_DOUBLE_EQ(repeated.fastlio_initial_pose->pose.position.x, 4.0);

  const auto selected = handoff.observe(
    ad_localization::Backend::kFastlio, odometry(4.1, 0.1, 0.02, 2.2), 2.2);
  EXPECT_EQ(selected.active_backend, ad_localization::Backend::kFastlio);
  EXPECT_EQ(selected.phase, ad_localization::HandoffPhase::kFastlioActive);
  EXPECT_EQ(selected.switch_count, 1U);
  ASSERT_TRUE(selected.canonical_odometry.has_value());
  EXPECT_NEAR(selected.canonical_odometry->pose.pose.position.x, 4.0, 1.0e-12);
  EXPECT_NEAR(selected.canonical_odometry->pose.pose.position.y, 0.0, 1.0e-12);
  EXPECT_NEAR(yaw(selected.canonical_odometry->pose.pose.orientation), 0.0, 1.0e-12);

  const auto after_switch = handoff.observe(
    ad_localization::Backend::kGnssImu, odometry(4.2, 0.0, 0.0, 2.3), 2.3);
  EXPECT_FALSE(after_switch.fastlio_initial_pose.has_value());
}

TEST(LocalizationHandoffTest, RejectsEntryOutsideRadiusDisagreementAndStaleReference)
{
  ad_localization::LocalizationHandoff handoff(default_config());
  handoff.observe(
    ad_localization::Backend::kGnssImu, odometry(0.0, 0.0, 0.0, 1.0), 1.0);

  auto outside = handoff.observe(
    ad_localization::Backend::kFastlio, odometry(0.1, 0.0, 0.0, 1.1), 1.1);
  EXPECT_EQ(outside.active_backend, ad_localization::Backend::kGnssImu);
  EXPECT_FALSE(outside.canonical_odometry.has_value());

  handoff.observe(
    ad_localization::Backend::kGnssImu, odometry(5.0, 0.0, 0.0, 2.0), 2.0);
  auto disagree = handoff.observe(
    ad_localization::Backend::kFastlio, odometry(8.0, 0.0, 0.0, 2.1), 2.1);
  EXPECT_EQ(disagree.active_backend, ad_localization::Backend::kGnssImu);

  auto stale = handoff.observe(
    ad_localization::Backend::kFastlio, odometry(5.1, 0.0, 0.0, 3.0), 3.0);
  EXPECT_EQ(stale.active_backend, ad_localization::Backend::kGnssImu);
  EXPECT_EQ(stale.switch_count, 0U);
}

TEST(LocalizationHandoffTest, SwitchesBackToGnssNearExitAndNeverReenters)
{
  auto handoff = fastlio_active_handoff();
  const auto tunnel = handoff.observe(
    ad_localization::Backend::kFastlio, odometry(82.0, 0.0, 0.0, 10.0), 10.0);
  EXPECT_EQ(tunnel.phase, ad_localization::HandoffPhase::kGnssRecovery);

  const auto recovered = handoff.observe(
    ad_localization::Backend::kGnssImu, odometry(82.1, 0.0, 0.01, 10.1), 10.1);
  EXPECT_EQ(recovered.active_backend, ad_localization::Backend::kGnssImu);
  EXPECT_EQ(recovered.phase, ad_localization::HandoffPhase::kGnssFinish);
  EXPECT_EQ(recovered.switch_count, 2U);
  ASSERT_TRUE(recovered.canonical_odometry.has_value());
  EXPECT_NEAR(recovered.canonical_odometry->pose.pose.position.x, 82.0, 1.0e-12);

  const auto ignored = handoff.observe(
    ad_localization::Backend::kFastlio, odometry(83.0, 0.0, 0.0, 10.2), 10.2);
  EXPECT_EQ(ignored.active_backend, ad_localization::Backend::kGnssImu);
  EXPECT_EQ(ignored.phase, ad_localization::HandoffPhase::kGnssFinish);
  EXPECT_EQ(ignored.switch_count, 2U);
  EXPECT_FALSE(ignored.canonical_odometry.has_value());
}

TEST(LocalizationHandoffTest, DecaysSwitchCorrectionToRawPose)
{
  ad_localization::LocalizationHandoff handoff(default_config());
  handoff.observe(
    ad_localization::Backend::kGnssImu, odometry(5.0, 0.0, 0.0, 1.0), 1.0);
  const auto switched = handoff.observe(
    ad_localization::Backend::kFastlio, odometry(5.5, 0.0, 0.10, 1.1), 1.1);
  ASSERT_TRUE(switched.canonical_odometry.has_value());
  EXPECT_NEAR(switched.canonical_odometry->pose.pose.position.x, 5.0, 1.0e-12);

  const auto halfway = handoff.observe(
    ad_localization::Backend::kFastlio, odometry(6.5, 0.0, 0.10, 2.1), 2.1);
  ASSERT_TRUE(halfway.canonical_odometry.has_value());
  EXPECT_NEAR(halfway.canonical_odometry->pose.pose.position.x, 6.25, 1.0e-12);
  EXPECT_NEAR(yaw(halfway.canonical_odometry->pose.pose.orientation), 0.05, 1.0e-12);
  EXPECT_GT(halfway.canonical_odometry->pose.covariance[0], 0.25);

  const auto complete = handoff.observe(
    ad_localization::Backend::kFastlio, odometry(7.5, 0.0, 0.10, 3.1), 3.1);
  ASSERT_TRUE(complete.canonical_odometry.has_value());
  EXPECT_NEAR(complete.canonical_odometry->pose.pose.position.x, 7.5, 1.0e-12);
  EXPECT_NEAR(yaw(complete.canonical_odometry->pose.pose.orientation), 0.10, 1.0e-12);
  EXPECT_DOUBLE_EQ(complete.canonical_odometry->pose.covariance[0], 0.25);
}

TEST(LocalizationHandoffTest, RejectsMalformedOrRegressingCandidates)
{
  ad_localization::LocalizationHandoff handoff(default_config());

  EXPECT_FALSE(
    handoff.observe(
      ad_localization::Backend::kGnssImu,
      odometry(0.0, 0.0, 0.0, 1.0, "map"), 1.0).canonical_odometry);
  EXPECT_FALSE(
    handoff.observe(
      ad_localization::Backend::kGnssImu,
      odometry(0.0, 0.0, 0.0, 1.0, "odom", "vehicle"), 1.0).canonical_odometry);

  auto invalid = odometry(0.0, 0.0, 0.0, 1.0);
  invalid.pose.pose.position.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(
    handoff.observe(
      ad_localization::Backend::kGnssImu, invalid, 1.0).canonical_odometry);

  const auto accepted = handoff.observe(
    ad_localization::Backend::kGnssImu, odometry(0.0, 0.0, 0.0, 2.0), 2.0);
  ASSERT_TRUE(accepted.canonical_odometry.has_value());
  EXPECT_FALSE(
    handoff.observe(
      ad_localization::Backend::kGnssImu, odometry(0.1, 0.0, 0.0, 2.0), 2.1)
    .canonical_odometry);
  EXPECT_FALSE(
    handoff.observe(
      ad_localization::Backend::kGnssImu, odometry(0.1, 0.0, 0.0, 2.2), 1.9)
    .canonical_odometry);
}

TEST(LocalizationHandoffTest, RejectsInvalidConfiguration)
{
  auto config = default_config();
  config.source_timeout_sec = 0.0;
  EXPECT_THROW(
    static_cast<void>(ad_localization::LocalizationHandoff{config}),
    std::invalid_argument);

  config = default_config();
  config.maximum_yaw_disagreement_rad = kPi + 0.1;
  EXPECT_THROW(
    static_cast<void>(ad_localization::LocalizationHandoff{config}),
    std::invalid_argument);
}

}  // namespace
