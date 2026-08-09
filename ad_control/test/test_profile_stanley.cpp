#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ad_control/lateral/stanley.hpp"

namespace
{
using namespace ad_control;

StanleyConfig stanley_config()
{
  StanleyConfig config;
  config.target_speed_mps = 10.0;
  config.cross_track_gain = 1.0;
  config.speed_softening_mps = 0.5;
  config.lookahead_time_s = 0.0;
  config.lookahead_min_m = 0.0;
  config.lookahead_max_m = 0.0;
  config.max_steer_rad = 0.4;
  config.forward_window = 10;
  config.max_laps = 1;
  config.pid = PidConfig{0.5, 0.0, 0.0, 1.0, 1.0};
  config.control_point_x_m = 0.0;
  return config;
}

RouteSpeedProfileConfig speed_profile_config()
{
  return RouteSpeedProfileConfig{
    1.0,
    20.0,
    0.25,
    100.0,
    100.0,
    5,
    1.0,
    LongitudinalProfile{}};
}

ProfileStanleyConfig profile_config()
{
  StanleyConfig stanley = stanley_config();
  stanley.speed_profile = speed_profile_config();
  stanley.launch_speed_mps = 0.75;
  stanley.launch_ramp_s = 4.0;
  return ProfileStanleyConfig{stanley, LongitudinalProfile{}};
}

Route line_route()
{
  return Route{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
    {3.0, 0.0, 0.0}, {4.0, 0.0, 0.0}}, false};
}

Route corner_route()
{
  return Route{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
    {2.0, 1.0, 0.0}, {2.0, 2.0, 0.0}, {2.0, 3.0, 0.0}}, false};
}

TEST(ProfileStanleyController, LaunchRampStartsAtConfiguredSpeed)
{
  ProfileStanleyController controller(line_route(), profile_config());

  const auto result = controller.update(Pose2{}, 0.0, 0.1, 1, 4);

  ASSERT_TRUE(result.valid);
  ASSERT_TRUE(result.target_speed_mps.has_value());
  EXPECT_DOUBLE_EQ(*result.target_speed_mps, 0.75);
  EXPECT_DOUBLE_EQ(result.command.accel, 0.375);
}

TEST(ProfileStanleyController, CurvatureCapsTargetBeforeCorner)
{
  ProfileStanleyController controller(corner_route(), profile_config());

  ControllerResult result;
  for (int update = 0; update <= 4; ++update) {
    result = controller.update(Pose2{}, 0.0, 1.0, 1, 4);
    ASSERT_TRUE(result.valid);
  }

  ASSERT_TRUE(result.target_speed_mps.has_value());
  EXPECT_DOUBLE_EQ(*result.target_speed_mps, 1.0);
  EXPECT_DOUBLE_EQ(result.command.accel, 0.5);
}

TEST(ProfileStanleyController, LowerProfileTargetCommandsBrake)
{
  ProfileStanleyController controller(corner_route(), profile_config());
  for (int update = 0; update <= 4; ++update) {
    ASSERT_TRUE(controller.update(Pose2{}, 0.0, 1.0, 1, 4).valid);
  }

  const auto result = controller.update(Pose2{}, 2.0, 0.1, 1, 4);

  ASSERT_TRUE(result.valid);
  ASSERT_TRUE(result.target_speed_mps.has_value());
  EXPECT_DOUBLE_EQ(*result.target_speed_mps, 1.0);
  EXPECT_DOUBLE_EQ(result.command.accel, 0.0);
  EXPECT_GT(result.command.brake, 0.0);
}

TEST(ProfileStanleyController, ExposesTheProfileUsedForControl)
{
  const Route route = corner_route();
  ProfileStanleyController controller(route, profile_config());

  const RouteSpeedProfile * profile = controller.route_speed_profile();

  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(profile->speed_mps.size(), route.points.size());
  EXPECT_EQ(profile->curvature_inv_m.size(), route.points.size());
  EXPECT_NEAR(profile->curvature_inv_m[2], std::sqrt(2.0), 1.0e-12);
}

TEST(ProfileStanleyController, UsesMeasuredLongitudinalEnvelopeWhenConfigured)
{
  ProfileStanleyConfig config = profile_config();
  config.longitudinal_profile = LongitudinalProfile{
    {0.0, 20.0}, {100.0, 100.0}, {1.0, 1.0}, 0.0};
  ProfileStanleyController controller(line_route(), config);

  const RouteSpeedProfile * profile = controller.route_speed_profile();

  ASSERT_NE(profile, nullptr);
  EXPECT_TRUE(profile->uses_longitudinal_profile);
  EXPECT_NEAR(profile->speed_mps.front(), std::sqrt(8.0), 1.0e-12);
}

TEST(ProfileStanleyController, UsesThePrecomputedSpatialTargetAfterPoseBurst)
{
  ProfileStanleyConfig config = profile_config();
  config.stanley.launch_speed_mps = 20.0;
  config.stanley.speed_profile.deceleration_mps2 = 10.0;
  config.longitudinal_profile = LongitudinalProfile{
    {0.0, 20.0}, {100.0, 100.0}, {1.0, 1.0}, 0.0};
  ProfileStanleyController controller(line_route(), config);

  const auto first = controller.update(Pose2{}, 3.0, 0.05, 1, 4);
  ASSERT_TRUE(first.valid);
  ASSERT_TRUE(first.target_speed_mps.has_value());
  EXPECT_NEAR(*first.target_speed_mps, std::sqrt(8.0), 1.0e-12);

  const auto after_pose_burst = controller.update(
    Pose2{2.0, 0.0, 0.0}, 3.0, 0.05, 1, 4);
  ASSERT_TRUE(after_pose_burst.valid);
  ASSERT_TRUE(after_pose_burst.target_speed_mps.has_value());
  EXPECT_DOUBLE_EQ(*after_pose_burst.target_speed_mps, 2.0);
}

TEST(ProfileStanleyController, RejectsInvalidLaunchConfiguration)
{
  ProfileStanleyConfig config = profile_config();
  config.stanley.launch_speed_mps = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
    ProfileStanleyController(line_route(), config), std::invalid_argument);

  config = profile_config();
  config.stanley.launch_speed_mps = -0.01;
  EXPECT_THROW(
    ProfileStanleyController(line_route(), config), std::invalid_argument);

  config = profile_config();
  config.stanley.launch_ramp_s = 0.0;
  EXPECT_THROW(
    ProfileStanleyController(line_route(), config), std::invalid_argument);
}

TEST(ProfileStanleyController, InvalidUpdateDoesNotAdvanceLaunchRamp)
{
  const ProfileStanleyConfig config = profile_config();
  ProfileStanleyController controller(line_route(), config);
  const auto first = controller.update(Pose2{}, 0.0, 1.0, 1, 4);
  ASSERT_TRUE(first.valid);
  ASSERT_TRUE(first.target_speed_mps.has_value());
  EXPECT_DOUBLE_EQ(*first.target_speed_mps, config.stanley.launch_speed_mps);

  const auto invalid = controller.update(
    Pose2{0.0, 0.0, std::numeric_limits<double>::quiet_NaN()},
    0.0, 1.0, 1, 4);
  EXPECT_FALSE(invalid.valid);

  const auto second = controller.update(Pose2{}, 0.0, 1.0, 1, 4);
  ASSERT_TRUE(second.valid);
  ASSERT_TRUE(second.target_speed_mps.has_value());
  EXPECT_DOUBLE_EQ(*second.target_speed_mps, 3.0625);
}
}  // namespace
