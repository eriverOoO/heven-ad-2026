#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "ad_control/lateral/stanley.hpp"

namespace
{
using namespace ad_control;

StanleyConfig config()
{
  StanleyConfig value;
  value.target_speed_mps = 2.0;
  value.cross_track_gain = 1.0;
  value.speed_softening_mps = 0.5;
  value.lookahead_time_s = 0.0;
  value.lookahead_min_m = 2.0;
  value.lookahead_max_m = 2.0;
  value.max_steer_rad = 0.4;
  value.forward_window = 2;
  value.max_laps = 1;
  value.pid = PidConfig{0.5, 0.0, 0.0, 1.0, 1.0};
  value.speed_profile = RouteSpeedProfileConfig{
    0.1, 100.0, 1000.0, 100.0, 100.0, 1, 1.0, LongitudinalProfile{}};
  value.launch_speed_mps = 100.0;
  value.launch_ramp_s = 4.0;
  return value;
}

Route line_route()
{
  return Route{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
    {3.0, 0.0, 0.0}, {4.0, 0.0, 0.0}}, false};
}

void expect_progress_equal(
  const RouteProgressState & actual, const RouteProgressState & expected)
{
  EXPECT_EQ(actual.initialized, expected.initialized);
  EXPECT_EQ(actual.target_index, expected.target_index);
  EXPECT_EQ(actual.lap_count, expected.lap_count);
  EXPECT_EQ(actual.terminal_full_brake, expected.terminal_full_brake);
}

TEST(StanleyController, UsesRequestedSpeedWithBaselineRouteProfile)
{
  StanleyConfig cfg = config();
  cfg.target_speed_mps = 2.0;
  cfg.pid.kp = 0.1;
  StanleyController controller(line_route(), cfg);

  const auto result = controller.update(Pose2{}, 0.0, 0.1, 1, 4);

  ASSERT_TRUE(result.valid);
  ASSERT_TRUE(result.target_speed_mps.has_value());
  EXPECT_DOUBLE_EQ(*result.target_speed_mps, 2.0);
  EXPECT_DOUBLE_EQ(result.command.accel, 0.2);
  EXPECT_DOUBLE_EQ(result.command.brake, 0.0);
  ASSERT_NE(controller.route_speed_profile(), nullptr);
  EXPECT_FALSE(controller.route_speed_profile()->uses_longitudinal_profile);
}

TEST(StanleyController, HighConfiguredLateralLimitDoesNotCapThisCorner)
{
  const Route corner{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
    {2.0, 1.0, 0.0}, {2.0, 2.0, 0.0}, {2.0, 3.0, 0.0}}, false};
  StanleyConfig cfg = config();
  cfg.target_speed_mps = 10.0;
  cfg.pid.kp = 0.05;
  StanleyController controller(corner, cfg);

  const auto result = controller.update(Pose2{}, 0.0, 0.1, 1, 4);

  ASSERT_TRUE(result.valid);
  ASSERT_TRUE(result.target_speed_mps.has_value());
  EXPECT_DOUBLE_EQ(*result.target_speed_mps, 10.0);
  EXPECT_DOUBLE_EQ(result.command.accel, 0.5);
}

TEST(StanleyController, UsesRadiansPreservesCrossTrackSignAndClampsPhysicalSteering)
{
  struct Case {double lateral_position; double expected_steering;};
  const double slew_per_update = 120.0 * std::acos(-1.0) / 180.0 * 0.1;
  const std::array<Case, 3> cases{
    {{1.0, -slew_per_update}, {-1.0, slew_per_update}, {0.0, 0.0}}};
  for (const auto & test : cases) {
    StanleyController controller(line_route(), config());
    const auto result = controller.update(
      Pose2{0.0, test.lateral_position, 0.0}, 0.0, 0.1, 1, 4);
    ASSERT_TRUE(result.valid);
    EXPECT_NEAR(result.command.steering_rad, test.expected_steering, 1e-12);
    EXPECT_NEAR(result.command.accel, 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(result.command.brake, 0.0);
  }
}

TEST(StanleyController, WrapsHeadingErrorAndSoftensZeroSpeed)
{
  Route vertical{{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 2.0, 0.0}}, false};
  StanleyController controller(vertical, config());
  const auto aligned = controller.update(Pose2{0.0, 0.0, std::acos(-1.0) / 2.0}, 0.0, 0.1, 1, 4);
  ASSERT_TRUE(aligned.valid);
  EXPECT_NEAR(aligned.command.steering_rad, 0.0, 1e-12);
}

TEST(StanleyController, HeadingErrorGainScalesHeadingCorrection)
{
  Route vertical{{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 2.0, 0.0}}, false};
  auto low_config = config();
  low_config.lookahead_min_m = 0.0;
  low_config.lookahead_max_m = 0.0;
  low_config.max_steer_rad = 2.0;
  low_config.heading_error_gain = 0.5;
  auto high_config = low_config;
  high_config.heading_error_gain = 1.0;
  StanleyController low(vertical, low_config);
  StanleyController high(vertical, high_config);

  const auto low_result = low.update(Pose2{}, 0.0, 1.0, 1, 4);
  const auto high_result = high.update(Pose2{}, 0.0, 1.0, 1, 4);

  ASSERT_TRUE(low_result.valid);
  ASSERT_TRUE(high_result.valid);
  EXPECT_NEAR(low_result.command.steering_rad, std::acos(-1.0) / 4.0, 1e-12);
  EXPECT_NEAR(high_result.command.steering_rad, std::acos(-1.0) / 2.0, 1e-12);
}

TEST(StanleyController, RejectsAReverseHeadingInsteadOfFollowingTheRouteBackward)
{
  StanleyController controller(line_route(), config());

  const auto result = controller.update(
    Pose2{0.0, 0.0, std::acos(-1.0)}, 0.0, 0.1, 1, 4);

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.reason, "route heading mismatch");
  EXPECT_DOUBLE_EQ(result.command.accel, 0.0);
  EXPECT_DOUBLE_EQ(result.command.brake, 1.0);
}

TEST(StanleyController, UsesPerUpdateTargetSpeedForSlowdown)
{
  Route long_route;
  for (int x = 0; x <= 20; ++x) {
    long_route.points.push_back(Point3{static_cast<double>(x), 0.0, 0.0});
  }
  auto cfg = config();
  cfg.target_speed_mps = 8.33;
  cfg.pid.kp = 0.1;
  StanleyController normal(long_route, cfg);
  StanleyController slowdown(long_route, cfg);

  for (int update = 0; update < 40; ++update) {
    ASSERT_TRUE(normal.update(Pose2{0.0, 0.0, 0.0}, 0.0, 0.1, 1, 4).valid);
    ASSERT_TRUE(
      slowdown.update(Pose2{0.0, 0.0, 0.0}, 0.0, 0.1, 1, 4, 5.56).valid);
  }

  const auto normal_result = normal.update(
    Pose2{0.0, 0.0, 0.0}, 0.0, 0.1, 1, 4);
  const auto slowdown_result = slowdown.update(
    Pose2{0.0, 0.0, 0.0}, 0.0, 0.1, 1, 4, 5.56);

  ASSERT_TRUE(normal_result.valid);
  ASSERT_TRUE(slowdown_result.valid);
  EXPECT_NEAR(normal_result.command.accel, 0.833, 1e-12);
  EXPECT_NEAR(slowdown_result.command.accel, 0.556, 1e-12);
  EXPECT_DOUBLE_EQ(slowdown_result.command.brake, 0.0);
}

TEST(StanleyController, UsesNearestSegmentForCrossTrackAndPreviewSegmentForHeading)
{
  Route corner{{{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}, {10.0, 10.0, 0.0}}, false};
  StanleyConfig cfg = config();
  cfg.lookahead_min_m = 11.0;
  cfg.lookahead_max_m = 11.0;
  cfg.max_steer_rad = 2.0;
  StanleyController controller(corner, cfg);

  const auto result = controller.update(
    Pose2{0.0, 1.0, 0.0}, 0.0, 1.0, 1, 4);

  ASSERT_TRUE(result.valid);
  EXPECT_NEAR(result.command.steering_rad, std::atan(0.5), 1e-12);
}

TEST(StanleyController, ScalesHeadingPreviewWithCurrentSpeed)
{
  Route corner{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
    {2.0, 1.0, 0.0}, {2.0, 2.0, 0.0}}, false};
  StanleyConfig cfg = config();
  cfg.lookahead_time_s = 1.0;
  cfg.lookahead_min_m = 0.0;
  cfg.lookahead_max_m = 10.0;
  cfg.max_steer_rad = 2.0;
  StanleyController slow(corner, cfg);
  StanleyController fast(corner, cfg);

  const auto slow_result = slow.update(Pose2{}, 1.0, 1.0, 1, 4);
  const auto fast_result = fast.update(Pose2{}, 3.0, 1.0, 1, 4);

  ASSERT_TRUE(slow_result.valid);
  ASSERT_TRUE(fast_result.valid);
  ASSERT_TRUE(slow_result.target.has_value());
  ASSERT_TRUE(fast_result.target.has_value());
  EXPECT_DOUBLE_EQ(slow_result.target->x, 1.0);
  EXPECT_DOUBLE_EQ(slow_result.target->y, 0.0);
  EXPECT_DOUBLE_EQ(fast_result.target->x, 2.0);
  EXPECT_DOUBLE_EQ(fast_result.target->y, 1.0);
}

TEST(StanleyController, LimitsSteeringChangeToOneHundredTwentyDegreesPerSecond)
{
  StanleyConfig cfg = config();
  cfg.lookahead_min_m = 0.0;
  cfg.lookahead_max_m = 0.0;
  cfg.max_steer_rad = 0.6981317;
  StanleyController controller(line_route(), cfg);
  const double maximum_delta = 120.0 * std::acos(-1.0) / 180.0 * 0.05;

  const auto first = controller.update(Pose2{0.0, 10.0, 0.0}, 0.0, 0.05, 1, 4);
  const auto second = controller.update(Pose2{0.0, 10.0, 0.0}, 0.0, 0.05, 1, 4);

  ASSERT_TRUE(first.valid);
  ASSERT_TRUE(second.valid);
  EXPECT_NEAR(first.command.steering_rad, -maximum_delta, 1e-12);
  EXPECT_NEAR(second.command.steering_rad, -2.0 * maximum_delta, 1e-12);
}

TEST(StanleyController, ClosedRouteLookaheadWrapsAcrossThePathSeam)
{
  Route square{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0},
    {0.0, 1.0, 0.0}}, true};
  StanleyConfig cfg = config();
  cfg.lookahead_min_m = 2.0;
  cfg.lookahead_max_m = 2.0;
  cfg.max_steer_rad = 4.0;
  cfg.forward_window = 4;
  StanleyController controller(square, cfg);

  const auto result = controller.update(
    Pose2{0.0, 1.0, std::acos(-1.0)}, 0.0, 1.0, 1, 4);

  ASSERT_TRUE(result.valid);
  ASSERT_TRUE(result.target.has_value());
  EXPECT_DOUBLE_EQ(result.target->x, 1.0);
  EXPECT_DOUBLE_EQ(result.target->y, 0.0);
}

TEST(StanleyController, ReseedsProgressAndLaunchRampAfterPoseJump)
{
  Route route;
  for (int x = 0; x <= 100; ++x) {
    route.points.push_back(Point3{static_cast<double>(x), 0.0, 0.0});
  }
  StanleyConfig cfg = config();
  cfg.target_speed_mps = 10.0;
  cfg.lookahead_min_m = 0.0;
  cfg.lookahead_max_m = 0.0;
  cfg.forward_window = 2;
  cfg.pid.kp = 0.1;
  cfg.launch_speed_mps = 1.0;
  cfg.launch_ramp_s = 4.0;
  StanleyController controller(route, cfg);

  ASSERT_TRUE(controller.update(Pose2{0.0, 0.0, 0.0}, 0.0, 1.0, 1, 4).valid);
  ASSERT_TRUE(controller.update(Pose2{2.0, 0.0, 0.0}, 0.0, 1.0, 1, 4).valid);
  const auto jumped = controller.update(Pose2{90.0, 0.0, 0.0}, 0.0, 1.0, 1, 4);

  ASSERT_TRUE(jumped.valid);
  ASSERT_TRUE(jumped.target.has_value());
  EXPECT_DOUBLE_EQ(jumped.target->x, 90.0);
  ASSERT_TRUE(jumped.target_speed_mps.has_value());
  EXPECT_DOUBLE_EQ(*jumped.target_speed_mps, 1.0);
  EXPECT_NEAR(jumped.command.accel, 0.1, 1e-12);
}

TEST(StanleyController, FullBrakesOnLocalizationJumpFarFromEveryRoutePoint)
{
  StanleyConfig cfg = config();
  cfg.lookahead_min_m = 0.0;
  cfg.lookahead_max_m = 0.0;
  StanleyController controller(line_route(), cfg);

  const auto result =
    controller.update(Pose2{100.0, 100.0, 0.0}, 1.0, 0.1, 1, 4);

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.reason, "route localization mismatch");
  EXPECT_DOUBLE_EQ(result.command.accel, 0.0);
  EXPECT_DOUBLE_EQ(result.command.brake, 1.0);
}

TEST(StanleyController, ZeroControlPointOffsetPreservesRearAxleGeometry)
{
  StanleyConfig cfg = config();
  cfg.control_point_x_m = 0.0;
  cfg.lookahead_min_m = 0.0;
  cfg.lookahead_max_m = 0.0;
  cfg.max_steer_rad = 2.0;
  StanleyController controller(line_route(), cfg);

  const auto result = controller.update(Pose2{1.0, 1.0, 0.0}, 0.0, 1.0, 1, 4);

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(controller.progress_state().target_index, 1U);
  EXPECT_NEAR(result.command.steering_rad, -std::atan(2.0), 1e-12);
}

TEST(StanleyController, ThreeMeterControlPointRotatesWithYaw)
{
  Route vertical;
  for (int y = 0; y <= 6; ++y) {
    vertical.points.push_back(Point3{0.0, static_cast<double>(y), 0.0});
  }
  StanleyConfig cfg = config();
  cfg.control_point_x_m = 3.0;
  cfg.lookahead_min_m = 0.0;
  cfg.lookahead_max_m = 0.0;
  StanleyController forward_x(vertical, cfg);
  StanleyController forward_y(vertical, cfg);

  ASSERT_TRUE(forward_x.update(Pose2{0.0, 0.0, 0.0}, 0.0, 1.0, 1, 4).valid);
  ASSERT_TRUE(
    forward_y.update(Pose2{0.0, 0.0, std::acos(-1.0) / 2.0}, 0.0, 1.0, 1, 4).valid);

  EXPECT_EQ(forward_x.progress_state().target_index, 0U);
  EXPECT_EQ(forward_y.progress_state().target_index, 3U);
}

TEST(StanleyController, ControlPointOffsetChangesRouteProgress)
{
  StanleyConfig rear_cfg = config();
  rear_cfg.control_point_x_m = 0.0;
  rear_cfg.lookahead_min_m = 0.0;
  rear_cfg.lookahead_max_m = 0.0;
  StanleyConfig front_cfg = rear_cfg;
  front_cfg.control_point_x_m = 3.0;
  StanleyController rear_controller(line_route(), rear_cfg);
  StanleyController front_controller(line_route(), front_cfg);

  ASSERT_TRUE(
    rear_controller.update(Pose2{0.0, 0.0, 0.0}, 0.0, 1.0, 1, 4).valid);
  ASSERT_TRUE(
    front_controller.update(Pose2{0.0, 0.0, 0.0}, 0.0, 1.0, 1, 4).valid);

  EXPECT_EQ(rear_controller.progress_state().target_index, 0U);
  EXPECT_EQ(front_controller.progress_state().target_index, 3U);
}

TEST(StanleyController, RejectsNanControlPointOffset)
{
  StanleyConfig cfg = config();
  cfg.control_point_x_m = std::numeric_limits<double>::quiet_NaN();

  EXPECT_THROW(StanleyController(line_route(), cfg), std::invalid_argument);
}

TEST(StanleyController, RejectsInfiniteControlPointOffset)
{
  StanleyConfig cfg = config();
  cfg.control_point_x_m = std::numeric_limits<double>::infinity();

  EXPECT_THROW(StanleyController(line_route(), cfg), std::invalid_argument);
}

TEST(StanleyController, AccumulatesLookaheadWithoutEarlyWrapAndStopsAtMaxLaps)
{
  auto route = line_route();
  route.closed = true;
  StanleyConfig cfg = config();
  cfg.lookahead_min_m = 2.1;
  cfg.lookahead_max_m = 2.1;
  cfg.forward_window = 4;
  cfg.max_laps = 1;
  StanleyController controller(route, cfg);

  auto first = controller.update(Pose2{0.0, 0.0, 0.0}, 1.0, 0.1, 1, 4);
  ASSERT_TRUE(first.valid);
  ASSERT_TRUE(first.target.has_value());
  EXPECT_DOUBLE_EQ(first.target->x, 3.0);
  EXPECT_EQ(controller.progress_state().lap_count, 0U);

  controller.update(Pose2{4.0, 0.0, std::acos(-1.0)}, 1.0, 0.1, 1, 4);
  const auto terminal = controller.update(Pose2{0.0, 0.0, 0.0}, 1.0, 0.1, 1, 4);
  EXPECT_TRUE(terminal.valid);
  EXPECT_DOUBLE_EQ(terminal.command.brake, 1.0);
  ASSERT_TRUE(terminal.target_speed_mps.has_value());
  EXPECT_DOUBLE_EQ(*terminal.target_speed_mps, 0.0);
  EXPECT_EQ(controller.progress_state().lap_count, 1U);
}

TEST(StanleyController, InvalidInputDoesNotMutateProgressPidOrLastValid)
{
  StanleyController controller(line_route(), config());
  const auto valid = controller.update(Pose2{0.0, 0.0, 0.0}, 1.0, 0.1, 1, 4);
  const auto progress = controller.progress_state();
  const auto pid = controller.pid_state();
  const auto invalid = controller.update(
    Pose2{0.0, 0.0, std::numeric_limits<double>::infinity()}, 1.0, 0.1, 9, 9);
  EXPECT_FALSE(invalid.valid);
  expect_progress_equal(controller.progress_state(), progress);
  EXPECT_EQ(controller.pid_state(), pid);
  EXPECT_EQ(controller.last_valid_result(), valid);
}

TEST(StanleyController, ExtremeGeometryFailureIsFiniteAndFullyTransactional)
{
  const double maximum = std::numeric_limits<double>::max();
  Route extreme{{{maximum, maximum, 0.0}, {-maximum, -maximum, 0.0}}, false};
  StanleyConfig cfg = config();
  cfg.lookahead_min_m = 0.0;
  cfg.lookahead_max_m = 0.0;
  StanleyController controller(extreme, cfg);
  const auto progress = controller.progress_state();
  const auto pid = controller.pid_state();
  const auto last_valid = controller.last_valid_result();

  const auto result = controller.update(Pose2{0.0, 0.0, 0.0}, 1.0, 0.1, 1, 4);

  EXPECT_FALSE(result.valid);
  EXPECT_TRUE(std::isfinite(result.command.steering_rad));
  EXPECT_DOUBLE_EQ(result.command.brake, 1.0);
  expect_progress_equal(controller.progress_state(), progress);
  EXPECT_EQ(controller.pid_state(), pid);
  EXPECT_EQ(controller.last_valid_result(), last_valid);
}
}  // namespace
