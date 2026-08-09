#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ad_control/command/curvature_command_adapter.hpp"
#include "ad_control/command/mppi_command_adapter.hpp"

namespace
{

using ad_control::CurvatureCommandAdapter;
using ad_control::CurvatureCommandAdapterConfig;
using ad_control::CurvatureCommandInput;
using ad_control::MppiCommandAdapter;
using ad_control::MppiCommandConfig;
using ad_control::PidConfig;

MppiCommandConfig config(const double steering_rate_limit_rad_s = 0.35)
{
  MppiCommandConfig result;
  result.wheelbase_m = 3.0;
  result.maximum_road_wheel_angle_rad = 0.588;
  result.steering_rate_limit_rad_s = steering_rate_limit_rad_s;
  result.near_zero_speed_mps = 0.05;
  result.allow_reverse = false;
  result.longitudinal_pid = PidConfig{0.5, 0.1, 0.0, 2.0, 3.0};
  return result;
}

void expect_full_brake(const ad_control::MppiCommandResult & result)
{
  EXPECT_FALSE(result.valid);
  EXPECT_DOUBLE_EQ(result.command.accel, 0.0);
  EXPECT_DOUBLE_EQ(result.command.brake, 1.0);
  EXPECT_TRUE(std::isfinite(result.command.steering_rad));
  EXPECT_DOUBLE_EQ(result.desired_speed_mps, 0.0);
  EXPECT_DOUBLE_EQ(result.desired_curvature_inv_m, 0.0);
}

TEST(MppiCommandAdapter, ConvertsForwardTwistAndDelegatesRoadWheelGeometry)
{
  MppiCommandAdapter adapter(config(20.0));

  const auto result = adapter.update(5.0, 1.0, 4.0, 0.05, 0.0, 1, 4);

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.reason, "ok");
  EXPECT_DOUBLE_EQ(result.desired_speed_mps, 5.0);
  EXPECT_DOUBLE_EQ(result.desired_curvature_inv_m, 0.2);
  EXPECT_NEAR(result.command.steering_rad, std::atan(3.0 * 0.2), 1e-12);
}

TEST(MppiCommandAdapter, NearZeroSpeedStopsWithoutDividingAndSlewsTowardStraight)
{
  MppiCommandAdapter adapter(config());

  const auto result = adapter.update(0.05, 100.0, 2.0, 0.1, 0.03, 1, 4);

  ASSERT_TRUE(result.valid);
  EXPECT_DOUBLE_EQ(result.desired_speed_mps, 0.0);
  EXPECT_DOUBLE_EQ(result.desired_curvature_inv_m, 0.0);
  EXPECT_DOUBLE_EQ(result.command.accel, 0.0);
  EXPECT_GT(result.command.brake, 0.0);
  EXPECT_DOUBLE_EQ(result.command.steering_rad, 0.0);
}

TEST(MppiCommandAdapter, ReverseOutsideStopBandIsFullBrakeAndDoesNotMutatePid)
{
  MppiCommandAdapter adapter(config());
  ASSERT_TRUE(adapter.update(1.0, 0.1, 0.0, 0.1, 0.0, 1, 4).valid);
  const auto state_before = adapter.pid_state();

  const auto result = adapter.update(-1.0, 0.1, 1.0, 0.1, 0.0, 1, 4);

  expect_full_brake(result);
  EXPECT_EQ(result.reason, "reverse_disabled");
  EXPECT_EQ(adapter.pid_state(), state_before);
}

TEST(MppiCommandAdapter, ClampsSteeringThenAppliesCommonRateLimit)
{
  MppiCommandAdapter adapter(config());

  const auto result = adapter.update(1.0, 100.0, 0.0, 0.1, 0.1, 1, 4);

  ASSERT_TRUE(result.valid);
  EXPECT_NEAR(result.command.steering_rad, 0.135, 1e-12);
}

TEST(MppiCommandAdapter, InvalidInputsAndDivisionOverflowDoNotMutatePid)
{
  MppiCommandAdapter adapter(config());
  ASSERT_TRUE(adapter.update(1.0, 0.1, 0.0, 0.1, 0.0, 1, 4).valid);
  const auto state_before = adapter.pid_state();
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto inf = std::numeric_limits<double>::infinity();

  const auto assert_rejected =
    [&](const double vx, const double wz, const double actual,
      const double dt, const double previous) {
      const auto result =
        adapter.update(vx, wz, actual, dt, previous, 9, 9);
      expect_full_brake(result);
      EXPECT_EQ(adapter.pid_state(), state_before);
    };

  assert_rejected(nan, 0.0, 0.0, 0.1, 0.0);
  assert_rejected(1.0, inf, 0.0, 0.1, 0.0);
  assert_rejected(1.0, 0.0, nan, 0.1, 0.0);
  assert_rejected(1.0, 0.0, 0.0, 0.0, 0.0);
  assert_rejected(1.0, 0.0, 0.0, -0.1, 0.0);
  assert_rejected(1.0, 0.0, 0.0, 0.1, inf);
  assert_rejected(1.0, 0.0, 0.0, 0.1, 0.6);
  assert_rejected(
    std::nextafter(0.05, std::numeric_limits<double>::infinity()),
    std::numeric_limits<double>::max(), 0.0, 0.1, 0.0);
}

TEST(MppiCommandAdapter, UsesExistingLongitudinalPidAndSteeringSemantics)
{
  const auto mppi_config = config(5.0);
  MppiCommandAdapter adapter(mppi_config);
  CurvatureCommandAdapter direct(CurvatureCommandAdapterConfig{
    mppi_config.longitudinal_pid,
    mppi_config.wheelbase_m,
    mppi_config.maximum_road_wheel_angle_rad,
    mppi_config.steering_rate_limit_rad_s,
  });
  const double expected_curvature = 0.3 / 3.0;
  const auto input = CurvatureCommandInput{
    2.0, 3.0, expected_curvature, 0.1, 0.2, 7, 4};

  const auto expected = direct.update(input);
  const auto actual =
    adapter.update(3.0, 0.3, 2.0, 0.2, 0.1, 7, 4);

  ASSERT_TRUE(actual.valid);
  EXPECT_EQ(actual.valid, expected.valid);
  EXPECT_EQ(actual.reason, expected.reason);
  EXPECT_EQ(actual.command, expected.command);
  EXPECT_DOUBLE_EQ(actual.desired_speed_mps, input.target_speed_mps);
  EXPECT_DOUBLE_EQ(
    actual.desired_curvature_inv_m, input.desired_curvature_inv_m);
  EXPECT_EQ(adapter.pid_state(), direct.pid_state());
}

TEST(MppiCommandAdapter, ProjectsVelocityWithoutMutatingPidAndSharesUpdateSemantics)
{
  MppiCommandAdapter adapter(config(5.0));
  const auto state_before = adapter.pid_state();

  const auto projected = adapter.target_from_twist(3.0, 0.6);

  ASSERT_TRUE(projected.valid);
  EXPECT_EQ(projected.reason, "ok");
  EXPECT_DOUBLE_EQ(projected.desired_speed_mps, 3.0);
  EXPECT_DOUBLE_EQ(projected.desired_curvature_inv_m, 0.2);
  EXPECT_EQ(adapter.pid_state(), state_before);

  const auto updated =
    adapter.update(3.0, 0.6, 2.0, 0.1, 0.0, 1, 4);
  ASSERT_TRUE(updated.valid);
  EXPECT_DOUBLE_EQ(
    updated.desired_speed_mps, projected.desired_speed_mps);
  EXPECT_DOUBLE_EQ(
    updated.desired_curvature_inv_m,
    projected.desired_curvature_inv_m);

  const auto stopped = adapter.target_from_twist(0.05, 100.0);
  ASSERT_TRUE(stopped.valid);
  EXPECT_DOUBLE_EQ(stopped.desired_speed_mps, 0.0);
  EXPECT_DOUBLE_EQ(stopped.desired_curvature_inv_m, 0.0);

  const auto reverse = adapter.target_from_twist(-1.0, 0.1);
  EXPECT_FALSE(reverse.valid);
  EXPECT_EQ(reverse.reason, "reverse_disabled");
}

TEST(MppiCommandAdapter, RejectsInvalidStopBandAndReverseEnabledConfiguration)
{
  auto invalid = config();
  invalid.near_zero_speed_mps = -0.1;
  EXPECT_THROW((void)MppiCommandAdapter{invalid}, std::invalid_argument);

  invalid = config();
  invalid.near_zero_speed_mps =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)MppiCommandAdapter{invalid}, std::invalid_argument);

  invalid = config();
  invalid.allow_reverse = true;
  EXPECT_THROW((void)MppiCommandAdapter{invalid}, std::invalid_argument);
}

}  // namespace
