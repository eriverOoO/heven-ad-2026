#include <array>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "ad_control/longitudinal/pid.hpp"

namespace
{
using ad_control::PidConfig;
using ad_control::PidController;

TEST(PidController, InitializesDeterministicallyAndActivelyBrakesOverspeed)
{
  struct Case {double current; double target; double accel; double brake;};
  const std::array<Case, 2> cases{{
    {0.0, 1.0, 0.51, 0.0},
    {3.0, 1.0, 0.0, 1.0},
  }};
  for (const auto & test : cases) {
    PidController pid(PidConfig{0.5, 0.1, 0.2, 2.0, 3.0});
    const auto result = pid.update(test.current, test.target, 0.1, 1, 4);
    ASSERT_TRUE(result.valid);
    EXPECT_DOUBLE_EQ(result.command.accel, test.accel);
    EXPECT_DOUBLE_EQ(result.command.brake, test.brake);
  }
}

TEST(PidController, BoundsIntegralAndDerivativeAndResetsOnModeTransition)
{
  PidController pid(PidConfig{0.0, 1.0, 1.0, 0.25, 0.5});
  EXPECT_DOUBLE_EQ(pid.update(0.0, 1.0, 1.0, 1, 4).command.accel, 0.25);
  EXPECT_NEAR(pid.update(1.0, 0.0, 0.1, 1, 4).command.brake, 0.1, 1e-12);
  EXPECT_DOUBLE_EQ(pid.update(0.0, 1.0, 0.1, 2, 4).command.accel, 0.1);
  EXPECT_DOUBLE_EQ(pid.update(0.0, 1.0, 0.1, 2, 2).command.accel, 0.1);
}

TEST(PidController, LowPassFiltersDerivativeNoise)
{
  PidConfig config{0.0, 0.0, 1.0, 1.0, 10.0};
  config.derivative_filter_time_constant_s = 0.1;
  PidController pid(config);

  ASSERT_TRUE(pid.update(0.0, 1.0, 0.1, 1, 4).valid);
  const auto filtered = pid.update(0.0, 1.1, 0.1, 1, 4);

  ASSERT_TRUE(filtered.valid);
  EXPECT_NEAR(filtered.command.accel, 0.5, 1.0e-12);
  EXPECT_NEAR(pid.state().filtered_derivative, 0.5, 1.0e-12);
}

TEST(PidController, SeparateBrakeGainsUseOnlyDeadbandExcess)
{
  PidConfig config{0.9, 0.0, 0.0, 100.0, 100.0};
  config.error_scale = 3.6;
  config.brake_deadband = 1.0;
  config.use_separate_brake_gains = true;
  config.brake_kp = 0.2;
  PidController pid(config);

  const auto coast = pid.update(10.0, 9.8, 0.1, 1, 4);
  ASSERT_TRUE(coast.valid);
  EXPECT_DOUBLE_EQ(coast.command.brake, 0.0);

  const auto brake = pid.update(10.0, 9.5, 0.1, 1, 4);
  ASSERT_TRUE(brake.valid);
  EXPECT_DOUBLE_EQ(brake.command.accel, 0.0);
  EXPECT_NEAR(brake.command.brake, 0.16, 1.0e-12);
}

TEST(PidController, PositiveIntegralHistoryCannotSuppressOverspeedBraking)
{
  PidController pid(PidConfig{0.0, 0.1, 0.0, 10.0, 1.0});
  for (int sample = 0; sample < 5; ++sample) {
    ASSERT_TRUE(pid.update(0.0, 1.0, 1.0, 1, 4).valid);
  }
  ASSERT_GT(pid.state().integral, 0.0);

  const auto overspeed = pid.update(1.1, 1.0, 1.0, 1, 4);
  ASSERT_TRUE(overspeed.valid);
  EXPECT_DOUBLE_EQ(overspeed.command.accel, 0.0);
  EXPECT_GT(overspeed.command.brake, 0.0);
  EXPECT_LT(pid.state().integral, 0.0);
}

TEST(PidController, SaturationDoesNotWindIntegralFurtherIntoTheLimit)
{
  PidController pid(PidConfig{1.0, 1.0, 0.0, 10.0, 1.0});
  const auto saturated = pid.update(0.0, 1.0, 1.0, 1, 4);
  ASSERT_TRUE(saturated.valid);
  EXPECT_DOUBLE_EQ(saturated.command.accel, 1.0);
  EXPECT_DOUBLE_EQ(pid.state().integral, 0.0);
}

TEST(PidController, InvalidInputReturnsFullBrakeWithoutMutatingStateOrLastValid)
{
  PidController pid(PidConfig{0.5, 0.1, 0.2, 2.0, 3.0});
  const auto valid = pid.update(0.0, 1.0, 0.1, 1, 4);
  const auto before = pid.state();
  const auto invalid = pid.update(
    0.0, std::numeric_limits<double>::quiet_NaN(), 0.1, 9, 9);
  EXPECT_FALSE(invalid.valid);
  EXPECT_DOUBLE_EQ(invalid.command.brake, 1.0);
  EXPECT_EQ(pid.state(), before);
  EXPECT_EQ(pid.last_valid_result(), valid);
}

TEST(PidController, RejectsInvalidConfigurationAndNonPositiveDt)
{
  EXPECT_THROW(PidController(PidConfig{0.1, 0.0, 0.0, -1.0, 1.0}), std::invalid_argument);
  PidController pid(PidConfig{0.1, 0.0, 0.0, 1.0, 1.0});
  EXPECT_FALSE(pid.update(0.0, 1.0, 0.0, 1, 4).valid);
}

TEST(PidController, FiniteExtremeInputsCannotOverflowStoredState)
{
  PidController pid(PidConfig{0.5, 0.1, 0.2, 2.0, 3.0});
  const auto before = pid.state();
  const auto result = pid.update(
    -std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
    1.0, 1, 4);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(pid.state(), before);
}
}  // namespace
