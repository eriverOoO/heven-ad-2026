#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ad_control/command/curvature_command_adapter.hpp"

namespace
{
using ad_control::CurvatureCommandAdapter;
using ad_control::CurvatureCommandAdapterConfig;
using ad_control::CurvatureCommandInput;
using ad_control::PidConfig;

CurvatureCommandAdapterConfig config()
{
  return CurvatureCommandAdapterConfig{
    PidConfig{0.5, 0.1, 0.0, 2.0, 3.0}, 2.8, 0.5, 2.0};
}

CurvatureCommandInput input()
{
  return CurvatureCommandInput{0.0, 1.0, 0.1, 0.0, 0.1, 1, 4};
}

TEST(CurvatureCommandAdapter, PositiveSpeedErrorAcceleratesWithFiniteSteering)
{
  CurvatureCommandAdapter adapter(config());
  const auto result = adapter.update(input());

  ASSERT_TRUE(result.valid);
  EXPECT_GT(result.command.accel, 0.0);
  EXPECT_DOUBLE_EQ(result.command.brake, 0.0);
  EXPECT_TRUE(std::isfinite(result.command.steering_rad));
}

TEST(CurvatureCommandAdapter, NegativeSpeedErrorBrakes)
{
  CurvatureCommandAdapter adapter(config());
  auto command = input();
  command.current_speed_mps = 2.0;
  command.target_speed_mps = 1.0;

  const auto result = adapter.update(command);

  ASSERT_TRUE(result.valid);
  EXPECT_DOUBLE_EQ(result.command.accel, 0.0);
  EXPECT_GT(result.command.brake, 0.0);
}

TEST(CurvatureCommandAdapter, ConvertsCurvatureUsingRoadWheelGeometry)
{
  CurvatureCommandAdapter adapter(config());
  auto command = input();
  command.desired_curvature_inv_m = 0.05;
  command.dt_s = 1.0;

  const auto result = adapter.update(command);

  ASSERT_TRUE(result.valid);
  EXPECT_NEAR(result.command.steering_rad, std::atan(2.8 * 0.05), 1e-12);
}

TEST(CurvatureCommandAdapter, SaturatesRoadWheelSteeringAtConfiguredLimit)
{
  CurvatureCommandAdapter adapter(config());
  auto command = input();
  command.desired_curvature_inv_m = 100.0;
  command.dt_s = 1.0;

  const auto result = adapter.update(command);

  ASSERT_TRUE(result.valid);
  EXPECT_DOUBLE_EQ(result.command.steering_rad, 0.5);
}

TEST(CurvatureCommandAdapter, BoundsPositiveAndNegativeSteeringSlew)
{
  CurvatureCommandAdapter adapter(config());
  auto command = input();
  command.dt_s = 0.1;
  command.desired_curvature_inv_m = 10.0;
  command.previous_steering_rad = 0.1;

  const auto positive = adapter.update(command);
  ASSERT_TRUE(positive.valid);
  EXPECT_DOUBLE_EQ(positive.command.steering_rad, 0.3);

  command.desired_curvature_inv_m = -10.0;
  command.previous_steering_rad = -0.1;
  const auto negative = adapter.update(command);
  ASSERT_TRUE(negative.valid);
  EXPECT_DOUBLE_EQ(negative.command.steering_rad, -0.3);
}

TEST(CurvatureCommandAdapter, RejectsInvalidConstructorScalars)
{
  auto bad = config();
  bad.wheelbase_m = 0.0;
  EXPECT_THROW((void)CurvatureCommandAdapter{bad}, std::invalid_argument);
  bad = config();
  bad.maximum_road_wheel_steering_rad = -0.1;
  EXPECT_THROW((void)CurvatureCommandAdapter{bad}, std::invalid_argument);
  bad = config();
  bad.maximum_steering_rate_radps = std::numeric_limits<double>::infinity();
  EXPECT_THROW((void)CurvatureCommandAdapter{bad}, std::invalid_argument);
  bad = config();
  bad.pid.kp = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)CurvatureCommandAdapter{bad}, std::invalid_argument);
}

TEST(CurvatureCommandAdapter, InvalidInputPreservesPidStateAndLastValidResult)
{
  CurvatureCommandAdapter adapter(config());
  const auto valid = adapter.update(input());
  ASSERT_TRUE(valid.valid);
  const auto state_before = adapter.pid_state();
  const auto last_before = adapter.last_valid_result();

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const CurvatureCommandInput invalid_inputs[] = {
    CurvatureCommandInput{nan, 1.0, 0.0, 0.0, 0.1, 9, 9},
    CurvatureCommandInput{0.0, inf, 0.0, 0.0, 0.1, 9, 9},
    CurvatureCommandInput{-0.1, 1.0, 0.0, 0.0, 0.1, 9, 9},
    CurvatureCommandInput{0.0, -0.1, 0.0, 0.0, 0.1, 9, 9},
    CurvatureCommandInput{0.0, 1.0, nan, 0.0, 0.1, 9, 9},
    CurvatureCommandInput{0.0, 1.0, 0.0, inf, 0.1, 9, 9},
    CurvatureCommandInput{0.0, 1.0, 0.0, 0.6, 0.1, 9, 9},
    CurvatureCommandInput{0.0, 1.0, 0.0, 0.0, 0.0, 9, 9},
  };
  for (const auto & invalid_input : invalid_inputs) {
    const auto invalid = adapter.update(invalid_input);
    EXPECT_FALSE(invalid.valid);
    EXPECT_DOUBLE_EQ(invalid.command.brake, 1.0);
    EXPECT_EQ(adapter.pid_state(), state_before);
    EXPECT_EQ(adapter.last_valid_result(), last_before);
  }
}
}  // namespace
