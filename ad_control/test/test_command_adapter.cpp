#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ad_control/command/command_adapter.hpp"

namespace
{

using ad_control::AcknowledgedGear;
using ad_control::GearRequest;
using ad_control::PhysicalCommand;
using ad_control::kGearDriveCode;
using ad_control::kGearReverseCode;
using ad_control::make_ctrl_cmd;
using ad_morai_interfaces::msg::CtrlCmd;

std_msgs::msg::Header header()
{
  std_msgs::msg::Header value;
  value.stamp.sec = 123;
  value.stamp.nanosec = 456;
  value.frame_id = "base_link";
  return value;
}

TEST(CommandAdapter, PreservesTheExistingMoraiTypeOneMapping)
{
  AcknowledgedGear gear;
  ASSERT_TRUE(gear.update(CtrlCmd::GEAR_PARK));

  const auto message = make_ctrl_cmd(
    PhysicalCommand{0.25, 0.0, 0.34906585}, GearRequest::kKeep,
    gear, 0.6981317, header());

  EXPECT_EQ(message.header, header());
  EXPECT_EQ(message.ctrl_mode, CtrlCmd::CTRL_MODE_AUTO);
  EXPECT_EQ(message.gear, CtrlCmd::GEAR_PARK);
  EXPECT_EQ(message.long_cmd_type, CtrlCmd::LONG_CMD_THROTTLE);
  EXPECT_FLOAT_EQ(message.velocity, 0.0F);
  EXPECT_FLOAT_EQ(message.acceleration, 0.0F);
  EXPECT_FLOAT_EQ(message.accel, 0.25F);
  EXPECT_FLOAT_EQ(message.brake, 0.0F);
  EXPECT_FLOAT_EQ(message.steering, static_cast<float>(0.34906585 / 0.6981317));
}

TEST(CommandAdapter, MapsPhysicalCommandsAndGearRequestsFromRegressionTable)
{
  AcknowledgedGear gear;
  ASSERT_TRUE(gear.update(CtrlCmd::GEAR_PARK));

  struct Case
  {
    PhysicalCommand physical;
    GearRequest gear_request;
    double steering_limit_rad;
    std::int8_t expected_gear;
    float expected_accel;
    float expected_brake;
    float expected_steering;
  };
  const std::array<Case, 3> cases{{
    {{0.25, 0.0, 0.34906585}, GearRequest::kKeep, 0.6981317,
      CtrlCmd::GEAR_PARK, 0.25F, 0.0F, 0.5F},
    {{0.2, 0.0, 2.0}, GearRequest::kReverse, 0.5,
      kGearReverseCode, 0.2F, 0.0F, 1.0F},
    {{0.0, 1.0, -2.0}, GearRequest::kDrive, 0.5,
      kGearDriveCode, 0.0F, 1.0F, -1.0F},
  }};

  for (const auto & item : cases) {
    const auto message = make_ctrl_cmd(
      item.physical, item.gear_request, gear, item.steering_limit_rad, header());
    EXPECT_EQ(message.gear, item.expected_gear);
    EXPECT_FLOAT_EQ(message.accel, item.expected_accel);
    EXPECT_FLOAT_EQ(message.brake, item.expected_brake);
    EXPECT_FLOAT_EQ(message.steering, item.expected_steering);
  }
}

TEST(CommandAdapter, KeepsTheLastValidAcknowledgedGear)
{
  AcknowledgedGear gear;
  EXPECT_EQ(gear.resolve(GearRequest::kKeep), kGearDriveCode);
  ASSERT_TRUE(gear.update(CtrlCmd::GEAR_NEUTRAL));
  EXPECT_FALSE(gear.update(-1));
  EXPECT_FALSE(gear.update(6));
  EXPECT_EQ(gear.resolve(GearRequest::kKeep), CtrlCmd::GEAR_NEUTRAL);
}

TEST(CommandAdapter, RejectsAnInvalidPhysicalSteeringLimit)
{
  AcknowledgedGear gear;
  for (const double limit : {
      0.0, -1.0, std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN()})
  {
    EXPECT_THROW(
      make_ctrl_cmd(PhysicalCommand{}, GearRequest::kKeep, gear, limit, header()),
      std::invalid_argument);
  }
}

}  // namespace
