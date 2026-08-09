#include <gtest/gtest.h>

#include <limits>

#include "ad_planner/behavior/planner_context.hpp"
#include "ad_planner/common/vehicle_observation.hpp"

namespace
{

using ad_planner::AcknowledgedGear;
using ad_planner::GearRequest;
using ad_planner::kGearDriveCode;
using ad_planner::kGearReverseCode;
using ad_planner::validated_speed_mps;

TEST(AcknowledgedGear, KeepsStatusGearBeforeOdometryForFailsafeCommand)
{
  AcknowledgedGear gear;
  EXPECT_EQ(gear.resolve(GearRequest::kKeep), kGearDriveCode);

  EXPECT_TRUE(gear.update(1));

  EXPECT_EQ(gear.resolve(GearRequest::kKeep), 1);
}

TEST(AcknowledgedGear, InvalidStatusCannotReplaceLastValidGear)
{
  AcknowledgedGear gear;
  ASSERT_TRUE(gear.update(3));

  EXPECT_FALSE(gear.update(-1));
  EXPECT_FALSE(gear.update(6));
  EXPECT_EQ(gear.resolve(GearRequest::kKeep), 3);
  EXPECT_EQ(gear.resolve(GearRequest::kReverse), kGearReverseCode);
  EXPECT_EQ(gear.resolve(GearRequest::kDrive), kGearDriveCode);
}

TEST(VehicleObservation, UsesFiniteLongitudinalVehicleSpeedAsNonnegativeControlSpeed)
{
  ASSERT_TRUE(validated_speed_mps(-4.25).has_value());
  EXPECT_DOUBLE_EQ(*validated_speed_mps(-4.25), 4.25);
  ASSERT_TRUE(validated_speed_mps(7.0).has_value());
  EXPECT_DOUBLE_EQ(*validated_speed_mps(7.0), 7.0);
}

TEST(VehicleObservation, RejectsNonfiniteVehicleSpeed)
{
  EXPECT_FALSE(validated_speed_mps(std::numeric_limits<double>::quiet_NaN()).has_value());
  EXPECT_FALSE(validated_speed_mps(std::numeric_limits<double>::infinity()).has_value());
}

}  // namespace
