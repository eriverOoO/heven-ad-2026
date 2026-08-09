#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "ad_planner/local_planning/common/local_motion_validation.hpp"

namespace ad_planner {
namespace {

TEST(LocalMotionValidation, ConvertsValidRosHeaderStamp) {
  const auto stamp_ns = valid_ros_stamp_nanoseconds(42, 123U);

  ASSERT_TRUE(stamp_ns.has_value());
  EXPECT_EQ(*stamp_ns, 42'000'000'123LL);
}

TEST(LocalMotionValidation, RejectsZeroRosHeaderStamp) {
  EXPECT_FALSE(valid_ros_stamp_nanoseconds(0, 0U).has_value());
}

TEST(LocalMotionValidation, RejectsNegativeRosHeaderStampSeconds) {
  EXPECT_FALSE(valid_ros_stamp_nanoseconds(-1, 1U).has_value());
}

TEST(LocalMotionValidation, RejectsOutOfRangeRosHeaderStampNanoseconds) {
  EXPECT_FALSE(valid_ros_stamp_nanoseconds(1, 1'000'000'000U).has_value());
  EXPECT_FALSE(
      valid_ros_stamp_nanoseconds(1, std::numeric_limits<std::uint32_t>::max())
          .has_value());
}

TEST(LocalMotionValidation, ConvertsMaximumValidRosHeaderStampWithoutOverflow) {
  const auto stamp_ns = valid_ros_stamp_nanoseconds(
      std::numeric_limits<std::int32_t>::max(), 999'999'999U);

  ASSERT_TRUE(stamp_ns.has_value());
  EXPECT_EQ(*stamp_ns, 2'147'483'647'999'999'999LL);
}

TEST(LocalMotionValidation, AcceptsDirectCommandBoundaries) {
  constexpr double steering_limit = 0.52;

  EXPECT_TRUE(valid_direct_command(PhysicalCommand{1.0, 0.0, steering_limit},
                                   steering_limit));
  EXPECT_TRUE(valid_direct_command(PhysicalCommand{0.0, 1.0, -steering_limit},
                                   steering_limit));
}

TEST(LocalMotionValidation, AppliesEachCommandConsumersExactSteeringLimit) {
  const PhysicalCommand command{0.0, 0.0, 0.5};

  EXPECT_TRUE(valid_direct_command(command, 0.6));
  EXPECT_FALSE(valid_direct_command(command, 0.4));
}

TEST(LocalMotionValidation, RejectsNonfiniteDirectCommandFields) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  constexpr double steering_limit = 0.52;

  EXPECT_FALSE(
      valid_direct_command(PhysicalCommand{nan, 0.0, 0.0}, steering_limit));
  EXPECT_FALSE(
      valid_direct_command(PhysicalCommand{0.0, nan, 0.0}, steering_limit));
  EXPECT_FALSE(
      valid_direct_command(PhysicalCommand{0.0, 0.0, nan}, steering_limit));
}

TEST(LocalMotionValidation,
     RejectsDirectCommandRangeAndMutualExclusionViolations) {
  constexpr double steering_limit = 0.52;
  const std::vector<PhysicalCommand> invalid_commands{
      PhysicalCommand{-0.01, 0.0, 0.0},
      PhysicalCommand{1.01, 0.0, 0.0},
      PhysicalCommand{0.0, -0.01, 0.0},
      PhysicalCommand{0.0, 1.01, 0.0},
      PhysicalCommand{0.1, 0.1, 0.0},
      PhysicalCommand{0.0, 0.0, steering_limit + 0.01},
      PhysicalCommand{0.0, 0.0, -steering_limit - 0.01},
  };

  for (const auto &command : invalid_commands) {
    EXPECT_FALSE(valid_direct_command(command, steering_limit));
  }
}

} // namespace
} // namespace ad_planner
