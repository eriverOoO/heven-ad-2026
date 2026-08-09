#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include "ad_planner/local_planning/common/local_motion_timing.hpp"

namespace ad_planner {
namespace {

TEST(LocalMotionTiming, AcceptsInclusiveAgeAndSkewBoundaries) {
  const auto result = validate_local_motion_timing(
      1'000'000'000LL, 500'000'000LL, 600'000'000LL,
      LocalMotionTimingLimits{0.50, 0.40, 0.10});

  EXPECT_TRUE(result.valid);
  EXPECT_TRUE(result.reason.empty());
}

TEST(LocalMotionTiming, AcceptsExactDecimalInclusiveAgeBoundary) {
  const auto result = validate_local_motion_timing(
      1'000'000'000LL, 700'000'000LL, 700'000'000LL,
      LocalMotionTimingLimits{0.3, 0.3, 0.3});

  EXPECT_TRUE(result.valid);
  EXPECT_TRUE(result.reason.empty());
}

TEST(LocalMotionTiming, RejectsPositiveLimitThatRoundsToZeroNanoseconds) {
  const auto result = validate_local_motion_timing(
      1'000'000'000LL, 1'000'000'000LL, 1'000'000'000LL,
      LocalMotionTimingLimits{0.4e-9, 0.50, 0.10});

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason,
            "timing limits must be finite, positive, and representable");
}

TEST(LocalMotionTiming, RejectsNonpositiveNonfiniteAndOverflowingLimits) {
  const auto expect_invalid_limits = [](const LocalMotionTimingLimits &limits) {
    const auto result = validate_local_motion_timing(
        1'000'000'000LL, 900'000'000LL, 900'000'000LL, limits);
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason,
              "timing limits must be finite, positive, and representable");
  };

  expect_invalid_limits(LocalMotionTimingLimits{0.0, 0.50, 0.10});
  expect_invalid_limits(LocalMotionTimingLimits{-0.50, 0.50, 0.10});
  expect_invalid_limits(LocalMotionTimingLimits{
      std::numeric_limits<double>::infinity(), 0.50, 0.10});
  expect_invalid_limits(LocalMotionTimingLimits{
      0.50, std::numeric_limits<double>::quiet_NaN(), 0.10});
  expect_invalid_limits(LocalMotionTimingLimits{0.50, 0.50, 1.0e20});
}

TEST(LocalMotionTiming, RejectsInvalidOdometryStamp) {
  for (const std::int64_t stamp : {0LL, -1LL}) {
    const auto result =
        validate_local_motion_timing(1'000'000'000LL, stamp, 900'000'000LL, {});
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason, "odometry stamp must be positive");
  }
}

TEST(LocalMotionTiming, RejectsInvalidGridStamp) {
  for (const std::int64_t stamp : {0LL, -1LL}) {
    const auto result =
        validate_local_motion_timing(1'000'000'000LL, 900'000'000LL, stamp, {});
    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason, "grid stamp must be positive");
  }
}

TEST(LocalMotionTiming, RejectsFutureOdometryWithoutOverflow) {
  const auto result = validate_local_motion_timing(
      std::numeric_limits<std::int64_t>::min(),
      std::numeric_limits<std::int64_t>::max(), 1LL, {});

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "odometry stamp is in the future");
}

TEST(LocalMotionTiming, RejectsFutureGridWithoutOverflow) {
  const auto result = validate_local_motion_timing(
      1'000'000'000LL, 900'000'000LL, 1'000'000'001LL, {});

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "grid stamp is in the future");
}

TEST(LocalMotionTiming, RejectsStaleOdometryAtOneNanosecondPastLimit) {
  const auto result = validate_local_motion_timing(
      1'000'000'001LL, 500'000'000LL, 600'000'001LL,
      LocalMotionTimingLimits{0.50, 0.50, 0.10});

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "odometry stamp is stale");
}

TEST(LocalMotionTiming, RejectsStaleGridAtOneNanosecondPastLimit) {
  const auto result = validate_local_motion_timing(
      1'000'000'001LL, 600'000'001LL, 500'000'000LL,
      LocalMotionTimingLimits{0.50, 0.50, 0.11});

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "grid stamp is stale");
}

TEST(LocalMotionTiming, RejectsExcessiveSkewAtOneNanosecondPastLimit) {
  const auto result = validate_local_motion_timing(
      1'000'000'000LL, 900'000'000LL, 799'999'999LL,
      LocalMotionTimingLimits{0.50, 0.50, 0.10});

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "odometry and grid stamps exceed maximum skew");
}

} // namespace
} // namespace ad_planner
