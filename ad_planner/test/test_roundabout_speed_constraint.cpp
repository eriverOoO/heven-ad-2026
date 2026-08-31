#include "ad_planner/planning/roundabout_speed_constraint.hpp"

#include <cmath>
#include <limits>
#include <optional>

#include <gtest/gtest.h>

namespace {

using ad_planner::RoundaboutResponseConstraintInput;
using ad_planner::roundabout_response_speed_limit;

constexpr int kRelease = 0;
constexpr int kYield = 1;
constexpr int kHold = 2;

// A fresh, active YIELD frame: ego 8 m/s, ~40 m to entry (available 34 m after
// the 6 m standoff), comfortable stop distance 8^2 / (2 * 1.8) ~= 17.78 m.
RoundaboutResponseConstraintInput fresh_yield(double ego_speed = 8.0,
                                              double available = 34.0,
                                              double comfortable_stop = 17.7778) {
  RoundaboutResponseConstraintInput input;
  input.received = true;
  input.fresh = true;
  input.active = true;
  input.action = kYield;
  input.ego_speed_mps = ego_speed;
  input.available_distance_m = available;
  input.comfortable_stop_distance_m = comfortable_stop;
  return input;
}

RoundaboutResponseConstraintInput fresh_hold() {
  RoundaboutResponseConstraintInput input;
  input.received = true;
  input.fresh = true;
  input.active = true;
  input.action = kHold;
  input.ego_speed_mps = 8.0;
  input.available_distance_m = 8.1;
  input.comfortable_stop_distance_m = 17.7778;
  return input;
}

RoundaboutResponseConstraintInput fresh_release() {
  RoundaboutResponseConstraintInput input;
  input.received = true;
  input.fresh = true;
  input.active = true;
  input.action = kRelease;
  input.ego_speed_mps = 8.0;
  return input;
}

TEST(RoundaboutSpeedConstraint, NoMessageIsNoConstraint) {
  RoundaboutResponseConstraintInput input;
  EXPECT_FALSE(roundabout_response_speed_limit(input).has_value());
}

TEST(RoundaboutSpeedConstraint, StaleYieldIsNoConstraint) {
  auto input = fresh_yield();
  input.fresh = false;
  EXPECT_FALSE(roundabout_response_speed_limit(input).has_value());
}

TEST(RoundaboutSpeedConstraint, StaleHoldIsNoConstraint) {
  auto input = fresh_hold();
  input.fresh = false;
  EXPECT_FALSE(roundabout_response_speed_limit(input).has_value());
}

TEST(RoundaboutSpeedConstraint, StaleReleaseIsNoConstraint) {
  auto input = fresh_release();
  input.fresh = false;
  EXPECT_FALSE(roundabout_response_speed_limit(input).has_value());
}

TEST(RoundaboutSpeedConstraint, InactiveResponseIsNoConstraint) {
  for (const int action : {kRelease, kYield, kHold}) {
    auto input = fresh_yield();
    input.action = action;
    input.active = false;
    EXPECT_FALSE(roundabout_response_speed_limit(input).has_value())
        << "action " << action;
  }
}

TEST(RoundaboutSpeedConstraint, ReleaseIsNoConstraint) {
  EXPECT_FALSE(roundabout_response_speed_limit(fresh_release()).has_value());
}

TEST(RoundaboutSpeedConstraint, HoldReturnsExactlyZero) {
  const auto limit = roundabout_response_speed_limit(fresh_hold());
  ASSERT_TRUE(limit.has_value());
  EXPECT_DOUBLE_EQ(*limit, 0.0);
}

TEST(RoundaboutSpeedConstraint, YieldCapEqualsComfortableStopEnvelope) {
  // ego 8 m/s, available 34 m, comfortable stop 17.7778 m ->
  // 8 * sqrt(34 / 17.7778) = sqrt(2 * 1.8 * 34) = 11.0635...
  const auto limit = roundabout_response_speed_limit(fresh_yield());
  ASSERT_TRUE(limit.has_value());
  EXPECT_NEAR(*limit, std::sqrt(2.0 * 1.8 * 34.0), 1e-3);
  EXPECT_NEAR(*limit, 11.0635, 1e-3);
}

TEST(RoundaboutSpeedConstraint, YieldCapIsIndependentOfEgoSpeedGivenTheSameEnvelope) {
  // comfortable_stop scales with ego_speed^2, so the cap depends only on the
  // implied deceleration and the available distance.
  const auto slow = roundabout_response_speed_limit(
      fresh_yield(4.0, 34.0, 4.0 * 4.0 / (2.0 * 1.8)));
  const auto fast = roundabout_response_speed_limit(
      fresh_yield(12.0, 34.0, 12.0 * 12.0 / (2.0 * 1.8)));
  ASSERT_TRUE(slow.has_value());
  ASSERT_TRUE(fast.has_value());
  EXPECT_NEAR(*slow, *fast, 1e-6);
  EXPECT_NEAR(*slow, std::sqrt(2.0 * 1.8 * 34.0), 1e-3);
}

TEST(RoundaboutSpeedConstraint, YieldCapShrinksAsAvailableDistanceShrinks) {
  const double v = 8.0;
  const double a = 1.8;
  const auto far = roundabout_response_speed_limit(
      fresh_yield(v, 60.0, v * v / (2.0 * a)));
  const auto near = roundabout_response_speed_limit(
      fresh_yield(v, 25.0, v * v / (2.0 * a)));
  ASSERT_TRUE(far.has_value());
  ASSERT_TRUE(near.has_value());
  EXPECT_LT(*near, *far);
}

TEST(RoundaboutSpeedConstraint, YieldCapIsMonotonicNonDecreasingInAvailableDistance) {
  const double v = 7.5;
  const double comfortable_stop = v * v / (2.0 * 1.8);
  std::optional<double> previous;
  for (const double available : {5.0, 12.0, 20.0, 33.0, 50.0, 90.0}) {
    const auto limit = roundabout_response_speed_limit(
        fresh_yield(v, available, comfortable_stop));
    ASSERT_TRUE(limit.has_value());
    if (previous) {
      EXPECT_GE(*limit, *previous);
    }
    previous = limit;
  }
}

TEST(RoundaboutSpeedConstraint, HoldIsNeverWeakerThanYield) {
  const auto hold = roundabout_response_speed_limit(fresh_hold());
  const auto yield = roundabout_response_speed_limit(fresh_yield());
  ASSERT_TRUE(hold.has_value());
  ASSERT_TRUE(yield.has_value());
  EXPECT_LE(*hold, *yield);
  EXPECT_DOUBLE_EQ(*hold, 0.0);
}

TEST(RoundaboutSpeedConstraint, YieldWithZeroAvailableDistanceIsNoConstraint) {
  auto input = fresh_yield();
  input.available_distance_m = 0.0;
  EXPECT_FALSE(roundabout_response_speed_limit(input).has_value());
}

TEST(RoundaboutSpeedConstraint, YieldWithNegativeAvailableDistanceIsNoConstraint) {
  auto input = fresh_yield();
  input.available_distance_m = -3.0;
  EXPECT_FALSE(roundabout_response_speed_limit(input).has_value());
}

TEST(RoundaboutSpeedConstraint, YieldWithZeroComfortableStopIsNoConstraint) {
  auto input = fresh_yield();
  input.comfortable_stop_distance_m = 0.0;
  EXPECT_FALSE(roundabout_response_speed_limit(input).has_value());
}

TEST(RoundaboutSpeedConstraint, YieldWithNaNFactsIsNoConstraint) {
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  auto ego = fresh_yield();
  ego.ego_speed_mps = nan;
  EXPECT_FALSE(roundabout_response_speed_limit(ego).has_value());
  auto avail = fresh_yield();
  avail.available_distance_m = nan;
  EXPECT_FALSE(roundabout_response_speed_limit(avail).has_value());
  auto stop = fresh_yield();
  stop.comfortable_stop_distance_m = nan;
  EXPECT_FALSE(roundabout_response_speed_limit(stop).has_value());
}

TEST(RoundaboutSpeedConstraint, YieldWithInfFactsIsNoConstraint) {
  const auto inf = std::numeric_limits<double>::infinity();
  auto avail = fresh_yield();
  avail.available_distance_m = inf;
  const auto limit = roundabout_response_speed_limit(avail);
  // available_distance_m is finite-checked; an infinite value is rejected.
  EXPECT_FALSE(limit.has_value());
}

TEST(RoundaboutSpeedConstraint, YieldWithNegativeEgoSpeedIsNoConstraint) {
  auto input = fresh_yield();
  input.ego_speed_mps = -1.0;
  EXPECT_FALSE(roundabout_response_speed_limit(input).has_value());
}

TEST(RoundaboutSpeedConstraint, UnknownActionIsNoConstraint) {
  auto input = fresh_yield();
  input.action = 7;
  EXPECT_FALSE(roundabout_response_speed_limit(input).has_value());
}

TEST(RoundaboutSpeedConstraint, NegativeActionIsNoConstraint) {
  auto input = fresh_yield();
  input.action = -1;
  EXPECT_FALSE(roundabout_response_speed_limit(input).has_value());
}

TEST(RoundaboutSpeedConstraint, ResultIsNeverNegativeOrNonFinite) {
  for (const double ego : {0.0, 0.5, 4.0, 8.3, 16.0, 30.0}) {
    for (const double available : {-1.0, 0.0, 1.0, 17.0, 40.0, 200.0}) {
      for (const double stop : {-1.0, 0.0, 0.5, 17.78, 80.0}) {
        for (const int action : {kRelease, kYield, kHold, 9}) {
          auto input = fresh_yield(ego, available, stop);
          input.action = action;
          const auto limit = roundabout_response_speed_limit(input);
          if (limit) {
            EXPECT_GE(*limit, 0.0);
            EXPECT_TRUE(std::isfinite(*limit));
          }
        }
      }
    }
  }
}

TEST(RoundaboutSpeedConstraint, DeterministicForRepeatedInput) {
  const auto input = fresh_yield();
  const auto first = roundabout_response_speed_limit(input);
  const auto second = roundabout_response_speed_limit(input);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  EXPECT_DOUBLE_EQ(*first, *second);
}

} // namespace
