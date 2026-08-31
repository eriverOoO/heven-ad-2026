#include "ad_planner/planning/highway_merge_speed_constraint.hpp"

#include <cmath>
#include <limits>
#include <optional>

#include <gtest/gtest.h>

namespace {

using ad_planner::highway_merge_response_merge_authorized;
using ad_planner::highway_merge_response_speed_limit;
using ad_planner::HighwayMergeResponseConstraintInput;

constexpr int kMergeReady = 0;
constexpr int kWait = 1;
constexpr int kHold = 2;
constexpr double kComfortableDecel = 1.8;

// A fresh, active, matching-zone frame. Callers override `action` and the WAIT
// facts. The WAIT facts describe a genuine WAIT: ego moving at 8 m/s with 34 m
// to the decision boundary, comfortable stop distance 8^2 / (2 * 1.8).
HighwayMergeResponseConstraintInput frame(int action) {
  HighwayMergeResponseConstraintInput input;
  input.received = true;
  input.fresh = true;
  input.active = true;
  input.zone_matches = true;
  input.action = action;
  input.ego_speed_mps = 8.0;
  input.available_distance_m = 34.0;
  input.comfortable_stop_distance_m = 8.0 * 8.0 / (2.0 * kComfortableDecel);
  return input;
}

double wait_cap(const HighwayMergeResponseConstraintInput &input) {
  return input.ego_speed_mps * std::sqrt(input.available_distance_m /
                                         input.comfortable_stop_distance_m);
}

// ---------------------------------------------------------------------------
// Speed limit: disabled / missing / stale / inactive / wrong zone
// ---------------------------------------------------------------------------

TEST(HighwayMergeSpeedLimit, NotReceivedIsNoConstraint) {
  HighwayMergeResponseConstraintInput input;
  input.received = false;
  EXPECT_FALSE(highway_merge_response_speed_limit(input).has_value());
}

TEST(HighwayMergeSpeedLimit, StaleFrameIsNoConstraintForEveryAction) {
  for (const int action : {kMergeReady, kWait, kHold}) {
    auto input = frame(action);
    input.fresh = false;
    EXPECT_FALSE(highway_merge_response_speed_limit(input).has_value())
        << "action " << action;
  }
}

TEST(HighwayMergeSpeedLimit, InactiveFrameIsNoConstraintForEveryAction) {
  for (const int action : {kMergeReady, kWait, kHold}) {
    auto input = frame(action);
    input.active = false;
    EXPECT_FALSE(highway_merge_response_speed_limit(input).has_value())
        << "action " << action;
  }
}

TEST(HighwayMergeSpeedLimit, WrongZoneIsNoConstraintEvenForHold) {
  auto input = frame(kHold);
  input.zone_matches = false;
  EXPECT_FALSE(highway_merge_response_speed_limit(input).has_value());
}

// ---------------------------------------------------------------------------
// Speed limit: action mapping
// ---------------------------------------------------------------------------

TEST(HighwayMergeSpeedLimit, MergeReadyImposesNoCap) {
  EXPECT_FALSE(highway_merge_response_speed_limit(frame(kMergeReady)).has_value());
}

TEST(HighwayMergeSpeedLimit, HoldIsExactlyZero) {
  const auto limit = highway_merge_response_speed_limit(frame(kHold));
  ASSERT_TRUE(limit.has_value());
  EXPECT_DOUBLE_EQ(*limit, 0.0);
}

TEST(HighwayMergeSpeedLimit, WaitIsTheComfortableStopEnvelope) {
  const auto input = frame(kWait);
  const auto limit = highway_merge_response_speed_limit(input);
  ASSERT_TRUE(limit.has_value());
  EXPECT_NEAR(*limit, wait_cap(input), 1e-9);
  // Algebraically equal to sqrt(2 * a_comfortable * available) - the ego speed
  // terms cancel, so no deceleration constant is duplicated in the planner.
  EXPECT_NEAR(*limit, std::sqrt(2.0 * kComfortableDecel * input.available_distance_m),
              1e-9);
}

TEST(HighwayMergeSpeedLimit, UnknownActionIsNoConstraint) {
  for (const int action : {-1, 3, 7, 255}) {
    EXPECT_FALSE(highway_merge_response_speed_limit(frame(action)).has_value())
        << "action " << action;
  }
}

// ---------------------------------------------------------------------------
// Speed limit: malformed WAIT facts
// ---------------------------------------------------------------------------

TEST(HighwayMergeSpeedLimit, WaitWithNonFiniteFactsIsNoConstraint) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {nan, inf, -inf}) {
    auto ego = frame(kWait);
    ego.ego_speed_mps = bad;
    EXPECT_FALSE(highway_merge_response_speed_limit(ego).has_value());
    auto avail = frame(kWait);
    avail.available_distance_m = bad;
    EXPECT_FALSE(highway_merge_response_speed_limit(avail).has_value());
    auto stop = frame(kWait);
    stop.comfortable_stop_distance_m = bad;
    EXPECT_FALSE(highway_merge_response_speed_limit(stop).has_value());
  }
}

TEST(HighwayMergeSpeedLimit, WaitWithNegativeEgoSpeedIsNoConstraint) {
  auto input = frame(kWait);
  input.ego_speed_mps = -1.0;
  EXPECT_FALSE(highway_merge_response_speed_limit(input).has_value());
}

TEST(HighwayMergeSpeedLimit, WaitWithNonPositiveDistancesIsNoConstraint) {
  auto zero_available = frame(kWait);
  zero_available.available_distance_m = 0.0;
  EXPECT_FALSE(highway_merge_response_speed_limit(zero_available).has_value());
  auto negative_available = frame(kWait);
  negative_available.available_distance_m = -5.0;
  EXPECT_FALSE(highway_merge_response_speed_limit(negative_available).has_value());
  auto zero_stop = frame(kWait);
  zero_stop.comfortable_stop_distance_m = 0.0;
  EXPECT_FALSE(highway_merge_response_speed_limit(zero_stop).has_value());
}

// ---------------------------------------------------------------------------
// Speed limit: value properties
// ---------------------------------------------------------------------------

TEST(HighwayMergeSpeedLimit, WaitCapIsNonNegativeAndFiniteAcrossASweep) {
  for (double ego = 0.0; ego <= 30.0; ego += 2.5) {
    for (double available = 1.0; available <= 200.0; available += 7.0) {
      auto input = frame(kWait);
      input.ego_speed_mps = ego;
      input.available_distance_m = available;
      input.comfortable_stop_distance_m =
          std::max(0.1, ego * ego / (2.0 * kComfortableDecel));
      const auto limit = highway_merge_response_speed_limit(input);
      if (limit) {
        EXPECT_TRUE(std::isfinite(*limit));
        EXPECT_GE(*limit, 0.0);
      }
    }
  }
}

TEST(HighwayMergeSpeedLimit, WaitCapShrinksAsTheEgoApproaches) {
  auto far = frame(kWait);
  far.available_distance_m = 60.0;
  auto near = frame(kWait);
  near.available_distance_m = 20.0;
  const auto far_cap = highway_merge_response_speed_limit(far);
  const auto near_cap = highway_merge_response_speed_limit(near);
  ASSERT_TRUE(far_cap && near_cap);
  EXPECT_GT(*far_cap, *near_cap);
}

TEST(HighwayMergeSpeedLimit, WaitCapCanExceedNominal) {
  // available 194 m, comfortable stop 17.78 m -> cap ~= 26.4 m/s, above a
  // 16.25 m/s cruise target. The pure core still returns it; the planner
  // clamps it against the nominal and treats it as a no-op (test #17).
  auto input = frame(kWait);
  input.available_distance_m = 194.0;
  const auto limit = highway_merge_response_speed_limit(input);
  ASSERT_TRUE(limit.has_value());
  EXPECT_GT(*limit, 16.25);
}

TEST(HighwayMergeSpeedLimit, DeterministicForRepeatedInput) {
  const auto input = frame(kWait);
  const auto a = highway_merge_response_speed_limit(input);
  const auto b = highway_merge_response_speed_limit(input);
  ASSERT_TRUE(a && b);
  EXPECT_DOUBLE_EQ(*a, *b);
}

// ---------------------------------------------------------------------------
// Merge authorization
// ---------------------------------------------------------------------------

TEST(HighwayMergeAuthorization, FreshActiveMatchingZoneMergeReadyAuthorizes) {
  EXPECT_TRUE(highway_merge_response_merge_authorized(frame(kMergeReady)));
}

TEST(HighwayMergeAuthorization, InactiveMergeReadyEnumDoesNotAuthorize) {
  // The enum-0 trap: an inactive frame carries ACTION_MERGE_READY = 0.
  auto input = frame(kMergeReady);
  input.active = false;
  EXPECT_FALSE(highway_merge_response_merge_authorized(input));
  EXPECT_FALSE(highway_merge_response_speed_limit(input).has_value());
}

TEST(HighwayMergeAuthorization, StaleMergeReadyDoesNotAuthorize) {
  auto input = frame(kMergeReady);
  input.fresh = false;
  EXPECT_FALSE(highway_merge_response_merge_authorized(input));
}

TEST(HighwayMergeAuthorization, NotReceivedDoesNotAuthorize) {
  HighwayMergeResponseConstraintInput input;
  input.received = false;
  input.active = true;
  input.zone_matches = true;
  input.action = kMergeReady;
  EXPECT_FALSE(highway_merge_response_merge_authorized(input));
}

TEST(HighwayMergeAuthorization, WrongZoneMergeReadyDoesNotAuthorize) {
  auto input = frame(kMergeReady);
  input.zone_matches = false;
  EXPECT_FALSE(highway_merge_response_merge_authorized(input));
}

TEST(HighwayMergeAuthorization, WaitDoesNotAuthorizeRegardlessOfFacts) {
  // Even a WAIT frame with generous gap facts is unauthorized: the response
  // action is the canonical policy output and is never re-decided here.
  auto input = frame(kWait);
  input.available_distance_m = 500.0;
  EXPECT_FALSE(highway_merge_response_merge_authorized(input));
}

TEST(HighwayMergeAuthorization, HoldDoesNotAuthorize) {
  EXPECT_FALSE(highway_merge_response_merge_authorized(frame(kHold)));
}

TEST(HighwayMergeAuthorization, UnknownActionDoesNotAuthorize) {
  for (const int action : {-1, 3, 99}) {
    EXPECT_FALSE(highway_merge_response_merge_authorized(frame(action)));
  }
}

TEST(HighwayMergeAuthorization, ReadyThenWaitRevokesImmediately) {
  EXPECT_TRUE(highway_merge_response_merge_authorized(frame(kMergeReady)));
  EXPECT_FALSE(highway_merge_response_merge_authorized(frame(kWait)));
}

TEST(HighwayMergeAuthorization, ReadyThenStaleRevokesImmediately) {
  EXPECT_TRUE(highway_merge_response_merge_authorized(frame(kMergeReady)));
  auto stale = frame(kMergeReady);
  stale.fresh = false;
  EXPECT_FALSE(highway_merge_response_merge_authorized(stale));
}

TEST(HighwayMergeAuthorization, DeterministicForRepeatedInput) {
  const auto input = frame(kMergeReady);
  EXPECT_EQ(highway_merge_response_merge_authorized(input),
            highway_merge_response_merge_authorized(input));
}

} // namespace
