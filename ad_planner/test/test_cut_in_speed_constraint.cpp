#include "ad_planner/planning/cut_in_speed_constraint.hpp"

#include <cmath>
#include <limits>
#include <optional>

#include <gtest/gtest.h>

namespace {

using ad_planner::CutInResponseConstraintInput;
using ad_planner::cut_in_response_speed_limit;

constexpr int kNone = 0;
constexpr int kSlowdown = 1;
constexpr int kHold = 2;

// A fresh, active, well-formed SLOWDOWN asking for 6.0 m/s.
CutInResponseConstraintInput fresh_slowdown(double speed = 6.0) {
  CutInResponseConstraintInput input;
  input.received = true;
  input.fresh = true;
  input.active = true;
  input.action = kSlowdown;
  input.requested_max_speed_valid = true;
  input.requested_max_speed_mps = speed;
  return input;
}

CutInResponseConstraintInput fresh_hold() {
  CutInResponseConstraintInput input;
  input.received = true;
  input.fresh = true;
  input.active = true;
  input.action = kHold;
  input.requested_max_speed_valid = true;
  input.requested_max_speed_mps = 0.0;
  return input;
}

TEST(CutInSpeedConstraint, NoMessageIsNoConstraint) {
  CutInResponseConstraintInput input;
  EXPECT_FALSE(cut_in_response_speed_limit(input).has_value());
}

TEST(CutInSpeedConstraint, StaleMessageIsNoConstraint) {
  auto input = fresh_slowdown();
  input.fresh = false;
  EXPECT_FALSE(cut_in_response_speed_limit(input).has_value());
}

TEST(CutInSpeedConstraint, StaleHoldIsNoConstraint) {
  auto input = fresh_hold();
  input.fresh = false;
  EXPECT_FALSE(cut_in_response_speed_limit(input).has_value());
}

TEST(CutInSpeedConstraint, InactiveMessageIsNoConstraint) {
  auto input = fresh_slowdown();
  input.active = false;
  EXPECT_FALSE(cut_in_response_speed_limit(input).has_value());
}

TEST(CutInSpeedConstraint, ActionNoneIsNoConstraint) {
  auto input = fresh_slowdown();
  input.action = kNone;
  input.requested_max_speed_valid = false;
  input.requested_max_speed_mps = 0.0;
  EXPECT_FALSE(cut_in_response_speed_limit(input).has_value());
}

TEST(CutInSpeedConstraint, SlowdownReturnsRequestedUpperBound) {
  const auto limit = cut_in_response_speed_limit(fresh_slowdown(6.57));
  ASSERT_TRUE(limit.has_value());
  EXPECT_DOUBLE_EQ(*limit, 6.57);
}

TEST(CutInSpeedConstraint, SlowdownWithInvalidFlagIsNoConstraint) {
  auto input = fresh_slowdown(6.0);
  input.requested_max_speed_valid = false;
  EXPECT_FALSE(cut_in_response_speed_limit(input).has_value());
}

TEST(CutInSpeedConstraint, SlowdownWithZeroSpeedIsRejectedNotHold) {
  auto input = fresh_slowdown(0.0);
  EXPECT_FALSE(cut_in_response_speed_limit(input).has_value());
}

TEST(CutInSpeedConstraint, SlowdownWithNegativeSpeedIsNoConstraint) {
  EXPECT_FALSE(cut_in_response_speed_limit(fresh_slowdown(-3.0)).has_value());
}

TEST(CutInSpeedConstraint, SlowdownWithNaNIsNoConstraint) {
  EXPECT_FALSE(cut_in_response_speed_limit(
                   fresh_slowdown(std::numeric_limits<double>::quiet_NaN()))
                   .has_value());
}

TEST(CutInSpeedConstraint, SlowdownWithInfIsNoConstraint) {
  EXPECT_FALSE(cut_in_response_speed_limit(
                   fresh_slowdown(std::numeric_limits<double>::infinity()))
                   .has_value());
}

TEST(CutInSpeedConstraint, HoldReturnsExactlyZero) {
  const auto limit = cut_in_response_speed_limit(fresh_hold());
  ASSERT_TRUE(limit.has_value());
  EXPECT_DOUBLE_EQ(*limit, 0.0);
}

TEST(CutInSpeedConstraint, HoldWithNonZeroSpeedIsNoConstraint) {
  auto input = fresh_hold();
  input.requested_max_speed_mps = 2.0;
  EXPECT_FALSE(cut_in_response_speed_limit(input).has_value());
}

TEST(CutInSpeedConstraint, HoldWithInvalidFlagIsNoConstraint) {
  auto input = fresh_hold();
  input.requested_max_speed_valid = false;
  EXPECT_FALSE(cut_in_response_speed_limit(input).has_value());
}

TEST(CutInSpeedConstraint, HoldWithNaNIsNoConstraint) {
  auto input = fresh_hold();
  input.requested_max_speed_mps = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(cut_in_response_speed_limit(input).has_value());
}

TEST(CutInSpeedConstraint, UnknownActionIsNoConstraint) {
  auto input = fresh_slowdown(6.0);
  input.action = 7;
  EXPECT_FALSE(cut_in_response_speed_limit(input).has_value());
}

TEST(CutInSpeedConstraint, NegativeActionIsNoConstraint) {
  auto input = fresh_slowdown(6.0);
  input.action = -1;
  EXPECT_FALSE(cut_in_response_speed_limit(input).has_value());
}

TEST(CutInSpeedConstraint, ResultIsNeverNegativeOrNonFinite) {
  for (double requested : {-10.0, -0.1, 0.0, 0.1, 1.0, 8.3, 25.0, 1e9}) {
    for (int action : {kNone, kSlowdown, kHold, 5}) {
      auto input = fresh_slowdown(requested);
      input.action = action;
      const auto limit = cut_in_response_speed_limit(input);
      if (limit) {
        EXPECT_GE(*limit, 0.0);
        EXPECT_TRUE(std::isfinite(*limit));
      }
    }
  }
}

TEST(CutInSpeedConstraint, SlowdownLimitIsMonotonicInRequestedSpeed) {
  const auto low = cut_in_response_speed_limit(fresh_slowdown(4.0));
  const auto high = cut_in_response_speed_limit(fresh_slowdown(9.0));
  ASSERT_TRUE(low.has_value());
  ASSERT_TRUE(high.has_value());
  EXPECT_LE(*low, *high);
}

TEST(CutInSpeedConstraint, SlowdownLimitEqualsRequestNeverExceedsIt) {
  const auto limit = cut_in_response_speed_limit(fresh_slowdown(12.34));
  ASSERT_TRUE(limit.has_value());
  EXPECT_LE(*limit, 12.34);
}

TEST(CutInSpeedConstraint, HoldIgnoresUnreadDiagnosticFields) {
  // The helper only reads action / active / freshness / requested_max_speed_*.
  auto input = fresh_hold();
  input.action = kHold;
  const auto limit = cut_in_response_speed_limit(input);
  ASSERT_TRUE(limit.has_value());
  EXPECT_DOUBLE_EQ(*limit, 0.0);
}

} // namespace
