#include "ad_planner/planning/external_speed_limit.hpp"

#include <cmath>
#include <limits>
#include <optional>

#include <gtest/gtest.h>

#include "ad_planner/planning/cut_in_speed_constraint.hpp"
#include "ad_planner/planning/highway_merge_speed_constraint.hpp"
#include "ad_planner/planning/roundabout_speed_constraint.hpp"

namespace {

using ad_planner::combine_speed_limits;
using ad_planner::CutInResponseConstraintInput;
using ad_planner::cut_in_response_speed_limit;
using ad_planner::highway_merge_response_speed_limit;
using ad_planner::HighwayMergeResponseConstraintInput;
using ad_planner::RoundaboutResponseConstraintInput;
using ad_planner::roundabout_response_speed_limit;

TEST(ExternalSpeedLimit, BothAbsentIsNoConstraint) {
  EXPECT_FALSE(combine_speed_limits(std::nullopt, std::nullopt).has_value());
}

TEST(ExternalSpeedLimit, SinglePresentValuePassesThrough) {
  const auto a = combine_speed_limits(4.0, std::nullopt);
  ASSERT_TRUE(a.has_value());
  EXPECT_DOUBLE_EQ(*a, 4.0);
  const auto b = combine_speed_limits(std::nullopt, 7.5);
  ASSERT_TRUE(b.has_value());
  EXPECT_DOUBLE_EQ(*b, 7.5);
}

TEST(ExternalSpeedLimit, MostRestrictiveWins) {
  const auto result = combine_speed_limits(6.0, 4.0);
  ASSERT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(*result, 4.0);
}

TEST(ExternalSpeedLimit, OrderDoesNotMatter) {
  for (const auto a : {std::optional<double>{}, std::optional<double>{0.0},
                       std::optional<double>{3.0}, std::optional<double>{9.0}}) {
    for (const auto b : {std::optional<double>{}, std::optional<double>{0.0},
                         std::optional<double>{2.0}, std::optional<double>{9.0}}) {
      const auto forward = combine_speed_limits(a, b);
      const auto reverse = combine_speed_limits(b, a);
      EXPECT_EQ(forward.has_value(), reverse.has_value());
      if (forward && reverse) {
        EXPECT_DOUBLE_EQ(*forward, *reverse);
      }
    }
  }
}

TEST(ExternalSpeedLimit, ZeroFromEitherSourceIsHonoured) {
  const auto a = combine_speed_limits(0.0, 5.0);
  ASSERT_TRUE(a.has_value());
  EXPECT_DOUBLE_EQ(*a, 0.0);
  const auto b = combine_speed_limits(5.0, 0.0);
  ASSERT_TRUE(b.has_value());
  EXPECT_DOUBLE_EQ(*b, 0.0);
}

TEST(ExternalSpeedLimit, NonFiniteInputIsIgnored) {
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto result = combine_speed_limits(nan, 4.0);
  ASSERT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(*result, 4.0);
  EXPECT_FALSE(combine_speed_limits(nan, std::nullopt).has_value());
}

TEST(ExternalSpeedLimit, EqualInputsReturnThatValue) {
  const auto result = combine_speed_limits(5.5, 5.5);
  ASSERT_TRUE(result.has_value());
  EXPECT_DOUBLE_EQ(*result, 5.5);
}

TEST(ExternalSpeedLimit, InitializerListFoldIsMostRestrictiveAndOrderFree) {
  EXPECT_FALSE(combine_speed_limits({}).has_value());
  EXPECT_FALSE(
      combine_speed_limits({std::nullopt, std::nullopt, std::nullopt}).has_value());
  const auto a = combine_speed_limits({7.0, std::nullopt, 4.0});
  ASSERT_TRUE(a.has_value());
  EXPECT_DOUBLE_EQ(*a, 4.0);
  const auto b = combine_speed_limits({4.0, 7.0, std::nullopt});
  ASSERT_TRUE(b.has_value());
  EXPECT_DOUBLE_EQ(*b, 4.0);
  const auto zero = combine_speed_limits({6.0, 0.0, 3.0});
  ASSERT_TRUE(zero.has_value());
  EXPECT_DOUBLE_EQ(*zero, 0.0);
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto ignored_nan = combine_speed_limits({nan, 5.0, std::nullopt});
  ASSERT_TRUE(ignored_nan.has_value());
  EXPECT_DOUBLE_EQ(*ignored_nan, 5.0);
}

// Phase 13: cut-in and roundabout are independent longitudinal upper bounds
// combined by min through the same helper the planner uses. Neither source can
// cancel a more restrictive one, and the order of composition never matters.
namespace {

CutInResponseConstraintInput cut_in_slowdown(double speed) {
  CutInResponseConstraintInput input;
  input.received = true;
  input.fresh = true;
  input.active = true;
  input.action = 1; // SLOWDOWN
  input.requested_max_speed_valid = true;
  input.requested_max_speed_mps = speed;
  return input;
}

CutInResponseConstraintInput cut_in_hold() {
  CutInResponseConstraintInput input;
  input.received = true;
  input.fresh = true;
  input.active = true;
  input.action = 2; // HOLD
  input.requested_max_speed_valid = true;
  input.requested_max_speed_mps = 0.0;
  return input;
}

CutInResponseConstraintInput cut_in_none() {
  CutInResponseConstraintInput input;
  input.received = true;
  input.fresh = true;
  input.active = false;
  input.action = 0;
  return input;
}

RoundaboutResponseConstraintInput roundabout(int action, double ego = 8.0,
                                             double available = 34.0,
                                             double stop = 17.7778) {
  RoundaboutResponseConstraintInput input;
  input.received = true;
  input.fresh = true;
  input.active = true;
  input.action = action;
  input.ego_speed_mps = ego;
  input.available_distance_m = available;
  input.comfortable_stop_distance_m = stop;
  return input;
}

std::optional<double> planner_bound(double nominal,
                                    std::optional<double> cut_in,
                                    std::optional<double> roundabout_limit) {
  // Mirrors AdPlannerNode: each source is first clamped against the nominal
  // cruise target (a value that does not lower it becomes nullopt), then the
  // two are combined by min.
  const auto clamp = [nominal](std::optional<double> raw) -> std::optional<double> {
    if (!raw) {
      return std::nullopt;
    }
    const double capped = std::min(nominal, *raw);
    return capped >= nominal ? std::nullopt : std::optional<double>{capped};
  };
  return combine_speed_limits(clamp(cut_in), clamp(roundabout_limit));
}

} // namespace

TEST(ExternalSpeedLimitComposition, CutInSlowdownWithRoundaboutRelease) {
  // A: nominal 10, cut-in 6, roundabout RELEASE -> 6.
  const auto bound = planner_bound(
      10.0, cut_in_response_speed_limit(cut_in_slowdown(6.0)),
      roundabout_response_speed_limit(roundabout(0)));
  ASSERT_TRUE(bound.has_value());
  EXPECT_DOUBLE_EQ(*bound, 6.0);
}

TEST(ExternalSpeedLimitComposition, CutInSlowdownBelowRoundaboutYield) {
  // B: nominal 10, cut-in 6, roundabout YIELD cap ~= 8.05 (ego 6, available 18,
  // stop 10) -> the smaller cut-in cap wins.
  const auto yield_input = roundabout(1, 6.0, 18.0, 6.0 * 6.0 / (2.0 * 1.8));
  const auto rb = roundabout_response_speed_limit(yield_input);
  ASSERT_TRUE(rb.has_value());
  EXPECT_GT(*rb, 6.0);
  const auto bound = planner_bound(
      10.0, cut_in_response_speed_limit(cut_in_slowdown(6.0)), rb);
  ASSERT_TRUE(bound.has_value());
  EXPECT_DOUBLE_EQ(*bound, 6.0);
}

TEST(ExternalSpeedLimitComposition, RoundaboutYieldWithoutCutIn) {
  // C: nominal 16.25, no cut-in, roundabout YIELD -> the YIELD cap wins.
  const auto rb = roundabout_response_speed_limit(roundabout(1));
  ASSERT_TRUE(rb.has_value());
  const auto bound = planner_bound(
      16.25, cut_in_response_speed_limit(cut_in_none()), rb);
  ASSERT_TRUE(bound.has_value());
  EXPECT_NEAR(*bound, std::sqrt(2.0 * 1.8 * 34.0), 1e-3);
}

TEST(ExternalSpeedLimitComposition, RoundaboutHoldBeatsCutInSlowdown) {
  // D: cut-in SLOWDOWN 4, roundabout HOLD 0 -> 0.
  const auto bound = planner_bound(
      10.0, cut_in_response_speed_limit(cut_in_slowdown(4.0)),
      roundabout_response_speed_limit(roundabout(2)));
  ASSERT_TRUE(bound.has_value());
  EXPECT_DOUBLE_EQ(*bound, 0.0);
}

TEST(ExternalSpeedLimitComposition, CutInHoldSurvivesRoundaboutRelease) {
  // E: cut-in HOLD 0, roundabout RELEASE -> 0 (RELEASE never lifts another cap).
  const auto bound = planner_bound(
      10.0, cut_in_response_speed_limit(cut_in_hold()),
      roundabout_response_speed_limit(roundabout(0)));
  ASSERT_TRUE(bound.has_value());
  EXPECT_DOUBLE_EQ(*bound, 0.0);
}

TEST(ExternalSpeedLimitComposition, BothHoldIsZero) {
  const auto bound = planner_bound(
      12.0, cut_in_response_speed_limit(cut_in_hold()),
      roundabout_response_speed_limit(roundabout(2)));
  ASSERT_TRUE(bound.has_value());
  EXPECT_DOUBLE_EQ(*bound, 0.0);
}

TEST(ExternalSpeedLimitComposition, OrderIndependentForEveryStatePair) {
  const std::optional<double> cut_ins[] = {
      cut_in_response_speed_limit(cut_in_none()),
      cut_in_response_speed_limit(cut_in_slowdown(5.0)),
      cut_in_response_speed_limit(cut_in_hold())};
  const std::optional<double> roundabouts[] = {
      roundabout_response_speed_limit(roundabout(0)),
      roundabout_response_speed_limit(roundabout(1)),
      roundabout_response_speed_limit(roundabout(2))};
  for (const auto &a : cut_ins) {
    for (const auto &b : roundabouts) {
      const auto forward = planner_bound(16.25, a, b);
      const auto swapped = planner_bound(16.25, b, a);
      EXPECT_EQ(forward.has_value(), swapped.has_value());
      if (forward && swapped) {
        EXPECT_DOUBLE_EQ(*forward, *swapped);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 14-17: cut-in + roundabout + highway merge compose conservatively and
// order-independently through the same helper the planner uses.
// ---------------------------------------------------------------------------
namespace {

HighwayMergeResponseConstraintInput
merge(int action, double ego = 8.0, double available = 34.0,
      double stop = 8.0 * 8.0 / (2.0 * 1.8)) {
  HighwayMergeResponseConstraintInput input;
  input.received = true;
  input.fresh = true;
  input.active = true;
  input.zone_matches = true;
  input.action = action;
  input.ego_speed_mps = ego;
  input.available_distance_m = available;
  input.comfortable_stop_distance_m = stop;
  return input;
}

std::optional<double> planner_bound3(double nominal,
                                     std::optional<double> cut_in,
                                     std::optional<double> roundabout_limit,
                                     std::optional<double> merge_limit) {
  const auto clamp =
      [nominal](std::optional<double> raw) -> std::optional<double> {
    if (!raw) {
      return std::nullopt;
    }
    const double capped = std::min(nominal, *raw);
    return capped >= nominal ? std::nullopt : std::optional<double>{capped};
  };
  return combine_speed_limits(
      {clamp(cut_in), clamp(roundabout_limit), clamp(merge_limit)});
}

} // namespace

TEST(ExternalSpeedLimitThreeWay, A_MergeReadyLeavesNominalUntouched) {
  const auto bound = planner_bound3(
      10.0, cut_in_response_speed_limit(cut_in_none()),
      roundabout_response_speed_limit(roundabout(0)),
      highway_merge_response_speed_limit(merge(0)));
  EXPECT_FALSE(bound.has_value());
}

TEST(ExternalSpeedLimitThreeWay, B_CutInSurvivesMergeReady) {
  const auto bound = planner_bound3(
      10.0, cut_in_response_speed_limit(cut_in_slowdown(6.0)),
      roundabout_response_speed_limit(roundabout(0)),
      highway_merge_response_speed_limit(merge(0)));
  ASSERT_TRUE(bound.has_value());
  EXPECT_DOUBLE_EQ(*bound, 6.0);
}

TEST(ExternalSpeedLimitThreeWay, C_MergeWaitBeatsRoundaboutYield) {
  // roundabout YIELD cap ~7, merge WAIT cap ~5 -> 5.
  const auto rb_cap =
      roundabout_response_speed_limit(roundabout(1, 8.0, 13.611));
  ASSERT_TRUE(rb_cap.has_value());
  EXPECT_NEAR(*rb_cap, 7.0, 0.05);
  const auto mg_cap = highway_merge_response_speed_limit(
      merge(1, 8.0, 6.944));
  ASSERT_TRUE(mg_cap.has_value());
  EXPECT_NEAR(*mg_cap, 5.0, 0.05);
  const auto bound = planner_bound3(
      16.25, cut_in_response_speed_limit(cut_in_none()), rb_cap, mg_cap);
  ASSERT_TRUE(bound.has_value());
  EXPECT_NEAR(*bound, 5.0, 0.05);
}

TEST(ExternalSpeedLimitThreeWay, D_CutInBeatsMergeWaitWithRoundaboutRelease) {
  const auto bound = planner_bound3(
      16.25, cut_in_response_speed_limit(cut_in_slowdown(4.0)),
      roundabout_response_speed_limit(roundabout(0)),
      highway_merge_response_speed_limit(merge(1, 8.0, 10.0)));
  ASSERT_TRUE(bound.has_value());
  EXPECT_DOUBLE_EQ(*bound, 4.0);
}

TEST(ExternalSpeedLimitThreeWay, E_RoundaboutHoldBeatsMergeReady) {
  const auto bound = planner_bound3(
      10.0, cut_in_response_speed_limit(cut_in_none()),
      roundabout_response_speed_limit(roundabout(2)),
      highway_merge_response_speed_limit(merge(0)));
  ASSERT_TRUE(bound.has_value());
  EXPECT_DOUBLE_EQ(*bound, 0.0);
}

TEST(ExternalSpeedLimitThreeWay, F_MergeHoldBeatsCutInAndRoundabout) {
  const auto bound = planner_bound3(
      16.25, cut_in_response_speed_limit(cut_in_slowdown(4.0)),
      roundabout_response_speed_limit(roundabout(1)),
      highway_merge_response_speed_limit(merge(2)));
  ASSERT_TRUE(bound.has_value());
  EXPECT_DOUBLE_EQ(*bound, 0.0);
}

TEST(ExternalSpeedLimitThreeWay, G_AllThreeHoldIsZero) {
  const auto bound = planner_bound3(
      16.25, cut_in_response_speed_limit(cut_in_hold()),
      roundabout_response_speed_limit(roundabout(2)),
      highway_merge_response_speed_limit(merge(2)));
  ASSERT_TRUE(bound.has_value());
  EXPECT_DOUBLE_EQ(*bound, 0.0);
}

TEST(ExternalSpeedLimitThreeWay, AllThreeActiveMostRestrictiveWins) {
  // nominal 16.25, cut-in 8, roundabout 6, merge WAIT ~10 -> 6.
  const auto merge_wait =
      highway_merge_response_speed_limit(merge(1, 8.0, 27.7778));
  ASSERT_TRUE(merge_wait.has_value());
  EXPECT_NEAR(*merge_wait, 10.0, 0.05);
  const auto bound = planner_bound3(
      16.25, cut_in_response_speed_limit(cut_in_slowdown(8.0)),
      roundabout_response_speed_limit(roundabout(1, 8.0, 10.0)), merge_wait);
  ASSERT_TRUE(bound.has_value());
  EXPECT_NEAR(*bound, 6.0, 0.05);
}

TEST(ExternalSpeedLimitThreeWay, OrderIndependentForEveryStateTriple) {
  const std::optional<double> cut_ins[] = {
      cut_in_response_speed_limit(cut_in_none()),
      cut_in_response_speed_limit(cut_in_slowdown(7.0)),
      cut_in_response_speed_limit(cut_in_hold())};
  const std::optional<double> roundabouts[] = {
      roundabout_response_speed_limit(roundabout(0)),
      roundabout_response_speed_limit(roundabout(1)),
      roundabout_response_speed_limit(roundabout(2))};
  const std::optional<double> merges[] = {
      highway_merge_response_speed_limit(merge(0)),
      highway_merge_response_speed_limit(merge(1)),
      highway_merge_response_speed_limit(merge(2))};
  for (const auto &c : cut_ins) {
    for (const auto &r : roundabouts) {
      for (const auto &m : merges) {
        const std::optional<double> perms[6][3] = {
            {c, r, m}, {c, m, r}, {r, c, m},
            {r, m, c}, {m, c, r}, {m, r, c}};
        const auto reference =
            planner_bound3(16.25, perms[0][0], perms[0][1], perms[0][2]);
        for (const auto &p : perms) {
          const auto result = planner_bound3(16.25, p[0], p[1], p[2]);
          EXPECT_EQ(result.has_value(), reference.has_value());
          if (result && reference) {
            EXPECT_DOUBLE_EQ(*result, *reference);
          }
        }
      }
    }
  }
}

TEST(ExternalSpeedLimitThreeWay, MergeReadyNeverOverridesCutInOrRoundabout) {
  const auto with_cut_in = planner_bound3(
      16.25, cut_in_response_speed_limit(cut_in_slowdown(5.0)),
      roundabout_response_speed_limit(roundabout(0)),
      highway_merge_response_speed_limit(merge(0)));
  ASSERT_TRUE(with_cut_in.has_value());
  EXPECT_DOUBLE_EQ(*with_cut_in, 5.0);
  const auto with_roundabout = planner_bound3(
      16.25, cut_in_response_speed_limit(cut_in_none()),
      roundabout_response_speed_limit(roundabout(1)),
      highway_merge_response_speed_limit(merge(0)));
  ASSERT_TRUE(with_roundabout.has_value());
  EXPECT_NEAR(*with_roundabout, std::sqrt(2.0 * 1.8 * 34.0), 1e-3);
}

} // namespace
