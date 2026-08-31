#include "ad_planner/planning/highway_merge_mission.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/common/local_motion.hpp"

namespace {

using ad_planner::derive_highway_merge_commit_station;
using ad_planner::HighwayMergeMissionGeometry;
using ad_planner::HighwayMergeMissionInput;
using ad_planner::HighwayMergeMissionState;
using ad_planner::ReferenceLane;
using ad_planner::ReferencePoint;
using ad_planner::step_highway_merge_mission;
using State = HighwayMergeMissionState;

// Source-grounded-shaped test geometry (kcity_highway_onramp numbers).
HighwayMergeMissionGeometry geometry() {
  HighwayMergeMissionGeometry g;
  g.approach_entry_route_s_m = 718.7418;
  g.zone_entry_route_s_m = 1118.7418;
  g.commit_route_s_m = 1210.0;
  g.merge_complete_route_s_m = 1286.1546;
  g.exit_route_s_m = 1306.1546;
  return g.validated();
}

HighwayMergeMissionGeometry raw_geometry(double a, double ze, double c,
                                         double mc, double ex) {
  HighwayMergeMissionGeometry g;
  g.approach_entry_route_s_m = a;
  g.zone_entry_route_s_m = ze;
  g.commit_route_s_m = c;
  g.merge_complete_route_s_m = mc;
  g.exit_route_s_m = ex;
  return g;
}

HighwayMergeMissionInput at(double s, bool authorized, State prev,
                            bool enabled = true) {
  HighwayMergeMissionInput input;
  input.enabled = enabled;
  input.route_progress_valid = true;
  input.ego_route_s_m = s;
  input.has_previous_route_s = true;
  input.previous_ego_route_s_m = s;
  input.route_s_reset_jump_m = 3.5;
  input.merge_authorized = authorized;
  input.previous_state = prev;
  return input;
}

State run(double s, bool authorized, State prev, bool enabled = true) {
  return step_highway_merge_mission(at(s, authorized, prev, enabled), geometry())
      .state;
}

// --------------------------------------------------------------------------
// Geometry validation + commit-station derivation
// --------------------------------------------------------------------------

TEST(HighwayMergeMissionGeometry, RejectsNonOrderedStations) {
  EXPECT_THROW(raw_geometry(700, 1118, 1300, 1286, 1306).validated(),
               std::invalid_argument);
  EXPECT_THROW(raw_geometry(1119, 1118, 1210, 1286, 1306).validated(),
               std::invalid_argument);
  EXPECT_THROW(raw_geometry(700, 1118, 1210, 1286,
                            std::numeric_limits<double>::quiet_NaN())
                   .validated(),
               std::invalid_argument);
  EXPECT_NO_THROW(geometry());
}

namespace {
ReferenceLane straight_lane(double x0, double y0, double dy, int n,
                            double s0 = 0.0) {
  ReferenceLane lane;
  for (int i = 0; i < n; ++i) {
    ReferencePoint p;
    p.pose.x = x0;
    p.pose.y = y0 + dy * i;
    p.pose.yaw_rad = dy > 0 ? M_PI / 2.0 : -M_PI / 2.0;
    p.route_s_m = s0 + std::abs(dy) * i;
    lane.points.push_back(p);
  }
  return lane;
}
} // namespace

TEST(HighwayMergeCommitStation, FindsFirstStationBelowSeparationThreshold) {
  // target: x = 0, southbound. source: starts 4 m east, tapers to 0 over its
  // span. commit threshold 3.5 m -> first source station whose lateral offset
  // drops below 3.5.
  ReferenceLane target = straight_lane(0.0, 300.0, -1.0, 200, 1100.0);
  ReferenceLane source;
  const int n = 168;
  for (int i = 0; i < n; ++i) {
    ReferencePoint p;
    const double frac = static_cast<double>(i) / (n - 1);
    p.pose.x = 4.0 * (1.0 - frac); // 4.0 -> 0.0
    p.pose.y = 256.0 - 1.0 * i;
    p.pose.yaw_rad = -M_PI / 2.0;
    p.route_s_m = 1118.7 + 1.0 * i;
    source.points.push_back(p);
  }
  const auto commit = derive_highway_merge_commit_station(source, target, 3.5,
                                                          1118.7, 1286.7);
  ASSERT_TRUE(commit.has_value());
  // lateral 3.5 at frac 0.125 -> i ~ 21 -> s ~ 1139.7
  EXPECT_GT(*commit, 1118.7);
  EXPECT_LT(*commit, 1286.7);
  EXPECT_NEAR(*commit, 1118.7 + 21.0, 2.0);
}

TEST(HighwayMergeCommitStation, NulloptWhenSeparationNeverCrossesThreshold) {
  ReferenceLane target = straight_lane(0.0, 300.0, -1.0, 200, 1100.0);
  ReferenceLane source = straight_lane(10.0, 256.0, -1.0, 100, 1118.7);
  EXPECT_FALSE(
      derive_highway_merge_commit_station(source, target, 3.5, 1118.7, 1286.7)
          .has_value());
}

TEST(HighwayMergeCommitStation, NulloptOnDegenerateInput) {
  ReferenceLane one_point;
  one_point.points.push_back(ReferencePoint{});
  ReferenceLane target = straight_lane(0.0, 300.0, -1.0, 10, 1100.0);
  EXPECT_FALSE(derive_highway_merge_commit_station(one_point, target, 3.5,
                                                   1118.7, 1286.7)
                   .has_value());
  EXPECT_FALSE(derive_highway_merge_commit_station(target, target, 0.0, 1118.7,
                                                   1286.7)
                   .has_value());
}

// --------------------------------------------------------------------------
// Phase 30 - core state transitions
// --------------------------------------------------------------------------

TEST(HighwayMergeMission, DisabledIsAlwaysInactive) {
  EXPECT_EQ(run(1150.0, true, State::kAuthorized, /*enabled=*/false),
            State::kInactive);
  EXPECT_EQ(run(1250.0, true, State::kCommitted, /*enabled=*/false),
            State::kInactive);
}

TEST(HighwayMergeMission, FarBeforeMergeIsInactive) {
  EXPECT_EQ(run(200.0, false, State::kInactive), State::kInactive);
  EXPECT_EQ(run(700.0, false, State::kInactive), State::kInactive); // just before approach
}

TEST(HighwayMergeMission, EnteringApproachWindow) {
  EXPECT_EQ(run(800.0, false, State::kInactive), State::kApproach);
  EXPECT_EQ(run(1100.0, false, State::kApproach), State::kApproach);
}

TEST(HighwayMergeMission, UnauthorizedDecisionRegionIsWaiting) {
  EXPECT_EQ(run(1150.0, false, State::kApproach), State::kWaiting);
  EXPECT_EQ(run(1200.0, false, State::kWaiting), State::kWaiting);
}

TEST(HighwayMergeMission, WaitingPlusAuthorizedBecomesAuthorized) {
  EXPECT_EQ(run(1150.0, true, State::kWaiting), State::kAuthorized);
}

TEST(HighwayMergeMission, AuthorizationRevokedBeforeCommitReturnsToWaiting) {
  EXPECT_EQ(run(1150.0, false, State::kAuthorized), State::kWaiting);
}

TEST(HighwayMergeMission, AuthorizationRegainedBeforeCommitReturnsToAuthorized) {
  EXPECT_EQ(run(1150.0, true, State::kWaiting), State::kAuthorized);
}

TEST(HighwayMergeMission, CrossesCommitWhileAuthorizedBecomesCommitted) {
  EXPECT_EQ(run(1215.0, true, State::kAuthorized), State::kCommitted);
  // exactly on the commit boundary
  EXPECT_EQ(run(1210.0, true, State::kAuthorized), State::kCommitted);
}

TEST(HighwayMergeMission, CrossesCommitUnauthorizedNeverCommits) {
  EXPECT_EQ(run(1250.0, false, State::kWaiting), State::kWaiting);
  EXPECT_EQ(run(1250.0, false, State::kAuthorized), State::kWaiting);
}

TEST(HighwayMergeMission, AuthorizationRevokedAfterCommitStaysCommitted) {
  EXPECT_EQ(run(1250.0, false, State::kCommitted), State::kCommitted);
  EXPECT_EQ(run(1280.0, false, State::kCommitted), State::kCommitted);
}

TEST(HighwayMergeMission, ReachesCompletionPastMergeComplete) {
  EXPECT_EQ(run(1287.0, true, State::kCommitted), State::kComplete);
  EXPECT_EQ(run(1287.0, false, State::kCommitted), State::kComplete);
}

TEST(HighwayMergeMission, LeavesMissionRegionAfterComplete) {
  EXPECT_EQ(run(1300.0, false, State::kComplete), State::kComplete); // still in [.., exit]
  EXPECT_EQ(run(1310.0, false, State::kComplete), State::kInactive); // past exit
}

TEST(HighwayMergeMission, SimResetRollbackReturnsToInactive) {
  HighwayMergeMissionInput input = at(1150.0, true, State::kCommitted);
  input.previous_ego_route_s_m = 1250.0; // ego s jumped backward > reset jump
  const auto out = step_highway_merge_mission(input, geometry());
  EXPECT_EQ(out.state, State::kInactive);
  EXPECT_TRUE(out.traversal_reset_detected);
}

TEST(HighwayMergeMission, SmallBackwardNoiseIsNotAReset) {
  HighwayMergeMissionInput input = at(1200.0, true, State::kAuthorized);
  input.previous_ego_route_s_m = 1201.0; // 1 m < 3.5 m reset jump
  const auto out = step_highway_merge_mission(input, geometry());
  EXPECT_FALSE(out.traversal_reset_detected);
  EXPECT_EQ(out.state, State::kAuthorized);
}

TEST(HighwayMergeMission, WrongMergeZoneIsInactive) {
  // "wrong zone" == ego at a route position outside the merge approach window.
  EXPECT_EQ(run(400.0, true, State::kInactive), State::kInactive);
  EXPECT_EQ(run(1800.0, true, State::kInactive), State::kInactive);
}

TEST(HighwayMergeMission, DeterministicForRepeatedInput) {
  const auto input = at(1150.0, true, State::kWaiting);
  const auto a = step_highway_merge_mission(input, geometry());
  const auto b = step_highway_merge_mission(input, geometry());
  EXPECT_EQ(a.state, b.state);
  EXPECT_EQ(a.committed, b.committed);
  EXPECT_EQ(a.merge_authorized_now, b.merge_authorized_now);
}

TEST(HighwayMergeMission, NoBackwardTransitionAfterCommit) {
  // Sweep every station and authorization value; from COMMITTED the state can
  // only be COMMITTED or COMPLETE (never APPROACH/WAITING/AUTHORIZED/INACTIVE
  // without a reset).
  for (double s = 700.0; s <= 1400.0; s += 5.0) {
    for (const bool auth : {false, true}) {
      const auto out = step_highway_merge_mission(
          at(s, auth, State::kCommitted), geometry());
      EXPECT_TRUE(out.state == State::kCommitted ||
                  out.state == State::kComplete)
          << "s=" << s << " auth=" << auth
          << " state=" << static_cast<int>(out.state);
    }
  }
}

TEST(HighwayMergeMission, HoldsPreviousStateWhenRouteProgressInvalid) {
  HighwayMergeMissionInput input = at(1150.0, false, State::kAuthorized);
  input.route_progress_valid = false;
  EXPECT_EQ(step_highway_merge_mission(input, geometry()).state,
            State::kAuthorized);
  input.previous_state = State::kCommitted;
  EXPECT_EQ(step_highway_merge_mission(input, geometry()).state,
            State::kCommitted);
  input.previous_state = State::kInactive;
  EXPECT_EQ(step_highway_merge_mission(input, geometry()).state,
            State::kInactive);
}

TEST(HighwayMergeMission, OutputFlagsAndFinitenessAcrossASweep) {
  for (double s = 600.0; s <= 1500.0; s += 3.0) {
    for (const bool auth : {false, true}) {
      for (const State prev :
           {State::kInactive, State::kApproach, State::kWaiting,
            State::kAuthorized, State::kCommitted, State::kComplete}) {
        const auto out =
            step_highway_merge_mission(at(s, auth, prev), geometry());
        EXPECT_EQ(out.active, out.state != State::kInactive);
        EXPECT_EQ(out.committed, out.state == State::kCommitted ||
                                     out.state == State::kComplete);
      }
    }
  }
}

// --------------------------------------------------------------------------
// Phase 31-34 - mission sequences mirroring the upstream response
// --------------------------------------------------------------------------

TEST(HighwayMergeMissionSequence, UnsafeThenAuthorizedBeforeCommit) {
  // rear closing unsafe -> merge_authorized false -> WAITING
  State st = run(1150.0, false, State::kApproach);
  EXPECT_EQ(st, State::kWaiting);
  // rear falls back / safe gap -> merge_authorized true -> AUTHORIZED
  st = run(1155.0, true, st);
  EXPECT_EQ(st, State::kAuthorized);
}

TEST(HighwayMergeMissionSequence, AuthorizedThenUnsafeBeforeCommit) {
  State st = run(1150.0, true, State::kWaiting);
  EXPECT_EQ(st, State::kAuthorized);
  st = run(1160.0, false, st); // upstream WAIT
  EXPECT_EQ(st, State::kWaiting);
}

TEST(HighwayMergeMissionSequence, AuthorizedThenCommittedThenUnsafeStaysCommitted) {
  State st = run(1150.0, true, State::kWaiting);
  EXPECT_EQ(st, State::kAuthorized);
  st = run(1215.0, true, st); // crosses commit
  EXPECT_EQ(st, State::kCommitted);
  const auto out = step_highway_merge_mission(
      at(1240.0, false, st), geometry()); // upstream WAIT after commit
  EXPECT_EQ(out.state, State::kCommitted);
  EXPECT_FALSE(out.merge_authorized_now); // upstream authorization is gone
  EXPECT_TRUE(out.committed);             // but the mission stays committed
}

TEST(HighwayMergeMissionSequence, CommitToComplete) {
  State st = State::kCommitted;
  st = run(1270.0, true, st);
  EXPECT_EQ(st, State::kCommitted);
  st = run(1290.0, true, st);
  EXPECT_EQ(st, State::kComplete);
  st = run(1330.0, true, st);
  EXPECT_EQ(st, State::kInactive);
}

} // namespace
