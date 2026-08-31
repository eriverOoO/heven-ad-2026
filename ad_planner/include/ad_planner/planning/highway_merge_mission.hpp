#ifndef AD_PLANNER__PLANNING__HIGHWAY_MERGE_MISSION_HPP_
#define AD_PLANNER__PLANNING__HIGHWAY_MERGE_MISSION_HPP_

#include <cstdint>
#include <optional>
#include <vector>

#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner {

// Highway Merge Mission Primitive v1 - the explicit mission-state layer between
// the revocable merge authorization (PlannerContext::highway_merge_authorized,
// from the Highway Merge Response Integration) and a future lateral executor.
//
// It is a pure function of (ego route progress, authorization, source-grounded
// merge geometry, previous state). It commands NOTHING: no steering, no
// lane-change, no CtrlCmd, no speed. It only tracks approach / waiting /
// authorization / commitment / completion so that:
//   * authorization stays revocable BEFORE the ego is physically committed, and
//   * a transient authorization loss AFTER commitment does not later cause a
//     lateral executor to reverse a merge maneuver mid-way.
//
// Independent safety systems (cut-in / roundabout longitudinal constraints,
// CollisionRecovery, TrafficStop, FailSafeBrake) are entirely separate and keep
// their own authority; COMMITTED never disables them.

enum class HighwayMergeMissionState : std::uint8_t {
  kInactive = 0,   // ego outside the highway-merge approach window this traversal
  kApproach = 1,   // ego inside the approach window, before the merge decision region
  kWaiting = 2,    // merge decision region is applicable but authorization is false
  kAuthorized = 3, // fresh merge_authorized == true and the ego has not yet committed
  kCommitted = 4,  // ego has crossed the source-grounded commit boundary while authorized
  kComplete = 5,   // ego has passed the merge-complete station (established on the target corridor)
};

// Source-grounded mission stations on the target corridor (route:0), all derived
// from highway_merge.json + the checksum-verified route corridor at startup.
// approach_entry < zone_entry < commit < merge_complete <= exit.
struct HighwayMergeMissionGeometry {
  double approach_entry_route_s_m{0.0};
  double zone_entry_route_s_m{0.0};
  double commit_route_s_m{0.0};
  double merge_complete_route_s_m{0.0};
  double exit_route_s_m{0.0};

  // Throws std::invalid_argument on a non-finite or non-ordered station set.
  HighwayMergeMissionGeometry validated() const;
};

// Derive the commit boundary: the first source-lane (route:0:left:1) station at
// which its lateral separation from the target lane (route:0) has fallen below
// commit_lateral_separation_m, i.e. the point past which the acceleration-lane
// geometry itself funnels a vehicle onto the mainline and the merge is
// physically committed. Returns std::nullopt when the source lane never crosses
// the threshold inside (zone_entry, merge_complete) - a geometry mismatch the
// caller must treat as "mission unavailable" (soft fail to INACTIVE).
std::optional<double> derive_highway_merge_commit_station(
    const ReferenceLane &source_lane, const ReferenceLane &target_lane,
    double commit_lateral_separation_m, double zone_entry_route_s_m,
    double merge_complete_route_s_m);

struct HighwayMergeMissionInput {
  bool enabled{false};
  // project_primary_route succeeded this tick. When false the mission holds its
  // previous state (never advances, never resets) - a persistent projection
  // failure is a bigger problem than a stale mission state.
  bool route_progress_valid{false};
  double ego_route_s_m{0.0};
  // Previous-tick ego route station, for sim-reset / route-loop-wrap detection.
  bool has_previous_route_s{false};
  double previous_ego_route_s_m{0.0};
  // A backward jump in ego_route_s_m larger than this in one tick is a sim reset
  // or the route loop wrapping; the mission resets to INACTIVE. Derived from
  // (control period) x (merge-region speed limit) with margin, not a round
  // number - see highway_merge_mission_primitive_v1.md.
  double route_s_reset_jump_m{3.5};
  // PlannerContext::highway_merge_authorized - already fresh, zone-matched, and
  // revocable upstream. The mission never re-reads the response topic.
  bool merge_authorized{false};
  HighwayMergeMissionState previous_state{HighwayMergeMissionState::kInactive};
};

struct HighwayMergeMissionOutput {
  HighwayMergeMissionState state{HighwayMergeMissionState::kInactive};
  bool active{false};    // state != kInactive
  bool committed{false}; // state == kCommitted || state == kComplete
  // Echo of the fresh upstream authorization this tick, kept separate from
  // `committed` so an observer can tell "mission is committed" from "upstream
  // still authorizing".
  bool merge_authorized_now{false};
  bool traversal_reset_detected{false};
};

// Pure deterministic mission-state transition. No ROS, no clock, no hidden
// state. Given identical inputs it returns an identical output.
HighwayMergeMissionOutput
step_highway_merge_mission(const HighwayMergeMissionInput &input,
                          const HighwayMergeMissionGeometry &geometry);

} // namespace ad_planner

#endif // AD_PLANNER__PLANNING__HIGHWAY_MERGE_MISSION_HPP_
