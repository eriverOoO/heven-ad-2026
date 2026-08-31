#include "ad_planner/planning/highway_merge_mission.hpp"

#include <cmath>
#include <stdexcept>

#include "ad_planner/local_planning/frenet/frenet_geometry.hpp"

namespace ad_planner {
namespace {

using State = HighwayMergeMissionState;

bool finite(double value) { return std::isfinite(value); }

} // namespace

HighwayMergeMissionGeometry HighwayMergeMissionGeometry::validated() const {
  if (!finite(approach_entry_route_s_m) || !finite(zone_entry_route_s_m) ||
      !finite(commit_route_s_m) || !finite(merge_complete_route_s_m) ||
      !finite(exit_route_s_m)) {
    throw std::invalid_argument(
        "highway merge mission geometry stations must be finite");
  }
  if (!(approach_entry_route_s_m < zone_entry_route_s_m) ||
      !(zone_entry_route_s_m < commit_route_s_m) ||
      !(commit_route_s_m < merge_complete_route_s_m) ||
      !(merge_complete_route_s_m <= exit_route_s_m)) {
    throw std::invalid_argument(
        "highway merge mission geometry stations must be strictly ordered "
        "approach < zone_entry < commit < merge_complete <= exit");
  }
  return *this;
}

std::optional<double> derive_highway_merge_commit_station(
    const ReferenceLane &source_lane, const ReferenceLane &target_lane,
    const double commit_lateral_separation_m, const double zone_entry_route_s_m,
    const double merge_complete_route_s_m) {
  if (!finite(commit_lateral_separation_m) ||
      commit_lateral_separation_m <= 0.0 || source_lane.points.size() < 2U ||
      target_lane.points.size() < 2U) {
    return std::nullopt;
  }
  // Each source-lane point's own route_s_m is a target-corridor station (the
  // source lane's stations were sampled on route:0 when the corridor cache was
  // built). Project it onto route:0 with the existing Frenet helper and read
  // the lateral offset.
  for (const auto &point : source_lane.points) {
    EgoState ego;
    ego.pose = point.pose;
    double lateral_m = 0.0;
    try {
      lateral_m = project_to_frenet(target_lane, ego).d_m;
    } catch (const std::exception &) {
      continue;
    }
    if (!finite(lateral_m)) {
      continue;
    }
    if (std::abs(lateral_m) < commit_lateral_separation_m) {
      const double station = point.route_s_m;
      if (!finite(station) || station <= zone_entry_route_s_m ||
          station >= merge_complete_route_s_m) {
        return std::nullopt;
      }
      return station;
    }
  }
  return std::nullopt;
}

HighwayMergeMissionOutput
step_highway_merge_mission(const HighwayMergeMissionInput &input,
                          const HighwayMergeMissionGeometry &geometry) {
  HighwayMergeMissionOutput out;
  out.merge_authorized_now = input.enabled && input.merge_authorized;

  const auto finish = [&](State state) {
    out.state = state;
    out.active = state != State::kInactive;
    out.committed = state == State::kCommitted || state == State::kComplete;
    return out;
  };

  if (!input.enabled) {
    return finish(State::kInactive);
  }

  // A large backward jump in ego route station is a sim reset or the route loop
  // wrapping: drop any carried mission state.
  const bool reset =
      input.has_previous_route_s && input.route_progress_valid &&
      finite(input.ego_route_s_m) && finite(input.previous_ego_route_s_m) &&
      input.ego_route_s_m <
          input.previous_ego_route_s_m - std::abs(input.route_s_reset_jump_m);
  out.traversal_reset_detected = reset;
  if (reset) {
    return finish(State::kInactive);
  }

  // No fresh route projection this tick: hold the previous state, never advance.
  if (!input.route_progress_valid || !finite(input.ego_route_s_m)) {
    return finish(input.previous_state);
  }

  const double s = input.ego_route_s_m;
  const State prev = input.previous_state;

  // Commitment is monotone within one traversal: once COMMITTED the mission only
  // advances to COMPLETE. A post-commit authorization loss does NOT reverse it.
  if (prev == State::kCommitted) {
    if (s >= geometry.merge_complete_route_s_m) {
      return finish(State::kComplete);
    }
    return finish(State::kCommitted);
  }
  if (prev == State::kComplete) {
    if (s < geometry.approach_entry_route_s_m ||
        s > geometry.exit_route_s_m) {
      return finish(State::kInactive);
    }
    return finish(State::kComplete);
  }

  // prev in {INACTIVE, APPROACH, WAITING, AUTHORIZED} - not yet committed.
  if (s < geometry.approach_entry_route_s_m ||
      s >= geometry.merge_complete_route_s_m) {
    // Outside the approach..merge-complete span and not committed: nothing to
    // track. (s past merge_complete without ever committing: the ego was
    // carried past the merge point - the mission is moot and exits cleanly.)
    return finish(State::kInactive);
  }

  if (s < geometry.zone_entry_route_s_m) {
    // In the approach corridor; the merge decision region is not yet applicable.
    // Authorization may already be true (the upstream response is active from
    // maximum_approach_distance_m out), in which case go straight to AUTHORIZED.
    return finish(input.merge_authorized ? State::kAuthorized
                                         : State::kApproach);
  }

  // s in [zone_entry, merge_complete): the merge decision region is applicable.
  if (input.merge_authorized) {
    if (s >= geometry.commit_route_s_m) {
      return finish(State::kCommitted);
    }
    return finish(State::kAuthorized);
  }
  // Not authorized in the decision region: WAITING. A prior AUTHORIZED is
  // revoked here (no latch before commit); crossing the commit boundary
  // unauthorized never commits.
  return finish(State::kWaiting);
}

} // namespace ad_planner
