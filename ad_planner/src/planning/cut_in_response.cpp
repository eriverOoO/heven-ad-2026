#include "ad_planner/planning/cut_in_response.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ad_planner
{
namespace
{

void require_positive(const double value, const char * const name)
{
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(
            std::string("cut-in response parameter must be finite and positive: ") + name);
  }
}

void require_finite(const double value, const char * const name)
{
  if (!std::isfinite(value)) {
    throw std::invalid_argument(
            std::string("cut-in response fact must be finite: ") + name);
  }
}

int action_rank(const CutInResponseAction action)
{
  return static_cast<int>(action);
}

// Order two candidate results for source selection: the more restrictive
// action wins, then the smaller requested speed, then the smaller UUID.
bool more_restrictive(
  const CutInResponseObjectResult & lhs, const CutInResponseObjectResult & rhs)
{
  if (action_rank(lhs.action) != action_rank(rhs.action)) {
    return action_rank(lhs.action) > action_rank(rhs.action);
  }
  if (lhs.requested_max_speed_mps != rhs.requested_max_speed_mps) {
    return lhs.requested_max_speed_mps < rhs.requested_max_speed_mps;
  }
  return lhs.input.object_id < rhs.input.object_id;
}

}  // namespace

CutInResponseParameters CutInResponseParameters::validated() const
{
  require_positive(longitudinal_standoff_m, "longitudinal_standoff_m");
  require_positive(comfortable_deceleration_mps2, "comfortable_deceleration_mps2");
  require_positive(maximum_deceleration_mps2, "maximum_deceleration_mps2");
  if (!(maximum_deceleration_mps2 >= comfortable_deceleration_mps2)) {
    throw std::invalid_argument(
            "cut-in response maximum_deceleration_mps2 must be >= comfortable_deceleration_mps2");
  }
  if (!std::isfinite(minimum_response_speed_mps) || minimum_response_speed_mps < 0.0) {
    throw std::invalid_argument(
            "cut-in response minimum_response_speed_mps must be finite and nonnegative");
  }
  if (maximum_risks == 0U) {
    throw std::invalid_argument("cut-in response maximum_risks must be positive");
  }
  return *this;
}

CutInResponseObjectResult evaluate_cut_in_response_object(
  const double ego_speed_mps,
  const CutInResponseObjectInput & object,
  const CutInResponseParameters & parameters)
{
  const CutInResponseParameters config = parameters.validated();
  require_finite(object.route_s_rel_m, "route_s_rel_m");
  require_finite(object.lateral_velocity_toward_corridor_mps, "lateral_velocity_toward_corridor_mps");
  if (object.predicted_entry_valid) {
    require_finite(object.predicted_entry_time_s, "predicted_entry_time_s");
    require_finite(object.predicted_entry_route_s_rel_m, "predicted_entry_route_s_rel_m");
  }
  if (object.ttc_valid) {
    require_finite(object.ttc_s, "ttc_s");
  }
  if (object.cpa_valid) {
    require_finite(object.cpa_distance_m, "cpa_distance_m");
    require_finite(object.cpa_time_s, "cpa_time_s");
  }
  if (object.predicted_min_separation_valid) {
    require_finite(object.predicted_min_separation_m, "predicted_min_separation_m");
  }

  CutInResponseObjectResult result;
  result.input = object;

  const double ego_speed = std::isfinite(ego_speed_mps) ? std::max(ego_speed_mps, 0.0) : 0.0;

  // Longitudinal headroom to the cut-in station. Only route-station facts (same
  // unit, signed the same way) are used here; the 2D separation facts are kept
  // as independent hold predicates below.
  double station_ahead_m = object.route_s_rel_m;
  if (object.predicted_entry_valid) {
    station_ahead_m = std::min(station_ahead_m, object.predicted_entry_route_s_rel_m);
  }

  if (station_ahead_m <= 0.0) {
    // The object is alongside or behind the ego origin. Longitudinal cut-in
    // response does not apply in v1; other planner safety layers still cover a
    // genuine collision.
    result.action = CutInResponseAction::kNone;
    result.reason = CutInResponseReason::kNone;
    result.required_deceleration_mps2 = 0.0;
    return result;
  }

  const double available_m = station_ahead_m - config.longitudinal_standoff_m;
  if (available_m <= 0.0) {
    result.action = CutInResponseAction::kHold;
    result.reason = CutInResponseReason::kCollisionConflict;
    result.required_deceleration_mps2 = config.maximum_deceleration_mps2;
  } else {
    const double required_decel = ego_speed * ego_speed / (2.0 * available_m);
    const double comfortable_speed =
      std::sqrt(2.0 * config.comfortable_deceleration_mps2 * available_m);
    result.required_deceleration_mps2 = required_decel;
    if (required_decel <= config.comfortable_deceleration_mps2) {
      result.action = CutInResponseAction::kNone;
      result.reason = CutInResponseReason::kNone;
    } else if (required_decel <= config.maximum_deceleration_mps2 &&
      comfortable_speed >= config.minimum_response_speed_mps)
    {
      result.action = CutInResponseAction::kSlowdown;
      result.reason = CutInResponseReason::kApproachingEntry;
      result.requested_max_speed_mps = comfortable_speed;
      result.requested_max_speed_valid = true;
    } else {
      result.action = CutInResponseAction::kHold;
      result.reason = CutInResponseReason::kCollisionConflict;
    }
  }

  // Time-to-collision is the one 2D risk fact used to strengthen the action: a
  // valid TTC means the circumscribed circles actually make contact under
  // constant relative velocity (a purely lateral merge that ends ahead of the
  // ego without contact leaves ttc_valid false), and a TTC inside the ego's own
  // maximum-braking time is an imminent longitudinal conflict -> hold.
  //
  // cpa_distance_m and predicted_min_separation_m are NOT used to set the
  // action: a completed cut-in always ends at a near-zero centroid separation
  // by construction, so thresholding them would collapse every candidate to a
  // hold. They are carried through as explainability facts only.
  const double maximum_brake_time_s = ego_speed > 1.0e-3 ?
    ego_speed / config.maximum_deceleration_mps2 :
    std::numeric_limits<double>::infinity();
  if (object.ttc_valid && object.ttc_s <= maximum_brake_time_s &&
    action_rank(result.action) < action_rank(CutInResponseAction::kHold))
  {
    result.action = CutInResponseAction::kHold;
    result.reason = CutInResponseReason::kCollisionConflict;
  }

  if (result.action == CutInResponseAction::kHold) {
    result.requested_max_speed_mps = 0.0;
    result.requested_max_speed_valid = true;
    if (available_m <= 0.0) {
      result.required_deceleration_mps2 = config.maximum_deceleration_mps2;
    }
  } else if (result.action == CutInResponseAction::kNone) {
    result.requested_max_speed_mps = 0.0;
    result.requested_max_speed_valid = false;
  }
  return result;
}

CutInResponseResult compute_cut_in_response(
  const double ego_speed_mps,
  const std::vector<CutInResponseObjectInput> & candidates,
  const CutInResponseParameters & parameters)
{
  const CutInResponseParameters config = parameters.validated();
  CutInResponseResult result;
  result.ego_speed_mps =
    std::isfinite(ego_speed_mps) ? std::max(ego_speed_mps, 0.0) : 0.0;

  std::vector<CutInResponseObjectResult> evaluated;
  evaluated.reserve(std::min(candidates.size(), config.maximum_risks));
  for (const auto & candidate : candidates) {
    if (evaluated.size() >= config.maximum_risks) {
      ++result.rejected_over_budget;
      continue;
    }
    evaluated.push_back(
      evaluate_cut_in_response_object(result.ego_speed_mps, candidate, config));
  }
  result.candidate_count = evaluated.size();
  if (evaluated.empty()) {
    return result;
  }

  const auto limiting = std::min_element(
    evaluated.begin(), evaluated.end(), more_restrictive);
  result.source = *limiting;
  result.has_source = true;
  result.action = limiting->action;
  result.reason = limiting->reason;
  result.active = limiting->action != CutInResponseAction::kNone;
  result.requested_max_speed_valid = limiting->requested_max_speed_valid;
  result.requested_max_speed_mps = limiting->requested_max_speed_mps;
  if (!result.active) {
    result.has_source = false;
    result.reason = CutInResponseReason::kNone;
  }
  return result;
}

}  // namespace ad_planner
