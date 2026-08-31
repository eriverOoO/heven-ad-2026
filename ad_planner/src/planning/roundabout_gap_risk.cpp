#include "ad_planner/planning/roundabout_gap_risk.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ad_planner
{
namespace
{

constexpr double kEgoStationToleranceM = 1.0e-3;

// Absorbs float round-trip when comparing the (discrete) prediction horizon
// against the ego exit ETA; far below the predictor sample spacing.
constexpr double kPredictionCoverageToleranceS = 1.0e-3;

// One contiguous "inside the conflict region" run over the discrete predicted
// centroids. end_valid = false means the object is still inside at the last
// predicted sample (open through the horizon).
struct ConflictInterval
{
  double start_s{0.0};
  bool end_valid{false};
  double end_s{0.0};
};

bool finite(const double value)
{
  return std::isfinite(value);
}

void require_finite(const double value, const char * const name)
{
  if (!finite(value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

double polygon_signed_area(const std::vector<ConflictPoint2> & polygon)
{
  double sum = 0.0;
  const std::size_t n = polygon.size();
  for (std::size_t i = 0U; i < n; ++i) {
    const auto & a = polygon[i];
    const auto & b = polygon[(i + 1U) % n];
    sum += a.x_m * b.y_m - b.x_m * a.y_m;
  }
  return 0.5 * sum;
}

double distance_point_to_segment(
  const double px, const double py,
  const double ax, const double ay, const double bx, const double by)
{
  const double abx = bx - ax;
  const double aby = by - ay;
  const double denom = abx * abx + aby * aby;
  double t = 0.0;
  if (denom > 0.0) {
    t = ((px - ax) * abx + (py - ay) * aby) / denom;
    t = std::clamp(t, 0.0, 1.0);
  }
  const double cx = ax + t * abx;
  const double cy = ay + t * aby;
  return std::hypot(px - cx, py - cy);
}

// Reconstructs the absolute map-frame position of a body-relative point. A
// DynamicObjectRiskState is relative to the constant-velocity ego rollout, so
// the caller adds ego_speed * time to x_rel first.
void relative_to_map(
  const Pose2 & ego_pose, const double x_rel_m, const double y_rel_m,
  double & x_m, double & y_m)
{
  const double c = std::cos(ego_pose.yaw_rad);
  const double s = std::sin(ego_pose.yaw_rad);
  x_m = ego_pose.x + c * x_rel_m - s * y_rel_m;
  y_m = ego_pose.y + s * x_rel_m + c * y_rel_m;
}

// Scans every discrete predicted centroid and returns every contiguous
// inside-the-conflict-region run, in time order. A currently-inside object
// contributes a run starting at 0.0. Discrete-sample semantics: an entry is the
// first predicted centroid inside the polygon, an exit the first subsequent
// centroid outside it -- no interpolation.
std::vector<ConflictInterval> extract_conflict_intervals(
  const RoundaboutConflictZone & conflict,
  const RoundaboutEgoState & ego,
  const RoundaboutObjectInput & object,
  const bool object_in_conflict_now)
{
  std::vector<ConflictInterval> intervals;
  bool inside_prev = object_in_conflict_now;
  if (object_in_conflict_now) {
    intervals.push_back(ConflictInterval{0.0, false, 0.0});
  }
  for (const auto & state : object.predicted_states) {
    double xk = 0.0;
    double yk = 0.0;
    relative_to_map(
      ego.pose,
      ego.longitudinal_speed_mps * state.time_s + state.x_rel_m,
      state.y_rel_m, xk, yk);
    const bool inside = point_in_conflict_polygon(conflict.polygon_m, xk, yk);
    if (inside && !inside_prev) {
      intervals.push_back(ConflictInterval{state.time_s, false, 0.0});
    } else if (!inside && inside_prev && !intervals.empty() &&
      !intervals.back().end_valid)
    {
      intervals.back().end_valid = true;
      intervals.back().end_s = state.time_s;
    }
    inside_prev = inside;
  }
  return intervals;
}

void validate_object(const RoundaboutObjectInput & object)
{
  if (!finite(static_cast<double>(object.classification_probability)) ||
    !finite(static_cast<double>(object.existence_probability)))
  {
    throw std::invalid_argument("roundabout object probabilities must be finite");
  }
  require_finite(object.x_rel_m, "roundabout object x");
  require_finite(object.y_rel_m, "roundabout object y");
  const auto finite_fact = [](const bool valid, const double value) {
      return finite(value) && (!valid || value >= 0.0);
    };
  if (!finite_fact(object.ttc_valid, object.ttc_s) ||
    !finite_fact(object.cpa_valid, object.cpa_time_s) ||
    !finite_fact(object.cpa_valid, object.cpa_distance_m) ||
    !finite_fact(
      object.predicted_min_separation_valid, object.predicted_min_separation_m) ||
    !finite_fact(
      object.predicted_min_separation_valid,
      object.predicted_min_separation_time_s))
  {
    throw std::invalid_argument("roundabout object risk facts are malformed");
  }
  double previous_time = 0.0;
  for (const auto & state : object.predicted_states) {
    if (!finite(state.time_s) || !(state.time_s > previous_time) ||
      !finite(state.x_rel_m) || !finite(state.y_rel_m))
    {
      throw std::invalid_argument(
              "roundabout predicted states must be finite, positive, and ordered");
    }
    previous_time = state.time_s;
  }
}

}  // namespace

bool point_in_conflict_polygon(
  const std::vector<ConflictPoint2> & polygon, const double x_m, const double y_m)
{
  if (polygon.size() < 3U || !finite(x_m) || !finite(y_m)) {
    return false;
  }
  bool inside = false;
  const std::size_t n = polygon.size();
  std::size_t j = n - 1U;
  for (std::size_t i = 0U; i < n; ++i) {
    const double xi = polygon[i].x_m;
    const double yi = polygon[i].y_m;
    const double xj = polygon[j].x_m;
    const double yj = polygon[j].y_m;
    const bool straddles = (yi > y_m) != (yj > y_m);
    if (straddles) {
      const double x_cross = (xj - xi) * (y_m - yi) / (yj - yi) + xi;
      if (x_m < x_cross) {
        inside = !inside;
      }
    }
    j = i;
  }
  return inside;
}

double distance_to_conflict_polygon(
  const std::vector<ConflictPoint2> & polygon, const double x_m, const double y_m)
{
  if (polygon.size() < 3U || !finite(x_m) || !finite(y_m)) {
    return 0.0;
  }
  if (point_in_conflict_polygon(polygon, x_m, y_m)) {
    return 0.0;
  }
  double best = std::numeric_limits<double>::infinity();
  const std::size_t n = polygon.size();
  for (std::size_t i = 0U; i < n; ++i) {
    const auto & a = polygon[i];
    const auto & b = polygon[(i + 1U) % n];
    best = std::min(
      best, distance_point_to_segment(x_m, y_m, a.x_m, a.y_m, b.x_m, b.y_m));
  }
  return finite(best) ? best : 0.0;
}

RoundaboutConflictZone RoundaboutConflictZone::validated() const
{
  if (polygon_m.size() < 3U) {
    throw std::invalid_argument("roundabout conflict polygon needs >= 3 vertices");
  }
  for (const auto & vertex : polygon_m) {
    if (!finite(vertex.x_m) || !finite(vertex.y_m)) {
      throw std::invalid_argument("roundabout conflict polygon vertex is not finite");
    }
  }
  if (std::abs(polygon_signed_area(polygon_m)) < 1.0e-6) {
    throw std::invalid_argument("roundabout conflict polygon is degenerate");
  }
  if (!finite(route_s_enter_m) || !finite(route_s_exit_m) ||
    !(route_s_exit_m > route_s_enter_m))
  {
    throw std::invalid_argument(
            "roundabout conflict route span must be finite and increasing");
  }
  return *this;
}

RoundaboutGapParameters RoundaboutGapParameters::validated() const
{
  if (!finite(ego_speed_epsilon_mps) || !(ego_speed_epsilon_mps > 0.0)) {
    throw std::invalid_argument("ego_speed_epsilon_mps must be finite and positive");
  }
  if (!finite(maximum_ego_approach_distance_m) ||
    !(maximum_ego_approach_distance_m > 0.0))
  {
    throw std::invalid_argument(
            "maximum_ego_approach_distance_m must be finite and positive");
  }
  if (maximum_objects == 0U) {
    throw std::invalid_argument("roundabout maximum_objects must be positive");
  }
  return *this;
}

RoundaboutEgoTiming compute_roundabout_ego_timing(
  const RoundaboutConflictZone & zone,
  const RoundaboutEgoState & ego,
  const RoundaboutGapParameters & parameters)
{
  const RoundaboutConflictZone conflict = zone.validated();
  const RoundaboutGapParameters config = parameters.validated();
  require_finite(ego.pose.x, "roundabout ego x");
  require_finite(ego.pose.y, "roundabout ego y");
  require_finite(ego.pose.yaw_rad, "roundabout ego yaw");
  require_finite(ego.longitudinal_speed_mps, "roundabout ego speed");
  require_finite(ego.route_s_m, "roundabout ego route station");

  RoundaboutEgoTiming timing;
  timing.in_conflict_now = point_in_conflict_polygon(
    conflict.polygon_m, ego.pose.x, ego.pose.y);
  timing.route_distance_to_entry_m = conflict.route_s_enter_m - ego.route_s_m;
  timing.route_distance_to_exit_m = conflict.route_s_exit_m - ego.route_s_m;

  const bool passed_exit =
    timing.route_distance_to_exit_m < -kEgoStationToleranceM;
  const bool too_far =
    timing.route_distance_to_entry_m > config.maximum_ego_approach_distance_m;
  const bool moving = ego.longitudinal_speed_mps >= config.ego_speed_epsilon_mps;
  if (passed_exit || too_far || !moving) {
    return timing;
  }

  const double distance_to_entry = std::max(0.0, timing.route_distance_to_entry_m);
  const double distance_to_exit = std::max(0.0, timing.route_distance_to_exit_m);
  timing.entry_valid = true;
  timing.entry_time_s = distance_to_entry / ego.longitudinal_speed_mps;
  timing.exit_valid = true;
  timing.exit_time_s = distance_to_exit / ego.longitudinal_speed_mps;
  return timing;
}

RoundaboutGapRiskResult compute_roundabout_gap_risk(
  const RoundaboutConflictZone & zone,
  const RoundaboutEgoState & ego,
  const RoundaboutEgoTiming & ego_timing,
  const RoundaboutObjectInput & object,
  const RoundaboutGapParameters & parameters)
{
  const RoundaboutConflictZone conflict = zone.validated();
  static_cast<void>(parameters.validated());
  require_finite(ego.pose.x, "roundabout ego x");
  require_finite(ego.pose.y, "roundabout ego y");
  require_finite(ego.pose.yaw_rad, "roundabout ego yaw");
  require_finite(ego.longitudinal_speed_mps, "roundabout ego speed");
  validate_object(object);

  RoundaboutGapRiskResult result;
  result.object_id = object.object_id;
  result.classification = object.classification;
  result.classification_probability = object.classification_probability;
  result.existence_probability = object.existence_probability;

  double x_now = 0.0;
  double y_now = 0.0;
  relative_to_map(ego.pose, object.x_rel_m, object.y_rel_m, x_now, y_now);
  result.object_in_conflict_now =
    point_in_conflict_polygon(conflict.polygon_m, x_now, y_now);
  result.object_map_distance_to_conflict_m = result.object_in_conflict_now ?
    0.0 : distance_to_conflict_polygon(conflict.polygon_m, x_now, y_now);

  // Scan every discrete predicted centroid into contiguous conflict-occupancy
  // intervals. Each state is relative to the constant-velocity ego rollout, so
  // ego_speed * time is added back before the map transform (inside the helper).
  const std::vector<ConflictInterval> intervals =
    extract_conflict_intervals(
    conflict, ego, object, result.object_in_conflict_now);

  // First-interval fields (unchanged contract): they describe intervals[0] only.
  if (!intervals.empty()) {
    result.object_entry_valid = true;
    result.object_entry_time_s = intervals.front().start_s;
    result.object_exit_valid = intervals.front().end_valid;
    result.object_exit_time_s =
      intervals.front().end_valid ? intervals.front().end_s : 0.0;
  }
  result.relevant_to_conflict =
    result.object_in_conflict_now || result.object_entry_valid;

  // All-interval summary. later_reentry_detected protects a future response
  // consumer from reading an early first exit as permanent clearance.
  result.predicted_conflict_interval_count = static_cast<std::uint16_t>(
    std::min<std::size_t>(
      intervals.size(), std::numeric_limits<std::uint16_t>::max()));
  result.later_reentry_detected = intervals.size() > 1U;
  result.prediction_horizon_s = object.predicted_states.empty() ?
    0.0 : object.predicted_states.back().time_s;
  result.prediction_covers_ego_exit = ego_timing.entry_valid &&
    ego_timing.exit_valid &&
    (result.prediction_horizon_s + kPredictionCoverageToleranceS >=
    ego_timing.exit_time_s);

  // Arrival delta (Phase 6/20).
  result.arrival_delta_valid =
    ego_timing.entry_valid && result.object_entry_valid;
  if (result.arrival_delta_valid) {
    result.arrival_delta_s =
      result.object_entry_time_s - ego_timing.entry_time_s;
  }

  // Occupancy overlap / temporal gap (Phase 7/19). An object still inside at
  // the horizon end has an unbounded upper occupancy bound.
  if (ego_timing.entry_valid && ego_timing.exit_valid &&
    result.object_entry_valid)
  {
    const double e0 = ego_timing.entry_time_s;
    const double e1 = ego_timing.exit_time_s;
    const double o0 = result.object_entry_time_s;
    const double o1 = result.object_exit_valid ?
      result.object_exit_time_s : std::numeric_limits<double>::infinity();
    result.temporal_gap_valid = true;
    const bool overlap = (o0 < e1) && (e0 < o1);
    if (overlap) {
      result.occupancy_overlap = true;
      result.temporal_gap_s = 0.0;
    } else if (o1 <= e0) {
      result.occupancy_overlap = false;
      result.temporal_gap_s = std::max(0.0, e0 - o1);
    } else {
      result.occupancy_overlap = false;
      result.temporal_gap_s = std::max(0.0, o0 - e1);
    }
  }

  // All-interval overlap / minimum separation (conservative aggregate over
  // every predicted occupancy interval, not just the first). An interval still
  // open at the horizon end has an unbounded upper bound.
  if (ego_timing.entry_valid && ego_timing.exit_valid && !intervals.empty()) {
    const double e0 = ego_timing.entry_time_s;
    const double e1 = ego_timing.exit_time_s;
    bool any_overlap = false;
    double min_gap = std::numeric_limits<double>::infinity();
    for (const auto & interval : intervals) {
      const double o0 = interval.start_s;
      const double o1 = interval.end_valid ?
        interval.end_s : std::numeric_limits<double>::infinity();
      if ((o0 < e1) && (e0 < o1)) {
        any_overlap = true;
        min_gap = 0.0;
        continue;
      }
      const double gap = (o1 <= e0) ? (e0 - o1) : (o0 - e1);
      min_gap = std::min(min_gap, std::max(0.0, gap));
    }
    result.any_occupancy_overlap = any_overlap;
    result.minimum_temporal_gap_valid = true;
    result.minimum_temporal_gap_s =
      std::isfinite(min_gap) ? std::max(0.0, min_gap) : 0.0;
  }

  result.ttc_valid = object.ttc_valid;
  result.ttc_s = object.ttc_valid ? object.ttc_s : 0.0;
  result.cpa_valid = object.cpa_valid;
  result.cpa_time_s = object.cpa_valid ? object.cpa_time_s : 0.0;
  result.cpa_distance_m = object.cpa_valid ? object.cpa_distance_m : 0.0;
  result.predicted_min_separation_valid = object.predicted_min_separation_valid;
  result.predicted_min_separation_m = object.predicted_min_separation_valid ?
    object.predicted_min_separation_m : 0.0;
  result.predicted_min_separation_time_s = object.predicted_min_separation_valid ?
    object.predicted_min_separation_time_s : 0.0;
  return result;
}

RoundaboutGapComputation compute_roundabout_gap_risks(
  const RoundaboutConflictZone & zone,
  const RoundaboutEgoState & ego,
  const std::vector<RoundaboutObjectInput> & objects,
  const RoundaboutGapParameters & parameters)
{
  const RoundaboutConflictZone conflict = zone.validated();
  const RoundaboutGapParameters config = parameters.validated();

  RoundaboutGapComputation computation;
  computation.ego = compute_roundabout_ego_timing(conflict, ego, config);
  computation.objects.reserve(std::min(objects.size(), config.maximum_objects));
  for (const auto & object : objects) {
    if (computation.objects.size() >= config.maximum_objects) {
      ++computation.rejected_over_budget;
      continue;
    }
    try {
      auto risk = compute_roundabout_gap_risk(
        conflict, ego, computation.ego, object, config);
      if (risk.relevant_to_conflict) {
        ++computation.relevant_object_count;
      }
      computation.objects.push_back(std::move(risk));
    } catch (const std::invalid_argument &) {
      ++computation.rejected_malformed_objects;
    } catch (const std::out_of_range &) {
      ++computation.rejected_malformed_objects;
    } catch (const std::domain_error &) {
      ++computation.rejected_malformed_objects;
    }
  }
  return computation;
}

}  // namespace ad_planner
