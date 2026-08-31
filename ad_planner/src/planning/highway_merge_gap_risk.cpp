#include "ad_planner/planning/highway_merge_gap_risk.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "ad_planner/local_planning/frenet/frenet_geometry.hpp"

namespace ad_planner
{
namespace
{

constexpr double kStationToleranceM = 1.0e-3;
constexpr double kPredictionCoverageToleranceS = 1.0e-3;
constexpr double kDeltaSNowEpsilonM = 1.0e-6;

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

struct LaneHalfWidths
{
  double left_m{0.0};
  double right_m{0.0};
};

// Linear width interpolation along the lane, clamped to the lane's own station
// range (the caller has already established relevance / seam windows).
LaneHalfWidths half_widths_at(const ReferenceLane & lane, double route_s_m)
{
  if (lane.points.size() < 2U) {
    throw std::invalid_argument("merge target lane needs at least two points");
  }
  const double first_s = lane.points.front().route_s_m;
  const double last_s = lane.points.back().route_s_m;
  route_s_m = std::clamp(route_s_m, first_s, last_s);
  const auto upper = std::upper_bound(
    lane.points.begin(), lane.points.end(), route_s_m,
    [](const double value, const ReferencePoint & point) {
      return value < point.route_s_m;
    });
  if (upper == lane.points.begin()) {
    return LaneHalfWidths{lane.points.front().left_width_m,
      lane.points.front().right_width_m};
  }
  if (upper == lane.points.end()) {
    return LaneHalfWidths{lane.points.back().left_width_m,
      lane.points.back().right_width_m};
  }
  const auto & previous = *(upper - 1);
  const auto & next = *upper;
  const double span = next.route_s_m - previous.route_s_m;
  const double ratio = span > 0.0 ? (route_s_m - previous.route_s_m) / span : 0.0;
  return LaneHalfWidths{
    previous.left_width_m + ratio * (next.left_width_m - previous.left_width_m),
    previous.right_width_m + ratio * (next.right_width_m - previous.right_width_m)};
}

// Reconstructs the absolute map-frame pose of a body-relative point (yaw kept
// as the ego yaw -- only used to seed a zero-speed projection).
Pose2 relative_to_map(
  const Pose2 & ego_pose, const double x_rel_m, const double y_rel_m)
{
  const double c = std::cos(ego_pose.yaw_rad);
  const double s = std::sin(ego_pose.yaw_rad);
  return Pose2{
    ego_pose.x + c * x_rel_m - s * y_rel_m,
    ego_pose.y + s * x_rel_m + c * y_rel_m,
    ego_pose.yaw_rad};
}

FrenetState project_pose(const ReferenceLane & lane, const Pose2 & pose)
{
  return project_to_frenet(lane, EgoState{pose, 0.0, 0.0});
}

void validate_object(const MergeObjectInput & object)
{
  if (!finite(static_cast<double>(object.classification_probability)) ||
    !finite(static_cast<double>(object.existence_probability)))
  {
    throw std::invalid_argument("merge object probabilities must be finite");
  }
  require_finite(object.x_rel_m, "merge object x");
  require_finite(object.y_rel_m, "merge object y");
  require_finite(object.vx_rel_mps, "merge object vx");
  require_finite(object.vy_rel_mps, "merge object vy");
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
    throw std::invalid_argument("merge object risk facts are malformed");
  }
  double previous_time = 0.0;
  for (const auto & state : object.predicted_states) {
    if (!finite(state.time_s) || !(state.time_s > previous_time) ||
      !finite(state.x_rel_m) || !finite(state.y_rel_m))
    {
      throw std::invalid_argument(
              "merge predicted states must be finite, positive, and ordered");
    }
    previous_time = state.time_s;
  }
}

bool lexicographically_less(
  const std::array<std::uint8_t, 16U> & a,
  const std::array<std::uint8_t, 16U> & b)
{
  return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
}

}  // namespace

MergeZone MergeZone::validated() const
{
  if (!finite(route_s_zone_entry_m) || !finite(route_s_merge_complete_m) ||
    !(route_s_merge_complete_m > route_s_zone_entry_m))
  {
    throw std::invalid_argument(
            "merge zone route span must be finite and increasing");
  }
  return *this;
}

HighwayMergeParameters HighwayMergeParameters::validated() const
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
  if (!finite(relevant_rear_window_m) || !(relevant_rear_window_m >= 0.0) ||
    !finite(relevant_front_window_m) || !(relevant_front_window_m >= 0.0))
  {
    throw std::invalid_argument("merge relevance windows must be finite and nonnegative");
  }
  if (!finite(alongside_longitudinal_band_m) ||
    !(alongside_longitudinal_band_m >= 0.0))
  {
    throw std::invalid_argument(
            "alongside_longitudinal_band_m must be finite and nonnegative");
  }
  if (!finite(target_corridor_lateral_margin_m) ||
    !(target_corridor_lateral_margin_m >= 0.0))
  {
    throw std::invalid_argument(
            "target_corridor_lateral_margin_m must be finite and nonnegative");
  }
  if (!finite(closing_speed_epsilon_mps) || !(closing_speed_epsilon_mps > 0.0)) {
    throw std::invalid_argument(
            "closing_speed_epsilon_mps must be finite and positive");
  }
  if (maximum_objects == 0U) {
    throw std::invalid_argument("merge maximum_objects must be positive");
  }
  return *this;
}

MergeEgoTiming compute_merge_ego_timing(
  const MergeZone & zone,
  const MergeEgoState & ego,
  const HighwayMergeParameters & parameters)
{
  const MergeZone merge = zone.validated();
  const HighwayMergeParameters config = parameters.validated();
  require_finite(ego.pose.x, "merge ego x");
  require_finite(ego.pose.y, "merge ego y");
  require_finite(ego.pose.yaw_rad, "merge ego yaw");
  require_finite(ego.longitudinal_speed_mps, "merge ego speed");
  require_finite(ego.route_s_m, "merge ego route station");
  require_finite(ego.route_longitudinal_speed_mps, "merge ego route speed");

  MergeEgoTiming timing;
  timing.route_distance_to_zone_entry_m =
    merge.route_s_zone_entry_m - ego.route_s_m;
  timing.route_distance_to_merge_m =
    merge.route_s_merge_complete_m - ego.route_s_m;
  timing.in_merge_zone_now =
    ego.route_s_m >= merge.route_s_zone_entry_m - kStationToleranceM &&
    ego.route_s_m <= merge.route_s_merge_complete_m + kStationToleranceM;

  const bool passed_merge =
    timing.route_distance_to_merge_m < -kStationToleranceM;
  const bool too_far =
    timing.route_distance_to_zone_entry_m > config.maximum_ego_approach_distance_m;
  const bool moving =
    ego.route_longitudinal_speed_mps >= config.ego_speed_epsilon_mps;
  if (passed_merge || too_far || !moving) {
    return timing;
  }

  timing.timing_valid = true;
  timing.merge_time_s =
    std::max(0.0, timing.route_distance_to_merge_m) /
    ego.route_longitudinal_speed_mps;
  return timing;
}

HighwayMergeGapRiskResult compute_highway_merge_gap_risk(
  const MergeZone & zone,
  const ReferenceLane & target_lane,
  const MergeEgoState & ego,
  const MergeEgoTiming & ego_timing,
  const MergeObjectInput & object,
  const HighwayMergeParameters & parameters)
{
  const MergeZone merge = zone.validated();
  const HighwayMergeParameters config = parameters.validated();
  require_finite(ego.pose.x, "merge ego x");
  require_finite(ego.pose.y, "merge ego y");
  require_finite(ego.pose.yaw_rad, "merge ego yaw");
  require_finite(ego.longitudinal_speed_mps, "merge ego speed");
  require_finite(ego.route_s_m, "merge ego route station");
  require_finite(ego.route_longitudinal_speed_mps, "merge ego route speed");
  validate_object(object);

  HighwayMergeGapRiskResult result;
  result.object_id = object.object_id;
  result.classification = object.classification;
  result.classification_probability = object.classification_probability;
  result.existence_probability = object.existence_probability;

  // Generic risk facts are always copied (identity of the underlying object).
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
  result.prediction_horizon_s = object.predicted_states.empty() ?
    0.0 : object.predicted_states.back().time_s;

  // Project the current centroid onto the target corridor. A far / degenerate
  // geometry projection is treated as "not relevant", never malformed.
  double object_route_s_m = 0.0;
  double object_lateral_offset_m = 0.0;
  try {
    const FrenetState frenet =
      project_pose(target_lane, relative_to_map(ego.pose, object.x_rel_m, object.y_rel_m));
    object_route_s_m = frenet.s_m;
    object_lateral_offset_m = frenet.d_m;
  } catch (const std::exception &) {
    return result;  // relevant_to_merge stays false; only identity + generic facts.
  }
  result.object_route_s_m = object_route_s_m;
  result.object_lateral_offset_m = object_lateral_offset_m;
  result.delta_s_now_m = object_route_s_m - ego.route_s_m;

  const auto in_corridor = [&](const double route_s, const double lateral) {
      const LaneHalfWidths widths = half_widths_at(target_lane, route_s);
      return lateral <= widths.left_m + config.target_corridor_lateral_margin_m &&
             lateral >= -(widths.right_m + config.target_corridor_lateral_margin_m);
    };
  const auto s_in_window = [&](const double route_s) {
      return route_s >= merge.route_s_zone_entry_m - config.relevant_rear_window_m &&
             route_s <= merge.route_s_merge_complete_m + config.relevant_front_window_m;
    };

  bool current_lateral_in_corridor = false;
  try {
    current_lateral_in_corridor = in_corridor(object_route_s_m, object_lateral_offset_m);
  } catch (const std::exception &) {
    return result;
  }
  result.object_in_target_corridor_now = current_lateral_in_corridor;
  const bool current_relevant =
    current_lateral_in_corridor && s_in_window(object_route_s_m);

  // Object longitudinal (route-tangential) speed from its CURRENT velocity.
  const double object_vx_map = object.vx_rel_mps + ego.longitudinal_speed_mps;
  const double object_vy_map = object.vy_rel_mps;
  const double object_speed = std::hypot(object_vx_map, object_vy_map);
  Pose2 motion_pose = relative_to_map(ego.pose, object.x_rel_m, object.y_rel_m);
  if (object_speed > 1.0e-9) {
    motion_pose.yaw_rad = ego.pose.yaw_rad + std::atan2(object_vy_map, object_vx_map);
  }
  try {
    const FrenetState motion_frenet =
      project_to_frenet(target_lane, EgoState{motion_pose, object_speed, 0.0});
    result.object_longitudinal_speed_mps = motion_frenet.s_dot_mps;
  } catch (const std::exception &) {
    result.object_longitudinal_speed_mps = 0.0;
  }
  result.relative_longitudinal_speed_mps =
    result.object_longitudinal_speed_mps - ego.route_longitudinal_speed_mps;

  // Discrete predicted centroids -> route stations on the target corridor.
  std::vector<double> predicted_time_s;
  std::vector<double> predicted_route_s_m;
  predicted_time_s.reserve(object.predicted_states.size());
  predicted_route_s_m.reserve(object.predicted_states.size());
  for (const auto & state : object.predicted_states) {
    Pose2 pose;
    try {
      pose = relative_to_map(
        ego.pose,
        ego.longitudinal_speed_mps * state.time_s + state.x_rel_m,
        state.y_rel_m);
      const FrenetState frenet = project_pose(target_lane, pose);
      const double pred_s = frenet.s_m;
      const double pred_d = frenet.d_m;
      predicted_time_s.push_back(state.time_s);
      predicted_route_s_m.push_back(pred_s);
      if (!result.predicted_to_enter_target_corridor &&
        s_in_window(pred_s) && in_corridor(pred_s, pred_d))
      {
        result.predicted_to_enter_target_corridor = true;
        result.predicted_corridor_entry_valid = true;
        result.predicted_corridor_entry_time_s = state.time_s;
      }
    } catch (const std::exception &) {
      // Skip an un-projectable predicted sample; it contributes no evidence.
    }
  }

  result.relevant_to_merge =
    current_relevant || result.predicted_to_enter_target_corridor;
  if (!result.relevant_to_merge) {
    // The route projection is against a station-bounded merge-corridor window,
    // so the numeric route/speed context of a far object would be window-edge
    // clamped rather than meaningful. Keep only the relevance flags, the
    // predicted-corridor-entry evidence, prediction_horizon_s, and the copied
    // generic risk facts; zero the rest.
    result.object_route_s_m = 0.0;
    result.object_lateral_offset_m = 0.0;
    result.delta_s_now_m = 0.0;
    result.object_longitudinal_speed_mps = 0.0;
    result.relative_longitudinal_speed_mps = 0.0;
    return result;
  }

  // --- Merge-interaction facts (relevant objects only) ---------------------

  // Longitudinal gap dynamics from the current relative longitudinal speed.
  const double sign_now = std::abs(result.delta_s_now_m) < kDeltaSNowEpsilonM ?
    0.0 : (result.delta_s_now_m > 0.0 ? 1.0 : -1.0);
  result.longitudinal_closing_speed_mps =
    -sign_now * result.relative_longitudinal_speed_mps;
  result.longitudinal_gap_closing =
    result.longitudinal_closing_speed_mps > config.closing_speed_epsilon_mps;
  if (result.longitudinal_gap_closing) {
    result.time_to_route_coincidence_valid = true;
    result.time_to_route_coincidence_s =
      std::abs(result.delta_s_now_m) / result.longitudinal_closing_speed_mps;
  }

  // Minimum longitudinal separation on the target corridor over the shared
  // prediction horizon (constant-current-speed ego route rollout).
  if (!predicted_time_s.empty()) {
    double min_gap = std::numeric_limits<double>::infinity();
    double min_gap_time = 0.0;
    for (std::size_t k = 0U; k < predicted_time_s.size(); ++k) {
      const double ego_s_k = ego.route_s_m +
        ego.route_longitudinal_speed_mps * predicted_time_s[k];
      const double gap = std::abs(predicted_route_s_m[k] - ego_s_k);
      if (gap < min_gap) {
        min_gap = gap;
        min_gap_time = predicted_time_s[k];
      }
    }
    if (finite(min_gap)) {
      result.predicted_min_route_gap_valid = true;
      result.predicted_min_route_gap_m = min_gap;
      result.predicted_min_route_gap_time_s = min_gap_time;
    }
  }

  // Predicted longitudinal ordering at the ego merge time.
  if (ego_timing.timing_valid) {
    const double t_eval = ego_timing.merge_time_s;
    const bool has_samples = !predicted_time_s.empty();
    const double horizon = result.prediction_horizon_s;
    result.prediction_covers_merge_time =
      has_samples && (horizon + kPredictionCoverageToleranceS >= t_eval);

    double object_s_at_merge = 0.0;
    if (has_samples && t_eval <= horizon + kPredictionCoverageToleranceS) {
      // Interpolate over [(0, s_now), (t_k, s_k)...].
      if (t_eval <= predicted_time_s.front()) {
        const double t1 = predicted_time_s.front();
        const double ratio = t1 > 0.0 ? t_eval / t1 : 0.0;
        object_s_at_merge = object_route_s_m +
          ratio * (predicted_route_s_m.front() - object_route_s_m);
      } else {
        std::size_t upper = predicted_time_s.size() - 1U;
        for (std::size_t k = 1U; k < predicted_time_s.size(); ++k) {
          if (predicted_time_s[k] >= t_eval) {
            upper = k;
            break;
          }
        }
        const double t0 = predicted_time_s[upper - 1U];
        const double t1 = predicted_time_s[upper];
        const double span = t1 - t0;
        const double ratio = span > 0.0 ? (t_eval - t0) / span : 0.0;
        object_s_at_merge = predicted_route_s_m[upper - 1U] +
          ratio * (predicted_route_s_m[upper] - predicted_route_s_m[upper - 1U]);
      }
    } else {
      // Constant-current-longitudinal-speed extrapolation from the current
      // station (prediction does not span the ego merge time).
      object_s_at_merge =
        object_route_s_m + result.object_longitudinal_speed_mps * t_eval;
    }

    result.delta_s_at_merge_valid = true;
    result.delta_s_at_merge_m =
      object_s_at_merge - merge.route_s_merge_complete_m;
    result.is_alongside_at_merge =
      std::abs(result.delta_s_at_merge_m) <= config.alongside_longitudinal_band_m;
    result.is_ahead_at_merge =
      result.delta_s_at_merge_m > config.alongside_longitudinal_band_m;
    result.is_behind_at_merge =
      result.delta_s_at_merge_m < -config.alongside_longitudinal_band_m;
  }

  return result;
}

HighwayMergeGapComputation compute_highway_merge_gap_risks(
  const MergeZone & zone,
  const ReferenceLane & target_lane,
  const MergeEgoState & ego,
  const std::vector<MergeObjectInput> & objects,
  const HighwayMergeParameters & parameters)
{
  const MergeZone merge = zone.validated();
  const HighwayMergeParameters config = parameters.validated();

  HighwayMergeGapComputation computation;
  computation.merge_reference_route_s_m = merge.route_s_merge_complete_m;
  computation.ego = compute_merge_ego_timing(merge, ego, config);
  computation.objects.reserve(std::min(objects.size(), config.maximum_objects));

  bool leading_set = false;
  bool trailing_set = false;
  for (const auto & object : objects) {
    if (computation.objects.size() >= config.maximum_objects) {
      ++computation.rejected_over_budget;
      continue;
    }
    HighwayMergeGapRiskResult risk;
    try {
      risk = compute_highway_merge_gap_risk(
        merge, target_lane, ego, computation.ego, object, config);
    } catch (const std::invalid_argument &) {
      ++computation.rejected_malformed_objects;
      continue;
    } catch (const std::out_of_range &) {
      ++computation.rejected_malformed_objects;
      continue;
    } catch (const std::domain_error &) {
      ++computation.rejected_malformed_objects;
      continue;
    }

    if (risk.relevant_to_merge) {
      ++computation.relevant_object_count;
      if (risk.delta_s_at_merge_valid) {
        const double delta = risk.delta_s_at_merge_m;
        if (delta > 0.0) {
          const bool closer = !leading_set ||
            delta < computation.nearest_leading_delta_s_at_merge_m ||
            (delta == computation.nearest_leading_delta_s_at_merge_m &&
            lexicographically_less(
              risk.object_id, computation.nearest_leading_object_id));
          if (closer) {
            leading_set = true;
            computation.nearest_leading_valid = true;
            computation.nearest_leading_object_id = risk.object_id;
            computation.nearest_leading_delta_s_at_merge_m = delta;
          }
        } else if (delta < 0.0) {
          // Nearest behind = largest (least negative) delta_s = smallest |delta|.
          const bool closer = !trailing_set ||
            delta > computation.nearest_trailing_delta_s_at_merge_m ||
            (delta == computation.nearest_trailing_delta_s_at_merge_m &&
            lexicographically_less(
              risk.object_id, computation.nearest_trailing_object_id));
          if (closer) {
            trailing_set = true;
            computation.nearest_trailing_valid = true;
            computation.nearest_trailing_object_id = risk.object_id;
            computation.nearest_trailing_delta_s_at_merge_m = delta;
          }
        }
      }
    }
    computation.objects.push_back(std::move(risk));
  }

  if (computation.nearest_leading_valid && computation.nearest_trailing_valid) {
    computation.merge_gap_valid = true;
    computation.merge_gap_m =
      computation.nearest_leading_delta_s_at_merge_m -
      computation.nearest_trailing_delta_s_at_merge_m;
  }
  return computation;
}

}  // namespace ad_planner
