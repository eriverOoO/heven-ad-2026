#include "ad_planner/planning/cut_in_risk.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "ad_planner/local_planning/frenet/frenet_geometry.hpp"

namespace ad_planner
{
namespace
{

constexpr double kBoundaryEpsilonM = 1.0e-6;
constexpr double kStationToleranceM = 1.0e-6;

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

struct LaneWidths
{
  double left_m{0.0};
  double right_m{0.0};
};

LaneWidths widths_at(const ReferenceLane & lane, double route_s_m)
{
  if (lane.points.size() < 2U) {
    throw std::invalid_argument("cut-in primary lane needs at least two points");
  }
  require_finite(route_s_m, "cut-in route station");
  const double first_s = lane.points.front().route_s_m;
  const double last_s = lane.points.back().route_s_m;
  if (route_s_m < first_s - kStationToleranceM ||
    route_s_m > last_s + kStationToleranceM)
  {
    throw std::out_of_range("cut-in projection leaves primary lane");
  }
  route_s_m = std::clamp(route_s_m, first_s, last_s);
  const auto upper = std::upper_bound(
    lane.points.begin(), lane.points.end(), route_s_m,
    [](const double value, const ReferencePoint & point) {
      return value < point.route_s_m;
    });
  if (upper == lane.points.begin()) {
    return LaneWidths{upper->left_width_m, upper->right_width_m};
  }
  if (upper == lane.points.end()) {
    return LaneWidths{
      lane.points.back().left_width_m, lane.points.back().right_width_m};
  }
  const auto & previous = *(upper - 1);
  const auto & next = *upper;
  const double ratio = (route_s_m - previous.route_s_m) /
    (next.route_s_m - previous.route_s_m);
  const LaneWidths widths{
    previous.left_width_m + ratio * (next.left_width_m - previous.left_width_m),
    previous.right_width_m + ratio * (next.right_width_m - previous.right_width_m)};
  if (!finite(widths.left_m) || !finite(widths.right_m) ||
    !(widths.left_m > 0.0) || !(widths.right_m > 0.0))
  {
    throw std::invalid_argument("cut-in primary lane widths are invalid");
  }
  return widths;
}

Pose2 relative_to_route_pose(
  const CutInEgoState & ego, const double x_rel_m, const double y_rel_m)
{
  require_finite(x_rel_m, "cut-in relative x");
  require_finite(y_rel_m, "cut-in relative y");
  const double cosine = std::cos(ego.pose.yaw_rad);
  const double sine = std::sin(ego.pose.yaw_rad);
  return Pose2{
    ego.pose.x + cosine * x_rel_m - sine * y_rel_m,
    ego.pose.y + sine * x_rel_m + cosine * y_rel_m,
    ego.pose.yaw_rad};
}

FrenetState project_position(
  const ReferenceLane & lane, const Pose2 & pose)
{
  return project_to_frenet(lane, EgoState{pose, 0.0, 0.0});
}

bool inside(const double lateral_m, const LaneWidths & widths)
{
  return lateral_m <= widths.left_m + kBoundaryEpsilonM &&
         lateral_m >= -widths.right_m - kBoundaryEpsilonM;
}

CutInSide side_for(const double lateral_m, const LaneWidths & widths)
{
  if (lateral_m > widths.left_m + kBoundaryEpsilonM) {
    return CutInSide::kLeft;
  }
  if (lateral_m < -widths.right_m - kBoundaryEpsilonM) {
    return CutInSide::kRight;
  }
  return CutInSide::kNone;
}

double boundary_gap(
  const double lateral_m, const LaneWidths & widths, const CutInSide side)
{
  if (side == CutInSide::kLeft) {
    return lateral_m - widths.left_m;
  }
  if (side == CutInSide::kRight) {
    return -widths.right_m - lateral_m;
  }
  return std::min(
    std::abs(widths.left_m - lateral_m),
    std::abs(lateral_m + widths.right_m));
}

bool finite_risk_fact(const bool valid, const double value)
{
  return finite(value) && (!valid || value >= 0.0);
}

void validate_object(const CutInObjectInput & object)
{
  if (!finite(static_cast<double>(object.classification_probability)) ||
    !finite(static_cast<double>(object.existence_probability)))
  {
    throw std::invalid_argument("cut-in object probabilities must be finite");
  }
  require_finite(object.x_rel_m, "cut-in object x");
  require_finite(object.y_rel_m, "cut-in object y");
  require_finite(object.vx_rel_mps, "cut-in object vx");
  require_finite(object.vy_rel_mps, "cut-in object vy");
  if (!finite_risk_fact(object.ttc_valid, object.ttc_s) ||
    !finite_risk_fact(object.cpa_valid, object.cpa_time_s) ||
    !finite_risk_fact(object.cpa_valid, object.cpa_distance_m) ||
    !finite_risk_fact(
      object.predicted_min_separation_valid,
      object.predicted_min_separation_m) ||
    !finite_risk_fact(
      object.predicted_min_separation_valid,
      object.predicted_min_separation_time_s))
  {
    throw std::invalid_argument("cut-in object risk facts are malformed");
  }
  double previous_time = 0.0;
  for (const auto & state : object.predicted_states) {
    if (!finite(state.time_s) || !(state.time_s > previous_time) ||
      !finite(state.x_rel_m) || !finite(state.y_rel_m))
    {
      throw std::invalid_argument(
              "cut-in predicted states must be finite, positive, and ordered");
    }
    previous_time = state.time_s;
  }
}

}  // namespace

CutInParameters CutInParameters::validated() const
{
  const auto require_nonnegative = [](const double value, const char * name) {
      if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(
                std::string("cut-in parameter must be finite and nonnegative: ") + name);
      }
    };
  require_nonnegative(forward_analysis_distance_m, "forward_analysis_distance_m");
  require_nonnegative(rear_analysis_distance_m, "rear_analysis_distance_m");
  require_nonnegative(
    minimum_lateral_approach_speed_mps,
    "minimum_lateral_approach_speed_mps");
  require_nonnegative(boundary_margin_m, "boundary_margin_m");
  require_nonnegative(maximum_lateral_gap_m, "maximum_lateral_gap_m");
  require_nonnegative(
    minimum_sustained_entry_time_s, "minimum_sustained_entry_time_s");
  if (maximum_objects == 0U) {
    throw std::invalid_argument("cut-in maximum_objects must be positive");
  }
  return *this;
}

CutInRiskResult compute_cut_in_risk(
  const ReferenceLane & primary_lane,
  const CutInEgoState & ego,
  const CutInObjectInput & object,
  const CutInParameters & parameters)
{
  require_finite(ego.pose.x, "cut-in ego x");
  require_finite(ego.pose.y, "cut-in ego y");
  require_finite(ego.pose.yaw_rad, "cut-in ego yaw");
  require_finite(ego.longitudinal_speed_mps, "cut-in ego speed");
  validate_object(object);
  const CutInParameters config = parameters.validated();

  const FrenetState ego_frenet = project_to_frenet(
    primary_lane, EgoState{ego.pose, ego.longitudinal_speed_mps, 0.0});
  const Pose2 object_pose = relative_to_route_pose(
    ego, object.x_rel_m, object.y_rel_m);
  const double object_vx_base = object.vx_rel_mps + ego.longitudinal_speed_mps;
  const double object_vy_base = object.vy_rel_mps;
  const double object_speed = std::hypot(object_vx_base, object_vy_base);
  Pose2 motion_pose = object_pose;
  if (object_speed > 1.0e-9) {
    motion_pose.yaw_rad = ego.pose.yaw_rad +
      std::atan2(object_vy_base, object_vx_base);
  }
  const FrenetState object_frenet = project_to_frenet(
    primary_lane, EgoState{motion_pose, object_speed, 0.0});
  const LaneWidths widths = widths_at(primary_lane, object_frenet.s_m);

  CutInRiskResult result;
  result.object_id = object.object_id;
  result.classification = object.classification;
  result.classification_probability = object.classification_probability;
  result.existence_probability = object.existence_probability;
  result.route_s_rel_m = object_frenet.s_m - ego_frenet.s_m;
  result.lateral_offset_m = object_frenet.d_m;
  result.corridor_left_width_m = widths.left_m;
  result.corridor_right_width_m = widths.right_m;
  result.side = side_for(object_frenet.d_m, widths);
  result.inside_corridor = result.side == CutInSide::kNone;
  const double gap = boundary_gap(object_frenet.d_m, widths, result.side);
  result.near_boundary = gap <= config.boundary_margin_m + kBoundaryEpsilonM;
  result.adjacent_region = result.side != CutInSide::kNone &&
    gap <= config.maximum_lateral_gap_m + kBoundaryEpsilonM;
  result.lateral_velocity_mps = object_frenet.d_dot_mps;
  if (result.side == CutInSide::kLeft) {
    result.lateral_velocity_toward_corridor_mps = -object_frenet.d_dot_mps;
  } else if (result.side == CutInSide::kRight) {
    result.lateral_velocity_toward_corridor_mps = object_frenet.d_dot_mps;
  }
  result.approaching_corridor = result.side != CutInSide::kNone &&
    result.lateral_velocity_toward_corridor_mps >=
    config.minimum_lateral_approach_speed_mps;
  result.longitudinally_relevant =
    result.route_s_rel_m >= -config.rear_analysis_distance_m &&
    result.route_s_rel_m <= config.forward_analysis_distance_m;

  std::size_t entry_index = object.predicted_states.size();
  std::vector<bool> predicted_inside;
  predicted_inside.reserve(object.predicted_states.size());
  for (std::size_t index = 0U; index < object.predicted_states.size(); ++index) {
    const auto & state = object.predicted_states[index];
    const Pose2 predicted_pose = relative_to_route_pose(
      ego,
      ego.longitudinal_speed_mps * state.time_s + state.x_rel_m,
      state.y_rel_m);
    const FrenetState projected = project_position(primary_lane, predicted_pose);
    const LaneWidths predicted_widths = widths_at(primary_lane, projected.s_m);
    const bool state_inside = inside(projected.d_m, predicted_widths);
    predicted_inside.push_back(state_inside);
    if (state_inside && !result.predicted_entry_valid) {
      result.predicted_entry_valid = true;
      result.predicted_entry_time_s = state.time_s;
      result.predicted_entry_route_s_rel_m = projected.s_m - ego_frenet.s_m;
      result.predicted_entry_lateral_offset_m = projected.d_m;
      entry_index = index;
    }
  }
  if (result.predicted_entry_valid) {
    const bool stays_inside = std::all_of(
      predicted_inside.begin() + static_cast<std::ptrdiff_t>(entry_index),
      predicted_inside.end(), [](const bool value) {return value;});
    const double inside_duration =
      object.predicted_states.back().time_s - result.predicted_entry_time_s;
    result.predicted_entry_sustained = stays_inside &&
      inside_duration + 1.0e-9 >= config.minimum_sustained_entry_time_s;
  }

  result.ttc_valid = object.ttc_valid;
  result.ttc_s = object.ttc_valid ? object.ttc_s : 0.0;
  result.cpa_valid = object.cpa_valid;
  result.cpa_time_s = object.cpa_valid ? object.cpa_time_s : 0.0;
  result.cpa_distance_m = object.cpa_valid ? object.cpa_distance_m : 0.0;
  result.predicted_min_separation_valid =
    object.predicted_min_separation_valid;
  result.predicted_min_separation_m = object.predicted_min_separation_valid ?
    object.predicted_min_separation_m : 0.0;
  result.predicted_min_separation_time_s =
    object.predicted_min_separation_valid ?
    object.predicted_min_separation_time_s : 0.0;

  result.cut_in_candidate = result.adjacent_region &&
    result.approaching_corridor && result.predicted_entry_valid &&
    result.predicted_entry_sustained && result.longitudinally_relevant;
  return result;
}

CutInComputation compute_cut_in_risks(
  const ReferenceLane & primary_lane,
  const CutInEgoState & ego,
  const std::vector<CutInObjectInput> & objects,
  const CutInParameters & parameters)
{
  CutInComputation computation;
  const CutInParameters config = parameters.validated();
  // Validate required route/ego context even for a valid zero-object frame.
  static_cast<void>(project_to_frenet(
    primary_lane, EgoState{ego.pose, ego.longitudinal_speed_mps, 0.0}));
  computation.risks.reserve(std::min(objects.size(), config.maximum_objects));
  for (const auto & object : objects) {
    if (computation.risks.size() >= config.maximum_objects) {
      ++computation.rejected_over_budget;
      continue;
    }
    try {
      computation.risks.push_back(
        compute_cut_in_risk(primary_lane, ego, object, config));
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
