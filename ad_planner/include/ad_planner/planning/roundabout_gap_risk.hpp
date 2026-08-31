#ifndef AD_PLANNER__PLANNING__ROUNDABOUT_GAP_RISK_HPP_
#define AD_PLANNER__PLANNING__ROUNDABOUT_GAP_RISK_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ad_planner/common/types.hpp"

namespace ad_planner
{

// Two-dimensional map-frame point (matches the roundabout_conflicts.json
// polygon vertex order).
struct ConflictPoint2
{
  double x_m{0.0};
  double y_m{0.0};
};

// A fixed shared roundabout conflict region. The polygon is the physical area
// (map frame) that both the ego route and circulating traffic traverse; the
// route stations bracket the same region on the primary route and are used
// only for the ego arrival estimate.
struct RoundaboutConflictZone
{
  std::string id;
  std::vector<ConflictPoint2> polygon_m;
  double route_s_enter_m{0.0};
  double route_s_exit_m{0.0};

  // Throws std::invalid_argument on a degenerate polygon (< 3 vertices,
  // non-finite vertex, zero area) or a non-increasing / non-finite route span.
  RoundaboutConflictZone validated() const;
};

struct RoundaboutGapParameters
{
  double ego_speed_epsilon_mps{0.5};
  double maximum_ego_approach_distance_m{400.0};
  std::size_t maximum_objects{256U};

  RoundaboutGapParameters validated() const;
};

struct RoundaboutPredictedStateInput
{
  double time_s{0.0};
  double x_rel_m{0.0};
  double y_rel_m{0.0};
};

struct RoundaboutObjectInput
{
  std::array<std::uint8_t, 16U> object_id{};
  std::uint8_t classification{0U};
  float classification_probability{0.0F};
  float existence_probability{0.0F};

  // Current object centroid relative to the ego body frame (+x forward,
  // +y left), copied from DynamicObjectRisk.
  double x_rel_m{0.0};
  double y_rel_m{0.0};

  // Discrete predicted future centroids, each relative to a constant-velocity
  // ego rollout at time_s in the source-stamp body axes (DynamicObjectRiskState).
  std::vector<RoundaboutPredictedStateInput> predicted_states;

  bool ttc_valid{false};
  double ttc_s{0.0};
  bool cpa_valid{false};
  double cpa_time_s{0.0};
  double cpa_distance_m{0.0};
  bool predicted_min_separation_valid{false};
  double predicted_min_separation_m{0.0};
  double predicted_min_separation_time_s{0.0};
};

struct RoundaboutEgoState
{
  Pose2 pose;                       // map frame
  double longitudinal_speed_mps{0.0};
  double route_s_m{0.0};            // ego station on the primary route
};

// Constant-current-speed arrival estimate of the ego onto the conflict region.
struct RoundaboutEgoTiming
{
  bool in_conflict_now{false};
  bool entry_valid{false};
  double entry_time_s{0.0};
  bool exit_valid{false};
  double exit_time_s{0.0};
  double route_distance_to_entry_m{0.0};
  double route_distance_to_exit_m{0.0};
};

struct RoundaboutGapRiskResult
{
  std::array<std::uint8_t, 16U> object_id{};
  std::uint8_t classification{0U};
  float classification_probability{0.0F};
  float existence_probability{0.0F};

  bool relevant_to_conflict{false};
  double object_map_distance_to_conflict_m{0.0};

  bool object_in_conflict_now{false};
  bool object_entry_valid{false};
  double object_entry_time_s{0.0};
  bool object_exit_valid{false};
  double object_exit_time_s{0.0};

  bool arrival_delta_valid{false};
  double arrival_delta_s{0.0};

  bool temporal_gap_valid{false};
  double temporal_gap_s{0.0};
  bool occupancy_overlap{false};

  bool ttc_valid{false};
  double ttc_s{0.0};
  bool cpa_valid{false};
  double cpa_time_s{0.0};
  double cpa_distance_m{0.0};
  bool predicted_min_separation_valid{false};
  double predicted_min_separation_m{0.0};
  double predicted_min_separation_time_s{0.0};
};

struct RoundaboutGapComputation
{
  RoundaboutEgoTiming ego;
  std::vector<RoundaboutGapRiskResult> objects;
  std::size_t relevant_object_count{0U};
  std::size_t rejected_malformed_objects{0U};
  std::size_t rejected_over_budget{0U};
};

// True when the map-frame point is inside the (simple) conflict polygon.
bool point_in_conflict_polygon(
  const std::vector<ConflictPoint2> & polygon, double x_m, double y_m);

// 0.0 when the point is inside the polygon, otherwise the shortest distance to
// any polygon edge.
double distance_to_conflict_polygon(
  const std::vector<ConflictPoint2> & polygon, double x_m, double y_m);

RoundaboutEgoTiming compute_roundabout_ego_timing(
  const RoundaboutConflictZone & zone,
  const RoundaboutEgoState & ego,
  const RoundaboutGapParameters & parameters);

RoundaboutGapRiskResult compute_roundabout_gap_risk(
  const RoundaboutConflictZone & zone,
  const RoundaboutEgoState & ego,
  const RoundaboutEgoTiming & ego_timing,
  const RoundaboutObjectInput & object,
  const RoundaboutGapParameters & parameters);

RoundaboutGapComputation compute_roundabout_gap_risks(
  const RoundaboutConflictZone & zone,
  const RoundaboutEgoState & ego,
  const std::vector<RoundaboutObjectInput> & objects,
  const RoundaboutGapParameters & parameters);

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__ROUNDABOUT_GAP_RISK_HPP_
