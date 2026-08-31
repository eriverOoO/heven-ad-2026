#ifndef AD_PLANNER__PLANNING__HIGHWAY_MERGE_GAP_RISK_HPP_
#define AD_PLANNER__PLANNING__HIGHWAY_MERGE_GAP_RISK_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ad_planner/common/types.hpp"
#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

// A fixed source-grounded highway merge zone. The two lane ids are the
// tapering acceleration lane (source) and the mainline through lane (target)
// that bound it; the route stations bracket the merge on the target corridor.
struct MergeZone
{
  std::string id;
  std::string source_lane_id;
  std::string target_lane_id;
  double route_s_zone_entry_m{0.0};
  double route_s_merge_complete_m{0.0};

  // Throws std::invalid_argument on a non-finite / non-increasing route span.
  MergeZone validated() const;
};

struct HighwayMergeParameters
{
  double ego_speed_epsilon_mps{0.5};
  double maximum_ego_approach_distance_m{400.0};
  // Longitudinal relevance window around the merge zone on the target corridor.
  // Generous "which objects to report" bounds (a fast closing vehicle can be
  // well behind the zone entry over a merge horizon), not a policy threshold.
  double relevant_rear_window_m{150.0};
  double relevant_front_window_m{120.0};
  // |delta_s_at_merge_m| <= this band -> "alongside" (neither clearly ahead nor
  // behind). A geometric longitudinal band, not a lane width.
  double alongside_longitudinal_band_m{5.0};
  // Added to the target lane half-width when deciding "in the target corridor".
  double target_corridor_lateral_margin_m{0.5};
  double closing_speed_epsilon_mps{0.1};
  std::size_t maximum_objects{256U};

  HighwayMergeParameters validated() const;
};

struct MergePredictedStateInput
{
  double time_s{0.0};
  double x_rel_m{0.0};
  double y_rel_m{0.0};
};

struct MergeObjectInput
{
  std::array<std::uint8_t, 16U> object_id{};
  std::uint8_t classification{0U};
  float classification_probability{0.0F};
  float existence_probability{0.0F};

  // Current object centroid relative to the ego body frame (+x forward,
  // +y left), copied from DynamicObjectRisk.
  double x_rel_m{0.0};
  double y_rel_m{0.0};
  // Current relative velocity along the ego forward / left axes.
  double vx_rel_mps{0.0};
  double vy_rel_mps{0.0};

  // Discrete predicted future centroids, each relative to a constant-velocity
  // ego rollout at time_s in the source-stamp body axes (DynamicObjectRiskState).
  std::vector<MergePredictedStateInput> predicted_states;

  bool ttc_valid{false};
  double ttc_s{0.0};
  bool cpa_valid{false};
  double cpa_time_s{0.0};
  double cpa_distance_m{0.0};
  bool predicted_min_separation_valid{false};
  double predicted_min_separation_m{0.0};
  double predicted_min_separation_time_s{0.0};
};

struct MergeEgoState
{
  Pose2 pose;                        // map frame
  double longitudinal_speed_mps{0.0};
  double route_s_m{0.0};             // ego station on the target merge corridor
  double route_longitudinal_speed_mps{0.0};  // ego route-tangential speed (s_dot)
};

// Constant-current-speed arrival estimate of the ego onto the merge reference
// station of the target corridor.
struct MergeEgoTiming
{
  bool in_merge_zone_now{false};
  bool timing_valid{false};
  double merge_time_s{0.0};
  double route_distance_to_zone_entry_m{0.0};
  double route_distance_to_merge_m{0.0};
};

struct HighwayMergeGapRiskResult
{
  std::array<std::uint8_t, 16U> object_id{};
  std::uint8_t classification{0U};
  float classification_probability{0.0F};
  float existence_probability{0.0F};

  bool relevant_to_merge{false};
  double object_route_s_m{0.0};
  double object_lateral_offset_m{0.0};
  bool object_in_target_corridor_now{false};
  bool predicted_to_enter_target_corridor{false};
  bool predicted_corridor_entry_valid{false};
  double predicted_corridor_entry_time_s{0.0};

  double delta_s_now_m{0.0};
  double object_longitudinal_speed_mps{0.0};
  double relative_longitudinal_speed_mps{0.0};

  bool delta_s_at_merge_valid{false};
  double delta_s_at_merge_m{0.0};
  bool is_ahead_at_merge{false};
  bool is_behind_at_merge{false};
  bool is_alongside_at_merge{false};

  bool longitudinal_gap_closing{false};
  double longitudinal_closing_speed_mps{0.0};
  bool time_to_route_coincidence_valid{false};
  double time_to_route_coincidence_s{0.0};

  bool predicted_min_route_gap_valid{false};
  double predicted_min_route_gap_m{0.0};
  double predicted_min_route_gap_time_s{0.0};

  double prediction_horizon_s{0.0};
  bool prediction_covers_merge_time{false};

  bool ttc_valid{false};
  double ttc_s{0.0};
  bool cpa_valid{false};
  double cpa_time_s{0.0};
  double cpa_distance_m{0.0};
  bool predicted_min_separation_valid{false};
  double predicted_min_separation_m{0.0};
  double predicted_min_separation_time_s{0.0};
};

struct HighwayMergeGapComputation
{
  MergeEgoTiming ego;
  double merge_reference_route_s_m{0.0};
  std::vector<HighwayMergeGapRiskResult> objects;
  std::size_t relevant_object_count{0U};
  std::size_t rejected_malformed_objects{0U};
  std::size_t rejected_over_budget{0U};

  bool nearest_leading_valid{false};
  std::array<std::uint8_t, 16U> nearest_leading_object_id{};
  double nearest_leading_delta_s_at_merge_m{0.0};
  bool nearest_trailing_valid{false};
  std::array<std::uint8_t, 16U> nearest_trailing_object_id{};
  double nearest_trailing_delta_s_at_merge_m{0.0};
  bool merge_gap_valid{false};
  double merge_gap_m{0.0};
};

MergeEgoTiming compute_merge_ego_timing(
  const MergeZone & zone,
  const MergeEgoState & ego,
  const HighwayMergeParameters & parameters);

HighwayMergeGapRiskResult compute_highway_merge_gap_risk(
  const MergeZone & zone,
  const ReferenceLane & target_lane,
  const MergeEgoState & ego,
  const MergeEgoTiming & ego_timing,
  const MergeObjectInput & object,
  const HighwayMergeParameters & parameters);

HighwayMergeGapComputation compute_highway_merge_gap_risks(
  const MergeZone & zone,
  const ReferenceLane & target_lane,
  const MergeEgoState & ego,
  const std::vector<MergeObjectInput> & objects,
  const HighwayMergeParameters & parameters);

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__HIGHWAY_MERGE_GAP_RISK_HPP_
