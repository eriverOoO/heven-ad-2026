#ifndef AD_PLANNER__PLANNING__HIGHWAY_MERGE_GAP_RESPONSE_HPP_
#define AD_PLANNER__PLANNING__HIGHWAY_MERGE_GAP_RESPONSE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ad_planner
{

enum class HighwayMergeGapResponseAction : std::uint8_t
{
  kMergeReady = 0U,
  kWait = 1U,
  kHold = 2U,
};

enum class HighwayMergeGapResponseReason : std::uint8_t
{
  kNone = 0U,
  kClearGap = 1U,
  kFrontGap = 2U,
  kRearGap = 3U,
  kRearClosing = 4U,
  kAlongside = 5U,
  kPredictedRouteConflict = 6U,
  kInsufficientPrediction = 7U,
  kInvalidEgoState = 8U,
};

struct HighwayMergeGapResponseParameters
{
  // No existing highway following-headway parameter exists in the planner, so
  // these are competition-v1 tuning values, not a universal safety guarantee.
  // Provenance is documented in docs/planning/highway_merge_gap_response_v1.md
  // and config/highway_merge_gap_response.yaml.
  double minimum_front_time_headway_s{1.5};
  double minimum_rear_time_headway_s{2.0};
  double minimum_rear_closing_time_s{3.0};
  double minimum_predicted_route_gap_m{6.0};
  // Places the WAIT/HOLD decision boundary merge_standoff_m before the merge
  // reference station. Mirrors the cut-in / roundabout response standoff.
  double merge_standoff_m{6.0};
  // Canonical planner/perception comfortable braking value
  // (perception.braking_deceleration_mps2 in config/planner.yaml).
  double comfortable_deceleration_mps2{1.8};
  // Mirrors the upstream Highway Merge Gap Risk ego-speed epsilon.
  double stopped_speed_threshold_mps{0.5};
  // Denominator floor for the time-headway facts.
  double speed_epsilon_mps{0.5};
  std::size_t maximum_relevant_objects{256U};

  HighwayMergeGapResponseParameters validated() const;
};

struct HighwayMergeGapResponseObjectInput
{
  std::array<std::uint8_t, 16U> object_id{};
  double delta_s_now_m{0.0};
  double object_longitudinal_speed_mps{0.0};
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
  bool prediction_covers_merge_time{false};
};

struct HighwayMergeGapResponseInput
{
  // The ego is in the configured highway-merge approach / merge zone for the
  // current traversal (not far before the zone, not past the merge reference).
  bool applicable{false};
  bool ego_merge_timing_valid{false};
  double ego_merge_time_s{0.0};
  double ego_speed_mps{0.0};
  double ego_route_distance_to_merge_m{0.0};
  // Relevant objects the frame builder could not admit (over the policy
  // budget). Any nonzero value forbids MERGE_READY.
  std::size_t relevant_objects_omitted{0U};
  std::vector<HighwayMergeGapResponseObjectInput> relevant_objects;
};

struct HighwayMergeGapResponseResult
{
  HighwayMergeGapResponseAction action{HighwayMergeGapResponseAction::kMergeReady};
  HighwayMergeGapResponseReason reason{HighwayMergeGapResponseReason::kNone};
  bool active{false};
  bool has_source{false};
  std::array<std::uint8_t, 16U> source_object_id{};
  std::size_t relevant_object_count{0U};

  double ego_speed_mps{0.0};
  double ego_route_distance_to_merge_m{0.0};
  bool ego_merge_timing_valid{false};
  double ego_merge_time_s{0.0};
  double available_distance_m{0.0};
  double comfortable_stop_distance_m{0.0};
  bool complete_prediction_coverage{false};

  bool front_object_valid{false};
  std::array<std::uint8_t, 16U> front_object_id{};
  double front_gap_m{0.0};
  bool front_time_headway_valid{false};
  double front_time_headway_s{0.0};

  bool rear_object_valid{false};
  std::array<std::uint8_t, 16U> rear_object_id{};
  double rear_gap_m{0.0};
  bool rear_time_headway_valid{false};
  double rear_time_headway_s{0.0};

  bool rear_closing_object_valid{false};
  std::array<std::uint8_t, 16U> rear_closing_object_id{};
  bool rear_closing_time_valid{false};
  double rear_closing_time_s{0.0};

  bool predicted_route_clearance_valid{false};
  double minimum_predicted_route_gap_m{0.0};

  std::size_t rejected_over_budget{0U};
};

// Stateless merge-decision advisory. An inactive MERGE_READY is a
// non-restrictive not-applicable value for an ego far before the merge zone or
// past the merge reference station. MERGE_READY requires valid ego merge
// timing, complete prediction coverage for every relevant object, no alongside
// conflict, acceptable front and rear target-lane spacing, an acceptable rear
// closing time, and sufficient predicted route clearance. A large total merge
// span never overrides a failed front / rear condition.
HighwayMergeGapResponseResult compute_highway_merge_gap_response(
  const HighwayMergeGapResponseInput & input,
  const HighwayMergeGapResponseParameters & parameters);

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__HIGHWAY_MERGE_GAP_RESPONSE_HPP_
