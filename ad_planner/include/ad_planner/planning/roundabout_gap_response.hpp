#ifndef AD_PLANNER__PLANNING__ROUNDABOUT_GAP_RESPONSE_HPP_
#define AD_PLANNER__PLANNING__ROUNDABOUT_GAP_RESPONSE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ad_planner
{

enum class RoundaboutGapResponseAction : std::uint8_t
{
  kRelease = 0U,
  kYield = 1U,
  kHold = 2U,
};

enum class RoundaboutGapResponseReason : std::uint8_t
{
  kNone = 0U,
  kClearGap = 1U,
  kOverlap = 2U,
  kGapTooSmall = 3U,
  kInsufficientPrediction = 4U,
  kInvalidEgoState = 5U,
};

struct RoundaboutGapResponseParameters
{
  double minimum_release_gap_s{2.0};
  double entry_standoff_m{6.0};
  double comfortable_deceleration_mps2{1.8};
  double stopped_speed_threshold_mps{0.5};
  std::size_t maximum_relevant_objects{256U};

  RoundaboutGapResponseParameters validated() const;
};

struct RoundaboutGapResponseObjectInput
{
  std::array<std::uint8_t, 16U> object_id{};
  bool any_occupancy_overlap{false};
  bool minimum_temporal_gap_valid{false};
  double minimum_temporal_gap_s{0.0};
  bool prediction_covers_ego_exit{false};
};

struct RoundaboutGapResponseInput
{
  bool ego_in_conflict_now{false};
  bool ego_entry_valid{false};
  double ego_entry_time_s{0.0};
  bool ego_exit_valid{false};
  double ego_exit_time_s{0.0};
  double ego_route_distance_to_entry_m{0.0};
  double ego_route_distance_to_exit_m{0.0};
  double ego_speed_mps{0.0};
  std::vector<RoundaboutGapResponseObjectInput> relevant_objects;
};

struct RoundaboutGapResponseResult
{
  RoundaboutGapResponseAction action{RoundaboutGapResponseAction::kRelease};
  RoundaboutGapResponseReason reason{RoundaboutGapResponseReason::kNone};
  bool active{false};
  bool has_source{false};
  std::array<std::uint8_t, 16U> source_object_id{};
  std::size_t relevant_object_count{0U};
  double ego_speed_mps{0.0};
  double ego_route_distance_to_entry_m{0.0};
  double available_distance_m{0.0};
  double comfortable_stop_distance_m{0.0};
  bool limiting_gap_valid{false};
  double limiting_gap_s{0.0};
  bool conflict_overlap_present{false};
  bool complete_prediction_coverage{false};
  std::size_t rejected_over_budget{0U};
};

// Stateless entry-gating policy. Inactive RELEASE is a non-restrictive
// not-applicable value for an ego already inside or past the conflict region.
RoundaboutGapResponseResult compute_roundabout_gap_response(
  const RoundaboutGapResponseInput & input,
  const RoundaboutGapResponseParameters & parameters);

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__ROUNDABOUT_GAP_RESPONSE_HPP_
