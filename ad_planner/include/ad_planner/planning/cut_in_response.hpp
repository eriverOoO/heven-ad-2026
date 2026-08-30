#ifndef AD_PLANNER__PLANNING__CUT_IN_RESPONSE_HPP_
#define AD_PLANNER__PLANNING__CUT_IN_RESPONSE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ad_planner
{

enum class CutInResponseAction : std::uint8_t
{
  kNone = 0U,
  kSlowdown = 1U,
  kHold = 2U,
};

enum class CutInResponseReason : std::uint8_t
{
  kNone = 0U,
  kApproachingEntry = 1U,
  kCollisionConflict = 2U,
  kSmallPredictedClearance = 3U,
};

// Physically interpretable policy parameters. No opaque score weights.
struct CutInResponseParameters
{
  // Longitudinal clearance the ego keeps to the cut-in station: an object at or
  // within this route-relative distance is a hold case, not a slowdown case.
  double longitudinal_standoff_m{6.0};
  // Comfortable steady deceleration; mirrors perception.braking_deceleration_mps2.
  double comfortable_deceleration_mps2{1.8};
  // Upper deceleration used to decide slowdown vs hold; within the measured
  // IONIQ 5 brake envelope.
  double maximum_deceleration_mps2{3.0};
  // A slowdown request below this speed is escalated to a hold instead.
  double minimum_response_speed_mps{1.0};
  std::size_t maximum_risks{256U};

  CutInResponseParameters validated() const;
};

// Per-object cut-in facts, copied verbatim from a CutInRisk whose
// cut_in_candidate flag is already true. The response policy never re-derives
// geometry; it only consumes these facts.
struct CutInResponseObjectInput
{
  std::array<std::uint8_t, 16U> object_id{};
  std::uint8_t side{0U};
  double route_s_rel_m{0.0};
  double lateral_velocity_toward_corridor_mps{0.0};
  bool predicted_entry_valid{false};
  double predicted_entry_time_s{0.0};
  double predicted_entry_route_s_rel_m{0.0};
  bool ttc_valid{false};
  double ttc_s{0.0};
  bool cpa_valid{false};
  double cpa_time_s{0.0};
  double cpa_distance_m{0.0};
  bool predicted_min_separation_valid{false};
  double predicted_min_separation_m{0.0};
  double predicted_min_separation_time_s{0.0};
};

struct CutInResponseObjectResult
{
  CutInResponseObjectInput input;
  CutInResponseAction action{CutInResponseAction::kNone};
  CutInResponseReason reason{CutInResponseReason::kNone};
  double requested_max_speed_mps{0.0};
  bool requested_max_speed_valid{false};
  double required_deceleration_mps2{0.0};
};

struct CutInResponseResult
{
  CutInResponseAction action{CutInResponseAction::kNone};
  CutInResponseReason reason{CutInResponseReason::kNone};
  bool active{false};
  std::size_t candidate_count{0U};

  bool has_source{false};
  CutInResponseObjectResult source;

  bool requested_max_speed_valid{false};
  double requested_max_speed_mps{0.0};

  double ego_speed_mps{0.0};
  std::size_t rejected_over_budget{0U};
};

// Evaluate the single-object longitudinal policy. ego_speed_mps is clamped to
// be non-negative (the ego never runs cut-in response while reversing in v1).
CutInResponseObjectResult evaluate_cut_in_response_object(
  double ego_speed_mps,
  const CutInResponseObjectInput & object,
  const CutInResponseParameters & parameters);

// Aggregate the per-object policy over every candidate in a frame. The
// aggregate action is the most restrictive; the requested speed is the smallest
// among objects at that action; the source is the object producing that
// speed, ties broken by the lexicographically smallest UUID.
CutInResponseResult compute_cut_in_response(
  double ego_speed_mps,
  const std::vector<CutInResponseObjectInput> & candidates,
  const CutInResponseParameters & parameters);

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__CUT_IN_RESPONSE_HPP_
