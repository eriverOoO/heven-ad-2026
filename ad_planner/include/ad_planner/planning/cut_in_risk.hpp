#ifndef AD_PLANNER__PLANNING__CUT_IN_RISK_HPP_
#define AD_PLANNER__PLANNING__CUT_IN_RISK_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

enum class CutInSide : std::uint8_t
{
  kNone = 0U,
  kLeft = 1U,
  kRight = 2U,
};

struct CutInParameters
{
  double forward_analysis_distance_m{80.0};
  double rear_analysis_distance_m{5.0};
  double minimum_lateral_approach_speed_mps{0.20};
  double boundary_margin_m{0.25};
  double maximum_lateral_gap_m{5.0};
  double minimum_sustained_entry_time_s{0.50};
  std::size_t maximum_objects{256U};

  CutInParameters validated() const;
};

struct CutInPredictedStateInput
{
  double time_s{0.0};
  double x_rel_m{0.0};
  double y_rel_m{0.0};
};

struct CutInObjectInput
{
  std::array<std::uint8_t, 16U> object_id{};
  std::uint8_t classification{0U};
  float classification_probability{0.0F};
  float existence_probability{0.0F};
  double x_rel_m{0.0};
  double y_rel_m{0.0};
  double vx_rel_mps{0.0};
  double vy_rel_mps{0.0};
  std::vector<CutInPredictedStateInput> predicted_states;

  bool ttc_valid{false};
  double ttc_s{0.0};
  bool cpa_valid{false};
  double cpa_time_s{0.0};
  double cpa_distance_m{0.0};
  bool predicted_min_separation_valid{false};
  double predicted_min_separation_m{0.0};
  double predicted_min_separation_time_s{0.0};
};

struct CutInEgoState
{
  Pose2 pose;
  double longitudinal_speed_mps{0.0};
};

struct CutInRiskResult
{
  std::array<std::uint8_t, 16U> object_id{};
  std::uint8_t classification{0U};
  float classification_probability{0.0F};
  float existence_probability{0.0F};

  double route_s_rel_m{0.0};
  double lateral_offset_m{0.0};
  double corridor_left_width_m{0.0};
  double corridor_right_width_m{0.0};
  bool inside_corridor{false};
  bool near_boundary{false};
  bool adjacent_region{false};
  CutInSide side{CutInSide::kNone};

  double lateral_velocity_mps{0.0};
  double lateral_velocity_toward_corridor_mps{0.0};
  bool approaching_corridor{false};
  bool longitudinally_relevant{false};

  bool predicted_entry_valid{false};
  double predicted_entry_time_s{0.0};
  double predicted_entry_route_s_rel_m{0.0};
  double predicted_entry_lateral_offset_m{0.0};
  bool predicted_entry_sustained{false};

  bool ttc_valid{false};
  double ttc_s{0.0};
  bool cpa_valid{false};
  double cpa_time_s{0.0};
  double cpa_distance_m{0.0};
  bool predicted_min_separation_valid{false};
  double predicted_min_separation_m{0.0};
  double predicted_min_separation_time_s{0.0};

  bool cut_in_candidate{false};
};

struct CutInComputation
{
  std::vector<CutInRiskResult> risks;
  std::size_t rejected_malformed_objects{0U};
  std::size_t rejected_over_budget{0U};
};

CutInRiskResult compute_cut_in_risk(
  const ReferenceLane & primary_lane,
  const CutInEgoState & ego,
  const CutInObjectInput & object,
  const CutInParameters & parameters);

CutInComputation compute_cut_in_risks(
  const ReferenceLane & primary_lane,
  const CutInEgoState & ego,
  const std::vector<CutInObjectInput> & objects,
  const CutInParameters & parameters);

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNING__CUT_IN_RISK_HPP_
