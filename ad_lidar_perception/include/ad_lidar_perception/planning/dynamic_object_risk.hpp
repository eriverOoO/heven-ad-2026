#ifndef AD_LIDAR_PERCEPTION__PLANNING__DYNAMIC_OBJECT_RISK_HPP_
#define AD_LIDAR_PERCEPTION__PLANNING__DYNAMIC_OBJECT_RISK_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ad_lidar_perception::planning
{

// Tunable semantics for the risk computation. No scenario-specific TTC
// thresholds live here -- those belong to later policy nodes.
struct RiskParameters
{
  // Relative motion below this magnitude is treated as "not closing" for the
  // radial rate and as "no relative motion" for TTC / CPA.
  double closing_speed_epsilon_mps{0.05};
  // Constant-velocity rollout horizon for CPA and predicted-min-separation.
  double cpa_horizon_s{6.0};
  // A modelled contact beyond this time is reported as ttc_valid = false.
  double ttc_horizon_s{6.0};
  // Ego circumscribed-circle radius = hypot(half_length, half_width).
  double ego_half_length_m{2.5};
  double ego_half_width_m{1.1};
  // Floor on the object circumscribed radius, guarding degenerate dimensions.
  double minimum_object_radius_m{0.30};
  // Hard upper bound on objects processed per frame (safety / compute bound).
  std::size_t maximum_objects{256};

  // Returns a copy with the values validated; throws std::invalid_argument on
  // a non-finite or out-of-range entry so the caller can fail closed.
  RiskParameters validated() const;
};

// Ego motion in the prediction frame (odom). Lateral and yaw-rate components
// are not observed by the canonical localization output and are treated as 0
// (non-holonomic assumption); see the interface doc.
struct EgoState
{
  double x_m{0.0};
  double y_m{0.0};
  double yaw_rad{0.0};
  double longitudinal_speed_mps{0.0};  // base_link +x
};

// One discrete predicted centroid, in the prediction frame (odom).
struct PredictedPoint
{
  double time_s{0.0};
  double x_m{0.0};
  double y_m{0.0};
};

// One canonical predicted object, already unpacked from
// ad_interfaces/PredictedObject (odom frame, world-frame initial twist).
struct PredictedObjectInput
{
  std::array<std::uint8_t, 16> object_id{};
  std::uint8_t classification{0};
  float classification_probability{0.0F};
  float existence_probability{0.0F};
  double x_m{0.0};
  double y_m{0.0};
  double length_m{0.0};
  double width_m{0.0};
  double vx_world_mps{0.0};
  double vy_world_mps{0.0};
  // initial_pose xy covariance as a row-major 2x2 [xx, xy, yx, yy]; all zero
  // means "source reports no position uncertainty".
  std::array<double, 4> position_covariance_xy{};
  std::vector<PredictedPoint> predicted_points{};
};

// Result mirrors ad_interfaces/DynamicObjectRisk. All vectors are in base_link
// (+x forward, +y left). No field is ever non-finite.
struct DynamicObjectRiskResult
{
  std::array<std::uint8_t, 16> object_id{};
  std::uint8_t classification{0};
  float classification_probability{0.0F};
  float existence_probability{0.0F};

  double x_rel_m{0.0};
  double y_rel_m{0.0};
  double distance_m{0.0};
  double vx_rel_mps{0.0};
  double vy_rel_mps{0.0};
  double relative_speed_mps{0.0};

  double range_rate_mps{0.0};             // positive => range decreasing
  double longitudinal_closing_mps{0.0};   // positive => forward gap decreasing

  bool ttc_valid{false};
  double ttc_s{0.0};

  bool cpa_valid{false};
  double cpa_time_s{0.0};
  double cpa_distance_m{0.0};

  bool predicted_min_separation_valid{false};
  double predicted_min_separation_m{0.0};
  double predicted_min_separation_time_s{0.0};

  double position_uncertainty_m{0.0};
};

// Reasons an individual object was dropped from the output.
enum class ObjectRejectReason
{
  kNonFiniteState,
  kNonFiniteDimensions,
  kOverObjectBudget,
};

struct DynamicObjectRiskComputation
{
  std::vector<DynamicObjectRiskResult> objects;
  std::size_t rejected_non_finite_state{0};
  std::size_t rejected_non_finite_dimensions{0};
  std::size_t rejected_over_budget{0};
};

// Computes one risk record. Throws std::invalid_argument only when the ego
// state itself is non-finite (nothing can be computed). A non-finite object
// is the caller's concern -- see compute_dynamic_object_risks.
DynamicObjectRiskResult compute_object_risk(
  const EgoState & ego,
  const PredictedObjectInput & object,
  const RiskParameters & parameters);

// Batch entry point. Throws std::invalid_argument on a non-finite ego state.
// Non-finite / degenerate objects are skipped and counted, never fabricated.
DynamicObjectRiskComputation compute_dynamic_object_risks(
  const EgoState & ego,
  const std::vector<PredictedObjectInput> & objects,
  const RiskParameters & parameters);

}  // namespace ad_lidar_perception::planning

#endif  // AD_LIDAR_PERCEPTION__PLANNING__DYNAMIC_OBJECT_RISK_HPP_
