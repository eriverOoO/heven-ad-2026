#include "ad_lidar_perception/planning/dynamic_object_risk.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ad_lidar_perception::planning
{
namespace
{

// Range below which the object is treated as coincident with the ego origin
// and the radial rate is undefined (reported 0). A length, not a speed --
// distinct from RiskParameters::closing_speed_epsilon_mps.
constexpr double kMinimumRangeM = 1.0e-3;

bool finite(const double value)
{
  return std::isfinite(value);
}

bool finite_ego(const EgoState & ego)
{
  return finite(ego.x_m) && finite(ego.y_m) && finite(ego.yaw_rad) &&
         finite(ego.longitudinal_speed_mps);
}

// Larger eigenvalue of the symmetric 2x2 [[xx, xy], [xy, yy]] (>= 0 clamp).
double larger_eigenvalue(const std::array<double, 4> & covariance_xy)
{
  const double xx = covariance_xy[0];
  const double xy = 0.5 * (covariance_xy[1] + covariance_xy[2]);
  const double yy = covariance_xy[3];
  if (!finite(xx) || !finite(xy) || !finite(yy)) {
    return 0.0;
  }
  const double half_trace = 0.5 * (xx + yy);
  const double determinant = xx * yy - xy * xy;
  const double radicand = std::max(0.0, half_trace * half_trace - determinant);
  return std::max(0.0, half_trace + std::sqrt(radicand));
}

double object_radius(const PredictedObjectInput & object, const RiskParameters & p)
{
  const double circumscribed =
    0.5 * std::hypot(std::abs(object.length_m), std::abs(object.width_m));
  return std::max(circumscribed, p.minimum_object_radius_m);
}

}  // namespace

RiskParameters RiskParameters::validated() const
{
  const auto require_positive = [](const double value, const char * name) {
      if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(
                std::string("dynamic object risk parameter must be positive: ") + name);
      }
    };
  require_positive(closing_speed_epsilon_mps, "closing_speed_epsilon_mps");
  require_positive(cpa_horizon_s, "cpa_horizon_s");
  require_positive(ttc_horizon_s, "ttc_horizon_s");
  require_positive(ego_half_length_m, "ego_half_length_m");
  require_positive(ego_half_width_m, "ego_half_width_m");
  require_positive(minimum_object_radius_m, "minimum_object_radius_m");
  if (maximum_objects == 0U) {
    throw std::invalid_argument("dynamic object risk maximum_objects must be > 0");
  }
  return *this;
}

DynamicObjectRiskResult compute_object_risk(
  const EgoState & ego,
  const PredictedObjectInput & object,
  const RiskParameters & parameters)
{
  if (!finite_ego(ego)) {
    throw std::invalid_argument("dynamic object risk ego state is non-finite");
  }

  DynamicObjectRiskResult result;
  result.object_id = object.object_id;
  result.classification = object.classification;
  result.classification_probability = object.classification_probability;
  result.existence_probability = object.existence_probability;

  const double cos_yaw = std::cos(ego.yaw_rad);
  const double sin_yaw = std::sin(ego.yaw_rad);

  // --- relative position: odom delta rotated into base_link (R(-yaw)) ---
  const double dx_world = object.x_m - ego.x_m;
  const double dy_world = object.y_m - ego.y_m;
  const double x_rel = cos_yaw * dx_world + sin_yaw * dy_world;
  const double y_rel = -sin_yaw * dx_world + cos_yaw * dy_world;
  result.x_rel_m = x_rel;
  result.y_rel_m = y_rel;
  result.distance_m = std::hypot(x_rel, y_rel);

  // --- relative velocity: (v_object_world - v_ego_world) rotated into base_link ---
  const double ego_vx_world = cos_yaw * ego.longitudinal_speed_mps;
  const double ego_vy_world = sin_yaw * ego.longitudinal_speed_mps;
  const double vx_world = object.vx_world_mps - ego_vx_world;
  const double vy_world = object.vy_world_mps - ego_vy_world;
  const double vx_rel = cos_yaw * vx_world + sin_yaw * vy_world;
  const double vy_rel = -sin_yaw * vx_world + cos_yaw * vy_world;
  result.vx_rel_mps = vx_rel;
  result.vy_rel_mps = vy_rel;
  result.relative_speed_mps = std::hypot(vx_rel, vy_rel);

  // --- closing metrics ---
  if (result.distance_m > kMinimumRangeM) {
    result.range_rate_mps =
      -(x_rel * vx_rel + y_rel * vy_rel) / result.distance_m;
  } else {
    result.range_rate_mps = 0.0;  // object effectively at the ego origin
  }
  const double sign_x = (x_rel > 0.0) ? 1.0 : ((x_rel < 0.0) ? -1.0 : 0.0);
  result.longitudinal_closing_mps = -sign_x * vx_rel;

  result.position_uncertainty_m = std::sqrt(larger_eigenvalue(object.position_covariance_xy));

  // --- TTC: constant-relative-velocity contact of two circumscribed circles ---
  const double radius_sum =
    std::hypot(parameters.ego_half_length_m, parameters.ego_half_width_m) +
    object_radius(object, parameters);
  const double a = vx_rel * vx_rel + vy_rel * vy_rel;
  const double b = 2.0 * (x_rel * vx_rel + y_rel * vy_rel);
  const double c = (x_rel * x_rel + y_rel * y_rel) - radius_sum * radius_sum;
  const double epsilon_sq =
    parameters.closing_speed_epsilon_mps * parameters.closing_speed_epsilon_mps;
  if (c <= 0.0) {
    result.ttc_valid = true;
    result.ttc_s = 0.0;  // already in contact / overlapping footprints
  } else if (a >= epsilon_sq) {
    const double discriminant = b * b - 4.0 * a * c;
    if (discriminant >= 0.0) {
      const double contact_time = (-b - std::sqrt(discriminant)) / (2.0 * a);
      if (contact_time >= 0.0 && contact_time <= parameters.ttc_horizon_s) {
        result.ttc_valid = true;
        result.ttc_s = contact_time;
      }
    }
  }

  // --- CPA: minimiser of ||r + v t||^2 over t in [0, cpa_horizon_s] ---
  if (a >= epsilon_sq) {
    double cpa_time = -(x_rel * vx_rel + y_rel * vy_rel) / a;
    cpa_time = std::clamp(cpa_time, 0.0, parameters.cpa_horizon_s);
    const double sx = x_rel + vx_rel * cpa_time;
    const double sy = y_rel + vy_rel * cpa_time;
    result.cpa_valid = true;
    result.cpa_time_s = cpa_time;
    result.cpa_distance_m = std::hypot(sx, sy);
  }

  // --- prediction-horizon minimum separation (uses discrete predicted states) ---
  bool any_point = false;
  double min_separation = 0.0;
  double min_separation_time = 0.0;
  for (const auto & point : object.predicted_points) {
    if (!finite(point.time_s) || !finite(point.x_m) || !finite(point.y_m)) {
      continue;
    }
    if (point.time_s <= 0.0 || point.time_s > parameters.cpa_horizon_s) {
      continue;
    }
    const double ego_x = ego.x_m + ego_vx_world * point.time_s;
    const double ego_y = ego.y_m + ego_vy_world * point.time_s;
    const double separation = std::hypot(point.x_m - ego_x, point.y_m - ego_y);
    if (!any_point || separation < min_separation) {
      any_point = true;
      min_separation = separation;
      min_separation_time = point.time_s;
    }
  }
  if (any_point) {
    result.predicted_min_separation_valid = true;
    result.predicted_min_separation_m = min_separation;
    result.predicted_min_separation_time_s = min_separation_time;
  }

  return result;
}

DynamicObjectRiskComputation compute_dynamic_object_risks(
  const EgoState & ego,
  const std::vector<PredictedObjectInput> & objects,
  const RiskParameters & parameters)
{
  if (!finite_ego(ego)) {
    throw std::invalid_argument("dynamic object risk ego state is non-finite");
  }

  DynamicObjectRiskComputation computation;
  computation.objects.reserve(std::min(objects.size(), parameters.maximum_objects));

  for (const auto & object : objects) {
    if (computation.objects.size() >= parameters.maximum_objects) {
      ++computation.rejected_over_budget;
      continue;
    }
    if (!finite(object.x_m) || !finite(object.y_m) ||
      !finite(object.vx_world_mps) || !finite(object.vy_world_mps))
    {
      ++computation.rejected_non_finite_state;
      continue;
    }
    if (!finite(object.length_m) || !finite(object.width_m) ||
      object.length_m <= 0.0 || object.width_m <= 0.0)
    {
      ++computation.rejected_non_finite_dimensions;
      continue;
    }
    computation.objects.push_back(compute_object_risk(ego, object, parameters));
  }

  return computation;
}

}  // namespace ad_lidar_perception::planning
