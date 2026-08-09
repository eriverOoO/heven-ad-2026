#pragma once

#include <Eigen/Core>

#include <cmath>
#include <limits>

namespace fast_lio {

class WheelPositionIncrement {
 public:
  struct Result {
    bool accepted{false};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    double distance_m{0.0};
    double forward_distance_m{0.0};
    double lateral_distance_m{0.0};
  };

  static bool cumulative_distance_advanced(
      double previous_forward_distance_m,
      double previous_lateral_distance_m,
      double current_forward_distance_m,
      double current_lateral_distance_m,
      double tolerance_m = 1.0e-9) {
    return std::isfinite(previous_forward_distance_m) &&
           std::isfinite(previous_lateral_distance_m) &&
           std::isfinite(current_forward_distance_m) &&
           std::isfinite(current_lateral_distance_m) &&
           std::isfinite(tolerance_m) && tolerance_m >= 0.0 &&
           (std::abs(current_forward_distance_m -
                     previous_forward_distance_m) > tolerance_m ||
            std::abs(current_lateral_distance_m -
                     previous_lateral_distance_m) > tolerance_m);
  }

  static double select_interval(
      double host_elapsed_sec,
      double configured_sensor_period_sec,
      double maximum_dt_sec) {
    if (!std::isfinite(host_elapsed_sec) ||
        !std::isfinite(configured_sensor_period_sec) ||
        !std::isfinite(maximum_dt_sec) ||
        host_elapsed_sec <= 0.0 || configured_sensor_period_sec < 0.0 ||
        maximum_dt_sec <= 0.0) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const double interval = configured_sensor_period_sec > 0.0 ?
        configured_sensor_period_sec : host_elapsed_sec;
    if (interval > maximum_dt_sec) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return interval;
  }

  static Result integrate(
      const Eigen::Matrix3d &world_R_body,
      const Eigen::Vector3d &previous_position,
      double previous_forward_speed_mps,
      double current_forward_speed_mps,
      double dt_sec,
      double maximum_dt_sec) {
    return integrate_planar(
        world_R_body, previous_position,
        previous_forward_speed_mps, 0.0,
        current_forward_speed_mps, 0.0,
        dt_sec, maximum_dt_sec);
  }

  static Result integrate_planar(
      const Eigen::Matrix3d &world_R_body,
      const Eigen::Vector3d &previous_position,
      double previous_forward_speed_mps,
      double previous_lateral_speed_mps,
      double current_forward_speed_mps,
      double current_lateral_speed_mps,
      double dt_sec,
      double maximum_dt_sec) {
    Result result;
    if (!world_R_body.allFinite() || !previous_position.allFinite() ||
        !std::isfinite(previous_forward_speed_mps) ||
        !std::isfinite(previous_lateral_speed_mps) ||
        !std::isfinite(current_forward_speed_mps) ||
        !std::isfinite(current_lateral_speed_mps) ||
        !std::isfinite(dt_sec) || !std::isfinite(maximum_dt_sec) ||
        dt_sec <= 0.0 || maximum_dt_sec <= 0.0 || dt_sec > maximum_dt_sec) {
      return result;
    }

    const double forward_distance_m = 0.5 *
        (previous_forward_speed_mps + current_forward_speed_mps) * dt_sec;
    const double lateral_distance_m = 0.5 *
        (previous_lateral_speed_mps + current_lateral_speed_mps) * dt_sec;
    return apply_displacement(
        world_R_body, previous_position,
        forward_distance_m, lateral_distance_m,
        std::numeric_limits<double>::infinity());
  }

  static Result apply_displacement(
      const Eigen::Matrix3d &world_R_body,
      const Eigen::Vector3d &previous_position,
      double forward_distance_m,
      double lateral_distance_m,
      double maximum_abs_distance_m) {
    Result result;
    if (!world_R_body.allFinite() || !previous_position.allFinite() ||
        !std::isfinite(forward_distance_m) ||
        !std::isfinite(lateral_distance_m) ||
        (!std::isfinite(maximum_abs_distance_m) &&
         maximum_abs_distance_m != std::numeric_limits<double>::infinity()) ||
        maximum_abs_distance_m <= 0.0 ||
        std::abs(forward_distance_m) > maximum_abs_distance_m ||
        std::abs(lateral_distance_m) > maximum_abs_distance_m) {
      return result;
    }
    Eigen::Vector3d world_forward = world_R_body.col(0);
    Eigen::Vector3d world_lateral = world_R_body.col(1);
    const double forward_norm = world_forward.norm();
    const double lateral_norm = world_lateral.norm();
    if (!std::isfinite(forward_norm) || forward_norm < 0.5 ||
        !std::isfinite(lateral_norm) || lateral_norm < 0.5) return result;
    world_forward /= forward_norm;
    world_lateral /= lateral_norm;
    result.forward_distance_m = forward_distance_m;
    result.lateral_distance_m = lateral_distance_m;
    result.distance_m = forward_distance_m;
    result.position = previous_position +
        world_forward * forward_distance_m +
        world_lateral * lateral_distance_m;
    result.accepted = result.position.allFinite();
    return result;
  }
};

}  // namespace fast_lio
