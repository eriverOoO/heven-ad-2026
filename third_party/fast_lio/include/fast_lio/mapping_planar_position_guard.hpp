#pragma once

#include <Eigen/Core>

#include <fast_lio/longitudinal_position_guard.hpp>

namespace fast_lio {

class MappingPlanarPositionGuard {
 public:
  using Covariance = LongitudinalPositionGuard::Covariance;

  struct Result {
    bool accepted{false};
    double removed_forward_correction_m{0.0};
    double removed_lateral_correction_m{0.0};
  };

  static Result apply(
      const Eigen::Matrix3d &world_R_body,
      const Eigen::Vector3d &wheel_predicted_world_position,
      const Covariance &predicted_covariance,
      double planar_variance_floor,
      Eigen::Vector3d &lidar_corrected_world_position,
      Covariance &lidar_corrected_covariance) {
    Result result;
    Eigen::Vector3d guarded_position = lidar_corrected_world_position;
    Covariance guarded_covariance = lidar_corrected_covariance;

    const auto forward = LongitudinalPositionGuard::apply(
        world_R_body, wheel_predicted_world_position, predicted_covariance,
        planar_variance_floor, guarded_position, guarded_covariance);
    if (!forward.accepted) return result;

    Eigen::Matrix3d world_R_lateral_guard;
    world_R_lateral_guard.col(0) = world_R_body.col(1);
    world_R_lateral_guard.col(1) = -world_R_body.col(0);
    world_R_lateral_guard.col(2) = world_R_body.col(2);
    const auto lateral = LongitudinalPositionGuard::apply(
        world_R_lateral_guard, wheel_predicted_world_position,
        predicted_covariance, planar_variance_floor, guarded_position,
        guarded_covariance);
    if (!lateral.accepted) return result;

    lidar_corrected_world_position = guarded_position;
    lidar_corrected_covariance = guarded_covariance;
    result.accepted = true;
    result.removed_forward_correction_m = forward.removed_correction_m;
    result.removed_lateral_correction_m = lateral.removed_correction_m;
    return result;
  }
};

}  // namespace fast_lio
