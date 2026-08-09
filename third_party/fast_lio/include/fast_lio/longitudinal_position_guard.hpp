#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace fast_lio {

class LongitudinalPositionGuard {
 public:
  static constexpr int kStateDimension = 23;
  using Covariance =
      Eigen::Matrix<double, kStateDimension, kStateDimension>;

  struct Result {
    bool accepted{false};
    double removed_correction_m{0.0};
  };

  static Result apply(
      const Eigen::Matrix3d &world_R_body,
      const Eigen::Vector3d &predicted_world_position,
      const Covariance &predicted_covariance,
      double longitudinal_variance_floor,
      Eigen::Vector3d &lidar_corrected_world_position,
      Covariance &lidar_corrected_covariance) {
    Result result;
    if (!world_R_body.allFinite() ||
        !predicted_world_position.allFinite() ||
        !predicted_covariance.allFinite() ||
        !lidar_corrected_world_position.allFinite() ||
        !lidar_corrected_covariance.allFinite() ||
        !std::isfinite(longitudinal_variance_floor) ||
        longitudinal_variance_floor <= 0.0) {
      return result;
    }

    Eigen::Vector3d world_forward = world_R_body.col(0);
    const double forward_norm = world_forward.norm();
    if (!std::isfinite(forward_norm) || forward_norm < 0.5) return result;
    world_forward /= forward_norm;

    const Eigen::Vector3d lidar_correction =
        lidar_corrected_world_position - predicted_world_position;
    result.removed_correction_m = world_forward.dot(lidar_correction);
    lidar_corrected_world_position -=
        world_forward * result.removed_correction_m;

    // Keep the LiDAR-updated covariance in the observable position plane,
    // discard its longitudinal cross-correlation, and restore an independent
    // longitudinal variance from the IMU prediction (with a safety floor).
    const Eigen::Matrix3d observable_position_projection =
        Eigen::Matrix3d::Identity() - world_forward * world_forward.transpose();
    Covariance state_projection = Covariance::Identity();
    state_projection.block<3, 3>(0, 0) = observable_position_projection;

    const double predicted_longitudinal_variance = world_forward.dot(
        predicted_covariance.block<3, 3>(0, 0) * world_forward);
    if (!std::isfinite(predicted_longitudinal_variance)) return result;
    const double restored_variance = std::max(
        predicted_longitudinal_variance, longitudinal_variance_floor);
    Eigen::Matrix<double, kStateDimension, 1> longitudinal_state_direction =
        Eigen::Matrix<double, kStateDimension, 1>::Zero();
    longitudinal_state_direction.head<3>() = world_forward;

    lidar_corrected_covariance =
        state_projection * lidar_corrected_covariance *
            state_projection.transpose() +
        restored_variance * longitudinal_state_direction *
            longitudinal_state_direction.transpose();
    lidar_corrected_covariance = 0.5 *
        (lidar_corrected_covariance + lidar_corrected_covariance.transpose());
    result.accepted = lidar_corrected_world_position.allFinite() &&
        lidar_corrected_covariance.allFinite();
    return result;
  }
};

}  // namespace fast_lio
