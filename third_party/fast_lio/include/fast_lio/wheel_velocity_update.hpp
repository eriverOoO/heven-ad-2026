#pragma once

#include <Eigen/Core>

#include <cmath>
#include <optional>

namespace fast_lio {

// FAST-LIO's error-state ordering is declared in use-ikfom.hpp.  Velocity is
// the three-element world-frame block beginning at index 12.
class WheelVelocityUpdate {
 public:
  static constexpr int kStateDimension = 23;
  static constexpr int kVelocityIndex = 12;
  using Covariance =
      Eigen::Matrix<double, kStateDimension, kStateDimension>;

  struct Result {
    bool accepted{false};
    Eigen::Vector3d innovation_body{Eigen::Vector3d::Zero()};
  };

  // Treat the current attitude as known and reset the measured body-velocity
  // components from external vehicle odometry.  This is deliberately a
  // kinematic state reset instead of a covariance-weighted Kalman correction:
  // scan-to-map matching can make FAST-LIO falsely overconfident in zero
  // longitudinal velocity inside a repetitive tunnel, which would otherwise
  // reject the actual vehicle speed.  The projection-form covariance reset is
  // positive semidefinite and removes stale cross-correlation only in the
  // measured velocity subspace.
  static Result apply(
      const Eigen::Matrix3d &world_R_body,
      double forward_speed_mps,
      double forward_variance,
      bool use_nonholonomic_constraints,
      double lateral_variance,
      double vertical_variance,
      Eigen::Vector3d &world_velocity,
      Covariance &covariance) {
    return apply(
        world_R_body, forward_speed_mps, forward_variance,
        std::nullopt, lateral_variance, use_nonholonomic_constraints,
        vertical_variance, world_velocity, covariance);
  }

  // A course-over-ground source can provide a real lateral component even
  // when the vehicle-status packet exposes only longitudinal speed.  In that
  // case reset x/y while leaving z unconstrained unless NHC is requested.
  static Result apply(
      const Eigen::Matrix3d &world_R_body,
      double forward_speed_mps,
      double forward_variance,
      std::optional<double> lateral_speed_mps,
      double lateral_variance,
      bool use_nonholonomic_constraints,
      double vertical_variance,
      Eigen::Vector3d &world_velocity,
      Covariance &covariance) {
    Result result;
    const bool measure_lateral = lateral_speed_mps.has_value();
    if (!world_R_body.allFinite() || !world_velocity.allFinite() ||
        !covariance.allFinite() || !std::isfinite(forward_speed_mps) ||
        !valid_variance(forward_variance) ||
        (measure_lateral && !std::isfinite(*lateral_speed_mps)) ||
        ((measure_lateral || use_nonholonomic_constraints) &&
         !valid_variance(lateral_variance)) ||
        (use_nonholonomic_constraints && !valid_variance(vertical_variance))) {
      return result;
    }

    const int measurement_dimension =
        use_nonholonomic_constraints ? 3 : (measure_lateral ? 2 : 1);
    Eigen::MatrixXd measurement_jacobian = Eigen::MatrixXd::Zero(
        measurement_dimension, kStateDimension);
    measurement_jacobian.block(0, kVelocityIndex, measurement_dimension, 3) =
        world_R_body.transpose().topRows(measurement_dimension);

    const Eigen::Vector3d predicted_body =
        world_R_body.transpose() * world_velocity;
    Eigen::VectorXd innovation = Eigen::VectorXd::Zero(measurement_dimension);
    innovation(0) = forward_speed_mps - predicted_body.x();
    if (measure_lateral || use_nonholonomic_constraints) {
      innovation(1) = lateral_speed_mps.value_or(0.0) - predicted_body.y();
    }
    if (use_nonholonomic_constraints) {
      innovation(2) = -predicted_body.z();
    }
    result.innovation_body.head(measurement_dimension) = innovation;

    Eigen::MatrixXd measurement_covariance =
        Eigen::MatrixXd::Zero(measurement_dimension, measurement_dimension);
    measurement_covariance(0, 0) = forward_variance;
    if (measure_lateral || use_nonholonomic_constraints) {
      measurement_covariance(1, 1) = lateral_variance;
    }
    if (use_nonholonomic_constraints) {
      measurement_covariance(2, 2) = vertical_variance;
    }

    // The rows of H_v are orthonormal because they are rows of a rotation
    // matrix.  H_v^T is therefore the exact reset gain for those components.
    // With NHC disabled, lateral and vertical body velocity remain untouched.
    Eigen::MatrixXd reset_gain =
        Eigen::MatrixXd::Zero(kStateDimension, measurement_dimension);
    reset_gain.block(kVelocityIndex, 0, 3, measurement_dimension) =
        measurement_jacobian
            .block(0, kVelocityIndex, measurement_dimension, 3)
            .transpose();

    const Eigen::VectorXd correction = reset_gain * innovation;
    if (!correction.allFinite()) return result;
    world_velocity += correction.segment<3>(kVelocityIndex);

    const Covariance identity = Covariance::Identity();
    const Covariance residual_projection =
        identity - reset_gain * measurement_jacobian;
    covariance = residual_projection * covariance *
                     residual_projection.transpose() +
                 reset_gain * measurement_covariance * reset_gain.transpose();
    covariance = 0.5 * (covariance + covariance.transpose());
    result.accepted = world_velocity.allFinite() && covariance.allFinite();
    return result;
  }

 private:
  static bool valid_variance(double value) {
    return std::isfinite(value) && value > 0.0;
  }
};

}  // namespace fast_lio
