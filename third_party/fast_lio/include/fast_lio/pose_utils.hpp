// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>

#include <cmath>
#include <vector>

namespace fast_lio {

inline bool valid_pose(const geometry_msgs::msg::Pose &pose) {
  const Eigen::Vector3d position(
      pose.position.x, pose.position.y, pose.position.z);
  const Eigen::Quaterniond orientation(
      pose.orientation.w, pose.orientation.x,
      pose.orientation.y, pose.orientation.z);
  return position.allFinite() && orientation.coeffs().allFinite() &&
         orientation.norm() >= 1.0e-6;
}

inline bool valid_rotation_matrix(
    const std::vector<double> &values,
    const double tolerance = 1.0e-6) {
  if (values.size() != 9 || !std::isfinite(tolerance) || tolerance <= 0.0) {
    return false;
  }
  Eigen::Matrix3d rotation;
  rotation << values[0], values[1], values[2],
              values[3], values[4], values[5],
              values[6], values[7], values[8];
  return rotation.allFinite() &&
         (rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).norm() <=
             tolerance &&
         std::abs(rotation.determinant() - 1.0) <= tolerance;
}

inline bool finite_vector(const std::vector<double> &values) {
  for (const double value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

inline Eigen::Isometry3d to_isometry(const geometry_msgs::msg::Pose &pose) {
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  result.linear() = Eigen::Quaterniond(pose.orientation.w, pose.orientation.x,
                                       pose.orientation.y, pose.orientation.z)
                        .normalized()
                        .toRotationMatrix();
  return result;
}

inline geometry_msgs::msg::Pose to_pose(const Eigen::Isometry3d &transform) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = transform.translation().x();
  pose.position.y = transform.translation().y();
  pose.position.z = transform.translation().z();
  const Eigen::Quaterniond q(transform.rotation());
  pose.orientation.x = q.x();
  pose.orientation.y = q.y();
  pose.orientation.z = q.z();
  pose.orientation.w = q.w();
  return pose;
}

inline Eigen::Vector3d world_velocity_to_body(
    const Eigen::Matrix3d &world_R_body,
    const Eigen::Vector3d &world_velocity) {
  return world_R_body.transpose() * world_velocity;
}

inline Eigen::Isometry3d rigid_transform(const std::vector<double> &xyz_rpy) {
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.translation() = Eigen::Vector3d(xyz_rpy.at(0), xyz_rpy.at(1), xyz_rpy.at(2));
  result.linear() = (Eigen::AngleAxisd(xyz_rpy.at(5), Eigen::Vector3d::UnitZ()) *
                     Eigen::AngleAxisd(xyz_rpy.at(4), Eigen::Vector3d::UnitY()) *
                     Eigen::AngleAxisd(xyz_rpy.at(3), Eigen::Vector3d::UnitX()))
                        .toRotationMatrix();
  return result;
}

}  // namespace fast_lio
