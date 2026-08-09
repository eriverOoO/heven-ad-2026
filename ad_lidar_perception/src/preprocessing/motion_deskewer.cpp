#include "ad_lidar_perception/preprocessing/motion_deskewer.hpp"

#include "ad_lidar_perception/preprocessing/xyzirt_layout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ad_lidar_perception::preprocessing
{
namespace
{

using Matrix3 = std::array<double, 9>;
using Vector3 = std::array<double, 3>;
using FloatVector3 = std::array<float, 3>;

Matrix3 multiply(const Matrix3 & left, const Matrix3 & right)
{
  Matrix3 result{};
  for (std::size_t row = 0U; row < 3U; ++row) {
    for (std::size_t column = 0U; column < 3U; ++column) {
      for (std::size_t inner = 0U; inner < 3U; ++inner) {
        result[row * 3U + column] +=
          left[row * 3U + inner] * right[inner * 3U + column];
      }
    }
  }
  return result;
}

Vector3 multiply(const Matrix3 & matrix, const Vector3 & vector)
{
  return {
    matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
    matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
    matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
  };
}

Vector3 add(const Vector3 & left, const Vector3 & right)
{
  return {left[0] + right[0], left[1] + right[1], left[2] + right[2]};
}

Vector3 subtract(const Vector3 & left, const Vector3 & right)
{
  return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

Matrix3 transpose(const Matrix3 & matrix)
{
  return {
    matrix[0], matrix[3], matrix[6],
    matrix[1], matrix[4], matrix[7],
    matrix[2], matrix[5], matrix[8],
  };
}

Matrix3 skew(const Vector3 & vector)
{
  return {
    0.0, -vector[2], vector[1],
    vector[2], 0.0, -vector[0],
    -vector[1], vector[0], 0.0,
  };
}

Matrix3 scaled_add(
  const Matrix3 & identity, const Matrix3 & first, const double first_scale,
  const Matrix3 & second, const double second_scale)
{
  Matrix3 result{};
  for (std::size_t index = 0U; index < result.size(); ++index) {
    result[index] =
      identity[index] + first_scale * first[index] + second_scale * second[index];
  }
  return result;
}

RigidTransform3d compose(
  const RigidTransform3d & left, const RigidTransform3d & right)
{
  return {
    multiply(left.rotation, right.rotation),
    add(left.translation, multiply(left.rotation, right.translation)),
  };
}

Vector3 transform_point(const RigidTransform3d & transform, const Vector3 & point)
{
  return add(multiply(transform.rotation, point), transform.translation);
}

Vector3 inverse_transform_point(
  const RigidTransform3d & transform, const Vector3 & point)
{
  return multiply(transpose(transform.rotation), subtract(point, transform.translation));
}

RigidTransform3d exponential(
  const double longitudinal_velocity_mps,
  const std::array<double, 3> & angular_velocity_rps,
  const double delta_sec)
{
  const Matrix3 identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const Vector3 phi{
    angular_velocity_rps[0] * delta_sec,
    angular_velocity_rps[1] * delta_sec,
    angular_velocity_rps[2] * delta_sec,
  };
  const Vector3 rho{longitudinal_velocity_mps * delta_sec, 0.0, 0.0};
  const auto phi_hat = skew(phi);
  const auto phi_hat_squared = multiply(phi_hat, phi_hat);
  const auto theta_squared = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];

  double rotation_first = 1.0;
  double rotation_second = 0.5;
  double jacobian_first = 0.5;
  double jacobian_second = 1.0 / 6.0;
  if (theta_squared > 1.0e-16) {
    const auto theta = std::sqrt(theta_squared);
    rotation_first = std::sin(theta) / theta;
    rotation_second = (1.0 - std::cos(theta)) / theta_squared;
    jacobian_first = rotation_second;
    jacobian_second = (theta - std::sin(theta)) / (theta_squared * theta);
  } else {
    rotation_first -= theta_squared / 6.0;
    rotation_second -= theta_squared / 24.0;
    jacobian_first -= theta_squared / 24.0;
    jacobian_second -= theta_squared / 120.0;
  }
  const auto rotation = scaled_add(
    identity, phi_hat, rotation_first, phi_hat_squared, rotation_second);
  const auto left_jacobian = scaled_add(
    identity, phi_hat, jacobian_first, phi_hat_squared, jacobian_second);
  return {rotation, multiply(left_jacobian, rho)};
}

void validate_transform(const RigidTransform3d & transform)
{
  if (!std::all_of(
      transform.rotation.begin(), transform.rotation.end(),
      [](const double value) {return std::isfinite(value);} ) ||
    !std::all_of(
      transform.translation.begin(), transform.translation.end(),
      [](const double value) {return std::isfinite(value);} ))
  {
    throw std::invalid_argument("base-from-LiDAR transform must be finite");
  }
  const auto transposed = transpose(transform.rotation);
  const auto product = multiply(transposed, transform.rotation);
  const Matrix3 identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  for (std::size_t index = 0U; index < product.size(); ++index) {
    if (std::abs(product[index] - identity[index]) > 1.0e-9) {
      throw std::invalid_argument("base-from-LiDAR rotation must be orthonormal");
    }
  }
  const auto determinant =
    transform.rotation[0] *
    (transform.rotation[4] * transform.rotation[8] -
    transform.rotation[5] * transform.rotation[7]) -
    transform.rotation[1] *
    (transform.rotation[3] * transform.rotation[8] -
    transform.rotation[5] * transform.rotation[6]) +
    transform.rotation[2] *
    (transform.rotation[3] * transform.rotation[7] -
    transform.rotation[4] * transform.rotation[6]);
  if (std::abs(determinant - 1.0) > 1.0e-9) {
    throw std::invalid_argument("base-from-LiDAR rotation must be proper");
  }
}

void validate_options(const MotionDeskewOptions & options)
{
  if (!std::isfinite(options.maximum_scan_duration_sec) ||
    options.maximum_scan_duration_sec <= 0.0 ||
    !std::isfinite(options.maximum_imu_gap_sec) ||
    options.maximum_imu_gap_sec <= 0.0 ||
    !std::isfinite(options.maximum_wheel_gap_sec) ||
    options.maximum_wheel_gap_sec <= 0.0 ||
    !std::isfinite(options.maximum_integration_step_sec) ||
    options.maximum_integration_step_sec <= 0.0 ||
    options.maximum_point_count == 0U)
  {
    throw std::invalid_argument("deskew limits must be finite and positive");
  }
}

double scan_start_seconds(const sensor_msgs::msg::PointCloud2 & cloud)
{
  return ros_stamp_seconds(cloud.header.stamp);
}

template<typename T>
void write_value(std::vector<std::uint8_t> & data, const std::size_t offset, const T value)
{
  std::memcpy(data.data() + offset, &value, sizeof(T));
}

MotionDeskewResult failure(const std::exception & error)
{
  return {std::nullopt, MotionDeskewRetryability::kPermanent, error.what()};
}

}  // namespace

std::int64_t ros_stamp_nanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  if (stamp.sec < 0 || stamp.nanosec >= 1000000000U) {
    throw std::invalid_argument("ROS timestamp must be valid and nonnegative");
  }
  return static_cast<std::int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
}

double ros_stamp_seconds(const builtin_interfaces::msg::Time & stamp)
{
  static_cast<void>(ros_stamp_nanoseconds(stamp));
  return static_cast<double>(stamp.sec) +
         static_cast<double>(stamp.nanosec) * 1.0e-9;
}

PendingDeskewAction pending_deskew_action(const MotionDeskewResult & result) noexcept
{
  if (result.cloud.has_value()) {
    return PendingDeskewAction::kPublish;
  }
  if (result.retryability == MotionDeskewRetryability::kRetryable) {
    return PendingDeskewAction::kRetry;
  }
  return PendingDeskewAction::kDrop;
}

RigidTransform3d rigid_transform_from_quaternion(
  const std::array<double, 3> & translation,
  const std::array<double, 4> & quaternion_xyzw)
{
  if (!std::all_of(
      translation.begin(), translation.end(),
      [](const double value) {return std::isfinite(value);} ) ||
    !std::all_of(
      quaternion_xyzw.begin(), quaternion_xyzw.end(),
      [](const double value) {return std::isfinite(value);} ))
  {
    throw std::invalid_argument("transform components must be finite");
  }
  const auto scale = std::max({
    std::abs(quaternion_xyzw[0]), std::abs(quaternion_xyzw[1]),
    std::abs(quaternion_xyzw[2]), std::abs(quaternion_xyzw[3])});
  if (scale == 0.0) {
    throw std::invalid_argument("transform quaternion must be nonzero");
  }
  const auto qx_scaled = quaternion_xyzw[0] / scale;
  const auto qy_scaled = quaternion_xyzw[1] / scale;
  const auto qz_scaled = quaternion_xyzw[2] / scale;
  const auto qw_scaled = quaternion_xyzw[3] / scale;
  const auto inverse_norm = 1.0 / std::sqrt(
    qx_scaled * qx_scaled + qy_scaled * qy_scaled +
    qz_scaled * qz_scaled + qw_scaled * qw_scaled);
  const auto x = qx_scaled * inverse_norm;
  const auto y = qy_scaled * inverse_norm;
  const auto z = qz_scaled * inverse_norm;
  const auto w = qw_scaled * inverse_norm;
  RigidTransform3d result{
    {
      1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
      2.0 * (x * z + y * w),
      2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
      2.0 * (y * z - x * w),
      2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
      1.0 - 2.0 * (x * x + y * y),
    },
    translation,
  };
  validate_transform(result);
  return result;
}

MotionDeskewResult deskew_xyzirt_cloud(
  const sensor_msgs::msg::PointCloud2 & input, const MotionHistory & history,
  const RigidTransform3d & base_from_lidar, const MotionDeskewOptions & options)
{
  try {
    validate_options(options);
    validate_transform(base_from_lidar);
    const XyzirtCloudView view(input);
    if (view.size() > options.maximum_point_count) {
      throw std::invalid_argument("cloud point count exceeds configured maximum");
    }
    if (view.size() == 0U) {
      return {input, MotionDeskewRetryability::kNotApplicable, ""};
    }

    const auto start_sec = scan_start_seconds(input);
    std::vector<double> point_stamps;
    std::vector<Vector3> input_points;
    std::vector<bool> finite_points;
    point_stamps.reserve(view.size());
    input_points.reserve(view.size());
    finite_points.reserve(view.size());
    double maximum_offset = 0.0;
    for (std::size_t index = 0U; index < view.size(); ++index) {
      const auto point = view.point(index);
      if (!std::isfinite(point.time) || point.time < 0.0F) {
        throw std::invalid_argument("point times must be finite nonnegative scan-start offsets");
      }
      const auto offset = static_cast<double>(point.time);
      if (
        offset > static_cast<double>(
          static_cast<float>(options.maximum_scan_duration_sec)))
      {
        throw std::invalid_argument("scan duration exceeds configured maximum");
      }
      const auto finite_xyz =
        std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
      if (!finite_xyz && input.is_dense) {
        throw std::invalid_argument("every point must contain finite XYZ coordinates");
      }
      maximum_offset = std::max(maximum_offset, offset);
      point_stamps.push_back(start_sec + offset);
      input_points.push_back({point.x, point.y, point.z});
      finite_points.push_back(finite_xyz);
    }
    const auto end_sec = start_sec + maximum_offset;
    const auto coverage = history.coverage(
      start_sec, end_sec, options.maximum_wheel_gap_sec,
      options.maximum_imu_gap_sec);
    if (coverage.status != MotionCoverageStatus::kCovered) {
      const auto retryability =
        coverage.status == MotionCoverageStatus::kAwaitingFuture ?
        MotionDeskewRetryability::kRetryable : MotionDeskewRetryability::kPermanent;
      return {std::nullopt, retryability, coverage.reason};
    }

    auto knots = history.knots(start_sec, end_sec);
    knots.insert(knots.end(), point_stamps.begin(), point_stamps.end());
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());

    std::vector<std::size_t> point_time_order;
    point_time_order.reserve(point_stamps.size());
    for (std::size_t index = 0U; index < point_stamps.size(); ++index) {
      point_time_order.push_back(index);
    }
    std::sort(
      point_time_order.begin(), point_time_order.end(),
      [&point_stamps](const std::size_t left, const std::size_t right) {
        if (point_stamps[left] != point_stamps[right]) {
          return point_stamps[left] < point_stamps[right];
        }
        return left < right;
      });

    RigidTransform3d motion;
    std::vector<RigidTransform3d> point_motion(view.size());
    std::size_t ordered_point_index = 0U;
    double current_sec = start_sec;
    for (const auto knot_sec : knots) {
      while (current_sec < knot_sec) {
        const auto step = std::min(
          options.maximum_integration_step_sec, knot_sec - current_sec);
        const auto control = history.interpolate(current_sec + 0.5 * step);
        auto angular_rate = control.angular_velocity_rps;
        if (options.mode == DeskewMode::kTwoDimensional) {
          angular_rate[0] = 0.0;
          angular_rate[1] = 0.0;
        }
        motion = compose(
          motion,
          exponential(control.longitudinal_velocity_mps, angular_rate, step));
        current_sec += step;
        if (knot_sec - current_sec < 1.0e-12) {
          current_sec = knot_sec;
        }
      }
      while (
        ordered_point_index < point_time_order.size() &&
        point_stamps[point_time_order[ordered_point_index]] == knot_sec)
      {
        point_motion[point_time_order[ordered_point_index]] = motion;
        ++ordered_point_index;
      }
    }
    if (ordered_point_index != point_stamps.size()) {
      throw std::runtime_error("internal point-time integration mismatch");
    }

    std::vector<FloatVector3> output_points;
    output_points.reserve(input_points.size());
    for (std::size_t index = 0U; index < input_points.size(); ++index) {
      if (!finite_points[index]) {
        output_points.push_back({});
        continue;
      }
      const auto point_in_base = transform_point(base_from_lidar, input_points[index]);
      const auto point_in_start_base = transform_point(point_motion[index], point_in_base);
      const auto point_in_start_lidar =
        inverse_transform_point(base_from_lidar, point_in_start_base);
      FloatVector3 float_point{};
      for (std::size_t coordinate = 0U; coordinate < float_point.size(); ++coordinate) {
        const auto value = point_in_start_lidar[coordinate];
        if (
          !std::isfinite(value) ||
          std::abs(value) > static_cast<double>(std::numeric_limits<float>::max()))
        {
          throw std::runtime_error("deskew coordinate exceeds finite float range");
        }
        float_point[coordinate] = static_cast<float>(value);
        if (!std::isfinite(float_point[coordinate])) {
          throw std::runtime_error("deskew float conversion produced a nonfinite coordinate");
        }
      }
      output_points.push_back(float_point);
    }

    auto output = input;
    for (std::size_t index = 0U; index < output_points.size(); ++index) {
      if (!finite_points[index]) {
        continue;
      }
      const auto offset = view.point_offset(index);
      write_value(output.data, offset + XyzirtCloudView::kXOffset,
        output_points[index][0]);
      write_value(output.data, offset + XyzirtCloudView::kYOffset,
        output_points[index][1]);
      write_value(output.data, offset + XyzirtCloudView::kZOffset,
        output_points[index][2]);
    }
    return {
      std::move(output), MotionDeskewRetryability::kNotApplicable, ""};
  } catch (const std::exception & error) {
    return failure(error);
  }
}

}  // namespace ad_lidar_perception::preprocessing
