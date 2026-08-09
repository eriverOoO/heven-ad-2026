#include "ad_planner/local_planning/frenet/frenet_geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace ad_planner
{
namespace
{

constexpr long double kPi = 3.141592653589793238462643383279502884L;
constexpr double kMinimumSegmentLengthM = 1e-12;
constexpr double kMinimumFrenetDenominator = 1e-9;
constexpr double kMinimumVelocityMps = 1e-12;

bool finite(const double value)
{
  return std::isfinite(value);
}

void require_finite(const double value, const char * const name)
{
  if (!finite(value)) {
    throw std::invalid_argument(std::string(name) + " must be finite");
  }
}

double checked_difference(
  const double lhs, const double rhs, const char * const name)
{
  const double result = lhs - rhs;
  require_finite(result, name);
  return result;
}

double checked_sum(const double lhs, const double rhs, const char * const name)
{
  const double result = lhs + rhs;
  require_finite(result, name);
  return result;
}

double wrap_yaw(const double yaw_rad)
{
  require_finite(yaw_rad, "yaw");
  return std::remainder(yaw_rad, static_cast<double>(2.0L * kPi));
}

double angle_difference(const double lhs_rad, const double rhs_rad)
{
  const double lhs_wrapped = wrap_yaw(lhs_rad);
  const double rhs_wrapped = wrap_yaw(rhs_rad);
  return wrap_yaw(lhs_wrapped - rhs_wrapped);
}

void validate_lane(const ReferenceLane & lane)
{
  if (lane.points.size() < 2U) {
    throw std::invalid_argument("reference lane must contain at least two points");
  }

  for (std::size_t index = 0; index < lane.points.size(); ++index) {
    const auto & point = lane.points[index];
    require_finite(point.pose.x, "reference x");
    require_finite(point.pose.y, "reference y");
    require_finite(point.pose.yaw_rad, "reference yaw");
    require_finite(point.route_s_m, "reference route_s_m");
    require_finite(point.curvature_inv_m, "reference curvature");
    require_finite(point.left_width_m, "reference left width");
    require_finite(point.right_width_m, "reference right width");
    require_finite(point.speed_limit_mps, "reference speed limit");

    if (index == 0U) {
      continue;
    }

    const auto & previous = lane.points[index - 1U];
    const double route_delta = checked_difference(
      point.route_s_m, previous.route_s_m, "reference route_s_m delta");
    if (!(route_delta > 0.0)) {
      throw std::invalid_argument("reference route_s_m must be strictly increasing");
    }
    const double segment_x = checked_difference(
      point.pose.x, previous.pose.x, "reference segment x delta");
    const double segment_y = checked_difference(
      point.pose.y, previous.pose.y, "reference segment y delta");
    const double segment_length = std::hypot(
      segment_x, segment_y);
    require_finite(segment_length, "reference segment length");
    if (!(segment_length > kMinimumSegmentLengthM)) {
      throw std::invalid_argument("reference lane contains a zero-length geometry segment");
    }
    static_cast<void>(angle_difference(point.pose.yaw_rad, previous.pose.yaw_rad));
    static_cast<void>(checked_difference(
      point.curvature_inv_m, previous.curvature_inv_m,
      "reference curvature delta"));
  }

  static_cast<void>(checked_difference(
    lane.points.back().route_s_m, lane.points.front().route_s_m,
    "reference route span"));
}

struct ReferenceSample
{
  double x_m{0.0};
  double y_m{0.0};
  double yaw_rad{0.0};
  double curvature_inv_m{0.0};
  double curvature_derivative_inv_m2{0.0};
};

ReferenceSample interpolate_reference(const ReferenceLane & lane, double route_s_m)
{
  const double first_s = lane.points.front().route_s_m;
  const double last_s = lane.points.back().route_s_m;
  const double route_span = checked_difference(last_s, first_s, "reference route span");
  const double tolerance = 1e-10 * std::max(1.0, std::abs(route_span));
  if (route_s_m < first_s) {
    const double gap = first_s - route_s_m;
    if (!finite(gap) || gap > tolerance) {
      throw std::out_of_range("Frenet route progress is outside the reference lane");
    }
    route_s_m = first_s;
  } else if (route_s_m > last_s) {
    const double gap = route_s_m - last_s;
    if (!finite(gap) || gap > tolerance) {
      throw std::out_of_range("Frenet route progress is outside the reference lane");
    }
    route_s_m = last_s;
  }

  const auto upper = std::upper_bound(
    lane.points.begin(), lane.points.end(), route_s_m,
    [](const double value, const ReferencePoint & point) {
      return value < point.route_s_m;
    });
  const std::size_t upper_index =
    upper == lane.points.end() ? lane.points.size() - 1U :
    static_cast<std::size_t>(std::distance(lane.points.begin(), upper));
  const std::size_t lower_index = upper_index - 1U;
  const auto & lower = lane.points[lower_index];
  const auto & next = lane.points[upper_index];
  const double delta_s = checked_difference(
    next.route_s_m, lower.route_s_m, "reference route_s_m delta");
  const double progress_from_lower = checked_difference(
    route_s_m, lower.route_s_m, "reference interpolation progress");
  const double ratio = progress_from_lower / delta_s;
  require_finite(ratio, "reference interpolation ratio");
  const double x_delta = checked_difference(
    next.pose.x, lower.pose.x, "reference segment x delta");
  const double y_delta = checked_difference(
    next.pose.y, lower.pose.y, "reference segment y delta");
  const double yaw_delta = angle_difference(next.pose.yaw_rad, lower.pose.yaw_rad);
  const double curvature_delta = checked_difference(
    next.curvature_inv_m, lower.curvature_inv_m,
    "reference curvature delta");

  ReferenceSample result;
  result.x_m = checked_sum(
    lower.pose.x, ratio * x_delta, "interpolated reference x");
  result.y_m = checked_sum(
    lower.pose.y, ratio * y_delta, "interpolated reference y");
  result.yaw_rad = wrap_yaw(lower.pose.yaw_rad + ratio * yaw_delta);
  result.curvature_inv_m = checked_sum(
    lower.curvature_inv_m, ratio * curvature_delta,
    "interpolated reference curvature");
  result.curvature_derivative_inv_m2 = curvature_delta / delta_s;
  require_finite(
    result.curvature_derivative_inv_m2,
    "reference curvature derivative");
  return result;
}

void validate_frenet_state(const FrenetState & state)
{
  require_finite(state.s_m, "Frenet s");
  require_finite(state.s_dot_mps, "Frenet s_dot");
  require_finite(state.s_ddot_mps2, "Frenet s_ddot");
  require_finite(state.d_m, "Frenet d");
  require_finite(state.d_dot_mps, "Frenet d_dot");
  require_finite(state.d_ddot_mps2, "Frenet d_ddot");
}

void require_valid_denominator(const double denominator)
{
  if (!finite(denominator) || !(denominator > kMinimumFrenetDenominator)) {
    throw std::domain_error("Frenet transform is singular at this lateral offset");
  }
}

template<std::size_t Size>
void require_finite_coefficients(const std::array<long double, Size> & coefficients)
{
  for (const long double coefficient : coefficients) {
    if (!std::isfinite(coefficient) ||
      !std::isfinite(static_cast<double>(coefficient)))
    {
      throw std::invalid_argument("polynomial coefficients are not finite");
    }
  }
}

std::array<long double, 3> solve_three_by_three(
  std::array<std::array<long double, 4>, 3> matrix)
{
  for (std::size_t column = 0; column < 3U; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1U; row < 3U; ++row) {
      if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
        pivot = row;
      }
    }
    if (matrix[pivot][column] == 0.0L || !std::isfinite(matrix[pivot][column])) {
      throw std::invalid_argument("quintic boundary system is singular");
    }
    if (pivot != column) {
      std::swap(matrix[pivot], matrix[column]);
    }

    for (std::size_t row = column + 1U; row < 3U; ++row) {
      const long double scale = matrix[row][column] / matrix[column][column];
      for (std::size_t entry = column; entry < 4U; ++entry) {
        matrix[row][entry] -= scale * matrix[column][entry];
      }
    }
  }

  std::array<long double, 3> solution{};
  for (std::size_t reverse = 0; reverse < 3U; ++reverse) {
    const std::size_t row = 2U - reverse;
    long double residual = matrix[row][3U];
    for (std::size_t column = row + 1U; column < 3U; ++column) {
      residual -= matrix[row][column] * solution[column];
    }
    solution[row] = residual / matrix[row][row];
    if (!std::isfinite(solution[row])) {
      throw std::invalid_argument("quintic boundary system produced a nonfinite result");
    }
  }
  return solution;
}

template<std::size_t Size>
double evaluate_polynomial(
  const std::array<long double, Size> & coefficients, const double t_s,
  const std::size_t derivative_order)
{
  require_finite(t_s, "polynomial time");
  const long double time = static_cast<long double>(t_s);
  long double value = 0.0L;
  for (std::size_t reverse = 0; reverse < Size - derivative_order; ++reverse) {
    const std::size_t power = Size - 1U - reverse;
    long double coefficient = coefficients[power];
    for (std::size_t derivative = 0; derivative < derivative_order; ++derivative) {
      coefficient *= static_cast<long double>(power - derivative);
    }
    value = value * time + coefficient;
  }
  if (!std::isfinite(value) || !std::isfinite(static_cast<double>(value))) {
    throw std::overflow_error("polynomial evaluation produced a nonfinite result");
  }
  return static_cast<double>(value);
}

}  // namespace

QuinticPolynomial::QuinticPolynomial(
  const double p0, const double v0, const double a0,
  const double p1, const double v1, const double a1, const double duration_s)
{
  require_finite(p0, "initial position");
  require_finite(v0, "initial velocity");
  require_finite(a0, "initial acceleration");
  require_finite(p1, "terminal position");
  require_finite(v1, "terminal velocity");
  require_finite(a1, "terminal acceleration");
  require_finite(duration_s, "duration");
  if (!(duration_s > 0.0)) {
    throw std::invalid_argument("duration must be greater than zero");
  }

  const long double duration = static_cast<long double>(duration_s);
  const long double duration2 = duration * duration;
  const long double duration3 = duration2 * duration;
  const long double duration4 = duration3 * duration;
  const long double duration5 = duration4 * duration;
  coefficients_[0] = static_cast<long double>(p0);
  coefficients_[1] = static_cast<long double>(v0);
  coefficients_[2] = static_cast<long double>(a0) / 2.0L;

  const long double position_residual =
    static_cast<long double>(p1) -
    (coefficients_[0] + coefficients_[1] * duration + coefficients_[2] * duration2);
  const long double velocity_residual =
    static_cast<long double>(v1) -
    (coefficients_[1] + 2.0L * coefficients_[2] * duration);
  const long double acceleration_residual =
    static_cast<long double>(a1) - 2.0L * coefficients_[2];
  const auto solution = solve_three_by_three({{
      {{duration3, duration4, duration5, position_residual}},
      {{3.0L * duration2, 4.0L * duration3, 5.0L * duration4, velocity_residual}},
      {{6.0L * duration, 12.0L * duration2, 20.0L * duration3, acceleration_residual}}
    }});
  coefficients_[3] = solution[0];
  coefficients_[4] = solution[1];
  coefficients_[5] = solution[2];
  require_finite_coefficients(coefficients_);
}

double QuinticPolynomial::position(const double t_s) const
{
  return evaluate_polynomial(coefficients_, t_s, 0U);
}

double QuinticPolynomial::velocity(const double t_s) const
{
  return evaluate_polynomial(coefficients_, t_s, 1U);
}

double QuinticPolynomial::acceleration(const double t_s) const
{
  return evaluate_polynomial(coefficients_, t_s, 2U);
}

double QuinticPolynomial::jerk(const double t_s) const
{
  return evaluate_polynomial(coefficients_, t_s, 3U);
}

QuarticPolynomial::QuarticPolynomial(
  const double p0, const double v0, const double a0,
  const double v1, const double a1, const double duration_s)
{
  require_finite(p0, "initial position");
  require_finite(v0, "initial velocity");
  require_finite(a0, "initial acceleration");
  require_finite(v1, "terminal velocity");
  require_finite(a1, "terminal acceleration");
  require_finite(duration_s, "duration");
  if (!(duration_s > 0.0)) {
    throw std::invalid_argument("duration must be greater than zero");
  }

  const long double duration = static_cast<long double>(duration_s);
  const long double duration2 = duration * duration;
  const long double duration3 = duration2 * duration;
  coefficients_[0] = static_cast<long double>(p0);
  coefficients_[1] = static_cast<long double>(v0);
  coefficients_[2] = static_cast<long double>(a0) / 2.0L;

  const long double velocity_residual =
    static_cast<long double>(v1) -
    (coefficients_[1] + 2.0L * coefficients_[2] * duration);
  const long double acceleration_residual =
    static_cast<long double>(a1) - 2.0L * coefficients_[2];
  const long double a = 3.0L * duration2;
  const long double b = 4.0L * duration3;
  const long double c = 6.0L * duration;
  const long double d = 12.0L * duration2;
  const long double determinant = a * d - b * c;
  if (determinant == 0.0L || !std::isfinite(determinant)) {
    throw std::invalid_argument("quartic boundary system is singular");
  }
  coefficients_[3] = (velocity_residual * d - b * acceleration_residual) / determinant;
  coefficients_[4] = (a * acceleration_residual - velocity_residual * c) / determinant;
  require_finite_coefficients(coefficients_);
}

double QuarticPolynomial::position(const double t_s) const
{
  return evaluate_polynomial(coefficients_, t_s, 0U);
}

double QuarticPolynomial::velocity(const double t_s) const
{
  return evaluate_polynomial(coefficients_, t_s, 1U);
}

double QuarticPolynomial::acceleration(const double t_s) const
{
  return evaluate_polynomial(coefficients_, t_s, 2U);
}

double QuarticPolynomial::jerk(const double t_s) const
{
  return evaluate_polynomial(coefficients_, t_s, 3U);
}

FrenetState project_to_frenet(const ReferenceLane & lane, const EgoState & ego)
{
  validate_lane(lane);
  require_finite(ego.pose.x, "ego x");
  require_finite(ego.pose.y, "ego y");
  require_finite(ego.pose.yaw_rad, "ego yaw");
  require_finite(ego.speed_mps, "ego speed");
  require_finite(ego.yaw_rate_radps, "ego yaw rate");

  double closest_distance = std::numeric_limits<double>::infinity();
  double closest_route_s = lane.points.front().route_s_m;
  for (std::size_t index = 1U; index < lane.points.size(); ++index) {
    const auto & start = lane.points[index - 1U];
    const auto & end = lane.points[index];
    const double segment_x = checked_difference(
      end.pose.x, start.pose.x, "reference segment x delta");
    const double segment_y = checked_difference(
      end.pose.y, start.pose.y, "reference segment y delta");
    const double segment_length = std::hypot(segment_x, segment_y);
    require_finite(segment_length, "reference segment length");
    const double unit_x = segment_x / segment_length;
    const double unit_y = segment_y / segment_length;
    require_finite(unit_x, "reference segment unit x");
    require_finite(unit_y, "reference segment unit y");
    const double relative_x = checked_difference(
      ego.pose.x, start.pose.x, "ego relative x");
    const double relative_y = checked_difference(
      ego.pose.y, start.pose.y, "ego relative y");
    const double along_segment = checked_sum(
      relative_x * unit_x, relative_y * unit_y,
      "ego along-segment projection");
    const double projection = std::clamp(
      along_segment / segment_length,
      0.0, 1.0);
    require_finite(projection, "ego segment projection ratio");
    const double x = checked_sum(
      start.pose.x, projection * segment_x, "projected reference x");
    const double y = checked_sum(
      start.pose.y, projection * segment_y, "projected reference y");
    const double residual_x = checked_difference(
      ego.pose.x, x, "ego projection residual x");
    const double residual_y = checked_difference(
      ego.pose.y, y, "ego projection residual y");
    const double distance = std::hypot(residual_x, residual_y);
    require_finite(distance, "ego projection distance");
    if (distance < closest_distance) {
      closest_distance = distance;
      const double route_delta = checked_difference(
        end.route_s_m, start.route_s_m, "reference route_s_m delta");
      closest_route_s = checked_sum(
        start.route_s_m, projection * route_delta,
        "projected route progress");
    }
  }
  require_finite(closest_distance, "closest projection distance");

  const ReferenceSample reference = interpolate_reference(lane, closest_route_s);
  const double cosine = std::cos(reference.yaw_rad);
  const double sine = std::sin(reference.yaw_rad);
  const double offset_x = checked_difference(
    ego.pose.x, reference.x_m, "ego reference offset x");
  const double offset_y = checked_difference(
    ego.pose.y, reference.y_m, "ego reference offset y");
  const double d = checked_sum(
    -sine * offset_x, cosine * offset_y,
    "Frenet lateral offset");
  const double heading_delta = angle_difference(ego.pose.yaw_rad, reference.yaw_rad);
  const double denominator = 1.0 - reference.curvature_inv_m * d;
  require_valid_denominator(denominator);

  FrenetState result;
  result.s_m = closest_route_s;
  result.d_m = d;
  result.s_dot_mps = ego.speed_mps * std::cos(heading_delta) / denominator;
  result.d_dot_mps = ego.speed_mps * std::sin(heading_delta);

  const double tangential_acceleration =
    -ego.speed_mps * ego.yaw_rate_radps * std::sin(heading_delta);
  const double normal_acceleration =
    ego.speed_mps * ego.yaw_rate_radps * std::cos(heading_delta);
  result.s_ddot_mps2 =
    (tangential_acceleration +
    reference.curvature_derivative_inv_m2 * d * result.s_dot_mps * result.s_dot_mps +
    2.0 * reference.curvature_inv_m * result.d_dot_mps * result.s_dot_mps) /
    denominator;
  result.d_ddot_mps2 =
    normal_acceleration -
    reference.curvature_inv_m * denominator * result.s_dot_mps * result.s_dot_mps;
  validate_frenet_state(result);
  return result;
}

TimedTrajectoryPoint frenet_to_cartesian(
  const ReferenceLane & lane, const FrenetState & state, const double time_s)
{
  validate_lane(lane);
  validate_frenet_state(state);
  require_finite(time_s, "trajectory time");
  if (time_s < 0.0) {
    throw std::invalid_argument("trajectory time must not be negative");
  }

  const ReferenceSample reference = interpolate_reference(lane, state.s_m);
  const double denominator = 1.0 - reference.curvature_inv_m * state.d_m;
  require_valid_denominator(denominator);
  const double cosine = std::cos(reference.yaw_rad);
  const double sine = std::sin(reference.yaw_rad);

  const double tangential_velocity = denominator * state.s_dot_mps;
  const double normal_velocity = state.d_dot_mps;
  const double velocity_x = cosine * tangential_velocity - sine * normal_velocity;
  const double velocity_y = sine * tangential_velocity + cosine * normal_velocity;
  const double speed = std::hypot(velocity_x, velocity_y);

  const double tangential_acceleration =
    denominator * state.s_ddot_mps2 -
    reference.curvature_derivative_inv_m2 * state.d_m *
    state.s_dot_mps * state.s_dot_mps -
    2.0 * reference.curvature_inv_m * state.d_dot_mps * state.s_dot_mps;
  const double normal_acceleration =
    reference.curvature_inv_m * denominator * state.s_dot_mps * state.s_dot_mps +
    state.d_ddot_mps2;
  const double acceleration_x =
    cosine * tangential_acceleration - sine * normal_acceleration;
  const double acceleration_y =
    sine * tangential_acceleration + cosine * normal_acceleration;

  TimedTrajectoryPoint result;
  result.pose.x = reference.x_m - sine * state.d_m;
  result.pose.y = reference.y_m + cosine * state.d_m;
  result.pose.yaw_rad = speed > kMinimumVelocityMps ?
    wrap_yaw(std::atan2(velocity_y, velocity_x)) : reference.yaw_rad;
  result.time_from_start_s = time_s;
  result.speed_mps = speed;
  if (speed > kMinimumVelocityMps) {
    result.curvature_inv_m =
      (velocity_x * acceleration_y - velocity_y * acceleration_x) /
      (speed * speed * speed);
  } else {
    result.curvature_inv_m = reference.curvature_inv_m / denominator;
  }

  require_finite(result.pose.x, "Cartesian x");
  require_finite(result.pose.y, "Cartesian y");
  require_finite(result.pose.yaw_rad, "Cartesian yaw");
  require_finite(result.speed_mps, "Cartesian speed");
  require_finite(result.curvature_inv_m, "Cartesian curvature");
  return result;
}

}  // namespace ad_planner
