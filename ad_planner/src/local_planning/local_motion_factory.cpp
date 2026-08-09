#include "ad_planner/local_planning/local_motion_factory.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "ad_control/longitudinal/pid.hpp"
#include "ad_planner/common/parameter_validation.hpp"
#include "ad_planner/local_planning/dwa/dwa.hpp"
#include "ad_planner/local_planning/dwa/dwa_backend.hpp"
#include "ad_planner/local_planning/frenet/frenet_lattice.hpp"
#include "ad_planner/local_planning/mppi_nav2/mppi_nav2_backend.hpp"

namespace ad_planner
{

namespace
{

DwaConfig make_dwa_config(LocalMotionParameterProvider & parameters)
{
  const int footprint_occupied_threshold = parameters.get_int(
    "dwa.footprint.occupied_threshold", 20);
  if (footprint_occupied_threshold < 0 || footprint_occupied_threshold > 100) {
    throw std::invalid_argument(
            "dwa.footprint.occupied_threshold must be between 0 and 100");
  }

  const FootprintConfig footprint{
    parameters.get_double("dwa.footprint.half_length_m", 2.3175),
    parameters.get_double("dwa.footprint.half_width_m", 0.945),
    parameters.get_double("dwa.footprint.clearance_m", 0.20),
    static_cast<std::int8_t>(footprint_occupied_threshold),
    positive_size_parameter(
      parameters.get_int("dwa.footprint.maximum_cells_to_check", 8192),
      "dwa.footprint.maximum_cells_to_check"),
    parameters.get_double("dwa.footprint.center_offset_x_m", 1.5275)};

  ad_control::PidConfig pid{
    parameters.get_double("dwa.speed_pid.kp", 0.3),
    parameters.get_double("dwa.speed_pid.ki", 0.0),
    parameters.get_double("dwa.speed_pid.kd", 0.01),
    parameters.get_double("dwa.speed_pid.integral_limit", 10.0),
    parameters.get_double("dwa.speed_pid.derivative_limit", 10.0)};
  pid.brake_deadband =
    parameters.get_double("dwa.speed_pid.brake_deadband_mps", 0.10);
  pid.use_separate_brake_gains = true;
  pid.brake_kp = parameters.get_double("dwa.brake_pid.kp", 0.25);
  pid.brake_ki = parameters.get_double("dwa.brake_pid.ki", 0.0);
  pid.brake_kd = parameters.get_double("dwa.brake_pid.kd", 0.01);
  pid.derivative_filter_time_constant_s = parameters.get_double(
    "dwa.speed_pid.derivative_filter_time_constant_s", 0.10);

  DwaConfig config;
  config.min_speed_mps =
    parameters.get_double("dwa.minimum_speed_mps", 0.0);
  config.max_speed_mps =
    parameters.get_double("dwa.maximum_speed_mps", 16.25);
  config.speed_step_mps =
    parameters.get_double("dwa.speed_step_mps", 1.0);
  config.min_steer_rad =
    parameters.get_double("dwa.minimum_steering_rad", -0.52);
  config.sampled_max_steer_rad =
    parameters.get_double("dwa.maximum_steering_rad", 0.52);
  config.steer_step_rad =
    parameters.get_double("dwa.steering_step_rad", 0.04);
  config.dt = parameters.get_double("dwa.simulation_dt", 0.2);
  config.horizon_s = parameters.get_double("dwa.horizon_sec", 1.5);
  config.wheelbase_m = parameters.get_double("dwa.wheelbase_m", 3.0);
  config.max_steer_rad =
    parameters.get_double("maximum_steering_rad", 0.6981317);
  config.control_period_s =
    parameters.get_double("control_period_sec", 0.05);
  config.dynamic_window_time_s =
    parameters.get_double("dwa.dynamic_window_time_sec", 0.50);
  config.maximum_acceleration_mps2 =
    parameters.get_double("dwa.maximum_acceleration_mps2", 5.0);
  config.maximum_deceleration_mps2 =
    parameters.get_double("dwa.maximum_deceleration_mps2", 1.8);
  config.emergency_deceleration_mps2 =
    parameters.get_double("dwa.emergency_deceleration_mps2", 6.0);
  config.initial_inflation_escape_s =
    parameters.get_double("dwa.initial_inflation_escape_sec", 0.6);
  config.maximum_steering_rate_radps =
    parameters.get_double("dwa.maximum_steering_rate_radps", 2.0943951023931953);
  config.maximum_lateral_acceleration_mps2 = parameters.get_double(
    "dwa.maximum_lateral_acceleration_mps2", 6.0);
  config.clearance_saturation_m =
    parameters.get_double("dwa.clearance_saturation_m", 8.0);
  config.maximum_path_distance_m =
    parameters.get_double("dwa.maximum_path_distance_m", 4.5);
  config.prediction_covariance_sigma =
    parameters.get_double("dwa.prediction.covariance_sigma", 2.0);
  config.prediction_minimum_margin_m =
    parameters.get_double("dwa.prediction.minimum_margin_m", 0.20);
  config.footprint = footprint;
  config.progress_weight =
    parameters.get_double("dwa.progress_weight", 1.0);
  config.goal_weight = parameters.get_double("dwa.goal_weight", 0.5);
  config.heading_weight =
    parameters.get_double("dwa.heading_weight", 1.0);
  config.clearance_weight =
    parameters.get_double("dwa.clearance_weight", 2.0);
  config.smoothness_weight =
    parameters.get_double("dwa.smoothness_weight", 0.15);
  config.path_distance_weight =
    parameters.get_double("dwa.path_distance_weight", 1.5);
  config.speed_weight = parameters.get_double("dwa.speed_weight", 0.5);
  config.pid = pid;
  return config;
}

FrenetLatticeConfig make_frenet_lattice_config(
  LocalMotionParameterProvider & parameters)
{
  const int occupied_threshold = parameters.get_int(
    "frenet_lattice.occupied_threshold", 100);
  if (occupied_threshold < 0 || occupied_threshold > 100) {
    throw std::invalid_argument(
            "frenet_lattice.occupied_threshold must be between 0 and 100");
  }

  FrenetLatticeConfig config;
  config.lateral_targets_m = parameters.get_double_array(
    "frenet_lattice.lateral_targets_m", {-1.0, 0.0, 1.0});
  config.target_speeds_mps = parameters.get_double_array(
    "frenet_lattice.target_speeds_mps", {2.0, 4.0, 6.0});
  config.durations_s = parameters.get_double_array(
    "frenet_lattice.durations_s", {2.0, 3.0, 4.0});
  config.sample_dt_s = parameters.get_double(
    "frenet_lattice.sample_dt_s", 0.1);
  config.maximum_curvature_inv_m = parameters.get_double(
    "frenet_lattice.maximum_curvature_inv_m", 0.2);
  config.maximum_acceleration_mps2 = parameters.get_double(
    "frenet_lattice.maximum_acceleration_mps2", 3.0);
  config.maximum_lateral_acceleration_mps2 = parameters.get_double(
    "frenet_lattice.maximum_lateral_acceleration_mps2", 3.0);
  config.maximum_jerk_mps3 = parameters.get_double(
    "frenet_lattice.maximum_jerk_mps3", 5.0);
  config.maximum_lateral_transition_m = parameters.get_double(
    "frenet_lattice.maximum_lateral_transition_m", 5.0);
  config.footprint_clearance_m = parameters.get_double(
    "frenet_lattice.footprint_clearance_m", 0.2);
  config.occupied_threshold = static_cast<std::int8_t>(occupied_threshold);
  config.maximum_cells_to_check = positive_size_parameter(
    parameters.get_int("frenet_lattice.maximum_cells_to_check", 4096),
    "frenet_lattice.maximum_cells_to_check");
  config.progress_weight = parameters.get_double(
    "frenet_lattice.progress_weight", 1.0);
  config.clearance_weight = parameters.get_double(
    "frenet_lattice.clearance_weight", 1.0);
  config.jerk_weight = parameters.get_double(
    "frenet_lattice.jerk_weight", 0.1);
  config.lateral_offset_weight = parameters.get_double(
    "frenet_lattice.lateral_offset_weight", 0.2);
  config.lane_change_weight = parameters.get_double(
    "frenet_lattice.lane_change_weight", 1.0);
  config.continuity_weight = parameters.get_double(
    "frenet_lattice.continuity_weight", 1.0);
  config.maximum_candidates = positive_size_parameter(
    parameters.get_int("frenet_lattice.maximum_candidates", 512),
    "frenet_lattice.maximum_candidates");
  return config;
}

MppiNav2BackendConfig make_mppi_nav2_config(
  LocalMotionParameterProvider & parameters)
{
  MppiNav2BackendConfig config;
  config.command_timeout_s = parameters.get_double(
    "mppi_nav2.command_timeout_s", 0.20);
  config.diagnostic_rollout_dt_s = parameters.get_double(
    "mppi_nav2.diagnostic_rollout_dt_s", 0.10);
  config.diagnostic_rollout_horizon_s = parameters.get_double(
    "mppi_nav2.diagnostic_rollout_horizon_s", 3.0);
  config.command = ad_control::MppiCommandConfig{
    parameters.get_double("mppi_nav2.wheelbase_m", 3.0),
    parameters.get_double(
      "mppi_nav2.maximum_road_wheel_angle_rad", 0.588),
    parameters.get_double(
      "mppi_nav2.steering_rate_limit_rad_s", 0.35),
    parameters.get_double("mppi_nav2.near_zero_speed_mps", 0.05),
    false,
    ad_control::PidConfig{
      parameters.get_double("speed_pid.kp", 0.3),
      parameters.get_double("speed_pid.ki", 0.0),
      parameters.get_double("speed_pid.kd", 0.01),
      parameters.get_double("speed_pid.integral_limit", 10.0),
      parameters.get_double("speed_pid.derivative_limit", 10.0)}};
  return config;
}

}  // namespace

LocalMotionBackendKind parse_local_motion_backend(const std::string & backend)
{
  if (backend == "dwa") {
    return LocalMotionBackendKind::kDwa;
  }
  if (backend == "frenet_lattice") {
    return LocalMotionBackendKind::kFrenetLattice;
  }
  if (backend == "mppi_nav2") {
    return LocalMotionBackendKind::kMppiNav2;
  }
  throw std::invalid_argument(
          "unsupported local_motion.backend '" + backend +
          "'; supported backends: dwa, frenet_lattice, mppi_nav2");
}

std::unique_ptr<LocalMotionBackend> make_local_motion_backend(
  LocalMotionBackendKind backend, LocalMotionParameterProvider & parameters)
{
  switch (backend) {
    case LocalMotionBackendKind::kDwa:
      return std::make_unique<DwaBackend>(make_dwa_config(parameters));
    case LocalMotionBackendKind::kFrenetLattice:
      return std::make_unique<FrenetLatticeBackend>(
        make_frenet_lattice_config(parameters));
    case LocalMotionBackendKind::kMppiNav2:
      return std::make_unique<MppiNav2Backend>(
        make_mppi_nav2_config(parameters));
  }
  throw std::invalid_argument("unsupported local motion backend enum");
}

std::unique_ptr<LocalMotionBackend> make_local_motion_backend(
  const std::string & backend, LocalMotionParameterProvider & parameters)
{
  return make_local_motion_backend(parse_local_motion_backend(backend), parameters);
}

}  // namespace ad_planner
