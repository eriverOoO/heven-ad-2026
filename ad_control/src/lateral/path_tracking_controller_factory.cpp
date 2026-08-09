#include "ad_control/lateral/path_tracking_controller_factory.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ad_control/lateral/stanley.hpp"
#include "ad_control/longitudinal/pid.hpp"

namespace ad_control {
namespace {
constexpr double kMetersPerSecondToKilometersPerHour = 3.6;
constexpr double kBrakeDeadbandKph = 1.0;

std::size_t positive_size(int value, const std::string &name) {
  if (value <= 0) {
    throw std::invalid_argument(name + " must be positive");
  }
  return static_cast<std::size_t>(value);
}

std::vector<RouteSpeedZone>
route_speed_zones(PathTrackingParameterProvider &parameters,
                  const std::string &prefix) {
  const int count = parameters.get_int(prefix + ".speed_zones.count", 0);
  if (count < 0) {
    throw std::invalid_argument(prefix + ".speed_zones.count must be nonnegative");
  }
  std::vector<RouteSpeedZone> zones;
  zones.reserve(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    const std::string zone =
        prefix + ".speed_zones." + std::to_string(index);
    const auto start =
        parameters.get_double_array(zone + ".start_xy_m", {});
    const auto end = parameters.get_double_array(zone + ".end_xy_m", {});
    if (start.size() != 2 || end.size() != 2) {
      throw std::invalid_argument(
          zone + ".start_xy_m and end_xy_m must each contain x and y");
    }
    zones.push_back(RouteSpeedZone{
        Point3{start[0], start[1], 0.0}, Point3{end[0], end[1], 0.0},
        parameters.get_double(zone + ".maximum_speed_mps", 16.25),
        parameters.get_double(zone + ".maximum_projection_distance_m", 5.0)});
  }
  return zones;
}

PidConfig pid_config(PathTrackingParameterProvider &parameters,
                     const std::string &prefix,
                     double default_integral_limit = 10.0,
                     double default_derivative_limit = 10.0,
                     double error_scale = 1.0, double brake_deadband = 0.0) {
  PidConfig config{
      parameters.get_double(prefix + ".kp", 1.08),
      parameters.get_double(prefix + ".ki", 0.0),
      parameters.get_double(prefix + ".kd", 0.036),
      parameters.get_double(prefix + ".integral_limit", default_integral_limit),
      parameters.get_double(prefix + ".derivative_limit",
                            default_derivative_limit)};
  config.error_scale = error_scale;
  config.brake_deadband = brake_deadband;
  config.derivative_filter_time_constant_s =
      parameters.get_double(prefix + ".derivative_filter_time_constant_s", 0.1);
  return config;
}

StanleyConfig stanley_config(PathTrackingParameterProvider &parameters,
                             const std::string &prefix) {
  StanleyConfig config;
  config.target_speed_mps =
      parameters.get_double(prefix + ".target_speed_mps", 16.25);
  config.cross_track_gain =
      parameters.get_double(prefix + ".cross_track_gain", 0.91);
  config.speed_softening_mps =
      parameters.get_double(prefix + ".speed_softening_mps", 2.57);
  config.lookahead_time_s =
      parameters.get_double(prefix + ".lookahead_time_s", 0.16);
  config.lookahead_min_m =
      parameters.get_double(prefix + ".lookahead_min_m", 1.5);
  config.lookahead_max_m =
      parameters.get_double(prefix + ".lookahead_max_m", 5.0);
  config.control_point_x_m =
      parameters.get_double(prefix + ".control_point_x_m", 0.0);
  config.heading_error_gain =
      parameters.get_double(prefix + ".heading_error_gain", 1.0);
  config.max_steer_rad =
      parameters.get_double("maximum_steering_rad", 0.6981317);
  config.forward_window =
      positive_size(parameters.get_int(prefix + ".forward_window", 200),
                    prefix + ".forward_window");
  config.max_laps =
      positive_size(parameters.get_int(prefix + ".maximum_laps", 1),
                    prefix + ".maximum_laps");
  config.pid =
      pid_config(parameters, prefix + ".speed_pid", 100.0, 10.0,
                 kMetersPerSecondToKilometersPerHour, kBrakeDeadbandKph);
  config.pid.use_separate_brake_gains = true;
  config.pid.brake_kp = parameters.get_double(prefix + ".brake_pid.kp", 0.2);
  config.pid.brake_ki = parameters.get_double(prefix + ".brake_pid.ki", 0.0);
  config.pid.brake_kd = parameters.get_double(prefix + ".brake_pid.kd", 0.01);
  config.speed_profile = RouteSpeedProfileConfig{
      parameters.get_double(prefix + ".minimum_speed_mps", 1.3888888889),
      parameters.get_double(prefix + ".maximum_speed_mps", 16.6666666667),
      parameters.get_double(prefix + ".lateral_acceleration_mps2", 6.0),
      parameters.get_double(prefix + ".acceleration_mps2", 2.0),
      parameters.get_double(prefix + ".deceleration_mps2", 2.0),
      positive_size(parameters.get_int(prefix + ".curvature_window_radius", 5),
                    prefix + ".curvature_window_radius"),
      parameters.get_double(prefix + ".curvature_lookahead_m", 1.0),
      LongitudinalProfile{}};
  config.speed_profile.speed_zones = route_speed_zones(parameters, prefix);
  config.launch_speed_mps =
      parameters.get_double(prefix + ".launch_speed_mps", 1.3888888889);
  config.launch_ramp_s = parameters.get_double(prefix + ".launch_ramp_s", 4.0);
  return config;
}

ProfileStanleyConfig
profile_stanley_config(PathTrackingParameterProvider &parameters) {
  const std::string prefix = "profile_stanley";
  ProfileStanleyConfig config;
  config.stanley = stanley_config(parameters, prefix);
  config.longitudinal_profile = LongitudinalProfile{
      parameters.get_double_array(prefix + ".longitudinal_profile.speed_mps",
                                  std::vector<double>{}),
      parameters.get_double_array(prefix +
                                      ".longitudinal_profile.acceleration_mps2",
                                  std::vector<double>{}),
      parameters.get_double_array(prefix +
                                      ".longitudinal_profile.deceleration_mps2",
                                  std::vector<double>{}),
      parameters.get_double(prefix + ".longitudinal_profile.braking_delay_s",
                            0.0)};
  return config;
}

} // namespace

PathTrackingBackend parse_path_tracking_backend(const std::string &backend) {
  if (backend == "stanley") {
    return PathTrackingBackend::kStanley;
  }
  if (backend == "profile_stanley") {
    return PathTrackingBackend::kProfileStanley;
  }
  throw std::invalid_argument(
      "path tracking backend must be 'stanley' or 'profile_stanley'");
}

std::unique_ptr<PathTrackingController>
make_path_tracking_controller(PathTrackingBackend backend, Route route,
                              PathTrackingParameterProvider &parameters) {
  switch (backend) {
  case PathTrackingBackend::kStanley:
    return std::make_unique<StanleyController>(
        std::move(route), stanley_config(parameters, "stanley"));
  case PathTrackingBackend::kProfileStanley:
    return std::make_unique<ProfileStanleyController>(
        std::move(route), profile_stanley_config(parameters));
  }
  throw std::invalid_argument("unknown path tracking backend");
}

std::unique_ptr<PathTrackingController>
make_path_tracking_controller(const std::string &backend, Route route,
                              PathTrackingParameterProvider &parameters) {
  return make_path_tracking_controller(parse_path_tracking_backend(backend),
                                       std::move(route), parameters);
}

} // namespace ad_control
