#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ad_control/lateral/path_tracking_controller_factory.hpp"
#include "ad_control/lateral/stanley.hpp"

namespace {
using namespace ad_control;

class RecordingParameters final : public PathTrackingParameterProvider {
public:
  double get_double(const std::string &name, double default_value) override {
    reads.push_back(name);
    const auto found = doubles.find(name);
    return found == doubles.end() ? default_value : found->second;
  }

  int get_int(const std::string &name, int default_value) override {
    reads.push_back(name);
    const auto found = integers.find(name);
    return found == integers.end() ? default_value : found->second;
  }

  std::vector<double>
  get_double_array(const std::string &name,
                   const std::vector<double> &default_value) override {
    reads.push_back(name);
    const auto found = arrays.find(name);
    return found == arrays.end() ? default_value : found->second;
  }

  std::map<std::string, double> doubles;
  std::map<std::string, int> integers;
  std::map<std::string, std::vector<double>> arrays;
  std::vector<std::string> reads;
};

Route line_route() {
  return Route{{{0.0, 0.0, 0.0}, {5.0, 0.0, 0.0}, {10.0, 0.0, 0.0}}, false};
}

void expect_exact_reads(const RecordingParameters &parameters,
                        std::initializer_list<const char *> expected) {
  const std::multiset<std::string> actual(parameters.reads.begin(),
                                          parameters.reads.end());
  const std::multiset<std::string> wanted(expected.begin(), expected.end());
  EXPECT_EQ(actual, wanted);
}

TEST(PathTrackingControllerFactory, ParsesOnlySupportedBackends) {
  EXPECT_EQ(parse_path_tracking_backend("stanley"),
            PathTrackingBackend::kStanley);
  EXPECT_EQ(parse_path_tracking_backend("profile_stanley"),
            PathTrackingBackend::kProfileStanley);
  EXPECT_THROW(parse_path_tracking_backend("pure_pursuit"),
               std::invalid_argument);
  EXPECT_THROW(parse_path_tracking_backend("teb"), std::invalid_argument);
}

TEST(PathTrackingControllerFactory, StanleyReadsExactParameterSet) {
  RecordingParameters parameters;
  auto controller = make_path_tracking_controller(PathTrackingBackend::kStanley,
                                                  line_route(), parameters);

  ASSERT_NE(controller, nullptr);
  expect_exact_reads(parameters,
                     {
                         "stanley.target_speed_mps",
                         "stanley.cross_track_gain",
                         "stanley.speed_softening_mps",
                         "stanley.lookahead_time_s",
                         "stanley.lookahead_min_m",
                         "stanley.lookahead_max_m",
                         "stanley.control_point_x_m",
                         "stanley.heading_error_gain",
                         "maximum_steering_rad",
                         "stanley.forward_window",
                         "stanley.maximum_laps",
                         "stanley.speed_pid.kp",
                         "stanley.speed_pid.ki",
                         "stanley.speed_pid.kd",
                         "stanley.speed_pid.integral_limit",
                         "stanley.speed_pid.derivative_limit",
                         "stanley.speed_pid.derivative_filter_time_constant_s",
                         "stanley.brake_pid.kp",
                         "stanley.brake_pid.ki",
                         "stanley.brake_pid.kd",
                         "stanley.minimum_speed_mps",
                         "stanley.maximum_speed_mps",
                         "stanley.lateral_acceleration_mps2",
                         "stanley.acceleration_mps2",
                         "stanley.deceleration_mps2",
                         "stanley.curvature_window_radius",
                         "stanley.curvature_lookahead_m",
                         "stanley.speed_zones.count",
                         "stanley.launch_speed_mps",
                         "stanley.launch_ramp_s",
                     });
  ASSERT_NE(controller->route_speed_profile(), nullptr);
  EXPECT_FALSE(controller->route_speed_profile()->uses_longitudinal_profile);
  EXPECT_TRUE(controller->update(Pose2{}, 0.0, 0.1, 0, 4).valid);
}

TEST(PathTrackingControllerFactory, ProfileStanleyReadsExactParameterSet) {
  RecordingParameters parameters;
  auto controller = make_path_tracking_controller(
      PathTrackingBackend::kProfileStanley, line_route(), parameters);

  ASSERT_NE(controller, nullptr);
  expect_exact_reads(
      parameters,
      {
          "profile_stanley.target_speed_mps",
          "profile_stanley.cross_track_gain",
          "profile_stanley.speed_softening_mps",
          "profile_stanley.lookahead_time_s",
          "profile_stanley.lookahead_min_m",
          "profile_stanley.lookahead_max_m",
          "profile_stanley.control_point_x_m",
          "profile_stanley.heading_error_gain",
          "maximum_steering_rad",
          "profile_stanley.forward_window",
          "profile_stanley.maximum_laps",
          "profile_stanley.speed_pid.kp",
          "profile_stanley.speed_pid.ki",
          "profile_stanley.speed_pid.kd",
          "profile_stanley.speed_pid.integral_limit",
          "profile_stanley.speed_pid.derivative_limit",
          "profile_stanley.speed_pid.derivative_filter_time_constant_s",
          "profile_stanley.brake_pid.kp",
          "profile_stanley.brake_pid.ki",
          "profile_stanley.brake_pid.kd",
          "profile_stanley.minimum_speed_mps",
          "profile_stanley.maximum_speed_mps",
          "profile_stanley.lateral_acceleration_mps2",
          "profile_stanley.acceleration_mps2",
          "profile_stanley.deceleration_mps2",
          "profile_stanley.curvature_window_radius",
          "profile_stanley.curvature_lookahead_m",
          "profile_stanley.speed_zones.count",
          "profile_stanley.launch_speed_mps",
          "profile_stanley.launch_ramp_s",
          "profile_stanley.longitudinal_profile.speed_mps",
          "profile_stanley.longitudinal_profile.acceleration_mps2",
          "profile_stanley.longitudinal_profile.deceleration_mps2",
          "profile_stanley.longitudinal_profile.braking_delay_s",
      });
  EXPECT_NE(controller->route_speed_profile(), nullptr);
  EXPECT_TRUE(controller->update(Pose2{}, 0.0, 0.1, 0, 4).valid);
}

TEST(PathTrackingControllerFactory, ProfileStanleyReadsSpeedZones) {
  RecordingParameters parameters;
  parameters.integers["profile_stanley.speed_zones.count"] = 1;
  parameters.arrays["profile_stanley.speed_zones.0.start_xy_m"] = {5.0, 0.0};
  parameters.arrays["profile_stanley.speed_zones.0.end_xy_m"] = {5.0, 0.0};
  parameters.doubles[
    "profile_stanley.speed_zones.0.maximum_speed_mps"] = 3.0;

  auto controller = make_path_tracking_controller(
    PathTrackingBackend::kProfileStanley, line_route(), parameters);

  const auto * profile = controller->route_speed_profile();
  ASSERT_NE(profile, nullptr);
  ASSERT_EQ(profile->speed_mps.size(), 3U);
  EXPECT_DOUBLE_EQ(profile->speed_mps[1], 3.0);
}

TEST(PathTrackingControllerFactory,
     UnknownBackendFailsBeforeReadingParameters) {
  RecordingParameters parameters;

  EXPECT_THROW(make_path_tracking_controller("mppi", line_route(), parameters),
               std::invalid_argument);
  EXPECT_TRUE(parameters.reads.empty());
}

TEST(PathTrackingControllerFactory, FactoryStanleyMatchesDirectController) {
  RecordingParameters parameters;
  auto factory = make_path_tracking_controller(PathTrackingBackend::kStanley,
                                               line_route(), parameters);
  StanleyConfig config;
  config.target_speed_mps = 16.25;
  config.cross_track_gain = 0.91;
  config.speed_softening_mps = 2.57;
  config.lookahead_time_s = 0.16;
  config.lookahead_min_m = 1.5;
  config.lookahead_max_m = 5.0;
  config.control_point_x_m = 0.0;
  config.heading_error_gain = 1.0;
  config.max_steer_rad = 0.6981317;
  config.forward_window = 200;
  config.max_laps = 1;
  config.pid =
      PidConfig{1.08, 0.0, 0.036, 100.0, std::numeric_limits<double>::max()};
  config.pid.error_scale = 3.6;
  config.pid.brake_deadband = 1.0;
  config.pid.use_separate_brake_gains = true;
  config.pid.brake_kp = 0.2;
  config.pid.brake_ki = 0.0;
  config.pid.brake_kd = 0.01;
  config.speed_profile = RouteSpeedProfileConfig{
      1.3888888889, 16.6666666667,        6.0, 2.0, 2.0, 5,
      1.0,          LongitudinalProfile{}};
  config.launch_speed_mps = 1.3888888889;
  config.launch_ramp_s = 4.0;
  StanleyController direct(line_route(), config);

  for (int step = 0; step < 5; ++step) {
    const Pose2 pose{0.5 * step, 0.25, 0.02};
    EXPECT_EQ(factory->update(pose, 1.0, 0.05, 0, 4),
              direct.update(pose, 1.0, 0.05, 0, 4));
  }
}
} // namespace
