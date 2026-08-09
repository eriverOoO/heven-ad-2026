#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/local_motion_factory.hpp"

namespace
{
using namespace ad_planner;

class RecordingParameterProvider final : public LocalMotionParameterProvider
{
public:
  double get_double(const std::string & name, double default_value) override
  {
    accessed.push_back(name);
    double_defaults[name] = default_value;
    const auto override = doubles.find(name);
    return override == doubles.end() ? default_value : override->second;
  }

  int get_int(const std::string & name, int default_value) override
  {
    accessed.push_back(name);
    int_defaults[name] = default_value;
    const auto override = integers.find(name);
    return override == integers.end() ? default_value : override->second;
  }

  std::vector<double> get_double_array(
    const std::string & name, const std::vector<double> & default_value) override
  {
    accessed.push_back(name);
    double_array_defaults[name] = default_value;
    const auto override = double_arrays.find(name);
    return override == double_arrays.end() ? default_value : override->second;
  }

  std::map<std::string, double> doubles;
  std::map<std::string, int> integers;
  std::map<std::string, std::vector<double>> double_arrays;
  std::map<std::string, double> double_defaults;
  std::map<std::string, int> int_defaults;
  std::map<std::string, std::vector<double>> double_array_defaults;
  std::vector<std::string> accessed;
};

RecordingParameterProvider dwa_parameters()
{
  RecordingParameterProvider parameters;
  parameters.doubles = {
    {"dwa.minimum_speed_mps", 1.0},
    {"dwa.maximum_speed_mps", 1.0},
    {"dwa.speed_step_mps", 1.0},
    {"dwa.minimum_steering_rad", 0.0},
    {"dwa.maximum_steering_rad", 0.0},
    {"dwa.steering_step_rad", 0.1},
    {"dwa.simulation_dt", 0.5},
    {"dwa.horizon_sec", 1.0},
    {"dwa.wheelbase_m", 1.0},
    {"maximum_steering_rad", 0.4},
    {"dwa.footprint.half_length_m", 0.2},
    {"dwa.footprint.half_width_m", 0.2},
    {"dwa.footprint.clearance_m", 0.0},
    {"dwa.footprint.center_offset_x_m", 0.0},
    {"dwa.speed_pid.kp", 0.5},
    {"dwa.speed_pid.ki", 0.0},
    {"dwa.speed_pid.kd", 0.0},
    {"dwa.speed_pid.integral_limit", 1.0},
    {"dwa.speed_pid.derivative_limit", 1.0},
  };
  parameters.integers = {
    {"dwa.footprint.occupied_threshold", 50},
    {"dwa.footprint.maximum_cells_to_check", 64},
  };
  return parameters;
}

TEST(LocalMotionBackendFactory, DwaBackendMaterializesItsParametersBehindTheFactory)
{
  auto parameters = dwa_parameters();

  const auto planner = make_local_motion_backend(
    LocalMotionBackendKind::kDwa, parameters);

  ASSERT_NE(planner, nullptr);
  const std::map<std::string, double> expected_double_defaults{
    {"control_period_sec", 0.05},
    {"dwa.minimum_speed_mps", 0.0},
    {"dwa.maximum_speed_mps", 16.25},
    {"dwa.speed_step_mps", 1.0},
    {"dwa.minimum_steering_rad", -0.52},
    {"dwa.maximum_steering_rad", 0.52},
    {"dwa.steering_step_rad", 0.04},
    {"dwa.simulation_dt", 0.2},
    {"dwa.horizon_sec", 1.5},
    {"dwa.dynamic_window_time_sec", 0.5},
    {"dwa.maximum_acceleration_mps2", 5.0},
    {"dwa.maximum_deceleration_mps2", 1.8},
    {"dwa.emergency_deceleration_mps2", 6.0},
    {"dwa.initial_inflation_escape_sec", 0.6},
    {"dwa.maximum_steering_rate_radps", 2.0943951023931953},
    {"dwa.maximum_lateral_acceleration_mps2", 6.0},
    {"dwa.clearance_saturation_m", 8.0},
    {"dwa.maximum_path_distance_m", 4.5},
    {"dwa.prediction.covariance_sigma", 2.0},
    {"dwa.prediction.minimum_margin_m", 0.20},
    {"dwa.wheelbase_m", 3.0},
    {"maximum_steering_rad", 0.6981317},
    {"dwa.footprint.half_length_m", 2.3175},
    {"dwa.footprint.half_width_m", 0.945},
    {"dwa.footprint.clearance_m", 0.20},
    {"dwa.footprint.center_offset_x_m", 1.5275},
    {"dwa.progress_weight", 1.0},
    {"dwa.goal_weight", 0.5},
    {"dwa.heading_weight", 1.0},
    {"dwa.clearance_weight", 2.0},
    {"dwa.smoothness_weight", 0.15},
    {"dwa.path_distance_weight", 1.5},
    {"dwa.speed_weight", 0.5},
    {"dwa.speed_pid.kp", 0.3},
    {"dwa.speed_pid.ki", 0.0},
    {"dwa.speed_pid.kd", 0.01},
    {"dwa.speed_pid.integral_limit", 10.0},
    {"dwa.speed_pid.derivative_limit", 10.0},
    {"dwa.speed_pid.derivative_filter_time_constant_s", 0.10},
    {"dwa.speed_pid.brake_deadband_mps", 0.10},
    {"dwa.brake_pid.kp", 0.25},
    {"dwa.brake_pid.ki", 0.0},
    {"dwa.brake_pid.kd", 0.01},
  };
  const std::map<std::string, int> expected_int_defaults{
    {"dwa.footprint.occupied_threshold", 20},
    {"dwa.footprint.maximum_cells_to_check", 8192},
  };
  EXPECT_EQ(parameters.double_defaults, expected_double_defaults);
  EXPECT_EQ(parameters.int_defaults, expected_int_defaults);
}

TEST(LocalMotionBackendFactory, DwaBackendOwnsFootprintParameterValidation)
{
  auto parameters = dwa_parameters();
  parameters.integers["dwa.footprint.occupied_threshold"] = 101;

  EXPECT_THROW(
    static_cast<void>(make_local_motion_backend("dwa", parameters)),
    std::invalid_argument);
}

TEST(LocalMotionBackendFactory, FrenetReadsOnlyItsOwnParameterNamespace)
{
  RecordingParameterProvider parameters;
  parameters.double_arrays = {
    {"frenet_lattice.lateral_targets_m", {0.0}},
    {"frenet_lattice.target_speeds_mps", {2.0}},
    {"frenet_lattice.durations_s", {2.0}},
  };

  const auto backend = make_local_motion_backend(
    LocalMotionBackendKind::kFrenetLattice, parameters);

  ASSERT_NE(backend, nullptr);
  ASSERT_FALSE(parameters.accessed.empty());
  for (const auto & name : parameters.accessed) {
    EXPECT_EQ(name.rfind("frenet_lattice.", 0U), 0U) << name;
  }
  EXPECT_TRUE(parameters.double_array_defaults.count(
      "frenet_lattice.lateral_targets_m"));
  EXPECT_TRUE(parameters.double_array_defaults.count(
      "frenet_lattice.target_speeds_mps"));
  EXPECT_TRUE(parameters.double_array_defaults.count(
      "frenet_lattice.durations_s"));
}

TEST(LocalMotionBackendFactory, DwaNeverReadsFrenetParameters)
{
  auto parameters = dwa_parameters();

  const auto backend = make_local_motion_backend(LocalMotionBackendKind::kDwa, parameters);

  ASSERT_NE(backend, nullptr);
  for (const auto & name : parameters.accessed) {
    EXPECT_NE(name.rfind("frenet_lattice.", 0U), 0U) << name;
  }
  EXPECT_TRUE(parameters.double_array_defaults.empty());
}

TEST(LocalMotionBackendFactory, MppiNav2ReadsOnlyItsNamespaceAndSharedPid)
{
  RecordingParameterProvider parameters;

  const auto backend = make_local_motion_backend(
    LocalMotionBackendKind::kMppiNav2, parameters);

  ASSERT_NE(backend, nullptr);
  const std::map<std::string, double> expected_double_defaults{
    {"mppi_nav2.command_timeout_s", 0.20},
    {"mppi_nav2.diagnostic_rollout_dt_s", 0.10},
    {"mppi_nav2.diagnostic_rollout_horizon_s", 3.0},
    {"mppi_nav2.wheelbase_m", 3.0},
    {"mppi_nav2.maximum_road_wheel_angle_rad", 0.588},
    {"mppi_nav2.steering_rate_limit_rad_s", 0.35},
    {"mppi_nav2.near_zero_speed_mps", 0.05},
    {"speed_pid.kp", 0.3},
    {"speed_pid.ki", 0.0},
    {"speed_pid.kd", 0.01},
    {"speed_pid.integral_limit", 10.0},
    {"speed_pid.derivative_limit", 10.0},
  };
  EXPECT_EQ(parameters.double_defaults, expected_double_defaults);
  EXPECT_TRUE(parameters.int_defaults.empty());
  EXPECT_TRUE(parameters.double_array_defaults.empty());
  for (const auto & name : parameters.accessed) {
    EXPECT_TRUE(
      name.rfind("mppi_nav2.", 0U) == 0U ||
      name.rfind("speed_pid.", 0U) == 0U) << name;
  }
}

}  // namespace
