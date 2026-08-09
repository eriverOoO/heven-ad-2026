#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/dwa/dwa.hpp"

namespace {
using namespace ad_planner;

static_assert(!std::is_polymorphic_v<DwaController>);

OccupancyGrid free_grid() {
  OccupancyGrid grid;
  grid.origin = Pose2{-10.0, -10.0, 0.0};
  grid.resolution = 0.25;
  grid.width = 80;
  grid.height = 80;
  grid.cells.assign(grid.width * grid.height, 0);
  grid.valid = true;
  grid.fresh = true;
  return grid;
}

OccupancyGrid drivable_strip(const OccupancyGrid &geometry, double minimum_y,
                             double maximum_y) {
  OccupancyGrid mask = geometry;
  std::fill(mask.cells.begin(), mask.cells.end(),
            static_cast<std::int8_t>(100));
  for (std::size_t y = 0; y < mask.height; ++y) {
    for (std::size_t x = 0; x < mask.width; ++x) {
      const double local_x = (static_cast<double>(x) + 0.5) * mask.resolution;
      const double local_y = (static_cast<double>(y) + 0.5) * mask.resolution;
      const double cosine = std::cos(mask.origin.yaw_rad);
      const double sine = std::sin(mask.origin.yaw_rad);
      const double world_y = mask.origin.y + sine * local_x + cosine * local_y;
      if (world_y >= minimum_y && world_y <= maximum_y) {
        mask.cells[y * mask.width + x] = 0;
      }
    }
  }
  return mask;
}

OccupancyGrid local_grid_with_centerline_obstacle() {
  OccupancyGrid grid;
  grid.origin = Pose2{-4.0, -10.0, 0.0};
  grid.resolution = 0.1;
  grid.width = 280;
  grid.height = 200;
  grid.cells.assign(grid.width * grid.height, 0);
  grid.valid = true;
  grid.fresh = true;

  const auto obstacle = world_to_cell(grid, Point3{12.0, 0.05, 0.0});
  EXPECT_TRUE(obstacle.has_value());
  if (obstacle) {
    grid.cells[obstacle->y * grid.width + obstacle->x] = 100;
  }
  return grid;
}

OccupancyGrid local_grid_with_close_offset_obstacle() {
  OccupancyGrid grid;
  grid.origin = Pose2{-4.0, -10.0, 0.0};
  grid.resolution = 0.1;
  grid.width = 280;
  grid.height = 200;
  grid.cells.assign(grid.width * grid.height, 0);
  grid.valid = true;
  grid.fresh = true;

  // Reproduces the nearest hard-occupied cells observed in the live MORAI OGM.
  for (double y = 0.25; y <= 1.75; y += 0.1) {
    const auto obstacle = world_to_cell(grid, Point3{5.15, y, 0.0});
    EXPECT_TRUE(obstacle.has_value());
    if (obstacle) {
      grid.cells[obstacle->y * grid.width + obstacle->x] = 100;
    }
  }
  return grid;
}

OccupancyGrid long_range_grid() {
  OccupancyGrid grid;
  grid.origin = Pose2{-4.0, -10.0, 0.0};
  grid.resolution = 0.2;
  grid.width = 520;
  grid.height = 100;
  grid.cells.assign(grid.width * grid.height, 0);
  grid.valid = true;
  grid.fresh = true;
  return grid;
}

OccupancyGrid long_range_grid_with_cross_track_barrier(double obstacle_x) {
  auto grid = long_range_grid();
  for (double y = -4.5; y <= 4.5; y += grid.resolution) {
    const auto obstacle = world_to_cell(grid, Point3{obstacle_x, y, 0.0});
    EXPECT_TRUE(obstacle.has_value());
    if (obstacle) {
      grid.cells[obstacle->y * grid.width + obstacle->x] = 100;
    }
  }
  return grid;
}

OccupancyGrid long_range_grid_with_center_box(double obstacle_x) {
  auto grid = long_range_grid();
  for (double x = obstacle_x - 0.5; x <= obstacle_x + 0.5;
       x += grid.resolution) {
    for (double y = -0.5; y <= 0.5; y += grid.resolution) {
      const auto obstacle = world_to_cell(grid, Point3{x, y, 0.0});
      EXPECT_TRUE(obstacle.has_value());
      if (obstacle) {
        grid.cells[obstacle->y * grid.width + obstacle->x] = 100;
      }
    }
  }
  return grid;
}

OccupancyGrid fine_grid_with_obstacle(double obstacle_x) {
  OccupancyGrid grid;
  grid.origin = Pose2{-2.0, -2.0, 0.0};
  grid.resolution = 0.05;
  grid.width = 500;
  grid.height = 80;
  grid.cells.assign(grid.width * grid.height, 0);
  grid.valid = true;
  grid.fresh = true;
  const auto obstacle = world_to_cell(grid, Point3{obstacle_x, 0.025, 0.0});
  EXPECT_TRUE(obstacle.has_value());
  if (obstacle) {
    grid.cells[obstacle->y * grid.width + obstacle->x] = 100;
  }
  return grid;
}

DwaConfig config(double steer = 0.0) {
  DwaConfig value;
  value.min_speed_mps = 1.0;
  value.max_speed_mps = 1.0;
  value.speed_step_mps = 1.0;
  value.min_steer_rad = steer;
  value.sampled_max_steer_rad = steer;
  value.steer_step_rad = 0.1;
  value.dt = 0.5;
  value.horizon_s = 1.0;
  value.wheelbase_m = 1.0;
  value.max_steer_rad = 0.4;
  value.footprint = FootprintConfig{0.2, 0.2, 0.0, 50, 64};
  value.progress_weight = 1.0;
  value.goal_weight = 1.0;
  value.heading_weight = 1.0;
  value.clearance_weight = 1.0;
  value.smoothness_weight = 1.0;
  value.pid = PidConfig{0.5, 0.0, 0.0, 1.0, 1.0};
  return value;
}

DwaConfig vehicle_config() {
  DwaConfig value;
  value.min_speed_mps = 0.0;
  value.max_speed_mps = 16.25;
  value.speed_step_mps = 1.0;
  value.min_steer_rad = -0.52;
  value.sampled_max_steer_rad = 0.52;
  value.steer_step_rad = 0.04;
  value.dt = 0.2;
  value.horizon_s = 1.5;
  value.wheelbase_m = 3.0;
  value.max_steer_rad = 0.6981317;
  value.control_period_s = 0.05;
  value.dynamic_window_time_s = 0.5;
  value.maximum_acceleration_mps2 = 5.0;
  value.maximum_deceleration_mps2 = 2.0;
  value.maximum_steering_rate_radps = 2.0943951023931953;
  value.maximum_lateral_acceleration_mps2 = 6.0;
  value.footprint = FootprintConfig{2.3175, 0.945, 0.2, 20, 8192, 1.5275};
  value.progress_weight = 1.0;
  value.goal_weight = 0.5;
  value.heading_weight = 1.0;
  value.clearance_weight = 2.0;
  value.smoothness_weight = 0.15;
  value.path_distance_weight = 1.5;
  value.speed_weight = 0.5;
  value.pid = PidConfig{0.3, 0.0, 0.01, 10.0, 10.0};
  return value;
}

TEST(DwaController,
     SamplesSiSpeedAndSteeringAndReturnsPhysicalCommandAndTrajectory) {
  DwaController controller(config());
  const auto result = controller.plan(free_grid(), Pose2{0.0, 0.0, 0.0},
                                      Point3{5.0, 0.0, 0.0}, 0.0, 0.0, 1, 4);
  ASSERT_TRUE(result.valid);
  EXPECT_DOUBLE_EQ(result.command.accel, 0.5);
  EXPECT_DOUBLE_EQ(result.command.brake, 0.0);
  EXPECT_DOUBLE_EQ(result.command.steering_rad, 0.0);
  ASSERT_TRUE(result.local_trajectory.has_value());
  ASSERT_EQ(result.local_trajectory->poses.size(), 2U);
  EXPECT_NEAR(result.local_trajectory->poses.back().x, 0.75, 1e-12);
  EXPECT_EQ(controller.candidate_trajectories().size(), 1U);
}

TEST(DwaController, UsesAverageSpeedForConservativeCompleteStopAdmission) {
  auto value = config();
  value.min_speed_mps = 1.0;
  value.max_speed_mps = 1.0;
  value.dt = 0.5;
  value.horizon_s = 1.0;
  value.emergency_deceleration_mps2 = value.maximum_deceleration_mps2;
  value.footprint.maximum_cells_to_check = 256;
  DwaController controller(value);

  const auto result = controller.plan(fine_grid_with_obstacle(5.1), Pose2{},
                                      Point3{20.0, 0.0, 0.0}, 4.0, 0.0, 1, 4);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "no safe DWA candidate");
}

TEST(DwaController, SimulatesAckermannBicycleWithSteeringRadians) {
  const std::array<double, 2> steering_cases{{-0.4, 0.4}};
  for (const double steering : steering_cases) {
    DwaController controller(config(steering));
    const auto result = controller.plan(free_grid(), Pose2{0.0, 0.0, 0.0},
                                        Point3{5.0, 2.0, 0.0}, 0.0, 0.0, 1, 4);
    ASSERT_TRUE(result.valid);
    ASSERT_TRUE(result.local_trajectory.has_value());
    const double expected_yaw =
        0.25 * std::tan(0.5 * steering) + 0.5 * std::tan(steering);
    EXPECT_NEAR(result.local_trajectory->poses.back().yaw_rad, expected_yaw,
                1e-12);
    EXPECT_NEAR(result.command.steering_rad,
                std::copysign(vehicle_config().maximum_steering_rate_radps *
                                  vehicle_config().control_period_s,
                              steering),
                1e-12);
  }
}

TEST(DwaController,
     FindsSafeResponseToCenterlineObstacleWithRuntimeVehicleGeometry) {
  const auto grid = local_grid_with_centerline_obstacle();
  DwaController controller(vehicle_config());

  const auto result =
      controller.plan(grid, Pose2{}, Point3{20.0, 0.0, 0.0}, 4.7, 0.0, 1, 4);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_TRUE(result.target_speed_mps.has_value());
  EXPECT_TRUE(std::abs(result.command.steering_rad) > 1.0e-9 ||
              *result.target_speed_mps < 4.7);
  ASSERT_TRUE(result.local_trajectory.has_value());
  for (const auto &pose : result.local_trajectory->poses) {
    EXPECT_TRUE(footprint_is_safe(grid, pose, vehicle_config().footprint));
  }
}

TEST(DwaController, LateObstacleStillReturnsOnlyCollisionFreeStopOrAvoidance) {
  const auto grid = local_grid_with_close_offset_obstacle();
  DwaController controller(vehicle_config());

  const auto result =
      controller.plan(grid, Pose2{}, Point3{8.0, 0.0, 0.0}, 0.0, 0.0, 1, 4);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_TRUE(result.local_trajectory.has_value());
  for (const auto &pose : result.local_trajectory->poses) {
    EXPECT_TRUE(footprint_is_safe(grid, pose, vehicle_config().footprint));
  }
}

TEST(DwaController, PrefersSafeForwardEscapeOverStationaryClearanceMaximum) {
  auto value = vehicle_config();
  value.max_speed_mps = 1.0;
  value.progress_weight = 0.0;
  value.goal_weight = 0.0;
  value.heading_weight = 0.0;
  value.clearance_weight = 100.0;
  value.smoothness_weight = 0.0;
  value.path_distance_weight = 0.0;
  value.speed_weight = 0.0;
  DwaController controller(value);

  const auto result = controller.plan_with_reference(
      fine_grid_with_obstacle(8.0), Pose2{}, Point3{20.0, 0.0, 0.0},
      {Point3{0.0, 0.0, 0.0}, Point3{20.0, 0.0, 0.0}}, 0.0, 0.0, 1, 4);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_TRUE(result.target_speed_mps.has_value());
  EXPECT_GT(*result.target_speed_mps, 0.0);
}

TEST(DwaController, PreviewsDynamicWindowSteeringButRateLimitsCommandTick) {
  auto value = vehicle_config();
  value.min_speed_mps = 1.0;
  value.max_speed_mps = 1.0;
  value.progress_weight = 0.0;
  value.goal_weight = 0.0;
  value.heading_weight = 0.0;
  value.clearance_weight = 0.0;
  value.smoothness_weight = 0.0;
  value.path_distance_weight = 0.0;
  value.speed_weight = 0.0;
  DwaController controller(value);

  const auto result = controller.plan_with_reference(
      long_range_grid(), Pose2{}, Point3{20.0, 0.0, 0.0},
      {Point3{0.0, 0.0, 0.0}, Point3{20.0, 0.0, 0.0}}, 0.0, 0.0, 1, 4);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_TRUE(result.local_trajectory.has_value());
  ASSERT_FALSE(result.local_trajectory->poses.empty());
  EXPECT_GT(std::abs(result.local_trajectory->poses.back().yaw_rad),
            value.maximum_steering_rate_radps * value.control_period_s);
  EXPECT_LE(std::abs(result.command.steering_rad),
            value.maximum_steering_rate_radps * value.control_period_s +
                1.0e-12);
}

TEST(DwaController,
     HighSpeedCandidateFitsRolloutAndCompleteStopInsideHundredMeterGrid) {
  const auto grid = long_range_grid();
  DwaController controller(vehicle_config());

  const auto result = controller.plan_with_reference(
      grid, Pose2{}, Point3{100.0, 0.0, 0.0},
      {Point3{0.0, 0.0, 0.0}, Point3{100.0, 0.0, 0.0}}, 55.0 / 3.6, 0.0, 1, 4);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_TRUE(result.target_speed_mps.has_value());
  EXPECT_GE(*result.target_speed_mps, 55.0 / 3.6 - 1.01);
  ASSERT_TRUE(result.local_trajectory.has_value());
  EXPECT_FALSE(result.local_trajectory->poses.empty());
  for (const auto &pose : result.local_trajectory->poses) {
    EXPECT_TRUE(footprint_is_safe(grid, pose, vehicle_config().footprint));
  }
}

TEST(DwaController,
     UsesEmergencyBrakeEnvelopeInsteadOfComfortableSpeedWindowForAdmission) {
  auto value = vehicle_config();
  value.maximum_deceleration_mps2 = 1.8;
  value.emergency_deceleration_mps2 = 6.0;
  DwaController controller(value);

  const auto result = controller.plan_with_reference(
      long_range_grid_with_cross_track_barrier(52.0), Pose2{},
      Point3{100.0, 0.0, 0.0}, {Point3{0.0, 0.0, 0.0}, Point3{100.0, 0.0, 0.0}},
      55.0 / 3.6, 0.0, 1, 4);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_TRUE(result.target_speed_mps.has_value());
  EXPECT_GE(*result.target_speed_mps, 55.0 / 3.6 - 1.01);
}

TEST(DwaController, HighSpeedCenterBoxProducesSteeredAvoidanceCandidate) {
  auto value = vehicle_config();
  value.maximum_deceleration_mps2 = 1.8;
  value.emergency_deceleration_mps2 = 6.0;
  DwaController controller(value);

  const auto result = controller.plan_with_reference(
      long_range_grid_with_center_box(34.0), Pose2{}, Point3{100.0, 0.0, 0.0},
      {Point3{0.0, 0.0, 0.0}, Point3{100.0, 0.0, 0.0}}, 55.0 / 3.6, 0.0, 1, 4);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_TRUE(result.local_trajectory.has_value());
  EXPECT_GT(std::abs(result.command.steering_rad), 1.0e-6);
  ASSERT_TRUE(result.target_speed_mps.has_value());
  EXPECT_GT(*result.target_speed_mps, 0.0);
}

TEST(DwaController, LateralAccelerationEnvelopeNarrowsHighSpeedSteeringWindow) {
  const auto grid = long_range_grid();
  DwaController controller(vehicle_config());

  const auto result = controller.plan(grid, Pose2{}, Point3{80.0, 0.0, 0.0},
                                      55.0 / 3.6, 0.0, 1, 4);

  ASSERT_TRUE(result.valid) << result.reason;
  const double maximum_target_steering =
      std::atan(vehicle_config().maximum_lateral_acceleration_mps2 *
                vehicle_config().wheelbase_m / ((55.0 / 3.6) * (55.0 / 3.6)));
  EXPECT_LE(std::abs(result.command.steering_rad),
            maximum_target_steering + 1e-12);
  EXPECT_LT(controller.candidate_trajectories().size(), 100U);
}

TEST(DwaController, RejectsOffRoadEscapeBeyondFixedPathDistance) {
  auto constrained = vehicle_config();
  constrained.maximum_path_distance_m = 0.5;
  DwaController controller(constrained);

  const auto result = controller.plan_with_reference(
      long_range_grid(), Pose2{}, Point3{20.0, 6.0, 0.0},
      {Point3{0.0, 0.0, 0.0}, Point3{80.0, 0.0, 0.0}}, 4.0, 0.0, 1, 4);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_TRUE(result.local_trajectory.has_value());
  for (const auto &pose : result.local_trajectory->poses) {
    EXPECT_LE(std::abs(pose.y), constrained.maximum_path_distance_m);
  }
}

TEST(DwaController, UnknownOrNoSafeCandidateReturnsInvalidFullBrake) {
  auto grid = free_grid();
  std::fill(grid.cells.begin(), grid.cells.end(), static_cast<std::int8_t>(-1));
  DwaController controller(config());
  const auto result = controller.plan(grid, Pose2{0.0, 0.0, 0.0},
                                      Point3{5.0, 0.0, 0.0}, 0.0, 0.0, 1, 4);
  EXPECT_FALSE(result.valid);
  EXPECT_DOUBLE_EQ(result.command.brake, 1.0);
  EXPECT_FALSE(result.local_trajectory.has_value());
}

TEST(DwaController, RejectsInitialFootprintOutsideDrivableMask) {
  const auto grid = free_grid();
  const auto mask = drivable_strip(grid, -0.5, 0.5);
  DwaController controller(config());

  const auto result = controller.plan_with_reference(
      grid, Pose2{0.0, 0.45, 0.0}, Point3{5.0, 0.0, 0.0}, {}, 0.0, 0.0, 1, 4,
      &mask);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "initial footprint outside drivable mask");
}

TEST(DwaController, RejectsDrivableMaskWithDifferentGridGeometry) {
  const auto grid = free_grid();
  auto mask = drivable_strip(grid, -1.0, 1.0);
  mask.origin.x += 0.25;
  DwaController controller(config());

  const auto result = controller.plan_with_reference(
      grid, Pose2{}, Point3{5.0, 0.0, 0.0}, {}, 0.0, 0.0, 1, 4, &mask);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "drivable mask geometry mismatch");
}

TEST(DwaController, KeepsEverySelectedPoseInsideDrivableMask) {
  const auto grid = free_grid();
  const auto mask = drivable_strip(grid, -0.75, 0.75);
  const auto value = config();
  DwaController controller(value);

  const auto result = controller.plan_with_reference(
      grid, Pose2{}, Point3{5.0, 0.0, 0.0}, {}, 0.0, 0.0, 1, 4, &mask);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_TRUE(result.local_trajectory.has_value());
  for (const auto &pose : result.local_trajectory->poses) {
    EXPECT_TRUE(footprint_is_safe(mask, pose, value.footprint));
  }
}

TEST(DwaController, RejectsCandidateWhenInitialFootprintIsOccupied) {
  auto map = free_grid();
  const auto initial_cell = world_to_cell(map, Point3{0.0, 0.0, 0.0});
  ASSERT_TRUE(initial_cell.has_value());
  map.cells[initial_cell->y * map.width + initial_cell->x] = 100;
  DwaController controller(config());

  const auto result = controller.plan(map, Pose2{0.0, 0.0, 0.0},
                                      Point3{5.0, 0.0, 0.0}, 0.0, 0.0, 1, 4);

  EXPECT_FALSE(result.valid);
  EXPECT_DOUBLE_EQ(result.command.brake, 1.0);
}

TEST(DwaController, EscapesInitialInflationWithoutCrossingLethalCells) {
  auto map = free_grid();
  const auto inflated_cell = world_to_cell(map, Point3{0.0, 0.0, 0.0});
  ASSERT_TRUE(inflated_cell.has_value());
  map.cells[inflated_cell->y * map.width + inflated_cell->x] = 50;
  auto value = config();
  value.initial_inflation_escape_s = 1.0;
  DwaController controller(value);

  const auto result = controller.plan(map, Pose2{0.0, 0.0, 0.0},
                                      Point3{5.0, 0.0, 0.0}, 0.0, 0.0, 1, 4);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_TRUE(result.local_trajectory.has_value());
  EXPECT_GT(result.local_trajectory->poses.back().x, 0.0);
}

TEST(DwaController, NonfiniteTargetDoesNotMutatePidOrLastValid) {
  DwaController controller(config());
  const auto valid = controller.plan(free_grid(), Pose2{0.0, 0.0, 0.0},
                                     Point3{5.0, 0.0, 0.0}, 0.0, 0.0, 1, 4);
  const auto state = controller.pid_state();
  const auto invalid = controller.plan(
      free_grid(), Pose2{0.0, 0.0, 0.0},
      Point3{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}, 0.0, 0.0, 9,
      9);
  EXPECT_FALSE(invalid.valid);
  EXPECT_EQ(controller.pid_state(), state);
  EXPECT_EQ(controller.last_valid_result(), valid);
}

TEST(DwaController, RejectsNonPositiveSimulationConfiguration) {
  auto cfg = config();
  cfg.dt = 0.0;
  EXPECT_THROW(DwaController{cfg}, std::invalid_argument);
}

TEST(DwaController, RejectsNonfiniteFootprintCenterOffset) {
  auto cfg = config();
  cfg.footprint.center_offset_x_m = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(DwaController{cfg}, std::invalid_argument);
}

TEST(DwaController, RejectsOccupiedThresholdOutsideOccupancyGridRange) {
  auto cfg = config();
  cfg.footprint.occupied_threshold = 101;
  EXPECT_THROW(DwaController{cfg}, std::invalid_argument);
}

TEST(DwaController,
     RejectsSamplingStepsThatCannotAdvanceAtConfiguredMagnitude) {
  auto cfg = config();
  cfg.min_speed_mps = 1.0e300;
  cfg.max_speed_mps = std::nextafter(cfg.min_speed_mps,
                                     std::numeric_limits<double>::infinity());
  cfg.speed_step_mps = std::numeric_limits<double>::denorm_min();
  EXPECT_THROW(DwaController{cfg}, std::invalid_argument);
}

TEST(DwaController, RejectsSimulationStepCountOverflow) {
  auto cfg = config();
  cfg.horizon_s = std::numeric_limits<double>::max();
  cfg.dt = std::numeric_limits<double>::denorm_min();
  EXPECT_THROW(DwaController{cfg}, std::invalid_argument);
}

TEST(DwaController, FiniteInputsCannotCollapseCandidateScoresToNonfinite) {
  auto cfg = config();
  cfg.min_steer_rad = -0.4;
  cfg.sampled_max_steer_rad = 0.4;
  cfg.steer_step_rad = 0.8;
  cfg.progress_weight = 0.0;
  cfg.goal_weight = 0.0;
  cfg.clearance_weight = 0.0;
  cfg.smoothness_weight = 0.0;
  DwaController controller(cfg);
  const double maximum = std::numeric_limits<double>::max();
  const auto result =
      controller.plan(free_grid(), Pose2{0.0, 0.0, 0.0},
                      Point3{maximum, maximum, 0.0}, 0.0, 0.0, 1, 4);
  ASSERT_TRUE(result.valid);
  EXPECT_LE(std::abs(result.command.steering_rad),
            cfg.maximum_steering_rate_radps * cfg.control_period_s + 1e-12);
}
} // namespace
