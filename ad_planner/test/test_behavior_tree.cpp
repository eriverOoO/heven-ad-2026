#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ad_planner/behavior/bt_nodes.hpp"
#include "ad_planner/behavior/planner_supervisor.hpp"
#include "ad_planner/local_planning/dwa/dwa.hpp"

namespace {

using ad_planner::CollisionObservation;
using ad_planner::ControllerResult;
using ad_planner::DwaConfig;
using ad_planner::DwaController;
using ad_planner::FutureRoadRiskObservation;
using ad_planner::FutureRoadRiskState;
using ad_planner::GearRequest;
using ad_planner::GridObservation;
using ad_planner::OccupancyGrid;
using ad_planner::PhysicalCommand;
using ad_planner::PlannerConfig;
using ad_planner::PlannerContext;
using ad_planner::PlannerSupervisor;
using ad_planner::Point3;
using ad_planner::Pose2;
using ad_planner::RecoveryPhase;
using ad_planner::RouteOccupancyObservation;
using ad_planner::RouteOccupancyState;
using ad_planner::SupervisorStatus;
using ad_planner::TrafficSignal;
using ad_planner::VehicleObservation;
using ad_planner::VehicleState;

std::string tree_xml() {
  const auto path = std::filesystem::path(AD_PLANNER_SOURCE_DIR) /
                    "behavior_trees" / "ad_planner.xml";
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot read " + path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

ControllerResult command(double accel) {
  ControllerResult result;
  result.valid = true;
  result.command = PhysicalCommand{accel, 0.0, 0.0};
  return result;
}

OccupancyGrid grid_with_cell(double x = 100.0, double y = 100.0) {
  OccupancyGrid grid;
  grid.origin = Pose2{-10.0, -10.0, 0.0};
  grid.resolution = 0.25;
  grid.width = 440;
  grid.height = 80;
  grid.cells.assign(grid.width * grid.height, 0);
  grid.valid = true;
  grid.fresh = true;

  const int column = static_cast<int>((x - grid.origin.x) / grid.resolution);
  const int row = static_cast<int>((y - grid.origin.y) / grid.resolution);
  if (column >= 0 && column < static_cast<int>(grid.width) && row >= 0 &&
      row < static_cast<int>(grid.height)) {
    grid.cells[static_cast<std::size_t>(row) * grid.width +
               static_cast<std::size_t>(column)] = 100;
  }
  return grid;
}

OccupancyGrid runtime_grid_with_cell(double x, double y) {
  OccupancyGrid grid;
  grid.origin = Pose2{-4.0, -10.0, 0.0};
  grid.resolution = 0.1;
  grid.width = 1040;
  grid.height = 200;
  grid.cells.assign(grid.width * grid.height, 0);
  grid.valid = true;
  grid.fresh = true;

  const auto column =
      static_cast<std::size_t>((x - grid.origin.x) / grid.resolution);
  const auto row =
      static_cast<std::size_t>((y - grid.origin.y) / grid.resolution);
  grid.cells[row * grid.width + column] = 100;
  return grid;
}

DwaConfig runtime_dwa_config() {
  DwaConfig config;
  config.min_speed_mps = 0.0;
  config.max_speed_mps = 16.25;
  config.speed_step_mps = 1.0;
  config.min_steer_rad = -0.52;
  config.sampled_max_steer_rad = 0.52;
  config.steer_step_rad = 0.04;
  config.dt = 0.2;
  config.horizon_s = 1.5;
  config.wheelbase_m = 3.0;
  config.max_steer_rad = 0.6981317;
  config.control_period_s = 0.05;
  config.dynamic_window_time_s = 0.5;
  config.maximum_acceleration_mps2 = 5.0;
  config.maximum_deceleration_mps2 = 2.0;
  config.maximum_steering_rate_radps = 2.0943951023931953;
  config.maximum_lateral_acceleration_mps2 = 6.0;
  config.clearance_saturation_m = 8.0;
  config.maximum_path_distance_m = 4.5;
  config.footprint =
      ad_planner::FootprintConfig{2.3175, 0.945, 0.2, 20, 8192, 1.5275};
  config.progress_weight = 1.0;
  config.goal_weight = 0.5;
  config.heading_weight = 1.0;
  config.clearance_weight = 2.0;
  config.smoothness_weight = 0.15;
  config.path_distance_weight = 1.5;
  config.speed_weight = 0.5;
  config.pid = ad_planner::PidConfig{0.3, 0.0, 0.01, 10.0, 10.0};
  return config;
}

PlannerContext nominal_context() {
  PlannerContext context;
  context.steady_time_s = 10.0;
  context.inputs.route_ready = true;
  context.inputs.status = VehicleObservation{
      true, true, 10.0, VehicleState{Pose2{}, 1.0, ad_planner::kGearDriveCode}};
  context.inputs.collisions = CollisionObservation{true, true, 10.0, {}};
  context.inputs.grid = GridObservation{true, true, 10.0, grid_with_cell()};
  context.inputs.route_occupancy = RouteOccupancyObservation{
      true, true, 10.0,
      RouteOccupancyState{true, 1.0, 99.0, 0U, 0U, std::nullopt}};
  context.inputs.future_road_risk =
      FutureRoadRiskObservation{true, true, 10.0, FutureRoadRiskState{}};
  context.callbacks.follow_global_path = []() { return command(0.2); };
  context.callbacks.perception_local_planner = []() { return command(0.3); };
  return context;
}

PlannerConfig test_config() {
  PlannerConfig config;
  config.freshness.status_timeout_s = 0.5;
  config.freshness.collision_timeout_s = 0.5;
  config.freshness.grid_timeout_s = 0.5;
  config.freshness.traffic_timeout_s = 0.5;
  config.freshness.stop_line_timeout_s = 0.5;
  config.collision.reverse_accel = 0.3;
  config.collision.reverse_duration_s = 2.0;
  config.collision.reverse_brake_duration_s = 1.0;
  config.collision.cooldown_s = 1.5;
  config.collision.rear_check_footprint.clearance_m = 0.0;
  config.perception.forward_check_footprint.clearance_m = 0.0;
  return config;
}

void refresh(PlannerContext &context, double now) {
  context.steady_time_s = now;
  context.inputs.status.receipt_time_s = now;
  context.inputs.collisions.receipt_time_s = now;
  if (context.inputs.grid.received) {
    context.inputs.grid.receipt_time_s = now;
  }
  if (context.inputs.route_occupancy.received) {
    context.inputs.route_occupancy.receipt_time_s = now;
  }
  if (context.inputs.future_road_risk.received) {
    context.inputs.future_road_risk.receipt_time_s = now;
  }
  if (context.inputs.traffic_signal.received) {
    context.inputs.traffic_signal.receipt_time_s = now;
  }
  if (context.inputs.stop_line.received) {
    context.inputs.stop_line.receipt_time_s = now;
  }
}

void expect_brake(const ad_planner::PlannerTickResult &result) {
  EXPECT_DOUBLE_EQ(result.command.motion.accel, 0.0);
  EXPECT_DOUBLE_EQ(result.command.motion.brake, 1.0);
}

std::unique_ptr<PlannerSupervisor>
make_supervisor(PlannerContext &context, PlannerConfig config = test_config()) {
  return std::make_unique<PlannerSupervisor>(context, config, tree_xml());
}

TEST(BehaviorTree, HasExpectedPriorityAndCruises) {
  const std::string expected = "<root main_tree_to_execute=\"AdPlanner\">\n"
                               "  <BehaviorTree ID=\"AdPlanner\">\n"
                               "    <ReactiveFallback>\n"
                               "      <ReactiveSequence>\n"
                               "        <InputsReady/>\n"
                               "        <ReactiveFallback>\n"
                               "          <CollisionRecovery/>\n"
                               "          <TrafficStop/>\n"
                               "          <PerceptionMission/>\n"
                               "          <FollowGlobalPath/>\n"
                               "        </ReactiveFallback>\n"
                               "      </ReactiveSequence>\n"
                               "      <FailSafeBrake/>\n"
                               "    </ReactiveFallback>\n"
                               "  </BehaviorTree>\n"
                               "</root>\n";
  EXPECT_EQ(tree_xml(), expected);

  auto context = nominal_context();
  auto supervisor = make_supervisor(context);
  const auto result = supervisor->tick();
  EXPECT_EQ(result.active_behavior, "follow_global_path");
  EXPECT_DOUBLE_EQ(result.command.motion.accel, 0.2);
  EXPECT_EQ(supervisor->registered_node_ids().size(), 6U);
}

TEST(BehaviorTree, WaitsForDriveAcknowledgementBeforeRunningPathTracker) {
  auto context = nominal_context();
  context.inputs.status.value.gear = 1;
  std::size_t path_tracker_calls = 0;
  context.callbacks.follow_global_path = [&]() {
    ++path_tracker_calls;
    return command(0.2);
  };
  auto supervisor = make_supervisor(context);

  const auto first = supervisor->tick();
  expect_brake(first);
  EXPECT_EQ(first.command.gear_request, GearRequest::kDrive);
  EXPECT_EQ(first.active_behavior, "follow_global_path");
  EXPECT_EQ(path_tracker_calls, 0U);

  refresh(context, 10.1);
  const auto waiting = supervisor->tick();
  expect_brake(waiting);
  EXPECT_EQ(waiting.command.gear_request, GearRequest::kDrive);
  EXPECT_EQ(path_tracker_calls, 0U);

  context.inputs.status.value.gear = ad_planner::kGearDriveCode;
  refresh(context, 10.2);
  const auto tracking = supervisor->tick();
  EXPECT_DOUBLE_EQ(tracking.command.motion.accel, 0.2);
  EXPECT_DOUBLE_EQ(tracking.command.motion.brake, 0.0);
  EXPECT_EQ(tracking.command.gear_request, GearRequest::kKeep);
  EXPECT_EQ(tracking.active_behavior, "follow_global_path");
  EXPECT_EQ(path_tracker_calls, 1U);
}

TEST(BehaviorTree, SemanticActionIdsDoNotNameControllerAlgorithms) {
  auto context = nominal_context();
  auto supervisor = make_supervisor(context);
  const auto ids = supervisor->registered_node_ids();

  EXPECT_NE(std::find(ids.begin(), ids.end(), "FollowGlobalPath"), ids.end());
  for (const auto &id : ids) {
    EXPECT_EQ(id.find("Stanley"), std::string::npos) << id;
    EXPECT_EQ(id.find("DWA"), std::string::npos) << id;
    EXPECT_EQ(id.find("Dwa"), std::string::npos) << id;
  }
  EXPECT_EQ(tree_xml().find("Stanley"), std::string::npos);
  EXPECT_EQ(tree_xml().find("DWA"), std::string::npos);
  EXPECT_EQ(tree_xml().find("Dwa"), std::string::npos);
}

TEST(BehaviorTree, StaleMandatoryInputFallsBackToBrake) {
  auto context = nominal_context();
  context.inputs.status.receipt_time_s = 9.0;
  auto supervisor = make_supervisor(context);

  const auto result = supervisor->tick();
  expect_brake(result);
  EXPECT_EQ(result.active_behavior, "fail_safe_brake");
}

TEST(BehaviorTree, InvalidOrThrowingControllerFallsBackToBrake) {
  auto context = nominal_context();
  context.callbacks.follow_global_path = []() { return command(2.0); };
  auto supervisor = make_supervisor(context);
  expect_brake(supervisor->tick());

  auto throwing_context = nominal_context();
  throwing_context.callbacks.follow_global_path = []() -> ControllerResult {
    throw std::runtime_error("controller failure");
  };
  auto throwing_supervisor = make_supervisor(throwing_context);
  const auto result = throwing_supervisor->tick();
  expect_brake(result);
  EXPECT_EQ(result.status, SupervisorStatus::kException);
}

TEST(BehaviorTree, NonStaticCollisionNeverRequestsReverse) {
  auto context = nominal_context();
  context.inputs.status.value.speed_mps = 0.0;
  context.inputs.collisions.object_types = {0};
  auto supervisor = make_supervisor(context);

  const auto first = supervisor->tick();
  expect_brake(first);
  EXPECT_EQ(first.command.gear_request, GearRequest::kKeep);
  refresh(context, 10.2);
  const auto second = supervisor->tick();
  expect_brake(second);
  EXPECT_EQ(second.command.gear_request, GearRequest::kKeep);
  EXPECT_EQ(supervisor->state().recovery_phase, RecoveryPhase::kHoldNonstatic);
}

TEST(BehaviorTree, StaticCollisionRunsBoundedReverseAndReturnsToCruise) {
  auto context = nominal_context();
  context.inputs.status.value.speed_mps = 0.0;
  context.inputs.collisions.object_types = {2};
  auto supervisor = make_supervisor(context);

  expect_brake(supervisor->tick());
  refresh(context, 10.1);
  EXPECT_EQ(supervisor->tick().command.gear_request, GearRequest::kReverse);

  refresh(context, 10.2);
  context.inputs.status.value.gear = ad_planner::kGearReverseCode;
  const auto reverse = supervisor->tick();
  EXPECT_DOUBLE_EQ(reverse.command.motion.accel, 0.3);
  EXPECT_EQ(reverse.command.gear_request, GearRequest::kReverse);

  refresh(context, 12.2);
  expect_brake(supervisor->tick());
  refresh(context, 13.2);
  expect_brake(supervisor->tick());

  context.inputs.collisions.object_types.clear();
  refresh(context, 13.3);
  const auto drive = supervisor->tick();
  expect_brake(drive);
  EXPECT_EQ(drive.command.gear_request, GearRequest::kDrive);

  context.inputs.status.value.gear = ad_planner::kGearDriveCode;
  refresh(context, 13.4);
  expect_brake(supervisor->tick());
  refresh(context, 14.9);
  const auto cruise = supervisor->tick();
  EXPECT_EQ(cruise.active_behavior, "follow_global_path");
  EXPECT_DOUBLE_EQ(cruise.command.motion.accel, 0.2);
}

TEST(BehaviorTree, OneShotStaticCollisionMayClearBeforeReverseAcknowledgement) {
  auto context = nominal_context();
  context.inputs.status.value.speed_mps = 0.0;
  context.inputs.collisions.object_types = {2};
  auto supervisor = make_supervisor(context);

  expect_brake(supervisor->tick());
  refresh(context, 10.1);
  EXPECT_EQ(supervisor->tick().command.gear_request, GearRequest::kReverse);

  context.inputs.collisions.object_types.clear();
  context.inputs.status.value.gear = ad_planner::kGearReverseCode;
  refresh(context, 10.2);
  const auto reverse = supervisor->tick();
  EXPECT_EQ(reverse.active_behavior, "collision_recovery");
  EXPECT_EQ(reverse.command.gear_request, GearRequest::kReverse);
  EXPECT_GT(reverse.command.motion.accel, 0.0);

  refresh(context, 10.3);
  EXPECT_GT(supervisor->tick().command.motion.accel, 0.0);
  EXPECT_EQ(supervisor->state().recovery_phase, RecoveryPhase::kReverseAccel);
}

TEST(BehaviorTree, InterruptedReverseBrakesAndRestoresDriveBeforeCruise) {
  auto context = nominal_context();
  context.inputs.status.value.speed_mps = 0.0;
  context.inputs.collisions.object_types = {2};
  auto supervisor = make_supervisor(context);

  expect_brake(supervisor->tick());
  refresh(context, 10.1);
  supervisor->tick();
  refresh(context, 10.2);
  context.inputs.status.value.gear = ad_planner::kGearReverseCode;
  EXPECT_GT(supervisor->tick().command.motion.accel, 0.0);

  context.inputs.route_ready = false;
  refresh(context, 10.3);
  expect_brake(supervisor->tick());

  context.inputs.route_ready = true;
  context.inputs.collisions.object_types.clear();
  refresh(context, 10.4);
  const auto drive = supervisor->tick();
  expect_brake(drive);
  EXPECT_EQ(drive.command.gear_request, GearRequest::kDrive);
  EXPECT_EQ(supervisor->state().recovery_phase, RecoveryPhase::kWaitDriveAck);
}

TEST(BehaviorTree, TrafficStopReleasesOnlyAfterStopAndFreshGreen) {
  auto context = nominal_context();
  auto config = test_config();
  config.traffic.zones = {Point3{0.0, 0.0, 0.0}};
  context.inputs.traffic_signal = {
      true, true, 10.0, static_cast<std::int8_t>(TrafficSignal::kRed)};
  auto supervisor = make_supervisor(context, config);

  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");
  refresh(context, 10.2);
  expect_brake(supervisor->tick());

  context.inputs.status.value.speed_mps = 0.0;
  refresh(context, 10.3);
  expect_brake(supervisor->tick());

  context.inputs.traffic_signal.value =
      static_cast<std::int8_t>(TrafficSignal::kUnknown);
  refresh(context, 10.4);
  expect_brake(supervisor->tick());

  context.inputs.traffic_signal.value =
      static_cast<std::int8_t>(TrafficSignal::kGreen);
  refresh(context, 10.5);
  expect_brake(supervisor->tick());

  refresh(context, 10.9);
  expect_brake(supervisor->tick());

  refresh(context, 11.0);
  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");
}

TEST(BehaviorTree, TrafficStopUsesMeasuredBrakeInsideStoppingDistance) {
  auto context = nominal_context();
  auto config = test_config();
  config.traffic.zones = {Point3{0.0, 0.0, 0.0}};
  config.traffic.front_bumper_x_m = 3.845;
  config.traffic.stopping_margin_m = 1.0;
  config.traffic.reaction_time_s = 0.11589156;
  config.traffic.braking_deceleration_mps2 = 1.8;
  config.traffic.brake_command = 0.2;
  context.inputs.status.value.pose.x = -75.0;
  context.inputs.status.value.speed_mps = 16.25;
  context.inputs.traffic_signal = {
      true, true, 10.0, static_cast<std::int8_t>(TrafficSignal::kRed)};

  auto supervisor = make_supervisor(context, config);
  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");
  refresh(context, 10.2);
  const auto result = supervisor->tick();

  EXPECT_EQ(result.active_behavior, "traffic_stop");
  EXPECT_DOUBLE_EQ(result.command.motion.accel, 0.0);
  EXPECT_DOUBLE_EQ(result.command.motion.brake, 0.2);
}

TEST(BehaviorTree, TrafficStopDoesNotBrakeBeforeStoppingDistance) {
  auto context = nominal_context();
  auto config = test_config();
  config.traffic.zones = {Point3{0.0, 0.0, 0.0}};
  config.traffic.front_bumper_x_m = 3.845;
  config.traffic.stopping_margin_m = 1.0;
  config.traffic.reaction_time_s = 0.11589156;
  config.traffic.braking_deceleration_mps2 = 1.8;
  config.traffic.brake_command = 0.2;
  context.inputs.status.value.pose.x = -85.0;
  context.inputs.status.value.speed_mps = 16.25;
  context.inputs.traffic_signal = {
      true, true, 10.0, static_cast<std::int8_t>(TrafficSignal::kRed)};

  auto supervisor = make_supervisor(context, config);

  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");
  refresh(context, 10.2);
  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");
}

TEST(BehaviorTree, TrafficUnknownDoesNotInitiateStop) {
  auto context = nominal_context();
  auto config = test_config();
  config.traffic.zones = {Point3{0.0, 0.0, 0.0}};
  context.inputs.status.value.pose.x = -75.0;
  context.inputs.status.value.speed_mps = 16.25;
  context.inputs.traffic_signal = {
      true, true, 10.0, static_cast<std::int8_t>(TrafficSignal::kUnknown)};

  auto supervisor = make_supervisor(context, config);

  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");
}

TEST(BehaviorTree, TrafficRedMustRemainContinuousBeforeInitiatingStop) {
  auto context = nominal_context();
  auto config = test_config();
  config.traffic.zones = {Point3{0.0, 0.0, 0.0}};
  context.inputs.status.value.pose.x = -75.0;
  context.inputs.status.value.speed_mps = 16.25;
  context.inputs.traffic_signal = {
      true, true, 10.0, static_cast<std::int8_t>(TrafficSignal::kRed)};
  auto supervisor = make_supervisor(context, config);

  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");

  refresh(context, 10.1);
  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");

  refresh(context, 10.2);
  EXPECT_EQ(supervisor->tick().active_behavior, "traffic_stop");
}

TEST(BehaviorTree, TrafficStopReleasesMovingApproachOnConfirmedGreen) {
  auto context = nominal_context();
  auto config = test_config();
  config.traffic.zones = {Point3{0.0, 0.0, 0.0}};
  config.traffic.front_bumper_x_m = 3.845;
  config.traffic.stopping_margin_m = 1.0;
  config.traffic.reaction_time_s = 0.11589156;
  config.traffic.braking_deceleration_mps2 = 1.8;
  config.traffic.brake_command = 0.2;
  context.inputs.status.value.pose.x = -50.0;
  context.inputs.status.value.speed_mps = 16.25;
  context.inputs.traffic_signal = {
      true, true, 10.0, static_cast<std::int8_t>(TrafficSignal::kRed)};
  auto supervisor = make_supervisor(context, config);
  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");

  refresh(context, 10.2);
  EXPECT_EQ(supervisor->tick().active_behavior, "traffic_stop");

  context.inputs.traffic_signal.value =
      static_cast<std::int8_t>(TrafficSignal::kGreen);
  refresh(context, 10.3);

  const auto result = supervisor->tick();
  EXPECT_EQ(result.active_behavior, "traffic_stop");
  EXPECT_GT(result.command.motion.brake, 0.0);

  refresh(context, 10.7);
  EXPECT_EQ(supervisor->tick().active_behavior, "traffic_stop");

  refresh(context, 10.8);
  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");
}

TEST(BehaviorTree, TrafficStopKeepsEmergencyBrakeAfterPassingLatchedLine) {
  auto context = nominal_context();
  auto config = test_config();
  config.traffic.zones = {Point3{0.0, 0.0, 0.0}};
  context.inputs.status.value.pose.x = -3.0;
  context.inputs.status.value.speed_mps = 10.0;
  context.inputs.traffic_signal = {
      true, true, 10.0, static_cast<std::int8_t>(TrafficSignal::kRed)};
  auto supervisor = make_supervisor(context, config);
  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");

  refresh(context, 10.2);
  expect_brake(supervisor->tick());

  context.inputs.status.value.pose.x = 1.0;
  refresh(context, 10.3);

  const auto result = supervisor->tick();
  EXPECT_EQ(result.active_behavior, "traffic_stop");
  expect_brake(result);
}

TEST(BehaviorTree, TrafficStopUsesEmergencyBrakeWhenDetectedTooLate) {
  auto context = nominal_context();
  auto config = test_config();
  config.traffic.zones = {Point3{0.0, 0.0, 0.0}};
  config.traffic.front_bumper_x_m = 3.845;
  config.traffic.stopping_margin_m = 1.0;
  config.traffic.reaction_time_s = 0.11589156;
  config.traffic.braking_deceleration_mps2 = 1.8;
  config.traffic.brake_command = 0.2;
  context.inputs.status.value.pose.x = -25.0;
  context.inputs.status.value.speed_mps = 15.0;
  context.inputs.traffic_signal = {
      true, true, 10.0, static_cast<std::int8_t>(TrafficSignal::kRed)};

  auto supervisor = make_supervisor(context, config);
  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");
  refresh(context, 10.2);
  const auto result = supervisor->tick();

  EXPECT_EQ(result.active_behavior, "traffic_stop");
  expect_brake(result);
}

TEST(BehaviorTree, PerceptionMissionUsesLocalMotionOnlyForBlockedPath) {
  auto context = nominal_context();
  auto supervisor = make_supervisor(context);
  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");

  context.inputs.grid.value = grid_with_cell(2.0, 0.0);
  refresh(context, 10.1);
  const auto result = supervisor->tick();
  EXPECT_EQ(result.active_behavior, "perception_mission");
  EXPECT_DOUBLE_EQ(result.command.motion.accel, 0.3);
}

TEST(BehaviorTree,
     RouteAlignedPerceptionIgnoresCellsOutsideTheEvaluatedRoadCorridor) {
  auto context = nominal_context();
  auto config = test_config();
  config.perception.route_aligned_activation = true;
  // The legacy base_link rectangle sees this cell, while the route-aligned
  // observation reports that the drivable corridor itself is clear.
  context.inputs.grid.value = grid_with_cell(2.0, 0.0);
  context.inputs.route_occupancy.value =
      RouteOccupancyState{true, 1.0, 99.0, 0U, 0U, std::nullopt};
  auto supervisor = make_supervisor(context, config);

  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");
}

TEST(BehaviorTree, RouteAlignedPerceptionActivatesForObstacleAroundACurve) {
  auto context = nominal_context();
  auto config = test_config();
  config.perception.route_aligned_activation = true;
  // The raw grid is clear in the straight base_link rectangle, but the route
  // query found an occupied road cell 12 m ahead around a curve.
  context.inputs.grid.value = grid_with_cell();
  context.inputs.route_occupancy.value =
      RouteOccupancyState{false, 1.0, 20.0, 3U, 0U, 12.0};
  auto supervisor = make_supervisor(context, config);

  EXPECT_EQ(supervisor->tick().active_behavior, "perception_mission");
}

TEST(BehaviorTree, RouteAlignedPerceptionFailsClosedWhenObservationIsStale) {
  auto context = nominal_context();
  auto config = test_config();
  config.perception.route_aligned_activation = true;
  context.inputs.route_occupancy.receipt_time_s = 9.0;
  auto supervisor = make_supervisor(context, config);

  const auto result = supervisor->tick();

  EXPECT_EQ(result.active_behavior, "perception_mission");
  EXPECT_EQ(result.failsafe_reason, "route occupancy is stale");
  expect_brake(result);
}

TEST(BehaviorTree,
     FutureRoadRiskActivatesPerceptionWhileCurrentRoadOccupancyIsClear) {
  auto context = nominal_context();
  auto config = test_config();
  config.perception.route_aligned_activation = true;
  config.perception.future_road_risk_required = true;
  context.inputs.future_road_risk.value =
      FutureRoadRiskState{true, 1U, 1U, 3U, 1.5};
  auto supervisor = make_supervisor(context, config);

  const auto result = supervisor->tick();

  EXPECT_EQ(result.active_behavior, "perception_mission");
  EXPECT_DOUBLE_EQ(result.command.motion.accel, 0.3);
}

TEST(BehaviorTree,
     RequiredFutureRoadRiskFailsClosedWhenPredictionDerivedStateIsInvalid) {
  auto context = nominal_context();
  auto config = test_config();
  config.perception.route_aligned_activation = true;
  config.perception.future_road_risk_required = true;
  context.inputs.future_road_risk.valid = false;
  auto supervisor = make_supervisor(context, config);

  const auto result = supervisor->tick();

  EXPECT_EQ(result.active_behavior, "perception_mission");
  EXPECT_EQ(result.failsafe_reason, "future road risk is stale or invalid");
  expect_brake(result);
}

TEST(BehaviorTree, ClearFreshFutureRoadRiskLeavesGlobalPathTrackingActive) {
  auto context = nominal_context();
  auto config = test_config();
  config.perception.route_aligned_activation = true;
  config.perception.future_road_risk_required = true;
  auto supervisor = make_supervisor(context, config);

  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");
}

TEST(BehaviorTree,
     PerceptionMissionRequiresTwoContinuousClearSecondsToRelease) {
  auto context = nominal_context();
  context.inputs.grid.value = grid_with_cell(2.0, 0.0);
  auto supervisor = make_supervisor(context);

  EXPECT_EQ(supervisor->tick().active_behavior, "perception_mission");

  context.inputs.grid.value = grid_with_cell();
  refresh(context, 10.1);
  EXPECT_EQ(supervisor->tick().active_behavior, "perception_mission");

  refresh(context, 10.7);
  EXPECT_EQ(supervisor->tick().active_behavior, "perception_mission");

  refresh(context, 12.0);
  EXPECT_EQ(supervisor->tick().active_behavior, "perception_mission");

  refresh(context, 12.2);
  EXPECT_EQ(supervisor->tick().active_behavior, "follow_global_path");
}

TEST(BehaviorTree,
     SpeedAwarePerceptionLookaheadCoversRoadSpeedStoppingEnvelope) {
  auto config = test_config().perception;
  const double speed_mps = 55.0 / 3.6;

  const auto road_speed =
      ad_planner::make_perception_forward_check(config, speed_mps);
  const double expected_far_x =
      config.front_bumper_x_m + config.stopping_margin_m +
      speed_mps * config.reaction_time_s +
      speed_mps * speed_mps / (2.0 * config.braking_deceleration_mps2);
  EXPECT_NEAR(road_speed.far_x_m, expected_far_x, 1.0e-12);
  EXPECT_NEAR(road_speed.pose.x - road_speed.footprint.half_length_m,
              config.near_x_m, 1.0e-12);
  EXPECT_NEAR(road_speed.pose.x + road_speed.footprint.half_length_m,
              expected_far_x, 1.0e-12);

  const auto maximum_speed =
      ad_planner::make_perception_forward_check(config, 100.0);
  EXPECT_DOUBLE_EQ(maximum_speed.far_x_m, config.maximum_lookahead_m);
}

TEST(BehaviorTree, PerceptionLatchKeepsDetectedLookaheadWhenVehicleSlowsDown) {
  auto context = nominal_context();
  context.inputs.status.value.speed_mps = 55.0 / 3.6;
  context.inputs.grid.value = grid_with_cell(70.0, 0.0);
  auto supervisor = make_supervisor(context);

  EXPECT_EQ(supervisor->tick().active_behavior, "perception_mission");

  context.inputs.status.value.speed_mps = 0.0;
  refresh(context, 10.1);
  EXPECT_EQ(supervisor->tick().active_behavior, "perception_mission");

  refresh(context, 13.0);
  EXPECT_EQ(supervisor->tick().active_behavior, "perception_mission");
}

TEST(BehaviorTree,
     PerceptionObstacleHandsOffEarlyEnoughForConfiguredLocalMotion) {
  auto context = nominal_context();
  context.inputs.status.value.speed_mps = 4.7;
  // At 4.7 m/s the configured trigger clamps to its 20 m minimum. Put the
  // obstacle inside that boundary and give DWA the same long reference target
  // used by the runtime corridor.
  context.inputs.grid.value = runtime_grid_with_cell(16.0, 0.05);
  DwaController configured_local_planner(runtime_dwa_config());
  std::size_t local_planner_calls = 0;
  context.callbacks.perception_local_planner = [&]() {
    ++local_planner_calls;
    return configured_local_planner.plan(
        context.inputs.grid.value, Pose2{}, Point3{80.0, 0.0, 0.0},
        context.inputs.status.value.speed_mps, 0.0, 1,
        context.inputs.status.value.gear);
  };
  auto supervisor = make_supervisor(context);

  const auto result = supervisor->tick();

  EXPECT_EQ(local_planner_calls, 1U);
  EXPECT_EQ(result.active_behavior, "perception_mission");
  EXPECT_NE(result.command.motion.steering_rad, 0.0)
      << "accel=" << result.command.motion.accel
      << ", brake=" << result.command.motion.brake
      << ", failsafe=" << result.failsafe_reason;
}

TEST(BehaviorTree, InvalidTimeBrakes) {
  auto context = nominal_context();
  auto supervisor = make_supervisor(context);
  static_cast<void>(supervisor->tick());
  refresh(context, 9.0);
  const auto result = supervisor->tick();
  expect_brake(result);
  EXPECT_EQ(result.status, SupervisorStatus::kException);
}

TEST(BehaviorTree, RejectsReverseLongerThanCompetitionLimit) {
  auto context = nominal_context();
  auto config = test_config();
  config.collision.reverse_duration_s = 2.1;
  EXPECT_THROW(PlannerSupervisor(context, config, tree_xml()),
               std::invalid_argument);
}

TEST(BehaviorTree, RejectsSteeringBeyondCompetitionLimit) {
  auto context = nominal_context();
  auto config = test_config();
  config.maximum_steering_rad = 0.7;
  EXPECT_THROW(PlannerSupervisor(context, config, tree_xml()),
               std::invalid_argument);
}

TEST(BehaviorTree, RejectsInvalidPerceptionTriggerGeometry) {
  auto context = nominal_context();

  auto nonfinite_pose = test_config();
  nonfinite_pose.perception.forward_check_pose.yaw_rad =
      std::numeric_limits<double>::infinity();
  EXPECT_THROW(PlannerSupervisor(context, nonfinite_pose, tree_xml()),
               std::invalid_argument);

  auto zero_release_duration = test_config();
  zero_release_duration.perception.clear_release_duration_s = 0.0;
  EXPECT_THROW(PlannerSupervisor(context, zero_release_duration, tree_xml()),
               std::invalid_argument);

  auto zero_length = test_config();
  zero_length.perception.forward_check_footprint.half_length_m = 0.0;
  EXPECT_THROW(PlannerSupervisor(context, zero_length, tree_xml()),
               std::invalid_argument);

  auto negative_clearance = test_config();
  negative_clearance.perception.forward_check_footprint.clearance_m = -0.1;
  EXPECT_THROW(PlannerSupervisor(context, negative_clearance, tree_xml()),
               std::invalid_argument);

  auto invalid_threshold = test_config();
  invalid_threshold.perception.forward_check_footprint.occupied_threshold = 101;
  EXPECT_THROW(PlannerSupervisor(context, invalid_threshold, tree_xml()),
               std::invalid_argument);

  auto zero_cell_limit = test_config();
  zero_cell_limit.perception.forward_check_footprint.maximum_cells_to_check = 0;
  EXPECT_THROW(PlannerSupervisor(context, zero_cell_limit, tree_xml()),
               std::invalid_argument);
}

} // namespace
