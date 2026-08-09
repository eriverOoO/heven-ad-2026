#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/frenet/frenet_lattice.hpp"

namespace {

using namespace ad_planner;

ReferencePoint reference_point(const double x, const double y,
                               const double route_s_m,
                               const double speed_limit_mps = 20.0) {
  ReferencePoint point;
  point.pose = Pose2{x, y, 0.0};
  point.route_s_m = route_s_m;
  point.curvature_inv_m = 0.0;
  point.left_width_m = 4.0;
  point.right_width_m = 4.0;
  point.speed_limit_mps = speed_limit_mps;
  return point;
}

ReferenceLane straight_lane(const std::string &id, const double y_m,
                            const double start_x_m = 0.0,
                            const double end_x_m = 40.0) {
  ReferenceLane lane;
  lane.lane_sequence_id = id;
  lane.points = {reference_point(start_x_m, y_m, 0.0),
                 reference_point(end_x_m, y_m, end_x_m - start_x_m)};
  return lane;
}

OccupancyGrid free_grid() {
  OccupancyGrid grid;
  grid.origin = Pose2{-20.0, -20.0, 0.0};
  grid.resolution = 0.25;
  grid.width = 320U;
  grid.height = 160U;
  grid.cells.assign(grid.width * grid.height, 0);
  grid.valid = true;
  grid.fresh = true;
  return grid;
}

void occupy(OccupancyGrid &grid, const double x_m, const double y_m) {
  const auto x = static_cast<std::size_t>(
      std::floor((x_m - grid.origin.x) / grid.resolution));
  const auto y = static_cast<std::size_t>(
      std::floor((y_m - grid.origin.y) / grid.resolution));
  ASSERT_LT(x, grid.width);
  ASSERT_LT(y, grid.height);
  grid.cells[y * grid.width + x] = 100;
}

VehicleConstraints vehicle_constraints() {
  VehicleConstraints constraints;
  constraints.wheelbase_m = 3.0;
  constraints.maximum_steering_rad = 0.6;
  constraints.maximum_speed_mps = 20.0;
  constraints.maximum_acceleration_mps2 = 20.0;
  constraints.maximum_deceleration_mps2 = 20.0;
  constraints.maximum_lateral_acceleration_mps2 = 20.0;
  constraints.maximum_jerk_mps3 = 100.0;
  constraints.footprint_front_m = 1.0;
  constraints.footprint_rear_m = 1.0;
  constraints.footprint_half_width_m = 0.45;
  return constraints;
}

FrenetLatticeConfig permissive_config() {
  FrenetLatticeConfig config;
  config.lateral_targets_m = {0.0};
  config.target_speeds_mps = {2.0};
  config.durations_s = {5.0};
  config.sample_dt_s = 0.25;
  config.maximum_curvature_inv_m = 1.0;
  config.maximum_acceleration_mps2 = 20.0;
  config.maximum_lateral_acceleration_mps2 = 20.0;
  config.maximum_jerk_mps3 = 100.0;
  config.maximum_lateral_transition_m = 5.0;
  config.footprint_clearance_m = 0.0;
  config.occupied_threshold = 100;
  config.maximum_cells_to_check = 4096U;
  config.progress_weight = 1.0;
  config.clearance_weight = 0.0;
  config.jerk_weight = 0.0;
  config.lateral_offset_weight = 0.0;
  config.lane_change_weight = 0.0;
  config.continuity_weight = 0.0;
  config.maximum_candidates = 512U;
  return config;
}

LocalPlanningRequest request_with(const ReferenceCorridor &corridor) {
  LocalPlanningRequest request;
  request.reference_corridor = corridor;
  request.ego = EgoState{Pose2{5.0, 0.0, 0.0}, 2.0, 0.0};
  request.occupancy_grid = free_grid();
  request.constraints = vehicle_constraints();
  request.dt_s = 0.05;
  return request;
}

ReferenceCorridor primary_corridor() {
  ReferenceCorridor corridor;
  corridor.frame_id = "odom";
  corridor.lanes = {straight_lane("primary", 0.0)};
  corridor.primary_lane_index = 0U;
  return corridor;
}

PredictedObject stationary_object(const double x_m, const double y_m,
                                  const double start_s = 0.0,
                                  const double end_s = 5.0) {
  PredictedObject object;
  object.object_id = "obstacle";
  object.footprints = {
      PredictedFootprint{start_s, Pose2{x_m, y_m, 0.0}, 1.0, 1.0, 0.0, 0.0},
      PredictedFootprint{end_s, Pose2{x_m, y_m, 0.0}, 1.0, 1.0, 0.0, 0.0}};
  return object;
}

TEST(FrenetLattice, SelectsStraightCandidateOnFreePrimaryLane) {
  auto config = permissive_config();
  config.durations_s = {2.0};
  config.sample_dt_s = 0.3;
  FrenetLatticeBackend backend(config);

  const auto result = backend.plan(request_with(primary_corridor()));

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_EQ(result.candidate_trajectories.size(), 1U);
  ASSERT_EQ(result.trajectory.frame_id, "odom");
  ASSERT_FALSE(result.trajectory.points.empty());
  EXPECT_DOUBLE_EQ(result.trajectory.points.front().time_from_start_s, 0.0);
  EXPECT_DOUBLE_EQ(result.trajectory.points.back().time_from_start_s, 2.0);
  EXPECT_NEAR(result.trajectory.points.back().pose.y, 0.0, 1e-9);
  EXPECT_GT(result.desired_speed_mps, 0.0);
  EXPECT_FALSE(result.direct_command.has_value());
}

TEST(FrenetLattice, AvoidsStaticGridObstacleWithinFootprint) {
  auto config = permissive_config();
  config.lateral_targets_m = {0.0, 2.0};
  FrenetLatticeBackend backend(config);
  auto request = request_with(primary_corridor());
  occupy(request.occupancy_grid, 12.5, 0.0);

  const auto result = backend.plan(request);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_EQ(result.candidate_trajectories.size(), 1U);
  EXPECT_GT(result.trajectory.points.back().pose.y, 1.5);
}

TEST(FrenetLattice, RejectsTimeIndexedPredictedObjectCollision) {
  auto config = permissive_config();
  config.lateral_targets_m = {0.0, 2.0};
  FrenetLatticeBackend backend(config);
  auto request = request_with(primary_corridor());
  PredictedObject crossing;
  crossing.object_id = "crossing";
  crossing.footprints = {
      PredictedFootprint{0.0, Pose2{10.0, -4.0, 0.0}, 1.0, 0.6, 0.0, 0.0},
      PredictedFootprint{2.25, Pose2{10.0, -4.0, 0.0}, 1.0, 0.6, 0.0, 0.0},
      PredictedFootprint{2.5, Pose2{10.0, 0.0, 0.0}, 1.0, 0.6, 0.0, 0.0},
      PredictedFootprint{2.75, Pose2{10.0, 4.0, 0.0}, 1.0, 0.6, 0.0, 0.0},
      PredictedFootprint{5.0, Pose2{10.0, 4.0, 0.0}, 1.0, 0.6, 0.0, 0.0}};
  request.predicted_objects = {crossing};

  const auto result = backend.plan(request);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_EQ(result.candidate_trajectories.size(), 1U);
  EXPECT_GT(result.trajectory.points.back().pose.y, 1.5);
}

TEST(FrenetLattice, InflatesPredictedObjectFootprintByCovariance) {
  auto config = permissive_config();
  config.lateral_targets_m = {0.0, -2.0};
  FrenetLatticeBackend backend(config);
  auto request = request_with(primary_corridor());
  PredictedObject object;
  object.object_id = "uncertain";
  object.footprints = {
      PredictedFootprint{0.0, Pose2{12.5, 1.1, 0.0}, 1.0, 0.2, 0.0, 0.16},
      PredictedFootprint{5.0, Pose2{12.5, 1.1, 0.0}, 1.0, 0.2, 0.0, 0.16}};
  request.predicted_objects = {object};

  const auto result = backend.plan(request);

  ASSERT_TRUE(result.valid) << result.reason;
  EXPECT_LT(result.trajectory.points.back().pose.y, -1.5);
}

TEST(FrenetLattice, AcceptsPredictionTimesAgedToRequestStamp) {
  FrenetLatticeBackend backend(permissive_config());
  auto request = request_with(primary_corridor());
  request.predicted_objects = {PredictedObject{
      "aged",
      {
          PredictedFootprint{-0.4, Pose2{200.0, 200.0, 0.0}, 1.0, 1.0, 0.0,
                             0.0},
          PredictedFootprint{0.1, Pose2{200.0, 200.0, 0.0}, 1.0, 1.0, 0.0, 0.0},
          PredictedFootprint{0.6, Pose2{200.0, 200.0, 0.0}, 1.0, 1.0, 0.0, 0.0},
      }}};

  const auto result = backend.plan(request);

  EXPECT_TRUE(result.valid) << result.reason;
}

TEST(FrenetLattice, UsesQuinticTransitionToPermittedAdjacentLane) {
  ReferenceCorridor corridor;
  corridor.frame_id = "odom";
  corridor.lanes = {straight_lane("primary", 0.0),
                    straight_lane("left-a", 3.0)};
  corridor.primary_lane_index = 0U;
  corridor.lanes[0].left_lane_indices = {1U};

  FrenetLatticeBackend backend(permissive_config());
  auto request = request_with(corridor);
  occupy(request.occupancy_grid, 12.5, 0.0);

  const auto result = backend.plan(request);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_EQ(result.candidate_trajectories.size(), 1U);
  EXPECT_NEAR(result.trajectory.points.front().pose.y, 0.0, 1e-6);
  EXPECT_NEAR(result.trajectory.points.back().pose.y, 3.0, 1e-6);
  EXPECT_GT(
      result.trajectory.points[result.trajectory.points.size() / 2U].pose.y,
      0.2);
}

TEST(FrenetLattice, NeverConnectsUnrelatedParallelLaneDirectly) {
  ReferenceCorridor corridor;
  corridor.frame_id = "odom";
  corridor.lanes = {straight_lane("primary", 0.0),
                    straight_lane("unrelated", 3.0)};
  corridor.primary_lane_index = 0U;

  FrenetLatticeBackend backend(permissive_config());
  auto request = request_with(corridor);
  occupy(request.occupancy_grid, 9.5, 0.0);

  const auto result = backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_TRUE(result.trajectory.points.empty());
  EXPECT_TRUE(result.candidate_trajectories.empty());
}

TEST(FrenetLattice,
     RejectsCurvatureAccelerationLateralAccelerationAndJerkLimits) {
  const auto corridor = primary_corridor();
  auto request = request_with(corridor);

  auto acceleration = permissive_config();
  acceleration.target_speeds_mps = {8.0};
  acceleration.durations_s = {1.0};
  acceleration.maximum_acceleration_mps2 = 0.1;
  EXPECT_FALSE(FrenetLatticeBackend(acceleration).plan(request).valid);

  auto jerk = permissive_config();
  jerk.target_speeds_mps = {6.0};
  jerk.durations_s = {1.0};
  jerk.maximum_jerk_mps3 = 0.1;
  EXPECT_FALSE(FrenetLatticeBackend(jerk).plan(request).valid);

  auto curvature = permissive_config();
  curvature.lateral_targets_m = {2.0};
  curvature.durations_s = {1.0};
  curvature.maximum_curvature_inv_m = 0.01;
  EXPECT_FALSE(FrenetLatticeBackend(curvature).plan(request).valid);

  auto lateral_acceleration = permissive_config();
  lateral_acceleration.lateral_targets_m = {2.0};
  lateral_acceleration.target_speeds_mps = {8.0};
  lateral_acceleration.durations_s = {1.0};
  lateral_acceleration.maximum_lateral_acceleration_mps2 = 0.1;
  EXPECT_FALSE(FrenetLatticeBackend(lateral_acceleration).plan(request).valid);
}

TEST(FrenetLattice, PrefersPreviousTrajectoryContinuityWhenCostsOtherwiseTie) {
  auto config = permissive_config();
  config.lateral_targets_m = {-1.0, 1.0};
  config.progress_weight = 0.0;
  config.continuity_weight = 10.0;
  FrenetLatticeBackend backend(config);
  auto request = request_with(primary_corridor());
  TimedTrajectory previous;
  previous.frame_id = "odom";
  previous.points = {
      TimedTrajectoryPoint{Pose2{5.0, 0.0, 0.0}, 0.0, 2.0, 0.0},
      TimedTrajectoryPoint{Pose2{8.0, 1.0, 0.0}, 1.5, 2.0, 0.0},
      TimedTrajectoryPoint{Pose2{11.0, 1.0, 0.0}, 3.0, 2.0, 0.0}};
  request.previous_trajectory = previous;

  const auto result = backend.plan(request);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_EQ(result.candidate_trajectories.size(), 2U);
  EXPECT_GT(result.trajectory.points.back().pose.y, 0.5);
}

TEST(FrenetLattice, NoCandidateReturnsInvalidAndNoStaleTrajectory) {
  auto config = permissive_config();
  FrenetLatticeBackend backend(config);
  const auto valid = backend.plan(request_with(primary_corridor()));
  ASSERT_TRUE(valid.valid);
  ASSERT_FALSE(valid.candidate_trajectories.empty());

  auto blocked = request_with(primary_corridor());
  for (double x = 4.0; x <= 12.0; x += 0.25) {
    for (double y = -1.0; y <= 1.0; y += 0.25) {
      occupy(blocked.occupancy_grid, x, y);
    }
  }
  const auto invalid = backend.plan(blocked);

  EXPECT_FALSE(invalid.valid);
  EXPECT_TRUE(invalid.trajectory.points.empty());
  EXPECT_TRUE(invalid.candidate_trajectories.empty());
  EXPECT_FALSE(invalid.direct_command.has_value());
}

TEST(FrenetLattice, CandidateCapAppliesToDeterministicNestedOrder) {
  auto config = permissive_config();
  config.durations_s = {2.0, 3.0};
  config.target_speeds_mps = {2.0, 3.0};
  config.lateral_targets_m = {0.0, 1.0};
  config.maximum_candidates = 1U;
  FrenetLatticeBackend backend(config);

  const auto result = backend.plan(request_with(primary_corridor()));

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_EQ(result.candidate_trajectories.size(), 1U);
  EXPECT_DOUBLE_EQ(result.trajectory.points.back().time_from_start_s, 2.0);
  EXPECT_NEAR(result.trajectory.points.back().pose.y, 0.0, 1e-9);
  EXPECT_NEAR(result.trajectory.points.back().speed_mps, 2.0, 1e-6);
}

TEST(FrenetLattice, CandidateCapAlwaysEvaluatesNonzeroPrimaryIndexFirst) {
  ReferenceCorridor corridor;
  corridor.frame_id = "odom";
  corridor.lanes = {straight_lane("adjacent-lower-index", 3.0),
                    straight_lane("primary", 0.0)};
  corridor.primary_lane_index = 1U;
  corridor.lanes[1U].left_lane_indices = {0U};
  auto config = permissive_config();
  config.maximum_candidates = 1U;
  FrenetLatticeBackend backend(config);

  const auto result = backend.plan(request_with(corridor));

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_EQ(result.candidate_trajectories.size(), 1U);
  EXPECT_NEAR(result.trajectory.points.back().pose.y, 0.0, 1e-9);
}

TEST(FrenetLattice, AcceptsForwardOnlyStationaryStopCandidate) {
  auto config = permissive_config();
  config.target_speeds_mps = {0.0};
  config.durations_s = {2.0};
  FrenetLatticeBackend backend(config);
  auto request = request_with(primary_corridor());
  request.ego.speed_mps = 0.0;

  const auto result = backend.plan(request);

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_FALSE(result.trajectory.points.empty());
  for (const auto &point : result.trajectory.points) {
    EXPECT_NEAR(point.pose.x, request.ego.pose.x, 1e-9);
    EXPECT_NEAR(point.pose.y, request.ego.pose.y, 1e-9);
    EXPECT_DOUBLE_EQ(point.speed_mps, 0.0);
  }
}

TEST(FrenetLattice, InvalidPredictionTimeOrderAndDimensionsFailClosed) {
  FrenetLatticeBackend backend(permissive_config());
  const double nan = std::numeric_limits<double>::quiet_NaN();

  std::vector<PredictedObject> invalid_objects{
      PredictedObject{"empty", {}},
      PredictedObject{
          "duplicate-time",
          {PredictedFootprint{1.0, Pose2{20.0, 0.0, 0.0}, 1.0, 1.0, 0.0, 0.0},
           PredictedFootprint{1.0, Pose2{21.0, 0.0, 0.0}, 1.0, 1.0, 0.0, 0.0}}},
      PredictedObject{
          "reverse-time",
          {PredictedFootprint{2.0, Pose2{20.0, 0.0, 0.0}, 1.0, 1.0, 0.0, 0.0},
           PredictedFootprint{1.0, Pose2{21.0, 0.0, 0.0}, 1.0, 1.0, 0.0, 0.0}}},
      PredictedObject{
          "nan-time",
          {PredictedFootprint{nan, Pose2{20.0, 0.0, 0.0}, 1.0, 1.0, 0.0, 0.0}}},
      PredictedObject{
          "zero-length",
          {PredictedFootprint{0.0, Pose2{20.0, 0.0, 0.0}, 0.0, 1.0, 0.0, 0.0}}},
      PredictedObject{"negative-width",
                      {PredictedFootprint{0.0, Pose2{20.0, 0.0, 0.0}, 1.0, -1.0,
                                          0.0, 0.0}}},
      PredictedObject{
          "nan-pose",
          {PredictedFootprint{0.0, Pose2{nan, 0.0, 0.0}, 1.0, 1.0, 0.0, 0.0}}},
      PredictedObject{
          "nan-covariance",
          {PredictedFootprint{0.0, Pose2{20.0, 0.0, 0.0}, 1.0, 1.0, nan, 0.0}}},
      PredictedObject{
          "unbounded-covariance",
          {PredictedFootprint{0.0, Pose2{20.0, 0.0, 0.0}, 1.0, 1.0, 1.0e200,
                              1.0e200, std::numeric_limits<double>::max()}}},
  };

  for (const auto &object : invalid_objects) {
    auto request = request_with(primary_corridor());
    request.predicted_objects = {object};
    const auto result = backend.plan(request);
    EXPECT_FALSE(result.valid) << object.object_id;
    EXPECT_TRUE(result.trajectory.points.empty()) << object.object_id;
    EXPECT_TRUE(result.candidate_trajectories.empty()) << object.object_id;
  }
}

TEST(FrenetLattice, InvalidConfigArraysAndLimitsAreRejected) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto base = permissive_config();

  for (int mutation = 0; mutation < 17; ++mutation) {
    auto config = base;
    switch (mutation) {
    case 0:
      config.lateral_targets_m.clear();
      break;
    case 1:
      config.target_speeds_mps.clear();
      break;
    case 2:
      config.durations_s.clear();
      break;
    case 3:
      config.lateral_targets_m = {nan};
      break;
    case 4:
      config.target_speeds_mps = {-1.0};
      break;
    case 5:
      config.durations_s = {0.0};
      break;
    case 6:
      config.sample_dt_s = 0.0;
      break;
    case 7:
      config.maximum_curvature_inv_m = 0.0;
      break;
    case 8:
      config.maximum_acceleration_mps2 = nan;
      break;
    case 9:
      config.maximum_lateral_acceleration_mps2 = -1.0;
      break;
    case 10:
      config.maximum_jerk_mps3 = 0.0;
      break;
    case 11:
      config.maximum_lateral_transition_m = -1.0;
      break;
    case 12:
      config.footprint_clearance_m = -1.0;
      break;
    case 13:
      config.occupied_threshold = -1;
      break;
    case 14:
      config.maximum_cells_to_check = 0U;
      break;
    case 15:
      config.progress_weight = -1.0;
      break;
    case 16:
      config.maximum_candidates = 0U;
      break;
    }
    EXPECT_THROW(static_cast<void>(FrenetLatticeBackend(config)),
                 std::invalid_argument)
        << mutation;
  }
}

TEST(FrenetLattice, RejectsEndpointClampedAndFarDisjointAdjacency) {
  auto config = permissive_config();
  config.maximum_lateral_transition_m = 5.0;

  for (const ReferenceLane adjacent :
       {straight_lane("endpoint", 3.0, 5.0, 40.0),
        straight_lane("far", 8.0, 0.0, 40.0)}) {
    ReferenceCorridor corridor;
    corridor.frame_id = "odom";
    corridor.lanes = {straight_lane("primary", 0.0), adjacent};
    corridor.primary_lane_index = 0U;
    corridor.lanes[0].left_lane_indices = {1U};
    FrenetLatticeBackend backend(config);
    auto request = request_with(corridor);
    occupy(request.occupancy_grid, 12.5, 0.0);

    const auto result = backend.plan(request);
    EXPECT_FALSE(result.valid) << adjacent.lane_sequence_id;
    EXPECT_TRUE(result.candidate_trajectories.empty())
        << adjacent.lane_sequence_id;
  }
}

TEST(FrenetLattice, UsesShortestYawInterpolationAndHoldsPredictionEnds) {
  auto config = permissive_config();
  config.lateral_targets_m = {0.0, 2.0};
  FrenetLatticeBackend backend(config);
  auto request = request_with(primary_corridor());
  PredictedObject object;
  object.object_id = "wrapped-yaw";
  object.footprints = {
      PredictedFootprint{1.0, Pose2{12.5, 0.0, 3.12413936106985}, 1.0, 1.0,
                         0.04, 0.04},
      PredictedFootprint{2.0, Pose2{12.5, 0.0, -3.12413936106985}, 1.0, 1.0,
                         0.04, 0.04}};
  request.predicted_objects = {object};

  const auto result = backend.plan(request);

  ASSERT_TRUE(result.valid) << result.reason;
  EXPECT_GT(result.trajectory.points.back().pose.y, 1.5);
}

TEST(FrenetLattice, InvalidRequestAndSpeedLimitFailClosed) {
  FrenetLatticeBackend backend(permissive_config());

  auto invalid_grid = request_with(primary_corridor());
  invalid_grid.occupancy_grid.valid = false;
  EXPECT_FALSE(backend.plan(invalid_grid).valid);

  auto reverse = request_with(primary_corridor());
  reverse.ego.speed_mps = -0.01;
  EXPECT_FALSE(backend.plan(reverse).valid);

  auto malformed_primary = request_with(primary_corridor());
  malformed_primary.reference_corridor.primary_lane_index = 1U;
  EXPECT_FALSE(backend.plan(malformed_primary).valid);

  auto low_speed_lane = primary_corridor();
  for (auto &point : low_speed_lane.lanes.front().points) {
    point.speed_limit_mps = 1.0;
  }
  EXPECT_FALSE(backend.plan(request_with(low_speed_lane)).valid);
}

TEST(FrenetLattice, IgnoresMalformedUnrelatedLaneWithoutSelectingIt) {
  ReferenceCorridor corridor = primary_corridor();
  auto unrelated = straight_lane("unrelated-singular", -1.0);
  for (auto &point : unrelated.points) {
    point.curvature_inv_m = 1.0;
  }
  corridor.lanes.push_back(unrelated);
  FrenetLatticeBackend backend(permissive_config());

  const auto result = backend.plan(request_with(corridor));

  ASSERT_TRUE(result.valid) << result.reason;
  ASSERT_EQ(result.candidate_trajectories.size(), 1U);
  EXPECT_NEAR(result.trajectory.points.back().pose.y, 0.0, 1e-9);
}

TEST(FrenetLattice,
     RejectsSingularAdjacentProjectionWithoutLosingPrimaryCandidate) {
  ReferenceCorridor corridor = primary_corridor();
  auto adjacent = straight_lane("singular-adjacent", -1.0);
  for (auto &point : adjacent.points) {
    point.curvature_inv_m = 1.0;
  }
  corridor.lanes.push_back(adjacent);
  corridor.lanes.front().left_lane_indices = {1U};
  FrenetLatticeBackend backend(permissive_config());

  const auto result = backend.plan(request_with(corridor));

  ASSERT_TRUE(result.valid) << result.reason;
  EXPECT_NEAR(result.trajectory.points.back().pose.y, 0.0, 1e-9);
  ASSERT_EQ(result.candidate_trajectories.size(), 1U);
}

} // namespace
