#include <cmath>
#include <initializer_list>
#include <limits>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/dwa/dwa.hpp"
#include "ad_planner/local_planning/dwa/dwa_backend.hpp"

namespace {
using namespace ad_planner;

DwaConfig test_dwa_config() {
  DwaConfig config;
  config.min_speed_mps = 1.0;
  config.max_speed_mps = 1.0;
  config.speed_step_mps = 0.5;
  config.min_steer_rad = 0.0;
  config.sampled_max_steer_rad = 0.0;
  config.steer_step_rad = 0.1;
  config.dt = 0.5;
  config.horizon_s = 1.0;
  config.wheelbase_m = 1.0;
  config.max_steer_rad = 0.4;
  config.prediction_covariance_sigma = 0.0;
  config.prediction_minimum_margin_m = 0.0;
  config.footprint = FootprintConfig{0.2, 0.2, 0.0, 50, 64};
  config.progress_weight = 1.0;
  config.goal_weight = 1.0;
  config.heading_weight = 1.0;
  config.clearance_weight = 1.0;
  config.smoothness_weight = 1.0;
  config.pid = PidConfig{0.5, 0.0, 0.0, 1.0, 1.0};
  return config;
}

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

LocalPlanningRequest straight_request(const OccupancyGrid &grid,
                                      const Point3 &target) {
  LocalPlanningRequest request;
  request.occupancy_grid = grid;
  request.ego.speed_mps = 0.0;
  request.previous_command.steering_rad = 0.0;
  request.behavior_id = 1;
  request.gear_id = 4;
  request.constraints =
      VehicleConstraints{1.0, 0.4, 1.0, 5.0, 6.0, 6.0, 5.0, 0.2, 0.2, 0.2};
  ReferenceLane lane;
  lane.points.push_back(ReferencePoint{Pose2{target.x, target.y, 0.0}});
  request.reference_corridor.lanes.push_back(lane);
  return request;
}

TEST(DwaBackend, AppliesSharedVehicleSpeedConstraint) {
  auto config = test_dwa_config();
  config.min_speed_mps = 0.0;
  DwaBackend backend(config);
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.constraints.maximum_speed_mps = 0.5;

  const auto result = backend.plan(request);

  ASSERT_TRUE(result.valid) << result.reason;
  EXPECT_LE(result.desired_speed_mps, 0.5);
  for (const auto &point : result.trajectory.points) {
    EXPECT_LE(point.speed_mps, 0.5);
  }
}

TEST(DwaBackend, RejectsInvalidSharedVehicleConstraints) {
  DwaBackend backend(test_dwa_config());
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.constraints.maximum_lateral_acceleration_mps2 = 0.0;

  const auto result = backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason,
            "DWA vehicle constraints must be finite and positive");
}

PredictedFootprint predicted_footprint(double time_s, double x, double y,
                                       double covariance_xx = 0.0,
                                       double covariance_yy = 0.0,
                                       double covariance_xy = 0.0) {
  PredictedFootprint footprint;
  footprint.time_from_start_s = time_s;
  footprint.pose = Pose2{x, y, 0.0};
  footprint.length_m = 0.2;
  footprint.width_m = 0.2;
  footprint.covariance_xx = covariance_xx;
  footprint.covariance_yy = covariance_yy;
  footprint.covariance_xy = covariance_xy;
  return footprint;
}

PredictedObject
predicted_object(std::initializer_list<PredictedFootprint> footprints) {
  return PredictedObject{"object", footprints};
}

OccupancyGrid drivable_mask(const OccupancyGrid &geometry, double minimum_y,
                            double maximum_y) {
  OccupancyGrid mask = geometry;
  std::fill(mask.cells.begin(), mask.cells.end(),
            static_cast<std::int8_t>(100));
  for (std::size_t y = 0U; y < mask.height; ++y) {
    for (std::size_t x = 0U; x < mask.width; ++x) {
      const double world_y =
          mask.origin.y + (static_cast<double>(y) + 0.5) * mask.resolution;
      if (world_y >= minimum_y && world_y <= maximum_y) {
        mask.cells[y * mask.width + x] = 0;
      }
    }
  }
  return mask;
}

TEST(DwaBackend, MatchesLegacyDwaCommandAndTrajectory) {
  DwaController legacy(test_dwa_config());
  DwaBackend backend(test_dwa_config());
  const auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});

  const auto expected =
      legacy.plan(request.occupancy_grid, Pose2{}, Point3{5.0, 0.0, 0.0},
                  request.ego.speed_mps, request.previous_command.steering_rad,
                  request.behavior_id, request.gear_id);
  const auto actual = backend.plan(request);

  ASSERT_TRUE(actual.valid);
  ASSERT_TRUE(actual.direct_command);
  ASSERT_TRUE(expected.valid);
  ASSERT_TRUE(expected.local_trajectory);
  EXPECT_EQ(*actual.direct_command, expected.command);
  ASSERT_EQ(actual.trajectory.points.size(),
            expected.local_trajectory->poses.size());
  for (std::size_t i = 0; i < actual.trajectory.points.size(); ++i) {
    EXPECT_EQ(actual.trajectory.points[i].pose,
              expected.local_trajectory->poses[i]);
    EXPECT_DOUBLE_EQ(actual.trajectory.points[i].time_from_start_s,
                     0.5 * (i + 1));
  }
}

TEST(DwaBackend, InvalidLegacyResultDoesNotExposeFallbackCommandOrTrajectory) {
  auto grid = free_grid();
  grid.cells.assign(grid.cells.size(), static_cast<std::int8_t>(-1));
  DwaBackend backend(test_dwa_config());

  const auto result =
      backend.plan(straight_request(grid, Point3{5.0, 0.0, 0.0}));

  EXPECT_FALSE(result.valid);
  EXPECT_TRUE(result.trajectory.points.empty());
  EXPECT_FALSE(result.direct_command.has_value());
}

TEST(DwaBackend, ForwardsDrivableMaskAsHardConstraint) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.drivable_mask = request.occupancy_grid;
  std::fill(request.drivable_mask->cells.begin(),
            request.drivable_mask->cells.end(), static_cast<std::int8_t>(100));
  DwaBackend backend(test_dwa_config());

  const auto result = backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "initial footprint outside drivable mask");
}

TEST(DwaBackend, RejectsPredictedCrossingAtInterpolatedCandidateTime) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.drivable_mask = drivable_mask(request.occupancy_grid, -1.0, 1.0);
  request.predicted_objects = {predicted_object({
      predicted_footprint(-0.5, 0.25, 3.0),
      predicted_footprint(0.5, 0.25, 0.0),
      predicted_footprint(2.0, 0.25, 3.0),
  })};
  DwaBackend backend(test_dwa_config());

  const auto result = backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "no safe DWA candidate");
}

TEST(DwaBackend, RejectsMovingObjectThatCrossesBetweenDwaEndpoints) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.drivable_mask = drivable_mask(request.occupancy_grid, -1.0, 1.0);
  request.predicted_objects = {predicted_object({
      predicted_footprint(0.0, 0.25, -2.0),
      predicted_footprint(0.5, 0.25, 2.0),
      predicted_footprint(2.0, 0.25, 2.0),
  })};
  DwaBackend backend(test_dwa_config());

  const auto result = backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "no safe DWA candidate");
}

TEST(DwaBackend, RejectsPredictedCollisionInEmergencyBrakingTail) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.drivable_mask = drivable_mask(request.occupancy_grid, -1.0, 1.0);
  request.predicted_objects = {predicted_object({
      predicted_footprint(0.0, 1.25, 0.0),
      predicted_footprint(2.0, 1.25, 0.0),
  })};
  DwaBackend backend(test_dwa_config());

  const auto result = backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "no safe DWA candidate");
}

TEST(DwaBackend, IgnoresTrackedPredictionsWhollyOutsideDrivableRoad) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.drivable_mask = drivable_mask(request.occupancy_grid, -0.5, 0.5);
  request.predicted_objects = {predicted_object({
      predicted_footprint(0.0, 0.25, 4.0),
      predicted_footprint(2.0, 0.25, 4.0),
  })};
  DwaBackend backend(test_dwa_config());

  const auto result = backend.plan(request);

  ASSERT_TRUE(result.valid) << result.reason;
}

TEST(DwaBackend, PredictedObjectsWithoutDrivableMaskFailClosed) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.predicted_objects = {predicted_object({
      predicted_footprint(0.0, 4.0, 4.0),
      predicted_footprint(1.0, 4.0, 4.0),
  })};
  DwaBackend backend(test_dwa_config());

  const auto result = backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "predicted objects require drivable mask");
}

TEST(DwaBackend, MalformedPredictedTimelineFailsClosed) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.drivable_mask = drivable_mask(request.occupancy_grid, -1.0, 1.0);
  request.predicted_objects = {predicted_object({
      predicted_footprint(0.5, 4.0, 0.0),
      predicted_footprint(0.5, 5.0, 0.0),
  })};
  DwaBackend backend(test_dwa_config());

  const auto result = backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "malformed predicted objects");
}

TEST(DwaBackend, NumericallyUnboundedPredictionCovarianceFailsClosed) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.drivable_mask = drivable_mask(request.occupancy_grid, -1.0, 1.0);
  request.predicted_objects = {predicted_object({
      predicted_footprint(0.0, 4.0, 0.0, 1.0e200, 1.0e200,
                          std::numeric_limits<double>::max()),
      predicted_footprint(2.0, 4.0, 0.0, 1.0e200, 1.0e200,
                          std::numeric_limits<double>::max()),
  })};
  DwaBackend backend(test_dwa_config());

  const auto result = backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "malformed predicted objects");
}

TEST(DwaBackend, CovarianceInflationIsAppliedToPredictedFootprint) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.drivable_mask = drivable_mask(request.occupancy_grid, -1.0, 1.0);
  request.predicted_objects = {predicted_object({
      predicted_footprint(0.0, 0.0, 0.75, 0.0, 0.09),
      predicted_footprint(2.0, 0.0, 0.75, 0.0, 0.09),
  })};

  auto no_inflation = test_dwa_config();
  no_inflation.prediction_covariance_sigma = 0.0;
  no_inflation.prediction_minimum_margin_m = 0.0;
  DwaBackend uninflated_backend(no_inflation);
  ASSERT_TRUE(uninflated_backend.plan(request).valid);

  auto inflated = no_inflation;
  inflated.prediction_covariance_sigma = 2.0;
  DwaBackend inflated_backend(inflated);
  const auto result = inflated_backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "initial footprint intersects predicted object");
}

TEST(DwaBackend, InterpolatesAgedNegativePredictionAtCurrentTime) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.drivable_mask = drivable_mask(request.occupancy_grid, -1.0, 1.0);
  request.predicted_objects = {predicted_object({
      predicted_footprint(-1.0, 0.0, 2.0),
      predicted_footprint(2.0, 0.0, -4.0),
  })};
  DwaBackend backend(test_dwa_config());

  const auto result = backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "initial footprint intersects predicted object");
}

TEST(DwaBackend, OnlyExactZeroMaskCellsMakePredictionRoadRelevant) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.drivable_mask = request.occupancy_grid;
  std::fill(request.drivable_mask->cells.begin(),
            request.drivable_mask->cells.end(), static_cast<std::int8_t>(1));
  request.predicted_objects = {predicted_object({
      predicted_footprint(0.0, 0.0, 0.0),
      predicted_footprint(2.0, 0.0, 0.0),
  })};
  DwaBackend backend(test_dwa_config());

  const auto result = backend.plan(request);

  ASSERT_TRUE(result.valid) << result.reason;
}

TEST(DwaBackend, ConservativelyRejectsNearTouchingPredictedFootprints) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.drivable_mask = drivable_mask(request.occupancy_grid, -1.0, 1.0);
  request.predicted_objects = {predicted_object({
      predicted_footprint(0.0, 0.0, 0.3000000000000001),
      predicted_footprint(2.0, 0.0, 0.3000000000000001),
  })};
  DwaBackend backend(test_dwa_config());

  const auto result = backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "initial footprint intersects predicted object");
}

TEST(DwaBackend, RejectsPredictionThatDoesNotCoverBrakingTail) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.drivable_mask = drivable_mask(request.occupancy_grid, -1.0, 1.0);
  request.predicted_objects = {predicted_object({
      predicted_footprint(0.0, 4.0, 0.0),
      predicted_footprint(1.0, 4.0, 0.0),
  })};
  DwaBackend backend(test_dwa_config());

  const auto result = backend.plan(request);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason.rfind(
                "prediction horizon is shorter than rollout and braking", 0U),
            0U);
}

TEST(DwaBackend, EmptyPredictionsDoNotAddTimelineOverflowToLegacyPath) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.ego.speed_mps = std::numeric_limits<double>::max();
  DwaBackend backend(test_dwa_config());
  LocalPlanningResult result;

  EXPECT_NO_THROW(result = backend.plan(request));
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "no safe DWA candidate");
}

TEST(DwaBackend, OversizedPredictionTimelineFailsClosedWithoutThrowing) {
  auto request = straight_request(free_grid(), Point3{5.0, 0.0, 0.0});
  request.ego.speed_mps = std::numeric_limits<double>::max();
  request.drivable_mask = drivable_mask(request.occupancy_grid, -1.0, 1.0);
  request.predicted_objects = {predicted_object({
      predicted_footprint(0.0, 4.0, 0.0),
      predicted_footprint(1.0, 4.0, 0.0),
  })};
  DwaBackend backend(test_dwa_config());
  LocalPlanningResult result;

  EXPECT_NO_THROW(result = backend.plan(request));
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "prediction timeline overflows");
}
} // namespace
