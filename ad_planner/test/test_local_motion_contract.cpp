#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/common/local_motion.hpp"
#include "ad_planner/local_planning/local_motion_factory.hpp"

namespace {
using namespace ad_planner;

static_assert(std::is_default_constructible_v<LocalPlanningRequest>);
static_assert(std::is_default_constructible_v<LocalPlanningResult>);

class CountingParameterProvider final : public LocalMotionParameterProvider {
public:
  double get_double(const std::string &, double default_value) override {
    ++access_count;
    return default_value;
  }

  int get_int(const std::string &, int default_value) override {
    ++access_count;
    return default_value;
  }

  int access_count{0};
};

class BackendWithoutExternalVelocitySupport final : public LocalMotionBackend {
public:
  LocalPlanningResult plan(const LocalPlanningRequest &) override {
    return LocalPlanningResult{};
  }
};

TEST(LocalMotionContract, InvalidResultHasNoCommandOrStaleTrajectory) {
  const LocalPlanningResult result;
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, "invalid");
  EXPECT_TRUE(result.trajectory.points.empty());
  EXPECT_TRUE(result.candidate_trajectories.empty());
  EXPECT_FALSE(result.direct_command.has_value());
}

TEST(LocalMotionContract, DrivableMaskIsSeparateFromObstacleOccupancy) {
  LocalPlanningRequest request;
  request.occupancy_grid.cells = {0};
  request.drivable_mask = OccupancyGrid{};
  request.drivable_mask->cells = {100};

  EXPECT_EQ(request.occupancy_grid.cells.front(), 0);
  EXPECT_EQ(request.drivable_mask->cells.front(), 100);
}

TEST(LocalMotionContract, ExternalVelocityObservationIsBackendNeutralAndOptIn) {
  BackendWithoutExternalVelocitySupport backend;
  const ExternalVelocityCommand command{1.0, 0.2, 123};

  EXPECT_FALSE(backend.observe_external_velocity_command(command));
}

TEST(LocalMotionContract,
     ReferenceLaneSupportsEveryAdjacentSequenceOnEachSide) {
  ReferenceLane lane;
  lane.left_lane_indices = {1U, 2U};
  lane.right_lane_indices = {3U, 4U, 5U};

  EXPECT_EQ(lane.left_lane_indices.size(), 2U);
  EXPECT_EQ(lane.right_lane_indices.size(), 3U);
}

TEST(LocalMotionContract, ParsesSupportedBackendNames) {
  EXPECT_EQ(parse_local_motion_backend("dwa"), LocalMotionBackendKind::kDwa);
  EXPECT_EQ(parse_local_motion_backend("frenet_lattice"),
            LocalMotionBackendKind::kFrenetLattice);
  EXPECT_EQ(parse_local_motion_backend("mppi_nav2"),
            LocalMotionBackendKind::kMppiNav2);
  EXPECT_THROW(parse_local_motion_backend("mppi_cuda"), std::invalid_argument);
  EXPECT_THROW(parse_local_motion_backend("teb"), std::invalid_argument);
}

TEST(LocalMotionContract, UnsupportedBackendFailsWithoutReadingParameters) {
  CountingParameterProvider parameters;
  EXPECT_THROW(
      static_cast<void>(make_local_motion_backend("mppi_cuda", parameters)),
      std::invalid_argument);
  EXPECT_EQ(parameters.access_count, 0);
}
} // namespace
