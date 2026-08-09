#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "ad_control/command/curvature_command_adapter.hpp"
#include "ad_planner/local_planning/common/local_motion_runtime.hpp"

namespace ad_planner {
namespace {

class StubBackend final : public LocalMotionBackend {
public:
  explicit StubBackend(LocalPlanningResult result)
      : result_(std::move(result)) {}

  LocalPlanningResult plan(const LocalPlanningRequest &request) override {
    last_request = request;
    ++plan_count;
    return result_;
  }

  bool observe_external_velocity_command(
      const ExternalVelocityCommand &command) override {
    last_external_command = command;
    return true;
  }

  LocalPlanningRequest last_request;
  ExternalVelocityCommand last_external_command;
  int plan_count{0};

private:
  LocalPlanningResult result_;
};

TimedTrajectory valid_trajectory() {
  return TimedTrajectory{
      "odom",
      {
          TimedTrajectoryPoint{Pose2{1.0, 0.0, 0.0}, 0.1, 2.0, 0.0},
          TimedTrajectoryPoint{Pose2{2.0, 0.0, 0.0}, 0.2, 2.0, 0.0},
      }};
}

LocalPlanningRequest valid_request() {
  LocalPlanningRequest request;
  request.reference_corridor.frame_id = "odom";
  request.ego.speed_mps = 2.0;
  request.constraints.maximum_steering_rad = 0.6;
  request.dt_s = 0.05;
  request.behavior_id = 1;
  request.gear_id = 4;
  return request;
}

std::unique_ptr<ad_control::CurvatureCommandAdapter> adapter() {
  return std::make_unique<ad_control::CurvatureCommandAdapter>(
      ad_control::CurvatureCommandAdapterConfig{
          ad_control::PidConfig{0.2, 0.0, 0.0, 10.0, 10.0}, 3.0, 0.6, 2.0});
}

TEST(LocalMotionRuntime, ReturnsBackendDirectCommandAndSelectedTrajectory) {
  LocalPlanningResult planned;
  planned.valid = true;
  planned.reason = "direct";
  planned.trajectory = valid_trajectory();
  planned.desired_speed_mps = 4.0;
  planned.desired_curvature_inv_m = 0.0;
  planned.direct_command = PhysicalCommand{0.3, 0.0, 0.1};
  auto backend = std::make_unique<StubBackend>(planned);

  LocalMotionRuntime runtime(std::move(backend), adapter(),
                             LocalMotionRuntimeConfig{0.6, 0.7});
  const auto result = runtime.plan(valid_request(), 0.0);

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.controller.command, *planned.direct_command);
  EXPECT_EQ(result.controller.reason, "direct");
  ASSERT_TRUE(result.controller.target_speed_mps.has_value());
  EXPECT_DOUBLE_EQ(*result.controller.target_speed_mps, 4.0);
  EXPECT_EQ(result.planning.trajectory.points.size(), 2U);
}

TEST(LocalMotionRuntime, AdaptsCurvatureOnlyBackendsToPhysicalCommand) {
  LocalPlanningResult planned;
  planned.valid = true;
  planned.reason = "curvature";
  planned.trajectory = valid_trajectory();
  planned.desired_speed_mps = 4.0;
  planned.desired_curvature_inv_m = 0.1;

  LocalMotionRuntime runtime(std::make_unique<StubBackend>(planned), adapter(),
                             LocalMotionRuntimeConfig{0.6, 0.7});
  const auto result = runtime.plan(valid_request(), 0.0);

  ASSERT_TRUE(result.valid);
  EXPECT_GT(result.controller.command.steering_rad, 0.0);
  EXPECT_LE(result.controller.command.steering_rad, 0.1);
  EXPECT_GT(result.controller.command.accel, 0.0);
  EXPECT_DOUBLE_EQ(*result.controller.target_speed_mps, 4.0);
}

TEST(LocalMotionRuntime, RejectsMalformedTrajectoryBeforeCommandAdmission) {
  LocalPlanningResult planned;
  planned.valid = true;
  planned.trajectory = valid_trajectory();
  planned.trajectory.frame_id = "map";
  planned.desired_speed_mps = 4.0;
  planned.direct_command = PhysicalCommand{0.3, 0.0, 0.1};

  LocalMotionRuntime runtime(std::make_unique<StubBackend>(planned), adapter(),
                             LocalMotionRuntimeConfig{0.6, 0.7});
  const auto result = runtime.plan(valid_request(), 0.0);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason,
            "local motion backend returned a malformed selected trajectory");
}

TEST(LocalMotionRuntime, RejectsCommandAgainstPlannerOutputLimit) {
  LocalPlanningResult planned;
  planned.valid = true;
  planned.trajectory = valid_trajectory();
  planned.desired_speed_mps = 4.0;
  planned.direct_command = PhysicalCommand{0.0, 0.0, 0.55};

  LocalMotionRuntime runtime(std::make_unique<StubBackend>(planned), adapter(),
                             LocalMotionRuntimeConfig{0.6, 0.5});
  const auto result = runtime.plan(valid_request(), 0.0);

  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason,
            "local motion controller result failed final command admission");
}

TEST(LocalMotionRuntime, ReplacesBackendWithoutRecreatingRuntime) {
  LocalPlanningResult first;
  first.valid = false;
  first.reason = "first";
  LocalMotionRuntime runtime(std::make_unique<StubBackend>(first), adapter(),
                             LocalMotionRuntimeConfig{0.6, 0.7});

  LocalPlanningResult second;
  second.valid = true;
  second.reason = "second";
  second.trajectory = valid_trajectory();
  second.desired_speed_mps = 3.0;
  second.direct_command = PhysicalCommand{0.2, 0.0, 0.0};
  runtime.replace_backend(std::make_unique<StubBackend>(second));

  const auto result = runtime.plan(valid_request(), 0.0);
  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.controller.reason, "second");
}

TEST(LocalMotionRuntime, ForwardsExternalVelocityObservation) {
  LocalPlanningResult planned;
  auto backend = std::make_unique<StubBackend>(planned);
  auto *backend_view = backend.get();
  LocalMotionRuntime runtime(std::move(backend), adapter(),
                             LocalMotionRuntimeConfig{0.6, 0.7});

  const ExternalVelocityCommand command{4.0, 0.2, 123};
  EXPECT_TRUE(runtime.observe_external_velocity_command(command));
  EXPECT_DOUBLE_EQ(backend_view->last_external_command.vx_mps, 4.0);
  EXPECT_EQ(backend_view->last_external_command.receipt_steady_ns, 123);
}

} // namespace
} // namespace ad_planner
