#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/mppi_nav2/mppi_nav2_backend.hpp"

namespace {

using ad_planner::ExternalVelocityCommand;
using ad_planner::LocalPlanningRequest;
using ad_planner::LocalPlanningResult;
using ad_planner::MppiNav2Backend;
using ad_planner::MppiNav2BackendConfig;
using ad_planner::Pose2;
using ad_planner::VehicleConstraints;

constexpr double kPi = 3.1415926535897932384626433832795;

MppiNav2BackendConfig backend_config(const double rollout_dt_s = 0.1,
                                     const double rollout_horizon_s = 0.3) {
  MppiNav2BackendConfig config;
  config.command_timeout_s = 0.2;
  config.diagnostic_rollout_dt_s = rollout_dt_s;
  config.diagnostic_rollout_horizon_s = rollout_horizon_s;
  config.command.wheelbase_m = 3.0;
  config.command.maximum_road_wheel_angle_rad = 0.588;
  config.command.steering_rate_limit_rad_s = 20.0;
  config.command.near_zero_speed_mps = 0.05;
  config.command.allow_reverse = false;
  config.command.longitudinal_pid =
      ad_control::PidConfig{0.5, 0.1, 0.0, 10.0, 10.0};
  return config;
}

VehicleConstraints constraints() {
  return VehicleConstraints{3.0,  0.588, 10.0, 3.0, 5.0,
                            30.0, 5.0,   3.8,  0.8, 0.95};
}

LocalPlanningRequest
request(const std::int64_t steady_time_ns = 1'100'000'000) {
  LocalPlanningRequest request;
  request.ego.pose = Pose2{1.0, -2.0, 0.25};
  request.ego.speed_mps = 1.0;
  request.ego.yaw_rate_radps = 0.0;
  request.previous_command = ad_control::PhysicalCommand{};
  request.constraints = constraints();
  request.stamp_ns = 1'000'000'000;
  request.steady_time_ns = steady_time_ns;
  request.dt_s = 0.05;
  request.behavior_id = 1;
  request.gear_id = 4;
  return request;
}

void expect_invalid(const LocalPlanningResult &result,
                    const std::string &reason) {
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.reason, reason);
  EXPECT_TRUE(result.trajectory.points.empty());
  EXPECT_TRUE(result.candidate_trajectories.empty());
  EXPECT_FALSE(result.direct_command.has_value());
  EXPECT_DOUBLE_EQ(result.desired_speed_mps, 0.0);
  EXPECT_DOUBLE_EQ(result.desired_curvature_inv_m, 0.0);
}

void observe(MppiNav2Backend &backend, const double vx_mps,
             const double wz_rad_s,
             const std::int64_t receipt_steady_ns = 1'000'000'000) {
  ASSERT_TRUE(backend.observe_external_velocity_command(
      ExternalVelocityCommand{vx_mps, wz_rad_s, receipt_steady_ns}));
}

TEST(MppiNav2Backend, FreshCommandProducesDirectCommandAndDiagnosticRollout) {
  MppiNav2Backend backend(backend_config());
  observe(backend, 2.0, 0.4);

  const auto result = backend.plan(request());

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.reason, "ok");
  ASSERT_TRUE(result.direct_command.has_value());
  EXPECT_GT(result.direct_command->accel, 0.0);
  EXPECT_DOUBLE_EQ(result.direct_command->brake, 0.0);
  EXPECT_DOUBLE_EQ(result.desired_speed_mps, 2.0);
  EXPECT_DOUBLE_EQ(result.desired_curvature_inv_m, 0.2);
  EXPECT_EQ(result.trajectory.frame_id, "odom");
  ASSERT_EQ(result.trajectory.points.size(), 4U);
  EXPECT_TRUE(result.candidate_trajectories.empty());
  ASSERT_EQ(result.costs.size(), 1U);
  EXPECT_EQ(result.costs.front().name, "trajectory_kind.command_rollout");
  EXPECT_DOUBLE_EQ(result.costs.front().value, 1.0);

  const auto &initial = result.trajectory.points.front();
  EXPECT_EQ(initial.pose, request().ego.pose);
  EXPECT_DOUBLE_EQ(initial.time_from_start_s, 0.0);
  EXPECT_DOUBLE_EQ(initial.speed_mps, 2.0);
  EXPECT_DOUBLE_EQ(initial.curvature_inv_m, 0.2);
  for (std::size_t index = 1U; index < result.trajectory.points.size();
       ++index) {
    EXPECT_GT(result.trajectory.points[index].time_from_start_s,
              result.trajectory.points[index - 1U].time_from_start_s);
  }
  EXPECT_DOUBLE_EQ(result.trajectory.points.back().time_from_start_s, 0.3);
}

TEST(MppiNav2Backend, NearZeroCommandProducesStoppedRolloutWithoutDivision) {
  MppiNav2Backend backend(backend_config());
  observe(backend, 0.05, 100.0);

  const auto result = backend.plan(request());

  ASSERT_TRUE(result.valid);
  EXPECT_DOUBLE_EQ(result.desired_speed_mps, 0.0);
  EXPECT_DOUBLE_EQ(result.desired_curvature_inv_m, 0.0);
  for (const auto &point : result.trajectory.points) {
    EXPECT_EQ(point.pose, request().ego.pose);
    EXPECT_DOUBLE_EQ(point.speed_mps, 0.0);
    EXPECT_DOUBLE_EQ(point.curvature_inv_m, 0.0);
  }
}

TEST(MppiNav2Backend, MissingFutureAndStaleCommandsHaveExactReasons) {
  MppiNav2Backend missing(backend_config());
  expect_invalid(missing.plan(request()), "mppi command unavailable");

  MppiNav2Backend future(backend_config());
  observe(future, 2.0, 0.0, 1'000'000'000);
  expect_invalid(future.plan(request(999'999'999)),
                 "mppi command receipt is in the future");

  MppiNav2Backend boundary(backend_config());
  observe(boundary, 2.0, 0.0, 1'000'000'000);
  EXPECT_TRUE(boundary.plan(request(1'200'000'000)).valid);

  MppiNav2Backend stale(backend_config());
  observe(stale, 2.0, 0.0, 1'000'000'000);
  expect_invalid(stale.plan(request(1'200'000'001)), "mppi command is stale");
}

TEST(MppiNav2Backend, DroppedObservationsNeverReplaceTheLastValidSample) {
  MppiNav2Backend backend(backend_config());
  observe(backend, 1.5, 0.3, 1'000'000'000);

  EXPECT_FALSE(backend.observe_external_velocity_command(
      ExternalVelocityCommand{9.0, 0.0, 1'000'000'000}));
  EXPECT_FALSE(backend.observe_external_velocity_command(
      ExternalVelocityCommand{9.0, 0.0, 999'999'999}));
  EXPECT_FALSE(
      backend.observe_external_velocity_command(ExternalVelocityCommand{
          std::numeric_limits<double>::quiet_NaN(), 0.0, 1'100'000'000}));
  EXPECT_FALSE(
      backend.observe_external_velocity_command(ExternalVelocityCommand{
          9.0, std::numeric_limits<double>::infinity(), 1'100'000'001}));
  EXPECT_FALSE(backend.observe_external_velocity_command(
      ExternalVelocityCommand{9.0, 0.0, 0}));

  const auto still_fresh = backend.plan(request(1'150'000'000));
  ASSERT_TRUE(still_fresh.valid);
  EXPECT_DOUBLE_EQ(still_fresh.desired_speed_mps, 1.5);
  EXPECT_DOUBLE_EQ(still_fresh.desired_curvature_inv_m, 0.2);

  expect_invalid(backend.plan(request(1'200'000'001)), "mppi command is stale");
}

TEST(MppiNav2Backend,
     AdapterInvalidNewerObservationsDoNotReplaceCommandOrMutatePid) {
  const auto config = backend_config();
  MppiNav2Backend subject(config);
  MppiNav2Backend control(config);
  observe(subject, 2.0, 0.4, 1'000'000'000);
  observe(control, 2.0, 0.4, 1'000'000'000);

  const auto subject_warmup = subject.plan(request(1'050'000'000));
  const auto control_warmup = control.plan(request(1'050'000'000));
  ASSERT_TRUE(subject_warmup.valid);
  ASSERT_TRUE(control_warmup.valid);
  EXPECT_EQ(subject_warmup.direct_command, control_warmup.direct_command);

  EXPECT_FALSE(subject.observe_external_velocity_command(
      ExternalVelocityCommand{-1.0, 0.1, 1'100'000'000}));
  EXPECT_FALSE(
      subject.observe_external_velocity_command(ExternalVelocityCommand{
          0.1, std::numeric_limits<double>::max(), 1'110'000'000}));

  const auto subject_after_drops = subject.plan(request(1'150'000'000));
  const auto control_after_drops = control.plan(request(1'150'000'000));
  ASSERT_TRUE(subject_after_drops.valid);
  ASSERT_TRUE(control_after_drops.valid);
  EXPECT_DOUBLE_EQ(subject_after_drops.desired_speed_mps, 2.0);
  EXPECT_DOUBLE_EQ(subject_after_drops.desired_curvature_inv_m, 0.2);
  EXPECT_EQ(subject_after_drops.direct_command,
            control_after_drops.direct_command);

  expect_invalid(subject.plan(request(1'200'000'001)), "mppi command is stale");
}

TEST(MppiNav2Backend, AdapterInvalidObservationsDoNotAdvanceReceiptWatermark) {
  MppiNav2Backend backend(backend_config());
  observe(backend, 1.0, 0.0, 1'000'000'000);

  EXPECT_FALSE(backend.observe_external_velocity_command(
      ExternalVelocityCommand{-1.0, 0.0, 1'200'000'000}));
  EXPECT_FALSE(
      backend.observe_external_velocity_command(ExternalVelocityCommand{
          0.1, std::numeric_limits<double>::max(), 1'300'000'000}));
  EXPECT_TRUE(backend.observe_external_velocity_command(
      ExternalVelocityCommand{2.0, 0.2, 1'100'000'000}));

  const auto result = backend.plan(request(1'150'000'000));
  ASSERT_TRUE(result.valid);
  EXPECT_DOUBLE_EQ(result.desired_speed_mps, 2.0);
  EXPECT_DOUBLE_EQ(result.desired_curvature_inv_m, 0.1);
}

TEST(MppiNav2Backend, InvalidRequestAndRolloutOverflowDoNotMutatePidState) {
  auto config = backend_config(1.0, 1.0);
  MppiNav2Backend subject(config);
  MppiNav2Backend control(config);
  observe(subject, 1.0, 0.0, 100);
  observe(control, 1.0, 0.0, 100);

  auto warmup = request(110);
  ASSERT_TRUE(subject.plan(warmup).valid);
  ASSERT_TRUE(control.plan(warmup).valid);

  auto invalid = request(120);
  invalid.ego.pose.x = std::numeric_limits<double>::infinity();
  expect_invalid(subject.plan(invalid), "mppi request ego state is invalid");

  auto zero_time = request(0);
  expect_invalid(subject.plan(zero_time),
                 "mppi request steady time is invalid");
  auto nonmonotonic_time = request(109);
  expect_invalid(subject.plan(nonmonotonic_time),
                 "mppi request steady time is not monotonic");

  ASSERT_TRUE(subject.observe_external_velocity_command(
      ExternalVelocityCommand{std::numeric_limits<double>::max(), 0.0, 200}));
  ASSERT_TRUE(control.observe_external_velocity_command(
      ExternalVelocityCommand{std::numeric_limits<double>::max(), 0.0, 200}));
  auto overflowing = request(210);
  overflowing.ego.pose.x = std::numeric_limits<double>::max();
  overflowing.ego.pose.y = 0.0;
  overflowing.ego.pose.yaw_rad = 0.0;
  overflowing.ego.speed_mps = 0.0;
  overflowing.constraints.maximum_speed_mps =
      std::numeric_limits<double>::max();
  expect_invalid(subject.plan(overflowing),
                 "mppi command rollout is not finite");

  ASSERT_TRUE(subject.observe_external_velocity_command(
      ExternalVelocityCommand{1.0, 0.0, 300}));
  ASSERT_TRUE(control.observe_external_velocity_command(
      ExternalVelocityCommand{1.0, 0.0, 300}));
  const auto subject_next = subject.plan(request(310));
  const auto control_next = control.plan(request(310));
  ASSERT_TRUE(subject_next.valid);
  ASSERT_TRUE(control_next.valid);
  EXPECT_EQ(subject_next.direct_command, control_next.direct_command);
}

TEST(MppiNav2Backend, RejectsMalformedRequestsBeforeCommandAdaptation) {
  const auto rejected = [](LocalPlanningRequest malformed) {
    MppiNav2Backend backend(backend_config());
    observe(backend, 2.0, 0.2);
    const auto result = backend.plan(malformed);
    EXPECT_FALSE(result.valid);
    EXPECT_FALSE(result.reason.empty());
    EXPECT_FALSE(result.direct_command.has_value());
  };

  auto malformed = request();
  malformed.steady_time_ns = 0;
  rejected(malformed);
  malformed = request();
  malformed.stamp_ns = 0;
  rejected(malformed);
  malformed = request();
  malformed.ego.speed_mps = -1.0;
  rejected(malformed);
  malformed = request();
  malformed.ego.yaw_rate_radps = std::numeric_limits<double>::quiet_NaN();
  rejected(malformed);
  malformed = request();
  malformed.dt_s = 0.0;
  rejected(malformed);
  malformed = request();
  malformed.previous_command.accel = 0.5;
  malformed.previous_command.brake = 0.5;
  rejected(malformed);
  malformed = request();
  malformed.constraints.maximum_steering_rad = kPi / 2.0;
  rejected(malformed);
  malformed = request();
  malformed.behavior_id = -1;
  rejected(malformed);
  malformed = request();
  malformed.gear_id = 2;
  rejected(malformed);
}

TEST(MppiNav2Backend, EnforcesRequestSpeedAndLateralAccelerationBounds) {
  MppiNav2Backend speed_boundary(backend_config());
  observe(speed_boundary, 10.0, 0.0);
  EXPECT_TRUE(speed_boundary.plan(request()).valid);

  MppiNav2Backend too_fast(backend_config());
  observe(too_fast,
          std::nextafter(10.0, std::numeric_limits<double>::infinity()), 0.0);
  expect_invalid(too_fast.plan(request()),
                 "mppi command exceeds maximum speed");

  MppiNav2Backend too_much_lateral_acceleration(backend_config());
  observe(too_much_lateral_acceleration, 6.0, 6.0);
  expect_invalid(too_much_lateral_acceleration.plan(request()),
                 "mppi command exceeds maximum lateral acceleration");
}

TEST(MppiNav2Backend, RejectsIncoherentAdapterGeometryBeforePidMutation) {
  auto config = backend_config();
  config.command.maximum_road_wheel_angle_rad = 0.6;
  MppiNav2Backend steering_backend(config);
  observe(steering_backend, 2.0, 0.2);
  auto constrained_request = request();
  constrained_request.constraints.maximum_steering_rad = 0.5;

  expect_invalid(steering_backend.plan(constrained_request),
                 "mppi adapter steering limit exceeds request constraint");

  config = backend_config();
  config.command.wheelbase_m = 2.9;
  MppiNav2Backend wheelbase_backend(config);
  observe(wheelbase_backend, 2.0, 0.2);
  expect_invalid(wheelbase_backend.plan(request()),
                 "mppi adapter wheelbase does not match request constraint");
}

TEST(MppiNav2Backend, IntegratesStraightTurningAndWrappedYawDeterministically) {
  MppiNav2Backend straight(backend_config(0.5, 1.0));
  observe(straight, 2.0, 0.0);
  auto straight_request = request();
  straight_request.ego.pose = Pose2{0.0, 0.0, kPi / 2.0};
  const auto straight_result = straight.plan(straight_request);
  ASSERT_TRUE(straight_result.valid);
  ASSERT_EQ(straight_result.trajectory.points.size(), 3U);
  EXPECT_NEAR(straight_result.trajectory.points[1].pose.x, 0.0, 1e-12);
  EXPECT_NEAR(straight_result.trajectory.points[1].pose.y, 1.0, 1e-12);
  EXPECT_NEAR(straight_result.trajectory.points[2].pose.y, 2.0, 1e-12);

  MppiNav2Backend turning(backend_config(0.5, 1.0));
  observe(turning, 2.0, 1.0);
  auto turning_request = request();
  turning_request.ego.pose = Pose2{0.0, 0.0, 3.0};
  turning_request.constraints.maximum_lateral_acceleration_mps2 = 10.0;
  const auto turning_result = turning.plan(turning_request);
  ASSERT_TRUE(turning_result.valid);
  ASSERT_EQ(turning_result.trajectory.points.size(), 3U);
  EXPECT_NEAR(turning_result.trajectory.points[1].pose.x, std::cos(3.0), 1e-12);
  EXPECT_NEAR(turning_result.trajectory.points[1].pose.y, std::sin(3.0), 1e-12);
  EXPECT_NEAR(turning_result.trajectory.points[1].pose.yaw_rad,
              std::remainder(3.5, 2.0 * kPi), 1e-12);
  EXPECT_GE(turning_result.trajectory.points.back().pose.yaw_rad, -kPi);
  EXPECT_LE(turning_result.trajectory.points.back().pose.yaw_rad, kPi);
}

TEST(MppiNav2Backend, ReverseCommandIsRejectedByTheCommonMppiAdapter) {
  MppiNav2Backend backend(backend_config());
  EXPECT_FALSE(backend.observe_external_velocity_command(
      ExternalVelocityCommand{-1.0, 0.1, 1'000'000'000}));

  expect_invalid(backend.plan(request()), "mppi command unavailable");
}

TEST(MppiNav2Backend, ConstructorRejectsInvalidTimingAndRolloutCardinality) {
  const auto expect_bad_config = [](MppiNav2BackendConfig config) {
    EXPECT_THROW((void)MppiNav2Backend{config}, std::invalid_argument);
  };
  for (const double timeout :
       {0.0, -1.0, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    auto config = backend_config();
    config.command_timeout_s = timeout;
    expect_bad_config(config);
  }
  for (const double dt : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()}) {
    auto config = backend_config();
    config.diagnostic_rollout_dt_s = dt;
    expect_bad_config(config);
  }
  for (const double horizon :
       {0.0, -1.0, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    auto config = backend_config();
    config.diagnostic_rollout_horizon_s = horizon;
    expect_bad_config(config);
  }

  auto nonintegral = backend_config(0.2, 0.3);
  expect_bad_config(nonintegral);
  auto too_many = backend_config(0.001, 20.0);
  expect_bad_config(too_many);

  auto scale_tolerant = backend_config(
      0.1, std::nextafter(0.3, std::numeric_limits<double>::infinity()));
  EXPECT_NO_THROW((void)MppiNav2Backend{scale_tolerant});
}

TEST(MppiNav2Backend, ConstructorDelegatesCommandConfigurationValidation) {
  auto invalid = backend_config();
  invalid.command.near_zero_speed_mps = -0.1;
  EXPECT_THROW((void)MppiNav2Backend{invalid}, std::invalid_argument);

  invalid = backend_config();
  invalid.command.allow_reverse = true;
  EXPECT_THROW((void)MppiNav2Backend{invalid}, std::invalid_argument);
}

TEST(MppiNav2Backend, ConcurrentObservationAndPlanningRemainSemanticallySafe) {
  auto config = backend_config(0.01, 0.1);
  config.command_timeout_s = 1.0e20;
  MppiNav2Backend backend(config);
  observe(backend, 1.0, 0.1, 1);
  auto planning_request = request(std::numeric_limits<std::int64_t>::max());
  planning_request.constraints.maximum_lateral_acceleration_mps2 = 100.0;
  std::atomic<bool> failed{false};

  std::thread observer([&]() {
    for (std::int64_t index = 2; index < 1000; ++index) {
      const double speed = 1.0 + 0.001 * static_cast<double>(index % 500);
      if (!backend.observe_external_velocity_command(
              ExternalVelocityCommand{speed, 0.1 * speed, index})) {
        failed = true;
        return;
      }
    }
  });
  std::vector<std::thread> planners;
  for (int thread_index = 0; thread_index < 3; ++thread_index) {
    planners.emplace_back([&]() {
      for (int iteration = 0; iteration < 200; ++iteration) {
        const auto result = backend.plan(planning_request);
        if (!result.valid || !result.direct_command ||
            result.trajectory.points.size() != 11U ||
            result.desired_speed_mps < 1.0 || result.desired_speed_mps >= 1.5 ||
            std::abs(result.desired_curvature_inv_m - 0.1) > 1e-12) {
          failed = true;
          return;
        }
      }
    });
  }

  observer.join();
  for (auto &planner : planners) {
    planner.join();
  }
  EXPECT_FALSE(failed.load());
}

} // namespace
