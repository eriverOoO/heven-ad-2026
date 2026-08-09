#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <cmath>

#include <fast_lio/wheel_velocity_update.hpp>
#include <fast_lio/wheel_velocity_buffer.hpp>
#include <fast_lio/longitudinal_position_guard.hpp>
#include <fast_lio/mapping_planar_position_guard.hpp>
#include <fast_lio/wheel_position_increment.hpp>

namespace {

using Covariance = fast_lio::WheelVelocityUpdate::Covariance;

TEST(WheelVelocityUpdate, RotatesForwardSpeedIntoTheWorldFrame) {
  Eigen::Vector3d world_velocity = Eigen::Vector3d::Zero();
  Covariance covariance = Covariance::Identity();
  const Eigen::Matrix3d world_R_body =
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const auto result = fast_lio::WheelVelocityUpdate::apply(
      world_R_body, 2.0, 0.04, true, 0.04, 0.04,
      world_velocity, covariance);

  ASSERT_TRUE(result.accepted);
  EXPECT_NEAR(world_velocity.x(), 0.0, 1.0e-12);
  EXPECT_GT(world_velocity.y(), 1.9);
  EXPECT_NEAR(world_velocity.z(), 0.0, 1.0e-12);
  EXPECT_NEAR(result.innovation_body.x(), 2.0, 1.0e-12);
}

TEST(WheelVelocityUpdate, NonholonomicConstraintRemovesSideAndVerticalMotion) {
  Eigen::Vector3d world_velocity(3.0, 1.0, -0.5);
  Covariance covariance = Covariance::Identity();

  const auto result = fast_lio::WheelVelocityUpdate::apply(
      Eigen::Matrix3d::Identity(), 3.0, 0.04, true, 0.01, 0.01,
      world_velocity, covariance);

  ASSERT_TRUE(result.accepted);
  EXPECT_NEAR(world_velocity.x(), 3.0, 1.0e-12);
  EXPECT_LT(std::abs(world_velocity.y()), 0.02);
  EXPECT_LT(std::abs(world_velocity.z()), 0.01);
}

TEST(WheelVelocityUpdate, ForwardOnlyLeavesUnmeasuredAxesUntouched) {
  Eigen::Vector3d world_velocity(0.0, 1.0, -0.5);
  Covariance covariance = Covariance::Identity();

  const auto result = fast_lio::WheelVelocityUpdate::apply(
      Eigen::Matrix3d::Identity(), 2.0, 0.04, false, 0.01, 0.01,
      world_velocity, covariance);

  ASSERT_TRUE(result.accepted);
  EXPECT_GT(world_velocity.x(), 1.9);
  EXPECT_DOUBLE_EQ(world_velocity.y(), 1.0);
  EXPECT_DOUBLE_EQ(world_velocity.z(), -0.5);
}

TEST(WheelVelocityUpdate, MeasuredLateralVelocityOverridesOnlyPlanarComponents) {
  Eigen::Vector3d world_velocity(0.0, 1.0, -0.5);
  Covariance covariance = Covariance::Identity();

  const auto result = fast_lio::WheelVelocityUpdate::apply(
      Eigen::Matrix3d::Identity(), 2.0, 0.04,
      std::optional<double>(-0.15), 0.01, false, 0.04,
      world_velocity, covariance);

  ASSERT_TRUE(result.accepted);
  EXPECT_NEAR(world_velocity.x(), 2.0, 1.0e-12);
  EXPECT_NEAR(world_velocity.y(), -0.15, 1.0e-12);
  EXPECT_DOUBLE_EQ(world_velocity.z(), -0.5);
  EXPECT_NEAR(result.innovation_body.y(), -1.15, 1.0e-12);
}

TEST(WheelVelocityUpdate, RejectsMeasuredLateralVelocityWithoutVariance) {
  Eigen::Vector3d world_velocity = Eigen::Vector3d::Zero();
  Covariance covariance = Covariance::Identity();

  EXPECT_FALSE(fast_lio::WheelVelocityUpdate::apply(
      Eigen::Matrix3d::Identity(), 2.0, 0.04,
      std::optional<double>(-0.15), 0.0, false, 0.04,
      world_velocity, covariance).accepted);
}

TEST(WheelVelocityUpdate, ExternalOdometryOverridesAnOverconfidentStoppedState) {
  Eigen::Vector3d world_velocity = Eigen::Vector3d::Zero();
  Covariance covariance = Covariance::Identity();
  covariance.block<3, 3>(
      fast_lio::WheelVelocityUpdate::kVelocityIndex,
      fast_lio::WheelVelocityUpdate::kVelocityIndex) =
      1.0e-9 * Eigen::Matrix3d::Identity();

  const auto result = fast_lio::WheelVelocityUpdate::apply(
      Eigen::Matrix3d::Identity(), 1.0, 0.04, true, 0.04, 0.04,
      world_velocity, covariance);

  ASSERT_TRUE(result.accepted);
  EXPECT_NEAR(world_velocity.x(), 1.0, 1.0e-12);
  EXPECT_NEAR(world_velocity.y(), 0.0, 1.0e-12);
  EXPECT_NEAR(world_velocity.z(), 0.0, 1.0e-12);
}

TEST(WheelVelocityUpdate, RejectsNonfiniteAndInvalidVariance) {
  Eigen::Vector3d world_velocity = Eigen::Vector3d::Zero();
  Covariance covariance = Covariance::Identity();

  EXPECT_FALSE(fast_lio::WheelVelocityUpdate::apply(
      Eigen::Matrix3d::Identity(), std::nan(""), 0.04, true, 0.04, 0.04,
      world_velocity, covariance).accepted);
  EXPECT_FALSE(fast_lio::WheelVelocityUpdate::apply(
      Eigen::Matrix3d::Identity(), 1.0, 0.0, true, 0.04, 0.04,
      world_velocity, covariance).accepted);
}

TEST(WheelVelocityUpdate, KeepsCovarianceFiniteSymmetricAndPositive) {
  Eigen::Vector3d world_velocity = Eigen::Vector3d::Zero();
  Covariance covariance = Covariance::Identity();
  covariance.block<3, 3>(0, fast_lio::WheelVelocityUpdate::kVelocityIndex)
      .setConstant(0.05);
  covariance.block<3, 3>(fast_lio::WheelVelocityUpdate::kVelocityIndex, 0) =
      covariance.block<3, 3>(0, fast_lio::WheelVelocityUpdate::kVelocityIndex)
          .transpose();

  const auto result = fast_lio::WheelVelocityUpdate::apply(
      Eigen::Matrix3d::Identity(), 1.0, 0.04, true, 0.04, 0.04,
      world_velocity, covariance);

  ASSERT_TRUE(result.accepted);
  EXPECT_TRUE(covariance.allFinite());
  EXPECT_LT((covariance - covariance.transpose()).norm(), 1.0e-12);
  Eigen::SelfAdjointEigenSolver<Covariance> solver(covariance);
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_GT(solver.eigenvalues().minCoeff(), -1.0e-10);
  EXPECT_LT(
      covariance(fast_lio::WheelVelocityUpdate::kVelocityIndex,
                 fast_lio::WheelVelocityUpdate::kVelocityIndex),
      1.0);
}

TEST(LongitudinalPositionGuard, RemovesOnlyTheBodyForwardLidarCorrection) {
  using Guard = fast_lio::LongitudinalPositionGuard;
  const Eigen::Matrix3d world_R_body =
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Vector3d predicted_position(10.0, 20.0, 3.0);
  Eigen::Vector3d corrected_position(8.0, 22.0, 4.0);
  Guard::Covariance predicted_covariance = Guard::Covariance::Identity();
  Guard::Covariance corrected_covariance = Guard::Covariance::Identity();
  predicted_covariance(1, 1) = 0.25;

  const auto result = Guard::apply(
      world_R_body, predicted_position, predicted_covariance, 0.04,
      corrected_position, corrected_covariance);

  ASSERT_TRUE(result.accepted);
  EXPECT_NEAR(result.removed_correction_m, 2.0, 1.0e-12);
  EXPECT_NEAR(corrected_position.x(), 8.0, 1.0e-12);
  EXPECT_NEAR(corrected_position.y(), 20.0, 1.0e-12);
  EXPECT_NEAR(corrected_position.z(), 4.0, 1.0e-12);
  EXPECT_NEAR(corrected_covariance(1, 1), 0.25, 1.0e-12);
  EXPECT_TRUE(corrected_covariance.allFinite());
  EXPECT_LT(
      (corrected_covariance - corrected_covariance.transpose()).norm(),
      1.0e-12);
  Eigen::SelfAdjointEigenSolver<Guard::Covariance> solver(corrected_covariance);
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_GT(solver.eigenvalues().minCoeff(), -1.0e-10);
}

TEST(LongitudinalPositionGuard, AppliesTheConfiguredVarianceFloor) {
  using Guard = fast_lio::LongitudinalPositionGuard;
  Guard::Covariance predicted_covariance = Guard::Covariance::Identity();
  Guard::Covariance corrected_covariance = Guard::Covariance::Identity();
  predicted_covariance(0, 0) = 1.0e-9;
  Eigen::Vector3d corrected_position(1.0, 2.0, 3.0);

  const auto result = Guard::apply(
      Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(),
      predicted_covariance, 0.04, corrected_position, corrected_covariance);

  ASSERT_TRUE(result.accepted);
  EXPECT_NEAR(corrected_position.x(), 0.0, 1.0e-12);
  EXPECT_NEAR(corrected_covariance(0, 0), 0.04, 1.0e-12);
}

TEST(MappingPlanarPositionGuard, AnchorsPlanarWheelPositionAndKeepsUpCorrection) {
  using Guard = fast_lio::MappingPlanarPositionGuard;
  Guard::Covariance predicted_covariance = Guard::Covariance::Identity();
  Guard::Covariance corrected_covariance = Guard::Covariance::Identity();
  const Eigen::Vector3d wheel_predicted_position(10.0, 21.0, 3.0);
  Eigen::Vector3d corrected_position(12.0, 22.0, 4.0);

  const auto result = Guard::apply(
      Eigen::Matrix3d::Identity(), wheel_predicted_position,
      predicted_covariance, 0.04,
      corrected_position, corrected_covariance);

  ASSERT_TRUE(result.accepted);
  EXPECT_NEAR(result.removed_forward_correction_m, 2.0, 1.0e-12);
  EXPECT_NEAR(result.removed_lateral_correction_m, 1.0, 1.0e-12);
  EXPECT_NEAR(corrected_position.x(), 10.0, 1.0e-12);
  EXPECT_NEAR(corrected_position.y(), 21.0, 1.0e-12);
  EXPECT_NEAR(corrected_position.z(), 4.0, 1.0e-12);
}

TEST(MappingPlanarPositionGuard, KeepsCovarianceSymmetricAndPositive) {
  using Guard = fast_lio::MappingPlanarPositionGuard;
  Guard::Covariance predicted_covariance = Guard::Covariance::Identity();
  Guard::Covariance corrected_covariance = Guard::Covariance::Identity();
  corrected_covariance.block<3, 3>(0, 3).setConstant(0.05);
  corrected_covariance.block<3, 3>(3, 0) =
      corrected_covariance.block<3, 3>(0, 3).transpose();
  Eigen::Vector3d corrected_position(1.0, 2.0, 3.0);

  const auto result = Guard::apply(
      Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(),
      predicted_covariance, 0.04, corrected_position, corrected_covariance);

  ASSERT_TRUE(result.accepted);
  EXPECT_LT(
      (corrected_covariance - corrected_covariance.transpose()).norm(),
      1.0e-12);
  Eigen::SelfAdjointEigenSolver<Guard::Covariance> solver(corrected_covariance);
  ASSERT_EQ(solver.info(), Eigen::Success);
  EXPECT_GT(solver.eigenvalues().minCoeff(), -1.0e-10);
}

TEST(WheelPositionIncrement, HoldsLongitudinalPositionAtStandstill) {
  const Eigen::Vector3d previous_position(1.0, 2.0, 3.0);

  const auto result = fast_lio::WheelPositionIncrement::integrate(
      Eigen::Matrix3d::Identity(), previous_position,
      0.0, 0.0, 0.1, 0.5);

  ASSERT_TRUE(result.accepted);
  EXPECT_NEAR(result.distance_m, 0.0, 1.0e-12);
  EXPECT_TRUE(result.position.isApprox(previous_position, 1.0e-12));
}

TEST(WheelPositionIncrement, IntegratesTrapezoidalSpeedAlongBodyForward) {
  const Eigen::Matrix3d world_R_body =
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const auto result = fast_lio::WheelPositionIncrement::integrate(
      world_R_body, Eigen::Vector3d::Zero(), 1.0, 3.0, 0.5, 0.5);

  ASSERT_TRUE(result.accepted);
  EXPECT_NEAR(result.distance_m, 1.0, 1.0e-12);
  EXPECT_NEAR(result.position.x(), 0.0, 1.0e-12);
  EXPECT_NEAR(result.position.y(), 1.0, 1.0e-12);
  EXPECT_NEAR(result.position.z(), 0.0, 1.0e-12);
}

TEST(WheelPositionIncrement, IntegratesMeasuredPlanarBodyVelocity) {
  const Eigen::Matrix3d world_R_body =
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const auto result = fast_lio::WheelPositionIncrement::integrate_planar(
      world_R_body, Eigen::Vector3d::Zero(),
      1.0, -0.2, 3.0, -0.4, 0.5, 0.5);

  ASSERT_TRUE(result.accepted);
  EXPECT_NEAR(result.forward_distance_m, 1.0, 1.0e-12);
  EXPECT_NEAR(result.lateral_distance_m, -0.15, 1.0e-12);
  EXPECT_NEAR(result.position.x(), 0.15, 1.0e-12);
  EXPECT_NEAR(result.position.y(), 1.0, 1.0e-12);
  EXPECT_NEAR(result.position.z(), 0.0, 1.0e-12);
}

TEST(WheelPositionIncrement, RejectsInvalidOrExcessiveTimeSteps) {
  EXPECT_FALSE(fast_lio::WheelPositionIncrement::integrate(
      Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(),
      0.0, 0.0, 0.0, 0.5).accepted);
  EXPECT_FALSE(fast_lio::WheelPositionIncrement::integrate(
      Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(),
      0.0, 0.0, 0.6, 0.5).accepted);
}

TEST(WheelPositionIncrement, UsesConfiguredSensorPeriodInsteadOfHostArrivalDelta) {
  EXPECT_DOUBLE_EQ(
      fast_lio::WheelPositionIncrement::select_interval(0.124, 0.1, 0.5),
      0.1);
  EXPECT_DOUBLE_EQ(
      fast_lio::WheelPositionIncrement::select_interval(0.124, 0.0, 0.5),
      0.124);
  EXPECT_FALSE(std::isfinite(
      fast_lio::WheelPositionIncrement::select_interval(0.6, 0.0, 0.5)));
}

TEST(WheelPositionIncrement, AppliesMeasuredBodyFrameDistanceWithoutAClockDelta) {
  const Eigen::Matrix3d world_R_body =
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const auto result = fast_lio::WheelPositionIncrement::apply_displacement(
      world_R_body, Eigen::Vector3d(1.0, 2.0, 3.0),
      0.4, -0.1, 1.0);

  ASSERT_TRUE(result.accepted);
  EXPECT_NEAR(result.position.x(), 1.1, 1.0e-12);
  EXPECT_NEAR(result.position.y(), 2.4, 1.0e-12);
  EXPECT_NEAR(result.position.z(), 3.0, 1.0e-12);
}

TEST(WheelPositionIncrement, DetectsOnlyAnAdvancedCumulativeMeasurement) {
  EXPECT_FALSE(fast_lio::WheelPositionIncrement::cumulative_distance_advanced(
      1.0, -0.2, 1.0, -0.2));
  EXPECT_TRUE(fast_lio::WheelPositionIncrement::cumulative_distance_advanced(
      1.0, -0.2, 1.01, -0.2));
  EXPECT_TRUE(fast_lio::WheelPositionIncrement::cumulative_distance_advanced(
      1.0, -0.2, 1.0, -0.21));
}

TEST(WheelVelocityBuffer, SelectsLatestEligiblePastSampleAndRetainsFuture) {
  fast_lio::WheelVelocityBuffer buffer(8);
  ASSERT_TRUE(buffer.push({10.00, 0.8, 0.04}));
  ASSERT_TRUE(buffer.push({10.04, 0.9, 0.04}));
  ASSERT_TRUE(buffer.push({10.08, 1.0, 0.04}));

  const auto first = buffer.take_for_scan(10.05, 0.25, 0.01);
  ASSERT_TRUE(first.has_value());
  EXPECT_DOUBLE_EQ(first->stamp_sec, 10.04);
  EXPECT_DOUBLE_EQ(first->forward_speed_mps, 0.9);

  const auto second = buffer.take_for_scan(10.08, 0.25, 0.01);
  ASSERT_TRUE(second.has_value());
  EXPECT_DOUBLE_EQ(second->stamp_sec, 10.08);
}

TEST(WheelVelocityBuffer, PreservesOptionalLateralMeasurement) {
  fast_lio::WheelVelocityBuffer buffer(4);
  fast_lio::TimedWheelVelocity sample{10.0, 1.0, 0.04};
  sample.lateral_speed_mps = -0.12;
  sample.lateral_variance = 0.01;
  ASSERT_TRUE(buffer.push(sample));

  const auto selected = buffer.take_for_scan(10.0, 0.25, 0.01);
  ASSERT_TRUE(selected.has_value());
  ASSERT_TRUE(selected->lateral_speed_mps.has_value());
  EXPECT_DOUBLE_EQ(*selected->lateral_speed_mps, -0.12);
  EXPECT_DOUBLE_EQ(selected->lateral_variance, 0.01);
}

TEST(WheelVelocityBuffer, IntegratesDistanceInDeviceTimeButSelectsByArrivalTime) {
  fast_lio::WheelVelocityBuffer buffer(8);
  fast_lio::TimedWheelVelocity first{10.00, 1.0, 0.04};
  first.distance_stamp_sec = 100.00;
  first.lateral_speed_mps = -0.1;
  first.lateral_variance = 0.01;
  ASSERT_TRUE(buffer.push(first));

  fast_lio::TimedWheelVelocity second{10.20, 3.0, 0.04};
  second.distance_stamp_sec = 100.10;
  second.lateral_speed_mps = -0.3;
  second.lateral_variance = 0.01;
  ASSERT_TRUE(buffer.push(second));

  const auto before_arrival = buffer.take_for_scan(10.10, 0.25, 0.0);
  ASSERT_TRUE(before_arrival.has_value());
  ASSERT_TRUE(before_arrival->cumulative_forward_distance_m.has_value());
  EXPECT_NEAR(*before_arrival->cumulative_forward_distance_m, 0.0, 1.0e-12);

  const auto after_arrival = buffer.take_for_scan(10.20, 0.25, 0.0);
  ASSERT_TRUE(after_arrival.has_value());
  ASSERT_TRUE(after_arrival->cumulative_forward_distance_m.has_value());
  ASSERT_TRUE(after_arrival->cumulative_lateral_distance_m.has_value());
  EXPECT_NEAR(*after_arrival->cumulative_forward_distance_m, 0.2, 1.0e-12);
  EXPECT_NEAR(*after_arrival->cumulative_lateral_distance_m, -0.02, 1.0e-12);
}

TEST(WheelVelocityBuffer, RejectsDuplicateDeviceDistanceTimestamps) {
  fast_lio::WheelVelocityBuffer buffer(4);
  fast_lio::TimedWheelVelocity first{10.0, 1.0, 0.04};
  first.distance_stamp_sec = 100.0;
  ASSERT_TRUE(buffer.push(first));
  fast_lio::TimedWheelVelocity duplicate{10.1, 1.0, 0.04};
  duplicate.distance_stamp_sec = 100.0;
  EXPECT_FALSE(buffer.push(duplicate));
}

TEST(WheelVelocityBuffer, DoesNotIntegrateAcrossAnUnboundedDeviceGap) {
  fast_lio::WheelVelocityBuffer buffer(4);
  fast_lio::TimedWheelVelocity first{10.0, 2.0, 0.04};
  first.distance_stamp_sec = 100.0;
  ASSERT_TRUE(buffer.push(first));

  fast_lio::TimedWheelVelocity after_gap{11.0, 2.0, 0.04};
  after_gap.distance_stamp_sec = 101.0;
  ASSERT_TRUE(buffer.push(after_gap));

  const auto selected = buffer.take_for_scan(11.0, 0.25, 0.0);
  ASSERT_TRUE(selected.has_value());
  ASSERT_TRUE(selected->cumulative_forward_distance_m.has_value());
  EXPECT_NEAR(*selected->cumulative_forward_distance_m, 0.0, 1.0e-12);
}

TEST(WheelVelocityBuffer, RejectsInvalidOptionalLateralMeasurement) {
  fast_lio::WheelVelocityBuffer buffer(4);
  fast_lio::TimedWheelVelocity sample{10.0, 1.0, 0.04};
  sample.lateral_speed_mps = -0.12;
  sample.lateral_variance = 0.0;
  EXPECT_FALSE(buffer.push(sample));
}

TEST(WheelVelocityBuffer, ReusesTheLatestFreshSampleAcrossScans) {
  fast_lio::WheelVelocityBuffer buffer(8);
  ASSERT_TRUE(buffer.push({10.00, 0.8, 0.04}));

  const auto first = buffer.take_for_scan(10.00, 0.25, 0.01);
  ASSERT_TRUE(first.has_value());
  EXPECT_DOUBLE_EQ(first->forward_speed_mps, 0.8);

  const auto reused = buffer.take_for_scan(10.20, 0.25, 0.01);
  ASSERT_TRUE(reused.has_value());
  EXPECT_DOUBLE_EQ(reused->stamp_sec, 10.00);
  EXPECT_FALSE(buffer.take_for_scan(10.26, 0.25, 0.01).has_value());
}

TEST(WheelVelocityBuffer, ResetStartsANewTimestampEpoch) {
  fast_lio::WheelVelocityBuffer buffer(8);
  ASSERT_TRUE(buffer.push({100.0, 1.0, 0.04}));
  ASSERT_TRUE(buffer.take_for_scan(100.0, 0.25, 0.01).has_value());
  EXPECT_FALSE(buffer.push({1.0, 0.0, 0.04}));

  buffer.reset();
  ASSERT_TRUE(buffer.push({1.0, 0.0, 0.04}));
  const auto sample = buffer.take_for_scan(1.0, 0.25, 0.01);
  ASSERT_TRUE(sample.has_value());
  EXPECT_DOUBLE_EQ(sample->stamp_sec, 1.0);
}

TEST(WheelVelocityBuffer, RejectsDuplicateInvalidAndDropsStaleSamples) {
  fast_lio::WheelVelocityBuffer buffer(2);
  ASSERT_TRUE(buffer.push({10.0, 1.0, 0.04}));
  EXPECT_FALSE(buffer.push({10.0, 1.0, 0.04}));
  EXPECT_FALSE(buffer.push({9.0, 1.0, 0.04}));
  EXPECT_FALSE(buffer.push({11.0, std::nan(""), 0.04}));
  EXPECT_FALSE(buffer.take_for_scan(11.0, 0.25, 0.01).has_value());
}

TEST(WheelVelocityBuffer, BoundsMemoryWithoutDiscardingNewestSamples) {
  fast_lio::WheelVelocityBuffer buffer(2);
  ASSERT_TRUE(buffer.push({1.0, 1.0, 0.04}));
  ASSERT_TRUE(buffer.push({2.0, 2.0, 0.04}));
  ASSERT_TRUE(buffer.push({3.0, 3.0, 0.04}));

  const auto sample = buffer.take_for_scan(3.0, 10.0, 0.0);
  ASSERT_TRUE(sample.has_value());
  EXPECT_DOUBLE_EQ(sample->stamp_sec, 3.0);
}

}  // namespace
