#include <gtest/gtest.h>

#include <kalman_filter_localization/core/eskf_replay.hpp>
#include <kalman_filter_localization/core/imu_initializer.hpp>

namespace
{
using kalman_filter_localization::core::EKFEstimator;
using kalman_filter_localization::core::EskfReplay;
using kalman_filter_localization::core::ImuStationaryInitializer;

void configure(EKFEstimator & estimator)
{
  estimator.setPropagationModel(EKFEstimator::PropagationModel::kFast);
  estimator.setVarImuAcc(0.02);
  estimator.setVarImuGyro(0.001);
  estimator.setMaxPredictionDtSec(0.5);
}

EskfReplay::MeasurementFunction position_update(const Eigen::Vector3d & position)
{
  return [position](EKFEstimator & estimator) {
           return estimator.observationUpdateWithStatus(
             position, Eigen::Vector3d::Constant(0.25));
         };
}
}  // namespace

TEST(EskfFutureMeasurement, QueuesAndAppliesGnss150msAheadOfLatestImu)
{
  EKFEstimator estimator;
  configure(estimator);
  EskfReplay replay(
    estimator,
    /* history_duration_sec = */ 2.0,
    /* max_future_wait_sec = */ 0.25);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const Eigen::Vector3d acceleration(0.0, 0.0, 9.80665);

  ASSERT_EQ(
    replay.addImu(1.0, gyro, acceleration),
    EskfReplay::Status::kInitialized);
  ASSERT_EQ(
    replay.applyMeasurement(
      1.15, 1.01, 1U, position_update(Eigen::Vector3d(2.0, 0.0, 0.0))),
    EskfReplay::Status::kQueuedFuture);
  EXPECT_EQ(replay.counters().future_queued, 1U);
  EXPECT_EQ(replay.counters().measurements_applied, 0U);

  ASSERT_EQ(
    replay.addImu(1.10, gyro, acceleration),
    EskfReplay::Status::kApplied);
  EXPECT_EQ(replay.counters().measurements_applied, 0U);
  ASSERT_EQ(
    replay.addImu(1.15, gyro, acceleration),
    EskfReplay::Status::kApplied);

  EXPECT_EQ(replay.counters().measurements_applied, 1U);
  EXPECT_EQ(replay.counters().future, 0U);
  EXPECT_GT(estimator.getPosition().x(), 0.0);
}

TEST(EskfFutureMeasurement, DependencyOverlayClassifiesLargeImuGap)
{
  EKFEstimator estimator;
  configure(estimator);
  EskfReplay replay(estimator, /* history_duration_sec = */ 2.0);
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const Eigen::Vector3d acceleration(0.0, 0.0, 9.80665);

  ASSERT_EQ(
    replay.addImu(10.0, gyro, acceleration),
    EskfReplay::Status::kInitialized);
  EXPECT_EQ(
    replay.addImu(11.0, gyro, acceleration),
    EskfReplay::Status::kLargeGap);
  EXPECT_EQ(replay.counters().large_imu_gap, 1U);
  EXPECT_DOUBLE_EQ(replay.latestTime(), 10.0);
}

TEST(EskfStationaryInitialization, TwentyHertzUsesTimeSpanWithAStatisticalSampleFloor)
{
  const Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  const Eigen::Vector3d acceleration(0.0, 0.0, 9.80665);

  ImuStationaryInitializer::Config rate_coupled_config;
  rate_coupled_config.window_duration_sec = 1.5;
  rate_coupled_config.minimum_samples = 50U;
  ImuStationaryInitializer rate_coupled(rate_coupled_config);
  for (int index = 0; index <= 40; ++index) {
    EXPECT_EQ(
      rate_coupled.addSample(index * 0.05, gyro, acceleration),
      ImuStationaryInitializer::Status::kCollecting);
  }
  EXPECT_LT(rate_coupled.result().sample_count, 50U);

  ImuStationaryInitializer::Config robust_config;
  robust_config.window_duration_sec = 1.5;
  robust_config.minimum_samples = 25U;
  ImuStationaryInitializer robust(robust_config);
  const auto is_dropped_sample = [](const int index) {
      return index == 4 || index == 9 || index == 14 || index == 19 || index == 24;
    };
  for (int index = 0; index < 29; ++index) {
    if (!is_dropped_sample(index)) {
      EXPECT_EQ(
        robust.addSample(index * 0.05, gyro, acceleration),
        ImuStationaryInitializer::Status::kCollecting);
    }
  }
  EXPECT_EQ(
    robust.addSample(29 * 0.05, gyro, acceleration),
    ImuStationaryInitializer::Status::kInitialized);
  EXPECT_EQ(robust.result().sample_count, 25U);
}
