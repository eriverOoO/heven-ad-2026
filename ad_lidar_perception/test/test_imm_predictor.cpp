#include <ad_lidar_perception/tracking/imm_predictor.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <numeric>

namespace
{

using ad_lidar_perception::tracking::ImmConfig;
using ad_lidar_perception::tracking::ImmPredictor;
using ad_lidar_perception::tracking::MotionModel;
using ad_lidar_perception::tracking::TrackMeasurement2D;

constexpr std::int64_t kSecondNs = 1'000'000'000LL;

TrackMeasurement2D measurement(
  const double x, const double y, const double yaw,
  const double speed, const double yaw_rate)
{
  TrackMeasurement2D result;
  result.x_m = x;
  result.y_m = y;
  result.yaw_rad = yaw;
  result.vx_world_mps = speed * std::cos(yaw);
  result.vy_world_mps = speed * std::sin(yaw);
  result.yaw_rate_radps = yaw_rate;
  result.position_variance_m2 = 0.04;
  result.velocity_variance_m2ps2 = 0.09;
  result.yaw_variance_rad2 = 0.01;
  result.yaw_rate_variance_rad2ps2 = 0.01;
  return result;
}

ImmConfig test_config()
{
  ImmConfig config;
  config.horizons_s = {0.5, 1.0};
  config.initial_probabilities = {0.20, 0.60, 0.20};
  config.transition_probabilities = {0.90, 0.08, 0.02, 0.05, 0.90,
    0.05, 0.02, 0.08, 0.90};
  return config;
}

TEST(ImmPredictor, InitializesNormalizedModelProbabilities) {
  ImmPredictor predictor(test_config());
  const auto result =
    predictor.update(measurement(0.0, 0.0, 0.0, 5.0, 0.0), kSecondNs);

  const double sum = std::accumulate(
    result.model_probabilities.begin(),
    result.model_probabilities.end(), 0.0);
  EXPECT_NEAR(sum, 1.0, 1.0e-12);
  EXPECT_GT(
    result.model_probabilities[static_cast<std::size_t>(
      MotionModel::kConstantVelocity)],
    result.model_probabilities[static_cast<std::size_t>(
      MotionModel::kStationary)]);
  ASSERT_EQ(result.predicted_states.size(), 2U);
  EXPECT_NEAR(result.predicted_states.back().x_m, 5.0, 0.25);
}

TEST(ImmPredictor, StationaryEvidenceRaisesStationaryProbability) {
  ImmPredictor predictor(test_config());
  auto result =
    predictor.update(measurement(4.0, -2.0, 0.0, 0.0, 0.0), kSecondNs);
  for (std::int64_t index = 2; index <= 7; ++index) {
    result = predictor.update(
      measurement(4.0, -2.0, 0.0, 0.0, 0.0),
      index * kSecondNs);
  }

  const auto stationary = static_cast<std::size_t>(MotionModel::kStationary);
  const auto cv = static_cast<std::size_t>(MotionModel::kConstantVelocity);
  EXPECT_GT(
    result.model_probabilities[stationary],
    result.model_probabilities[cv]);
  EXPECT_NEAR(result.predicted_states.back().x_m, 4.0, 0.05);
  EXPECT_NEAR(result.predicted_states.back().y_m, -2.0, 0.05);
}

TEST(ImmPredictor, TurningEvidenceRaisesCoordinatedTurnProbability) {
  ImmPredictor predictor(test_config());
  constexpr double speed = 6.0;
  constexpr double yaw_rate = 0.30;
  constexpr double radius = speed / yaw_rate;
  auto result =
    predictor.update(measurement(0.0, 0.0, 0.0, speed, yaw_rate), kSecondNs);
  for (std::int64_t index = 2; index <= 8; ++index) {
    const double time = static_cast<double>(index - 1);
    const double yaw = yaw_rate * time;
    result = predictor.update(
      measurement(
        radius * std::sin(yaw),
        radius * (1.0 - std::cos(yaw)), yaw,
        speed, yaw_rate),
      index * kSecondNs);
  }

  const auto ctrv = static_cast<std::size_t>(MotionModel::kCoordinatedTurn);
  const auto cv = static_cast<std::size_t>(MotionModel::kConstantVelocity);
  EXPECT_GT(result.model_probabilities[ctrv], result.model_probabilities[cv]);
  ASSERT_EQ(result.predicted_states.size(), 2U);
  EXPECT_GT(result.predicted_states.back().y_m, result.fused_state.y_m);
  EXPECT_GT(result.predicted_states.back().yaw_rad, result.fused_state.yaw_rad);
}

TEST(ImmPredictor, ResetDiscardsPreviousMotionEvidence) {
  ImmPredictor predictor(test_config());
  (void)predictor.update(measurement(0.0, 0.0, 0.0, 8.0, 0.4), kSecondNs);
  (void)predictor.update(measurement(7.8, 1.6, 0.4, 8.0, 0.4), 2 * kSecondNs);
  predictor.reset();

  const auto result = predictor.update(
    measurement(100.0, 30.0, 0.0, 0.0, 0.0),
    100 * kSecondNs);

  EXPECT_NEAR(result.fused_state.x_m, 100.0, 1.0e-12);
  EXPECT_NEAR(result.fused_state.y_m, 30.0, 1.0e-12);
}

TEST(ImmPredictor, RejectsNonMonotonicMeasurementTime) {
  ImmPredictor predictor(test_config());
  (void)predictor.update(measurement(0.0, 0.0, 0.0, 1.0, 0.0), 2 * kSecondNs);
  EXPECT_THROW(
    (void)predictor.update(
      measurement(1.0, 0.0, 0.0, 1.0, 0.0),
      2 * kSecondNs),
    std::invalid_argument);
}

} // namespace
