#include <ad_lidar_perception/tracking/cv_predictor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

using ad_lidar_perception::tracking::CvPredictionConfig;
using ad_lidar_perception::tracking::TrackState2D;
using ad_lidar_perception::tracking::predict_tracks;
using ad_lidar_perception::tracking::rotate_object_local_motion_to_world;

constexpr double kPi = 3.14159265358979323846;

TrackState2D valid_track()
{
  TrackState2D track;
  for (std::size_t index = 0; index < track.id.size(); ++index) {
    track.id[index] = static_cast<std::uint8_t>(index + 1U);
  }
  track.classification = 1U;
  track.existence_probability = 0.8;
  track.classification_probability = 0.7;
  track.x_m = 1.0;
  track.y_m = 2.0;
  track.z_m = 0.5;
  track.yaw_rad = 0.25;
  track.vx_world_mps = 4.0;
  track.vy_world_mps = -2.0;
  track.length_m = 4.5;
  track.width_m = 1.8;
  track.height_m = 1.6;
  track.position_covariance_xy = {1.0, 0.2, 0.2, 2.0};
  track.velocity_covariance_xy = {3.0, 0.4, 0.4, 5.0};
  return track;
}

CvPredictionConfig valid_config()
{
  CvPredictionConfig config;
  config.horizons_s = {0.5, 1.0};
  config.acceleration_noise_std_mps2 = 2.0;
  return config;
}

TEST(CvPredictor, RotatesObjectLocalVelocityAndFullCovarianceToWorld)
{
  const auto motion = rotate_object_local_motion_to_world(
    kPi / 2.0, 3.0, 0.0, {4.0, 1.0, 1.0, 9.0});

  EXPECT_NEAR(motion.vx_world_mps, 0.0, 1.0e-12);
  EXPECT_NEAR(motion.vy_world_mps, 3.0, 1.0e-12);
  EXPECT_NEAR(motion.velocity_covariance_xy[0], 9.0, 1.0e-12);
  EXPECT_NEAR(motion.velocity_covariance_xy[1], -1.0, 1.0e-12);
  EXPECT_NEAR(motion.velocity_covariance_xy[2], -1.0, 1.0e-12);
  EXPECT_NEAR(motion.velocity_covariance_xy[3], 4.0, 1.0e-12);
}

TEST(CvPredictor, PredictsExactPositionsTimesAndCovariance)
{
  const auto predictions = predict_tracks({valid_track()}, valid_config());

  ASSERT_EQ(predictions.size(), 1U);
  ASSERT_EQ(predictions.front().states.size(), 2U);
  const auto & half_second = predictions.front().states[0];
  EXPECT_EQ(half_second.time_from_start_ns, 500000000);
  EXPECT_DOUBLE_EQ(half_second.x_m, 3.0);
  EXPECT_DOUBLE_EQ(half_second.y_m, 1.0);
  EXPECT_DOUBLE_EQ(half_second.z_m, 0.5);
  EXPECT_DOUBLE_EQ(half_second.yaw_rad, 0.25);
  EXPECT_NEAR(half_second.position_covariance_xy[0], 1.8125, 1.0e-12);
  EXPECT_NEAR(half_second.position_covariance_xy[1], 0.3, 1.0e-12);
  EXPECT_NEAR(half_second.position_covariance_xy[2], 0.3, 1.0e-12);
  EXPECT_NEAR(half_second.position_covariance_xy[3], 3.3125, 1.0e-12);

  const auto & one_second = predictions.front().states[1];
  EXPECT_EQ(one_second.time_from_start_ns, 1000000000);
  EXPECT_DOUBLE_EQ(one_second.x_m, 5.0);
  EXPECT_DOUBLE_EQ(one_second.y_m, 0.0);
  EXPECT_NEAR(one_second.position_covariance_xy[0], 5.0, 1.0e-12);
  EXPECT_NEAR(one_second.position_covariance_xy[1], 0.6, 1.0e-12);
  EXPECT_NEAR(one_second.position_covariance_xy[2], 0.6, 1.0e-12);
  EXPECT_NEAR(one_second.position_covariance_xy[3], 8.0, 1.0e-12);
}

TEST(CvPredictor, PreservesTrackIdentityShapeAndProbabilities)
{
  auto config = valid_config();
  config.acceleration_noise_std_mps2 = 0.0;
  const auto input = valid_track();

  const auto predictions = predict_tracks({input}, config);

  ASSERT_EQ(predictions.size(), 1U);
  const auto & initial = predictions.front().initial_state;
  EXPECT_EQ(initial.id, input.id);
  EXPECT_EQ(initial.classification, input.classification);
  EXPECT_DOUBLE_EQ(initial.existence_probability, input.existence_probability);
  EXPECT_DOUBLE_EQ(
    initial.classification_probability, input.classification_probability);
  EXPECT_DOUBLE_EQ(initial.length_m, input.length_m);
  EXPECT_DOUBLE_EQ(initial.width_m, input.width_m);
  EXPECT_DOUBLE_EQ(initial.height_m, input.height_m);
  EXPECT_EQ(initial.position_covariance_xy, input.position_covariance_xy);
  EXPECT_EQ(initial.velocity_covariance_xy, input.velocity_covariance_xy);
  EXPECT_DOUBLE_EQ(
    predictions.front().states.front().position_covariance_xy[0], 1.75);
}

TEST(CvPredictor, RejectsInvalidTrackFields)
{
  const auto config = valid_config();
  const auto expect_invalid = [&config](const TrackState2D & track) {
      EXPECT_THROW((void)predict_tracks({track}, config), std::invalid_argument);
    };

  auto track = valid_track();
  track.classification = 8U;
  expect_invalid(track);
  track = valid_track();
  track.existence_probability = -0.01;
  expect_invalid(track);
  track = valid_track();
  track.existence_probability = std::numeric_limits<double>::quiet_NaN();
  expect_invalid(track);
  track = valid_track();
  track.classification_probability = 1.01;
  expect_invalid(track);
  track = valid_track();
  track.x_m = std::numeric_limits<double>::infinity();
  expect_invalid(track);
  track = valid_track();
  track.yaw_rad = std::numeric_limits<double>::quiet_NaN();
  expect_invalid(track);
  track = valid_track();
  track.length_m = 0.0;
  expect_invalid(track);
  track = valid_track();
  track.width_m = -1.0;
  expect_invalid(track);
  track = valid_track();
  track.height_m = std::numeric_limits<double>::infinity();
  expect_invalid(track);
}

TEST(CvPredictor, RejectsInvalidCovariance)
{
  const auto config = valid_config();
  const auto expect_invalid = [&config](const TrackState2D & track) {
      EXPECT_THROW((void)predict_tracks({track}, config), std::invalid_argument);
    };

  auto track = valid_track();
  track.position_covariance_xy[1] = 0.3;
  expect_invalid(track);
  track = valid_track();
  track.position_covariance_xy = {1.0, 2.0, 2.0, 1.0};
  expect_invalid(track);
  track = valid_track();
  track.velocity_covariance_xy[0] = -0.1;
  expect_invalid(track);
  track = valid_track();
  track.velocity_covariance_xy[3] =
    std::numeric_limits<double>::quiet_NaN();
  expect_invalid(track);

  EXPECT_THROW(
    (void)rotate_object_local_motion_to_world(
      0.0, 1.0, 0.0, {1.0, 0.5, 0.0, 1.0}),
    std::invalid_argument);
}

TEST(CvPredictor, RejectsInvalidConfiguration)
{
  const auto track = valid_track();
  const auto expect_invalid = [&track](const CvPredictionConfig & config) {
      EXPECT_THROW((void)predict_tracks({track}, config), std::invalid_argument);
    };

  auto config = valid_config();
  config.horizons_s.clear();
  expect_invalid(config);
  config = valid_config();
  config.horizons_s = {0.0};
  expect_invalid(config);
  config = valid_config();
  config.horizons_s = {0.5, 0.5};
  expect_invalid(config);
  config = valid_config();
  config.horizons_s = {1.0, 0.5};
  expect_invalid(config);
  config = valid_config();
  config.horizons_s = {std::numeric_limits<double>::infinity()};
  expect_invalid(config);
  config = valid_config();
  config.horizons_s = {61.0};
  expect_invalid(config);
  config = valid_config();
  config.horizons_s.assign(65U, 0.5);
  for (std::size_t index = 0; index < config.horizons_s.size(); ++index) {
    config.horizons_s[index] = 0.5 + static_cast<double>(index) * 0.1;
  }
  expect_invalid(config);
  config = valid_config();
  config.acceleration_noise_std_mps2 = -0.1;
  expect_invalid(config);
  config = valid_config();
  config.acceleration_noise_std_mps2 =
    std::numeric_limits<double>::quiet_NaN();
  expect_invalid(config);
}

TEST(CvPredictor, RejectsNonfinitePredictionAndWholeArrayAtomically)
{
  auto overflowing = valid_track();
  overflowing.x_m = std::numeric_limits<double>::max();
  overflowing.vx_world_mps = std::numeric_limits<double>::max();
  auto retained = predict_tracks({valid_track()}, valid_config());
  ASSERT_EQ(retained.size(), 1U);

  EXPECT_THROW(
    retained = predict_tracks({valid_track(), overflowing}, valid_config()),
    std::overflow_error);
  ASSERT_EQ(retained.size(), 1U);
  EXPECT_EQ(retained.front().initial_state.id, valid_track().id);
}

}  // namespace
