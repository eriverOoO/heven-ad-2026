#include <ad_lidar_perception/tracking/cv_predictor.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ad_lidar_perception::tracking
{
namespace
{

constexpr double kSymmetryRelativeTolerance = 1.0e-10;
constexpr double kPsdRelativeTolerance = 1.0e-12;
constexpr std::uint8_t kMaximumClassification = 7U;
constexpr long double kNanosecondsPerSecond = 1000000000.0L;

bool finite_probability(const double value)
{
  return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

void validate_covariance(
  const std::array<double, 4> & covariance,
  const std::string & field_name)
{
  if (!std::all_of(
      covariance.begin(), covariance.end(),
      [](const double value) {return std::isfinite(value);}))
  {
    throw std::invalid_argument(field_name + " must be finite");
  }

  const double scale = std::max(
    {1.0, std::abs(covariance[0]), std::abs(covariance[1]),
      std::abs(covariance[2]), std::abs(covariance[3])});
  if (std::abs(covariance[1] - covariance[2]) >
    kSymmetryRelativeTolerance * scale)
  {
    throw std::invalid_argument(field_name + " must be symmetric");
  }

  const double diagonal_tolerance = kPsdRelativeTolerance * scale;
  if (covariance[0] < -diagonal_tolerance ||
    covariance[3] < -diagonal_tolerance)
  {
    throw std::invalid_argument(field_name + " must be positive semidefinite");
  }

  const double determinant =
    covariance[0] * covariance[3] - covariance[1] * covariance[2];
  if (!std::isfinite(determinant)) {
    throw std::invalid_argument(field_name + " determinant must be finite");
  }
  const double determinant_scale = std::max(
    {1.0, std::abs(covariance[0] * covariance[3]),
      std::abs(covariance[1] * covariance[2])});
  if (determinant < -kPsdRelativeTolerance * determinant_scale) {
    throw std::invalid_argument(field_name + " must be positive semidefinite");
  }
}

void validate_track(const TrackState2D & track)
{
  if (track.classification > kMaximumClassification) {
    throw std::invalid_argument("classification must be in the HEVEN range 0..7");
  }
  if (!finite_probability(track.existence_probability) ||
    !finite_probability(track.classification_probability))
  {
    throw std::invalid_argument("track probabilities must be finite and in [0, 1]");
  }

  const std::array<double, 7> state_values{
    track.x_m, track.y_m, track.z_m, track.yaw_rad,
    track.vx_world_mps, track.vy_world_mps, track.length_m};
  if (!std::all_of(
      state_values.begin(), state_values.end(),
      [](const double value) {return std::isfinite(value);}) ||
    !std::isfinite(track.width_m) || !std::isfinite(track.height_m))
  {
    throw std::invalid_argument("track state and dimensions must be finite");
  }
  if (track.length_m <= 0.0 || track.width_m <= 0.0 ||
    track.height_m <= 0.0)
  {
    throw std::invalid_argument("track dimensions must be strictly positive");
  }

  validate_covariance(track.position_covariance_xy, "position covariance");
  validate_covariance(track.velocity_covariance_xy, "velocity covariance");
}

void validate_config(const CvPredictionConfig & config)
{
  if (config.horizons_s.empty()) {
    throw std::invalid_argument("prediction horizons must not be empty");
  }
  if (config.horizons_s.size() > kMaximumPredictionHorizons) {
    throw std::invalid_argument("too many prediction horizons");
  }
  double previous = 0.0;
  for (const double horizon : config.horizons_s) {
    if (!std::isfinite(horizon) || horizon <= 0.0 ||
      horizon > kMaximumPredictionHorizonS)
    {
      throw std::invalid_argument("prediction horizon is outside the supported range");
    }
    if (horizon <= previous) {
      throw std::invalid_argument(
              "prediction horizons must be strictly increasing");
    }
    previous = horizon;
  }
  if (!std::isfinite(config.acceleration_noise_std_mps2) ||
    config.acceleration_noise_std_mps2 < 0.0)
  {
    throw std::invalid_argument(
            "acceleration noise standard deviation must be finite and nonnegative");
  }
}

std::int64_t checked_duration_ns(const double horizon_s)
{
  const long double nanoseconds =
    static_cast<long double>(horizon_s) * kNanosecondsPerSecond;
  if (!std::isfinite(nanoseconds) ||
    nanoseconds >
    static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
  {
    throw std::overflow_error("prediction duration overflows nanoseconds");
  }
  const long double rounded = std::round(nanoseconds);
  if (rounded <= 0.0L ||
    rounded >
    static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
  {
    throw std::overflow_error("prediction duration cannot be represented");
  }
  return static_cast<std::int64_t>(rounded);
}

double checked_sum(const double lhs, const double rhs, const char * field_name)
{
  const double result = lhs + rhs;
  if (!std::isfinite(result)) {
    throw std::overflow_error(std::string(field_name) + " became nonfinite");
  }
  return result;
}

PredictedState2D predict_state(
  const TrackState2D & track,
  const double horizon_s,
  const double acceleration_noise_variance)
{
  const double time_squared = horizon_s * horizon_s;
  const double time_fourth = time_squared * time_squared;
  const double process_variance =
    0.25 * acceleration_noise_variance * time_fourth;
  if (!std::isfinite(time_squared) || !std::isfinite(time_fourth) ||
    !std::isfinite(process_variance))
  {
    throw std::overflow_error("prediction process variance became nonfinite");
  }

  PredictedState2D prediction;
  prediction.time_from_start_ns = checked_duration_ns(horizon_s);
  prediction.x_m = checked_sum(
    track.x_m, track.vx_world_mps * horizon_s, "predicted x");
  prediction.y_m = checked_sum(
    track.y_m, track.vy_world_mps * horizon_s, "predicted y");
  prediction.z_m = track.z_m;
  prediction.yaw_rad = track.yaw_rad;
  for (std::size_t index = 0; index < 4U; ++index) {
    prediction.position_covariance_xy[index] = checked_sum(
      track.position_covariance_xy[index],
      time_squared * track.velocity_covariance_xy[index],
      "predicted covariance");
  }
  prediction.position_covariance_xy[0] = checked_sum(
    prediction.position_covariance_xy[0],
    process_variance, "predicted covariance xx");
  prediction.position_covariance_xy[3] = checked_sum(
    prediction.position_covariance_xy[3],
    process_variance, "predicted covariance yy");
  return prediction;
}

}  // namespace

WorldMotion2D rotate_object_local_motion_to_world(
  const double yaw_rad,
  const double vx_local_mps,
  const double vy_local_mps,
  const std::array<double, 4> & velocity_covariance_local_xy)
{
  if (!std::isfinite(yaw_rad) || !std::isfinite(vx_local_mps) ||
    !std::isfinite(vy_local_mps))
  {
    throw std::invalid_argument("local motion values must be finite");
  }
  validate_covariance(
    velocity_covariance_local_xy, "local velocity covariance");

  const double cosine = std::cos(yaw_rad);
  const double sine = std::sin(yaw_rad);
  const double covariance_xy =
    0.5 * (velocity_covariance_local_xy[1] +
    velocity_covariance_local_xy[2]);
  const double covariance_xx = velocity_covariance_local_xy[0];
  const double covariance_yy = velocity_covariance_local_xy[3];

  WorldMotion2D result;
  result.vx_world_mps =
    cosine * vx_local_mps - sine * vy_local_mps;
  result.vy_world_mps =
    sine * vx_local_mps + cosine * vy_local_mps;
  result.velocity_covariance_xy[0] =
    cosine * cosine * covariance_xx -
    2.0 * cosine * sine * covariance_xy +
    sine * sine * covariance_yy;
  result.velocity_covariance_xy[1] =
    cosine * sine * (covariance_xx - covariance_yy) +
    (cosine * cosine - sine * sine) * covariance_xy;
  result.velocity_covariance_xy[2] = result.velocity_covariance_xy[1];
  result.velocity_covariance_xy[3] =
    sine * sine * covariance_xx +
    2.0 * cosine * sine * covariance_xy +
    cosine * cosine * covariance_yy;

  if (!std::isfinite(result.vx_world_mps) ||
    !std::isfinite(result.vy_world_mps) ||
    !std::all_of(
      result.velocity_covariance_xy.begin(),
      result.velocity_covariance_xy.end(),
      [](const double value) {return std::isfinite(value);}))
  {
    throw std::overflow_error("world-frame motion became nonfinite");
  }
  return result;
}

std::vector<PredictedTrack2D> predict_tracks(
  const std::vector<TrackState2D> & tracks,
  const CvPredictionConfig & config)
{
  validate_config(config);
  for (const auto & track : tracks) {
    validate_track(track);
  }
  if (!tracks.empty() &&
    config.horizons_s.size() >
    std::numeric_limits<std::size_t>::max() / tracks.size())
  {
    throw std::overflow_error("prediction output size overflows");
  }

  const double acceleration_noise_variance =
    config.acceleration_noise_std_mps2 *
    config.acceleration_noise_std_mps2;
  if (!std::isfinite(acceleration_noise_variance)) {
    throw std::overflow_error("acceleration noise variance became nonfinite");
  }

  std::vector<PredictedTrack2D> predictions;
  predictions.reserve(tracks.size());
  for (const auto & track : tracks) {
    PredictedTrack2D prediction;
    prediction.initial_state = track;
    prediction.states.reserve(config.horizons_s.size());
    for (const double horizon : config.horizons_s) {
      prediction.states.push_back(
        predict_state(track, horizon, acceleration_noise_variance));
    }
    predictions.push_back(std::move(prediction));
  }
  return predictions;
}

}  // namespace ad_lidar_perception::tracking
