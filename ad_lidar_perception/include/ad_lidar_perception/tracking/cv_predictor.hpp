#ifndef AD_LIDAR_PERCEPTION__TRACKING__CV_PREDICTOR_HPP_
#define AD_LIDAR_PERCEPTION__TRACKING__CV_PREDICTOR_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ad_lidar_perception::tracking
{

inline constexpr std::size_t kMaximumPredictionHorizons = 64U;
inline constexpr double kMaximumPredictionHorizonS = 60.0;

struct TrackState2D
{
  std::array<std::uint8_t, 16> id{};
  std::uint8_t classification{0U};
  double existence_probability{0.0};
  double classification_probability{0.0};
  double x_m{0.0};
  double y_m{0.0};
  double z_m{0.0};
  double yaw_rad{0.0};
  double vx_world_mps{0.0};
  double vy_world_mps{0.0};
  double length_m{0.0};
  double width_m{0.0};
  double height_m{0.0};
  std::array<double, 4> position_covariance_xy{};
  std::array<double, 4> velocity_covariance_xy{};
};

struct CvPredictionConfig
{
  std::vector<double> horizons_s{0.5, 1.0};
  double acceleration_noise_std_mps2{1.5};
};

struct WorldMotion2D
{
  double vx_world_mps{0.0};
  double vy_world_mps{0.0};
  std::array<double, 4> velocity_covariance_xy{};
};

struct PredictedState2D
{
  std::int64_t time_from_start_ns{0};
  double x_m{0.0};
  double y_m{0.0};
  double z_m{0.0};
  double yaw_rad{0.0};
  std::array<double, 4> position_covariance_xy{};
};

struct PredictedTrack2D
{
  TrackState2D initial_state;
  std::vector<PredictedState2D> states;
};

WorldMotion2D rotate_object_local_motion_to_world(
  double yaw_rad,
  double vx_local_mps,
  double vy_local_mps,
  const std::array<double, 4> & velocity_covariance_local_xy);

// The HEVEN ABI has no position/velocity cross-covariance, so this model
// intentionally propagates only Ppos + t^2 * Pvel plus acceleration noise.
std::vector<PredictedTrack2D> predict_tracks(
  const std::vector<TrackState2D> & tracks,
  const CvPredictionConfig & config);

}  // namespace ad_lidar_perception::tracking

#endif  // AD_LIDAR_PERCEPTION__TRACKING__CV_PREDICTOR_HPP_
