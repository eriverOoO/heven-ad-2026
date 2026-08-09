#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ad_lidar_perception::tracking
{

enum class MotionModel : std::size_t
{
  kStationary = 0,
  kConstantVelocity = 1,
  kCoordinatedTurn = 2,
};

struct TrackMeasurement2D
{
  double x_m{0.0};
  double y_m{0.0};
  double yaw_rad{0.0};
  double vx_world_mps{0.0};
  double vy_world_mps{0.0};
  double yaw_rate_radps{0.0};
  double position_variance_m2{0.25};
  double velocity_variance_m2ps2{1.0};
  double yaw_variance_rad2{0.04};
  double yaw_rate_variance_rad2ps2{0.04};
};

struct ImmKinematicState
{
  double x_m{0.0};
  double y_m{0.0};
  double yaw_rad{0.0};
  double vx_world_mps{0.0};
  double vy_world_mps{0.0};
  double yaw_rate_radps{0.0};
};

struct ImmPredictedState
{
  double time_from_start_s{0.0};
  double x_m{0.0};
  double y_m{0.0};
  double yaw_rad{0.0};
};

struct ImmResult
{
  ImmKinematicState fused_state{};
  std::array<double, 3> model_probabilities{{0.0, 0.0, 0.0}};
  std::vector<ImmPredictedState> predicted_states;
};

struct ImmConfig
{
  std::vector<double> horizons_s{0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::array<double, 3> initial_probabilities{{0.20, 0.60, 0.20}};
  // Row-major Markov matrix: source model -> destination model.
  std::array<double, 9> transition_probabilities{
    {0.94, 0.05, 0.01, 0.03, 0.93, 0.04, 0.01, 0.06, 0.93}};
  double maximum_update_interval_s{2.0};
  double stationary_process_variance{0.02};
  double constant_velocity_process_variance{0.20};
  double coordinated_turn_process_variance{0.35};
};

class ImmPredictor
{
public:
  explicit ImmPredictor(ImmConfig config = {});
  ~ImmPredictor();

  ImmPredictor(ImmPredictor &&) noexcept;
  ImmPredictor & operator=(ImmPredictor &&) noexcept;
  ImmPredictor(const ImmPredictor & other);
  ImmPredictor & operator=(const ImmPredictor & other);

  ImmResult update(
    const TrackMeasurement2D & measurement,
    std::int64_t stamp_ns);
  void reset() noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ad_lidar_perception::tracking
