#ifndef AD_CONTROL__LATERAL__STANLEY_HPP_
#define AD_CONTROL__LATERAL__STANLEY_HPP_

#include <cstddef>
#include <optional>
#include <vector>

#include "ad_control/lateral/path_tracking_controller.hpp"
#include "ad_control/longitudinal/pid.hpp"
#include "ad_control/path/route_progress_tracker.hpp"
#include "ad_control/path/route_speed_profile.hpp"

namespace ad_control
{

struct StanleyConfig
{
  double target_speed_mps;
  double cross_track_gain;
  double speed_softening_mps;
  double lookahead_time_s;
  double lookahead_min_m;
  double lookahead_max_m;
  double max_steer_rad{0.6981317};
  std::size_t forward_window;
  std::size_t max_laps;
  PidConfig pid;
  double control_point_x_m{0.0};
  double heading_error_gain{1.0};
  RouteSpeedProfileConfig speed_profile;
  double launch_speed_mps{5.0 / 3.6};
  double launch_ramp_s{4.0};
};

struct ProfileStanleyConfig
{
  StanleyConfig stanley;
  LongitudinalProfile longitudinal_profile;
};

class StanleyController : public PathTrackingController
{
public:
  StanleyController(Route route, StanleyConfig config);
  ControllerResult update(
    const Pose2 & pose, double speed_mps, double dt, int behavior_id, int gear_id,
    std::optional<double> target_speed_mps = std::nullopt) override;
  const RouteProgressState & progress_state() const noexcept;
  const PidState & pid_state() const noexcept;
  const ControllerResult & last_valid_result() const noexcept;
  const RouteSpeedProfile * route_speed_profile() const noexcept override;

protected:
  StanleyController(Route route, ProfileStanleyConfig config);

private:
  Route route_;
  StanleyConfig config_;
  RouteProgressTracker progress_;
  PidController pid_;
  RouteSpeedProfile route_speed_profile_;
  double previous_steering_rad_{0.0};
  double launch_elapsed_s_{0.0};
  std::optional<Pose2> previous_control_pose_;
  ControllerResult last_valid_;
};

class ProfileStanleyController final : public StanleyController
{
public:
  ProfileStanleyController(Route route, ProfileStanleyConfig config);
};

}  // namespace ad_control
#endif  // AD_CONTROL__LATERAL__STANLEY_HPP_
