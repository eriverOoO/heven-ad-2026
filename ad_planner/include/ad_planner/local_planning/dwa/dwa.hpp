#ifndef AD_PLANNER__DWA_HPP_
#define AD_PLANNER__DWA_HPP_

#include <vector>

#include "ad_control/longitudinal/pid.hpp"
#include "ad_planner/local_planning/common/local_motion.hpp"
#include "ad_planner/local_planning/common/occupancy.hpp"

namespace ad_planner
{

using ad_control::PidConfig;
using ad_control::PidController;
using ad_control::PidState;

struct DwaConfig
{
  double min_speed_mps;
  double max_speed_mps;
  double speed_step_mps;
  double min_steer_rad;
  double sampled_max_steer_rad;
  double steer_step_rad;
  double dt;
  double horizon_s;
  double wheelbase_m;
  double max_steer_rad{0.6981317};
  double control_period_s{0.05};
  double dynamic_window_time_s{0.50};
  double maximum_acceleration_mps2{5.0};
  double maximum_deceleration_mps2{1.8};
  double emergency_deceleration_mps2{6.0};
  double initial_inflation_escape_s{0.6};
  double maximum_steering_rate_radps{2.0943951023931953};
  double maximum_lateral_acceleration_mps2{6.0};
  double clearance_saturation_m{8.0};
  double maximum_path_distance_m{4.5};
  double prediction_covariance_sigma{2.0};
  double prediction_minimum_margin_m{0.20};
  FootprintConfig footprint;
  double progress_weight;
  double goal_weight;
  double heading_weight;
  double clearance_weight;
  double smoothness_weight;
  double path_distance_weight{1.0};
  double speed_weight{0.5};
  PidConfig pid;
};

class DwaController
{
public:
  explicit DwaController(DwaConfig config);
  void apply_vehicle_constraints(const VehicleConstraints & constraints);
  ControllerResult plan(
    const OccupancyGrid & grid, const Pose2 & pose, const Point3 & target,
    double current_speed_mps, double previous_steering_rad,
    int behavior_id, int gear_id);
  ControllerResult plan_with_reference(
    const OccupancyGrid & grid, const Pose2 & pose, const Point3 & target,
    const std::vector<Point3> & reference_path,
    double current_speed_mps, double previous_steering_rad,
    int behavior_id, int gear_id,
    const OccupancyGrid * drivable_mask = nullptr,
    const PredictedObjectSet * predicted_objects = nullptr);
  const PidState & pid_state() const noexcept;
  const ControllerResult & last_valid_result() const noexcept;
  const std::vector<Trajectory> & candidate_trajectories() const noexcept;

private:
  DwaConfig base_config_;
  DwaConfig config_;
  PidController pid_;
  ControllerResult last_valid_;
  std::vector<Trajectory> candidate_trajectories_;
};

}  // namespace ad_planner
#endif  // AD_PLANNER__DWA_HPP_
