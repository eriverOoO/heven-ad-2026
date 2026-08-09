#ifndef AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_RUNTIME_HPP_
#define AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_RUNTIME_HPP_

#include <memory>
#include <string>

#include "ad_control/command/curvature_command_adapter.hpp"
#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

struct LocalMotionRuntimeConfig
{
  double backend_maximum_steering_rad{0.0};
  double output_maximum_steering_rad{0.0};
};

struct LocalMotionRuntimeResult
{
  bool valid{false};
  std::string reason{"invalid"};
  ControllerResult controller;
  LocalPlanningResult planning;
};

class LocalMotionRuntime
{
public:
  LocalMotionRuntime(
    std::unique_ptr<LocalMotionBackend> backend,
    std::unique_ptr<ad_control::CurvatureCommandAdapter> curvature_adapter,
    LocalMotionRuntimeConfig config);

  LocalMotionRuntimeResult plan(
    const LocalPlanningRequest & request, double previous_steering_rad);

  bool observe_external_velocity_command(
    const ExternalVelocityCommand & command);

  void replace_backend(std::unique_ptr<LocalMotionBackend> backend);

private:
  std::unique_ptr<LocalMotionBackend> backend_;
  std::unique_ptr<ad_control::CurvatureCommandAdapter> curvature_adapter_;
  LocalMotionRuntimeConfig config_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__LOCAL_MOTION_RUNTIME_HPP_
