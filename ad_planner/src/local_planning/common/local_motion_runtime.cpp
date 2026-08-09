#include "ad_planner/local_planning/common/local_motion_runtime.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

#include "ad_planner/local_planning/common/local_motion_validation.hpp"

namespace ad_planner
{
namespace
{

LocalMotionRuntimeResult failure(std::string reason)
{
  LocalMotionRuntimeResult result;
  result.reason = std::move(reason);
  result.controller.reason = result.reason;
  return result;
}

void validate_config(const LocalMotionRuntimeConfig & config)
{
  if (!std::isfinite(config.backend_maximum_steering_rad) ||
    config.backend_maximum_steering_rad <= 0.0 ||
    !std::isfinite(config.output_maximum_steering_rad) ||
    config.output_maximum_steering_rad <= 0.0)
  {
    throw std::invalid_argument(
            "local motion runtime steering limits must be finite and positive");
  }
}

}  // namespace

LocalMotionRuntime::LocalMotionRuntime(
  std::unique_ptr<LocalMotionBackend> backend,
  std::unique_ptr<ad_control::CurvatureCommandAdapter> curvature_adapter,
  LocalMotionRuntimeConfig config)
: backend_(std::move(backend)),
  curvature_adapter_(std::move(curvature_adapter)),
  config_(config)
{
  validate_config(config_);
  if (!backend_ || !curvature_adapter_) {
    throw std::invalid_argument(
            "local motion runtime requires a backend and curvature adapter");
  }
}

LocalMotionRuntimeResult LocalMotionRuntime::plan(
  const LocalPlanningRequest & request, const double previous_steering_rad)
{
  const auto planning = backend_->plan(request);
  if (!planning.valid) {
    return failure(
      planning.reason.empty() ?
      "local motion backend returned invalid" : planning.reason);
  }
  if (!valid_timed_trajectory(
      planning.trajectory, request.reference_corridor.frame_id) ||
    !std::isfinite(planning.desired_speed_mps) ||
    !std::isfinite(planning.desired_curvature_inv_m))
  {
    return failure(
      "local motion backend returned a malformed selected trajectory");
  }

  ControllerResult controller;
  if (planning.direct_command) {
    if (!valid_direct_command(
        *planning.direct_command, config_.backend_maximum_steering_rad))
    {
      return failure(
        "local motion backend returned an invalid direct command");
    }
    controller = ControllerResult{
      true, *planning.direct_command,
      planning.reason.empty() ? "ok" : planning.reason};
  } else {
    controller = curvature_adapter_->update(
      ad_control::CurvatureCommandInput{
        request.ego.speed_mps,
        planning.desired_speed_mps,
        planning.desired_curvature_inv_m,
        previous_steering_rad,
        request.dt_s,
        request.behavior_id,
        request.gear_id});
    if (!controller.valid) {
      return failure(
        "curvature command adapter rejected local motion result");
    }
  }
  if (!controller.valid ||
    !valid_direct_command(
      controller.command, config_.output_maximum_steering_rad))
  {
    return failure(
      "local motion controller result failed final command admission");
  }
  controller.target_speed_mps = planning.desired_speed_mps;

  LocalMotionRuntimeResult result;
  result.valid = true;
  result.reason = controller.reason;
  result.controller = std::move(controller);
  result.planning = planning;
  return result;
}

bool LocalMotionRuntime::observe_external_velocity_command(
  const ExternalVelocityCommand & command)
{
  return backend_->observe_external_velocity_command(command);
}

void LocalMotionRuntime::replace_backend(
  std::unique_ptr<LocalMotionBackend> backend)
{
  if (!backend) {
    throw std::invalid_argument("local motion backend must not be null");
  }
  backend_ = std::move(backend);
}

}  // namespace ad_planner
