#include "ad_planner/behavior/planner_supervisor.hpp"

#include <behaviortree_cpp_v3/bt_factory.h>

#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ad_planner/behavior/bt_nodes.hpp"

namespace ad_planner {
namespace {

bool positive(double value) { return std::isfinite(value) && value > 0.0; }

void validate_config(const PlannerConfig &config) {
  const auto &freshness = config.freshness;
  if (!positive(freshness.status_timeout_s) ||
      !positive(freshness.collision_timeout_s) ||
      !positive(freshness.grid_timeout_s) ||
      !positive(freshness.traffic_timeout_s) ||
      !positive(freshness.stop_line_timeout_s)) {
    throw std::invalid_argument("freshness timeouts must be positive");
  }

  const auto &recovery = config.collision;
  if (!std::isfinite(recovery.near_stop_speed_mps) ||
      recovery.near_stop_speed_mps < 0.0 ||
      recovery.near_stop_speed_mps > kMaximumAllowedNearStopSpeedMps ||
      !positive(recovery.reverse_accel) || recovery.reverse_accel > 1.0 ||
      !positive(recovery.reverse_duration_s) ||
      recovery.reverse_duration_s > kMaximumReverseDurationS ||
      !positive(recovery.reverse_brake_duration_s) ||
      !positive(recovery.cooldown_s)) {
    throw std::invalid_argument("collision recovery configuration is invalid");
  }

  if (!positive(config.traffic.zone_entry_radius_m) ||
      !positive(config.traffic.zone_exit_radius_m) ||
      config.traffic.zone_exit_radius_m <= config.traffic.zone_entry_radius_m ||
      !positive(config.traffic.front_bumper_x_m) ||
      !std::isfinite(config.traffic.stopping_margin_m) ||
      config.traffic.stopping_margin_m < 0.0 ||
      !std::isfinite(config.traffic.reaction_time_s) ||
      config.traffic.reaction_time_s < 0.0 ||
      !positive(config.traffic.braking_deceleration_mps2) ||
      !positive(config.traffic.brake_command) ||
      config.traffic.brake_command > 1.0 ||
      !std::isfinite(config.traffic.late_brake_deceleration_ratio) ||
      config.traffic.late_brake_deceleration_ratio < 1.0 ||
      !positive(config.traffic.red_confirmation_duration_s) ||
      !positive(config.traffic.green_release_duration_s) ||
      !positive(config.maximum_steering_rad) ||
      config.maximum_steering_rad > kMaximumAllowedSteeringRad) {
    throw std::invalid_argument("planner thresholds are invalid");
  }

  const auto &perception = config.perception;
  const auto &trigger_pose = perception.forward_check_pose;
  const auto &trigger_footprint = perception.forward_check_footprint;
  if (!positive(perception.clear_release_duration_s) ||
      !std::isfinite(perception.near_x_m) ||
      !positive(perception.minimum_lookahead_m) ||
      !positive(perception.maximum_lookahead_m) ||
      perception.minimum_lookahead_m <= perception.near_x_m ||
      perception.maximum_lookahead_m < perception.minimum_lookahead_m ||
      !std::isfinite(perception.front_bumper_x_m) ||
      !std::isfinite(perception.reaction_time_s) ||
      perception.reaction_time_s < 0.0 ||
      !positive(perception.braking_deceleration_mps2) ||
      !std::isfinite(perception.stopping_margin_m) ||
      perception.stopping_margin_m < 0.0 ||
      !positive(perception.future_road_risk_timeout_s) ||
      !positive(perception.future_road_risk_limits.maximum_horizon_s) ||
      perception.future_road_risk_limits.maximum_objects == 0U ||
      perception.future_road_risk_limits.maximum_footprints == 0U ||
      perception.future_road_risk_limits.maximum_corridor_segments == 0U ||
      !std::isfinite(perception.future_road_risk_limits.covariance_sigma) ||
      perception.future_road_risk_limits.covariance_sigma < 0.0 ||
      !std::isfinite(perception.future_road_risk_limits.minimum_margin_m) ||
      perception.future_road_risk_limits.minimum_margin_m < 0.0 ||
      !std::isfinite(trigger_pose.x) || !std::isfinite(trigger_pose.y) ||
      !std::isfinite(trigger_pose.yaw_rad) ||
      !positive(trigger_footprint.half_length_m) ||
      !positive(trigger_footprint.half_width_m) ||
      !std::isfinite(trigger_footprint.clearance_m) ||
      trigger_footprint.clearance_m < 0.0 ||
      !std::isfinite(trigger_footprint.center_offset_x_m) ||
      trigger_footprint.occupied_threshold < 0 ||
      trigger_footprint.occupied_threshold > 100 ||
      trigger_footprint.maximum_cells_to_check == 0) {
    throw std::invalid_argument("perception trigger configuration is invalid");
  }
}

bool valid_command(const PlannerCommand &command, double steering_limit,
                   const VehicleObservation &status) {
  const auto &motion = command.motion;
  const bool valid_motion =
      std::isfinite(motion.accel) && std::isfinite(motion.brake) &&
      std::isfinite(motion.steering_rad) && motion.accel >= 0.0 &&
      motion.accel <= 1.0 && motion.brake >= 0.0 && motion.brake <= 1.0 &&
      !(motion.accel > 0.0 && motion.brake > 0.0) &&
      std::abs(motion.steering_rad) <= steering_limit;
  if (!valid_motion || motion.accel == 0.0) {
    return valid_motion;
  }
  if (!status.received || !status.valid) {
    return false;
  }
  if (command.gear_request == GearRequest::kReverse) {
    return status.value.gear == kGearReverseCode;
  }
  return command.gear_request == GearRequest::kKeep &&
         status.value.gear == kGearDriveCode;
}

SupervisorStatus supervisor_status(BT::NodeStatus status) {
  if (status == BT::NodeStatus::SUCCESS) {
    return SupervisorStatus::kSuccess;
  }
  if (status == BT::NodeStatus::RUNNING) {
    return SupervisorStatus::kRunning;
  }
  return SupervisorStatus::kFailure;
}

PlannerTickResult make_result(const PlannerContext &context,
                              SupervisorStatus status) {
  return PlannerTickResult{context.staged_command(), status,
                           context.active_behavior(), context.failsafe_reason(),
                           context.claim_count()};
}

} // namespace

struct PlannerSupervisor::Impl {
  Impl(PlannerContext &context_in, PlannerConfig config_in,
       const std::string &tree_xml)
      : context(context_in), config(std::move(config_in)),
        environment(std::make_shared<BtNodeEnvironment>(
            BtNodeEnvironment{context, config, state})) {
    validate_config(config);
    register_ad_bt_nodes(factory, environment);
    tree = factory.createTreeFromText(tree_xml);
  }

  PlannerContext &context;
  PlannerConfig config;
  PlannerRuntimeState state;
  BtNodeEnvironmentPtr environment;
  BT::BehaviorTreeFactory factory;
  BT::Tree tree;
};

PlannerSupervisor::PlannerSupervisor(PlannerContext &context,
                                     PlannerConfig config,
                                     const std::string &tree_xml)
    : impl_(std::make_unique<Impl>(context, std::move(config), tree_xml)) {}

PlannerSupervisor::~PlannerSupervisor() = default;
PlannerSupervisor::PlannerSupervisor(PlannerSupervisor &&) noexcept = default;
PlannerSupervisor &
PlannerSupervisor::operator=(PlannerSupervisor &&) noexcept = default;

PlannerTickResult PlannerSupervisor::tick() {
  auto &context = impl_->context;
  auto &state = impl_->state;
  context.begin_tick();

  if (!std::isfinite(context.steady_time_s) || context.steady_time_s < 0.0 ||
      (state.tick_time_initialized &&
       context.steady_time_s < state.last_tick_time_s)) {
    impl_->tree.haltTree();
    context.force_full_brake("invalid or non-monotonic steady time");
    return make_result(context, SupervisorStatus::kException);
  }
  state.tick_time_initialized = true;
  state.last_tick_time_s = context.steady_time_s;

  SupervisorStatus status = SupervisorStatus::kFailure;
  try {
    status = supervisor_status(impl_->tree.tickRoot());
  } catch (const std::exception &error) {
    impl_->tree.haltTree();
    context.force_full_brake(std::string("behavior tree exception: ") +
                             error.what());
    return make_result(context, SupervisorStatus::kException);
  } catch (...) {
    impl_->tree.haltTree();
    context.force_full_brake("unknown behavior tree exception");
    return make_result(context, SupervisorStatus::kException);
  }

  if (context.claim_count() != 1 ||
      !valid_command(context.staged_command(),
                     impl_->config.maximum_steering_rad,
                     context.inputs.status)) {
    context.force_full_brake("behavior tree produced an invalid command");
  }
  return make_result(context, status);
}

void PlannerSupervisor::halt() { impl_->tree.haltTree(); }

const PlannerRuntimeState &PlannerSupervisor::state() const noexcept {
  return impl_->state;
}

std::vector<std::string> PlannerSupervisor::registered_node_ids() const {
  return ad_bt_node_ids();
}

} // namespace ad_planner
