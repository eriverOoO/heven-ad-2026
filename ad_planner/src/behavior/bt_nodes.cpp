#include "ad_planner/behavior/bt_nodes.hpp"

#include <behaviortree_cpp_v3/action_node.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace ad_planner {
namespace {

constexpr std::int16_t kStaticObject = 2;

template <typename Observation>
bool fresh(const Observation &observation, double now, double timeout) {
  return observation.received && observation.valid &&
         std::isfinite(observation.receipt_time_s) &&
         observation.receipt_time_s <= now &&
         now - observation.receipt_time_s <= timeout;
}

bool valid_vehicle(const VehicleState &vehicle) {
  return std::isfinite(vehicle.pose.x) && std::isfinite(vehicle.pose.y) &&
         std::isfinite(vehicle.pose.yaw_rad) &&
         std::isfinite(vehicle.speed_mps);
}

TrafficSignal update_confirmed_traffic_signal(BtNodeEnvironment &env) {
  auto &state = env.state;
  const auto &signal = env.context.inputs.traffic_signal;
  if (!fresh(signal, env.context.steady_time_s,
             env.config.freshness.traffic_timeout_s) ||
      (signal.value != static_cast<std::int8_t>(TrafficSignal::kGreen) &&
       signal.value != static_cast<std::int8_t>(TrafficSignal::kRed))) {
    state.traffic_signal_candidate = TrafficSignal::kUnknown;
    state.traffic_signal_confirmed = TrafficSignal::kUnknown;
    state.traffic_signal_timer_active = false;
    return state.traffic_signal_confirmed;
  }

  const auto observed = static_cast<TrafficSignal>(signal.value);
  if (!state.traffic_signal_timer_active ||
      observed != state.traffic_signal_candidate) {
    state.traffic_signal_candidate = observed;
    state.traffic_signal_candidate_started_s = env.context.steady_time_s;
    state.traffic_signal_timer_active = true;
    return state.traffic_signal_confirmed;
  }

  const double confirmation_duration_s =
      observed == TrafficSignal::kRed
          ? env.config.traffic.red_confirmation_duration_s
          : env.config.traffic.green_release_duration_s;
  if (env.context.steady_time_s - state.traffic_signal_candidate_started_s +
          1.0e-9 >=
      confirmation_duration_s) {
    state.traffic_signal_confirmed = observed;
  }
  return state.traffic_signal_confirmed;
}

bool grid_ready(const BtNodeEnvironment &env) {
  const auto &input = env.context.inputs.grid;
  return fresh(input, env.context.steady_time_s,
               env.config.freshness.grid_timeout_s) &&
         input.value.valid && input.value.fresh &&
         validate_occupancy_grid(input.value).valid;
}

bool route_occupancy_ready(const BtNodeEnvironment &env,
                           const double required_near_m,
                           const double required_far_m) {
  const auto &input = env.context.inputs.route_occupancy;
  const auto &value = input.value;
  if (!fresh(input, env.context.steady_time_s,
             env.config.freshness.grid_timeout_s)) {
    return false;
  }
  if (!std::isfinite(value.evaluated_near_m) ||
      !std::isfinite(value.evaluated_far_m) ||
      value.evaluated_near_m > required_near_m ||
      value.evaluated_far_m < required_far_m) {
    return false;
  }
  return !value.nearest_unsafe_distance_m ||
         (std::isfinite(*value.nearest_unsafe_distance_m) &&
          *value.nearest_unsafe_distance_m >= value.evaluated_near_m &&
          *value.nearest_unsafe_distance_m <= value.evaluated_far_m);
}

BT::NodeStatus brake(BtNodeEnvironment &env, const std::string &behavior,
                     const std::string &reason,
                     GearRequest gear = GearRequest::kKeep,
                     BT::NodeStatus status = BT::NodeStatus::SUCCESS) {
  env.context.claim_command(PlannerContext::full_brake_command(gear), behavior,
                            reason);
  return status;
}

BT::NodeStatus controlled_brake(BtNodeEnvironment &env,
                                const std::string &behavior,
                                const std::string &reason,
                                const double command) {
  env.context.claim_command(
      PlannerCommand{PhysicalCommand{0.0, command, 0.0}, GearRequest::kKeep},
      behavior, reason);
  return BT::NodeStatus::SUCCESS;
}

BT::NodeStatus run_controller(BtNodeEnvironment &env,
                              const std::function<ControllerResult()> &callback,
                              const std::string &behavior) {
  if (env.context.inputs.status.value.gear != kGearDriveCode) {
    return brake(env, behavior, "waiting for drive gear before path tracking",
                 GearRequest::kDrive);
  }
  if (!callback) {
    return brake(env, behavior, "controller is unavailable");
  }

  const auto result = callback();
  if (!result.valid) {
    return brake(env, behavior,
                 result.reason.empty() ? "controller returned no command"
                                       : result.reason);
  }

  env.context.claim_command(PlannerCommand{result.command, GearRequest::kKeep},
                            behavior);
  return BT::NodeStatus::SUCCESS;
}

bool collision_is_static(const CollisionObservation &collision) {
  return !collision.object_types.empty() &&
         std::all_of(collision.object_types.begin(),
                     collision.object_types.end(),
                     [](std::int16_t type) { return type == kStaticObject; });
}

bool rear_is_clear(const BtNodeEnvironment &env) {
  return grid_ready(env) &&
         footprint_is_safe(env.context.inputs.grid.value,
                           env.config.collision.rear_check_pose,
                           env.config.collision.rear_check_footprint);
}

void start_phase(BtNodeEnvironment &env, RecoveryPhase phase) {
  env.state.recovery_phase = phase;
  env.state.recovery_phase_started_s = env.context.steady_time_s;
}

bool phase_elapsed(const BtNodeEnvironment &env, double duration) {
  return env.context.steady_time_s - env.state.recovery_phase_started_s >=
         duration;
}

class CollisionRecoveryNode final : public BT::StatefulActionNode {
public:
  CollisionRecoveryNode(const std::string &name,
                        const BT::NodeConfiguration &config,
                        BtNodeEnvironmentPtr environment)
      : BT::StatefulActionNode(name, config),
        environment_(std::move(environment)) {}

  static BT::PortsList providedPorts() { return {}; }
  BT::NodeStatus onStart() override { return step(); }
  BT::NodeStatus onRunning() override { return step(); }

  void onHalted() override {
    auto &env = *environment_;
    if (env.state.recovery_phase != RecoveryPhase::kIdle &&
        env.state.recovery_phase != RecoveryPhase::kCooldown) {
      env.state.recovery_phase = RecoveryPhase::kWaitDriveAck;
      env.state.recovery_phase_started_s = env.state.last_tick_time_s;
    }
  }

private:
  BT::NodeStatus recovery_brake(const std::string &reason,
                                GearRequest gear = GearRequest::kKeep) {
    return brake(*environment_, "collision_recovery", reason, gear,
                 BT::NodeStatus::RUNNING);
  }

  BT::NodeStatus step() {
    auto &env = *environment_;
    auto &state = env.state;
    const auto &config = env.config.collision;
    const auto &collision = env.context.inputs.collisions;
    const auto &vehicle = env.context.inputs.status.value;
    const bool collision_present = !collision.object_types.empty();

    if (state.recovery_phase == RecoveryPhase::kIdle) {
      if (!collision_present) {
        return BT::NodeStatus::FAILURE;
      }
      if (!collision_is_static(collision)) {
        start_phase(env, RecoveryPhase::kHoldNonstatic);
        return recovery_brake("non-static collision");
      }
      start_phase(env, RecoveryPhase::kBrakingToStop);
      ++state.recovery_cycle_count;
      return recovery_brake("static collision: stopping");
    }

    if (state.recovery_phase == RecoveryPhase::kHoldNonstatic) {
      if (collision_present) {
        return recovery_brake("non-static collision remains");
      }
      start_phase(env, RecoveryPhase::kWaitDriveAck);
    }

    if (state.recovery_phase == RecoveryPhase::kBrakingToStop) {
      if (collision_present && !collision_is_static(collision)) {
        start_phase(env, RecoveryPhase::kHoldNonstatic);
        return recovery_brake("collision changed to non-static");
      }
      if (std::abs(vehicle.speed_mps) > config.near_stop_speed_mps) {
        return recovery_brake("stopping before reverse");
      }
      if (!rear_is_clear(env)) {
        return recovery_brake("rear path is not clear");
      }
      start_phase(env, RecoveryPhase::kWaitReverseAck);
      return recovery_brake("requesting reverse gear", GearRequest::kReverse);
    }

    if (state.recovery_phase == RecoveryPhase::kWaitReverseAck) {
      if ((collision_present && !collision_is_static(collision)) ||
          !rear_is_clear(env)) {
        start_phase(env, RecoveryPhase::kWaitDriveAck);
        return recovery_brake("reverse cancelled");
      }
      if (vehicle.gear != kGearReverseCode) {
        return recovery_brake("waiting for reverse gear",
                              GearRequest::kReverse);
      }
      start_phase(env, RecoveryPhase::kReverseAccel);
    }

    if (state.recovery_phase == RecoveryPhase::kReverseAccel) {
      if ((collision_present && !collision_is_static(collision)) ||
          !rear_is_clear(env) || vehicle.gear != kGearReverseCode) {
        start_phase(env, RecoveryPhase::kWaitDriveAck);
        return recovery_brake("reverse cancelled");
      }
      if (phase_elapsed(env, config.reverse_duration_s)) {
        start_phase(env, RecoveryPhase::kReverseBrake);
        return recovery_brake("reverse complete", GearRequest::kReverse);
      }
      env.context.claim_command(
          PlannerCommand{PhysicalCommand{config.reverse_accel, 0.0, 0.0},
                         GearRequest::kReverse},
          "collision_recovery");
      return BT::NodeStatus::RUNNING;
    }

    if (state.recovery_phase == RecoveryPhase::kReverseBrake) {
      if (std::abs(vehicle.speed_mps) <= config.near_stop_speed_mps &&
          phase_elapsed(env, config.reverse_brake_duration_s)) {
        start_phase(env, RecoveryPhase::kWaitDriveAck);
      } else {
        return recovery_brake("braking after reverse", GearRequest::kReverse);
      }
    }

    if (state.recovery_phase == RecoveryPhase::kWaitDriveAck) {
      if (collision_present) {
        return recovery_brake("waiting for collision clear");
      }
      if (std::abs(vehicle.speed_mps) > config.near_stop_speed_mps) {
        return recovery_brake("stopping before drive gear");
      }
      if (vehicle.gear != kGearDriveCode) {
        return recovery_brake("requesting drive gear", GearRequest::kDrive);
      }
      start_phase(env, RecoveryPhase::kCooldown);
      return recovery_brake("drive gear restored");
    }

    if (state.recovery_phase == RecoveryPhase::kCooldown) {
      if (collision_present) {
        state.recovery_phase_started_s = env.context.steady_time_s;
        return recovery_brake("waiting for collision clear");
      }
      if (!phase_elapsed(env, config.cooldown_s)) {
        return recovery_brake("collision cooldown");
      }
      state.recovery_phase = RecoveryPhase::kIdle;
      return BT::NodeStatus::FAILURE;
    }

    return recovery_brake("invalid recovery state");
  }

  BtNodeEnvironmentPtr environment_;
};

BT::NodeStatus inputs_ready(BtNodeEnvironment &env) {
  const auto &inputs = env.context.inputs;
  const auto &timeout = env.config.freshness;
  const bool ready = inputs.route_ready &&
                     fresh(inputs.status, env.context.steady_time_s,
                           timeout.status_timeout_s) &&
                     fresh(inputs.collisions, env.context.steady_time_s,
                           timeout.collision_timeout_s) &&
                     valid_vehicle(inputs.status.value);
  return ready ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

BT::NodeStatus traffic_stop(BtNodeEnvironment &env) {
  auto &context = env.context;
  auto &state = env.state;
  const auto &config = env.config.traffic;

  if (config.stop_line_enabled) {
    const auto &line = context.inputs.stop_line;
    if (!fresh(line, context.steady_time_s,
               env.config.freshness.stop_line_timeout_s)) {
      return brake(env, "traffic_stop", "stop-line input is stale");
    }
    if (!line.value) {
      state.stop_line_armed = true;
    } else if (state.stop_line_armed) {
      state.traffic_latched = true;
      state.stop_line_armed = false;
    }
  }

  const auto &pose = context.inputs.status.value.pose;
  const double speed_mps = std::abs(context.inputs.status.value.speed_mps);
  const double stopping_distance_m =
      config.front_bumper_x_m + config.stopping_margin_m +
      speed_mps * config.reaction_time_s +
      speed_mps * speed_mps / (2.0 * config.braking_deceleration_mps2);
  double nearest_forward_distance_m = std::numeric_limits<double>::infinity();
  std::optional<Point3> nearest_zone;
  for (const auto &zone : config.zones) {
    const double dx = zone.x - pose.x;
    const double dy = zone.y - pose.y;
    const double cosine = std::cos(pose.yaw_rad);
    const double sine = std::sin(pose.yaw_rad);
    const double forward_distance_m = cosine * dx + sine * dy;
    const double lateral_distance_m = -sine * dx + cosine * dy;
    if (forward_distance_m >= 0.0 &&
        std::abs(lateral_distance_m) <= config.zone_entry_radius_m &&
        forward_distance_m < nearest_forward_distance_m) {
      nearest_forward_distance_m = forward_distance_m;
      nearest_zone = zone;
    }
  }

  const TrafficSignal signal = update_confirmed_traffic_signal(env);
  if (!state.traffic_latched && signal == TrafficSignal::kRed &&
      nearest_forward_distance_m <= stopping_distance_m) {
    state.traffic_latched = true;
    state.traffic_stop_zone = nearest_zone;
  }

  if (!state.traffic_latched) {
    return BT::NodeStatus::FAILURE;
  }

  if (signal == TrafficSignal::kGreen) {
    state.traffic_latched = false;
    state.traffic_stopped = false;
    state.traffic_stop_zone.reset();
    return BT::NodeStatus::FAILURE;
  }

  double active_forward_distance_m = nearest_forward_distance_m;
  if (state.traffic_stop_zone) {
    const double dx = state.traffic_stop_zone->x - pose.x;
    const double dy = state.traffic_stop_zone->y - pose.y;
    active_forward_distance_m =
        std::cos(pose.yaw_rad) * dx + std::sin(pose.yaw_rad) * dy;
  }

  if (speed_mps > config.near_stop_speed_mps) {
    const double distance_available_for_braking_m =
        active_forward_distance_m - config.front_bumper_x_m -
        config.stopping_margin_m - speed_mps * config.reaction_time_s;
    const double required_deceleration_mps2 =
        distance_available_for_braking_m > 0.0
            ? speed_mps * speed_mps / (2.0 * distance_available_for_braking_m)
            : std::numeric_limits<double>::infinity();
    if (!std::isfinite(active_forward_distance_m) ||
        required_deceleration_mps2 > config.braking_deceleration_mps2 *
                                         config.late_brake_deceleration_ratio) {
      return brake(env, "traffic_stop",
                   "profile stop is late: emergency braking");
    }
    return controlled_brake(env, "traffic_stop",
                            "profile braking before traffic stop line",
                            config.brake_command);
  }
  state.traffic_stopped = true;
  return brake(env, "traffic_stop", "waiting for confirmed green signal");
}

BT::NodeStatus perception_mission(BtNodeEnvironment &env) {
  if (!env.config.perception.enabled) {
    env.state.perception_latched = false;
    env.state.perception_clear_timer_active = false;
    env.state.perception_latched_lookahead_m = 0.0;
    return BT::NodeStatus::FAILURE;
  }
  if (!grid_ready(env)) {
    return brake(env, "perception_mission", "occupancy grid is stale");
  }
  const auto &future_risk = env.context.inputs.future_road_risk;
  if (env.config.perception.future_road_risk_required &&
      !fresh(future_risk, env.context.steady_time_s,
             env.config.perception.future_road_risk_timeout_s)) {
    return brake(env, "perception_mission",
                 "future road risk is stale or invalid");
  }
  const auto forward_check = make_perception_forward_check(
      env.config.perception, env.context.inputs.status.value.speed_mps,
      env.state.perception_latched ? env.state.perception_latched_lookahead_m
                                   : 0.0);
  bool clear = false;
  if (env.config.perception.route_aligned_activation) {
    if (!route_occupancy_ready(env, env.config.perception.near_x_m,
                               forward_check.far_x_m)) {
      return brake(env, "perception_mission", "route occupancy is stale");
    }
    clear = env.context.inputs.route_occupancy.value.clear;
  } else {
    clear = footprint_is_safe(env.context.inputs.grid.value, forward_check.pose,
                              forward_check.footprint);
  }
  if (env.config.perception.future_road_risk_required &&
      future_risk.value.risk) {
    clear = false;
  }
  if (!clear) {
    env.state.perception_latched = true;
    env.state.perception_clear_timer_active = false;
    env.state.perception_latched_lookahead_m = std::max(
        env.state.perception_latched_lookahead_m, forward_check.far_x_m);
  } else if (!env.state.perception_latched) {
    return BT::NodeStatus::FAILURE;
  } else if (!env.state.perception_clear_timer_active) {
    env.state.perception_clear_timer_active = true;
    env.state.perception_clear_started_s = env.context.steady_time_s;
  } else if (env.context.steady_time_s - env.state.perception_clear_started_s >=
             env.config.perception.clear_release_duration_s) {
    env.state.perception_latched = false;
    env.state.perception_clear_timer_active = false;
    env.state.perception_latched_lookahead_m = 0.0;
    return BT::NodeStatus::FAILURE;
  }
  return run_controller(env, env.context.callbacks.perception_local_planner,
                        "perception_mission");
}

} // namespace

PerceptionForwardCheck
make_perception_forward_check(const PerceptionMissionConfig &config,
                              double speed_mps, double minimum_far_x_m) {
  if (!config.speed_aware_lookahead) {
    const double far_x = config.forward_check_pose.x +
                         config.forward_check_footprint.center_offset_x_m +
                         config.forward_check_footprint.half_length_m +
                         config.forward_check_footprint.clearance_m;
    return PerceptionForwardCheck{config.forward_check_pose,
                                  config.forward_check_footprint, far_x};
  }

  const double speed = std::max(0.0, speed_mps);
  const double stopping_distance =
      speed * config.reaction_time_s +
      speed * speed / (2.0 * config.braking_deceleration_mps2);
  const double speed_far_x = std::clamp(
      config.front_bumper_x_m + config.stopping_margin_m + stopping_distance,
      config.minimum_lookahead_m, config.maximum_lookahead_m);
  const double far_x =
      std::clamp(std::max(speed_far_x, minimum_far_x_m),
                 config.minimum_lookahead_m, config.maximum_lookahead_m);
  const double center_x = 0.5 * (config.near_x_m + far_x);
  auto footprint = config.forward_check_footprint;
  footprint.half_length_m = 0.5 * (far_x - config.near_x_m);
  footprint.center_offset_x_m = 0.0;
  return PerceptionForwardCheck{Pose2{center_x, config.forward_check_pose.y,
                                      config.forward_check_pose.yaw_rad},
                                footprint, far_x};
}

std::vector<std::string> ad_bt_node_ids() {
  return {"CollisionRecovery", "FailSafeBrake",     "FollowGlobalPath",
          "InputsReady",       "PerceptionMission", "TrafficStop"};
}

void register_ad_bt_nodes(BT::BehaviorTreeFactory &factory,
                          const BtNodeEnvironmentPtr &environment) {
  factory.registerSimpleCondition("InputsReady", [environment](BT::TreeNode &) {
    return inputs_ready(*environment);
  });
  factory.registerBuilder<CollisionRecoveryNode>(
      "CollisionRecovery",
      [environment](const std::string &name, const auto &config) {
        return std::make_unique<CollisionRecoveryNode>(name, config,
                                                       environment);
      });
  factory.registerSimpleAction("TrafficStop", [environment](BT::TreeNode &) {
    return traffic_stop(*environment);
  });
  factory.registerSimpleAction("PerceptionMission",
                               [environment](BT::TreeNode &) {
                                 return perception_mission(*environment);
                               });
  factory.registerSimpleAction(
      "FollowGlobalPath", [environment](BT::TreeNode &) {
        return run_controller(*environment,
                              environment->context.callbacks.follow_global_path,
                              "follow_global_path");
      });
  factory.registerSimpleAction("FailSafeBrake", [environment](BT::TreeNode &) {
    return brake(*environment, "fail_safe_brake",
                 "behavior tree selected fail-safe braking");
  });
}

} // namespace ad_planner
