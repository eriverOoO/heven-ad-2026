#ifndef AD_PLANNER__PLANNER_CONTEXT_HPP_
#define AD_PLANNER__PLANNER_CONTEXT_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ad_control/command/command_adapter.hpp"
#include "ad_planner/common/types.hpp"
#include "ad_planner/local_planning/common/future_road_risk.hpp"
#include "ad_planner/local_planning/common/occupancy.hpp"

namespace ad_planner {

inline constexpr double kMaximumAllowedSteeringRad = 0.6981317;
inline constexpr double kMaximumAllowedNearStopSpeedMps = 0.05;
inline constexpr double kMaximumReverseDurationS = 2.0;
using ad_control::AcknowledgedGear;
using ad_control::GearRequest;
using ad_control::kGearDriveCode;
using ad_control::kGearReverseCode;

enum class TrafficSignal : std::int8_t {
  kUnknown = -1,
  kGreen = 0,
  kRed = 1,
};

enum class RecoveryPhase {
  kIdle,
  kBrakingToStop,
  kWaitReverseAck,
  kReverseAccel,
  kReverseBrake,
  kWaitDriveAck,
  kCooldown,
  kHoldNonstatic,
};

enum class SupervisorStatus {
  kSuccess,
  kRunning,
  kFailure,
  kException,
};

struct PlannerCommand {
  PhysicalCommand motion{};
  GearRequest gear_request{GearRequest::kKeep};
};

inline bool operator==(const PlannerCommand &lhs, const PlannerCommand &rhs) {
  return lhs.motion == rhs.motion && lhs.gear_request == rhs.gear_request;
}

template <typename T> struct ReceiptObservation {
  bool received{false};
  bool valid{false};
  double receipt_time_s{0.0};
  T value{};
};

struct VehicleState {
  Pose2 pose;
  double speed_mps{0.0};
  std::int8_t gear{kGearDriveCode};
};

using VehicleObservation = ReceiptObservation<VehicleState>;
using GridObservation = ReceiptObservation<OccupancyGrid>;

struct RouteOccupancyState {
  bool clear{false};
  double evaluated_near_m{0.0};
  double evaluated_far_m{0.0};
  std::size_t occupied_cell_count{0U};
  std::size_t unknown_cell_count{0U};
  std::optional<double> nearest_unsafe_distance_m;
};

using RouteOccupancyObservation = ReceiptObservation<RouteOccupancyState>;
using FutureRoadRiskObservation = ReceiptObservation<FutureRoadRiskState>;

struct CollisionObservation {
  bool received{false};
  bool valid{false};
  double receipt_time_s{0.0};
  std::vector<std::int16_t> object_types;
};

struct PlannerInputs {
  bool route_ready{false};
  VehicleObservation status;
  CollisionObservation collisions;
  GridObservation grid;
  RouteOccupancyObservation route_occupancy;
  FutureRoadRiskObservation future_road_risk;
  ReceiptObservation<std::int8_t> traffic_signal;
  ReceiptObservation<bool> stop_line;
};

struct PlannerCallbacks {
  std::function<ControllerResult()> follow_global_path;
  std::function<ControllerResult()> perception_local_planner;
};

struct FreshnessConfig {
  double status_timeout_s{0.25};
  double collision_timeout_s{0.25};
  double grid_timeout_s{0.25};
  double traffic_timeout_s{0.5};
  double stop_line_timeout_s{0.25};
};

struct CollisionRecoveryConfig {
  double near_stop_speed_mps{kMaximumAllowedNearStopSpeedMps};
  double reverse_accel{0.2};
  double reverse_duration_s{kMaximumReverseDurationS};
  double reverse_brake_duration_s{1.0};
  double cooldown_s{2.0};
  Pose2 rear_check_pose{-2.0, 0.0, 0.0};
  FootprintConfig rear_check_footprint{0.75, 0.75, 0.1, 50, 2000};
};

struct TrafficStopConfig {
  double near_stop_speed_mps{kMaximumAllowedNearStopSpeedMps};
  double zone_entry_radius_m{2.0};
  double zone_exit_radius_m{3.0};
  double front_bumper_x_m{3.845};
  double stopping_margin_m{1.0};
  double reaction_time_s{0.11589156};
  double braking_deceleration_mps2{1.8};
  double brake_command{0.2};
  double late_brake_deceleration_ratio{1.15};
  double red_confirmation_duration_s{0.2};
  double green_release_duration_s{0.5};
  bool stop_line_enabled{false};
  std::vector<Point3> zones;
};

struct PerceptionMissionConfig {
  bool enabled{true};
  bool route_aligned_activation{false};
  double clear_release_duration_s{2.0};
  bool speed_aware_lookahead{true};
  double near_x_m{1.0};
  double minimum_lookahead_m{20.0};
  double maximum_lookahead_m{99.0};
  double front_bumper_x_m{3.845};
  double reaction_time_s{1.0};
  double braking_deceleration_mps2{1.8};
  double stopping_margin_m{5.0};
  bool future_road_risk_required{false};
  double future_road_risk_timeout_s{0.5};
  FutureRoadRiskLimits future_road_risk_limits;
  Pose2 forward_check_pose{5.5, 0.0, 0.0};
  FootprintConfig forward_check_footprint{4.2, 1.2, 0.1, 50, 32768};
};

struct PlannerConfig {
  FreshnessConfig freshness;
  CollisionRecoveryConfig collision;
  TrafficStopConfig traffic;
  PerceptionMissionConfig perception;
  double maximum_steering_rad{kMaximumAllowedSteeringRad};
};

struct PlannerRuntimeState {
  RecoveryPhase recovery_phase{RecoveryPhase::kIdle};
  std::size_t recovery_cycle_count{0};
  double recovery_phase_started_s{0.0};

  bool traffic_latched{false};
  bool traffic_stopped{false};
  bool traffic_zone_armed{true};
  bool stop_line_armed{true};
  std::optional<Point3> traffic_stop_zone;
  TrafficSignal traffic_signal_candidate{TrafficSignal::kUnknown};
  TrafficSignal traffic_signal_confirmed{TrafficSignal::kUnknown};
  bool traffic_signal_timer_active{false};
  double traffic_signal_candidate_started_s{0.0};

  bool perception_latched{false};
  bool perception_clear_timer_active{false};
  double perception_clear_started_s{0.0};
  double perception_latched_lookahead_m{0.0};

  bool tick_time_initialized{false};
  double last_tick_time_s{0.0};
};

struct PlannerTickResult {
  PlannerCommand command;
  SupervisorStatus status{SupervisorStatus::kFailure};
  std::string active_behavior;
  std::string failsafe_reason;
  std::size_t claim_count{0};
};

class PlannerContext {
public:
  double steady_time_s{0.0};
  PlannerInputs inputs;
  PlannerCallbacks callbacks;

  void begin_tick() {
    staged_command_ = full_brake_command();
    claim_count_ = 0;
    active_behavior_.clear();
    failsafe_reason_.clear();
  }

  bool claim_command(const PlannerCommand &command, std::string behavior,
                     std::string reason = {}) {
    ++claim_count_;
    if (claim_count_ != 1) {
      return false;
    }
    staged_command_ = command;
    active_behavior_ = std::move(behavior);
    failsafe_reason_ = std::move(reason);
    return true;
  }

  void force_full_brake(std::string reason) {
    staged_command_ = full_brake_command();
    active_behavior_ = "fail_safe_brake";
    failsafe_reason_ = std::move(reason);
  }

  static PlannerCommand
  full_brake_command(GearRequest gear_request = GearRequest::kKeep) {
    return PlannerCommand{PhysicalCommand{0.0, 1.0, 0.0}, gear_request};
  }

  const PlannerCommand &staged_command() const noexcept {
    return staged_command_;
  }
  std::size_t claim_count() const noexcept { return claim_count_; }
  const std::string &active_behavior() const noexcept {
    return active_behavior_;
  }
  const std::string &failsafe_reason() const noexcept {
    return failsafe_reason_;
  }

private:
  PlannerCommand staged_command_{full_brake_command()};
  std::size_t claim_count_{0};
  std::string active_behavior_;
  std::string failsafe_reason_;
};

} // namespace ad_planner

#endif // AD_PLANNER__PLANNER_CONTEXT_HPP_
