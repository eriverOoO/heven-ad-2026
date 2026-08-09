#ifndef AD_PLANNER__LOCAL_PLANNING__MPPI_NAV2_BACKEND_HPP_
#define AD_PLANNER__LOCAL_PLANNING__MPPI_NAV2_BACKEND_HPP_

#include <cstddef>
#include <mutex>
#include <optional>

#include "ad_control/command/mppi_command_adapter.hpp"
#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

struct MppiNav2BackendConfig
{
  double command_timeout_s{0.20};
  double diagnostic_rollout_dt_s{0.10};
  double diagnostic_rollout_horizon_s{3.0};
  ad_control::MppiCommandConfig command;
};

class MppiNav2Backend final : public LocalMotionBackend
{
public:
  explicit MppiNav2Backend(MppiNav2BackendConfig config);

  bool observe_external_velocity_command(
    const ExternalVelocityCommand & command) override;

  LocalPlanningResult plan(const LocalPlanningRequest & request) override;

private:
  MppiNav2BackendConfig config_;
  std::size_t rollout_step_count_{0U};
  std::mutex state_mutex_;
  std::optional<ExternalVelocityCommand> command_;
  std::int64_t last_successful_plan_steady_ns_{0};
  ad_control::MppiCommandAdapter command_adapter_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__MPPI_NAV2_BACKEND_HPP_
