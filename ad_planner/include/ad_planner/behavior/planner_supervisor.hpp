#ifndef AD_PLANNER__PLANNER_SUPERVISOR_HPP_
#define AD_PLANNER__PLANNER_SUPERVISOR_HPP_

#include <memory>
#include <string>
#include <vector>

#include "ad_planner/behavior/planner_context.hpp"

namespace ad_planner
{

class PlannerSupervisor
{
public:
  PlannerSupervisor(
    PlannerContext & context, PlannerConfig config, const std::string & tree_xml);
  ~PlannerSupervisor();

  PlannerSupervisor(const PlannerSupervisor &) = delete;
  PlannerSupervisor & operator=(const PlannerSupervisor &) = delete;
  PlannerSupervisor(PlannerSupervisor &&) noexcept;
  PlannerSupervisor & operator=(PlannerSupervisor &&) noexcept;

  PlannerTickResult tick();
  void halt();
  const PlannerRuntimeState & state() const noexcept;
  std::vector<std::string> registered_node_ids() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__PLANNER_SUPERVISOR_HPP_
