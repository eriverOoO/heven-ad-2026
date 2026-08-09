#ifndef AD_PLANNER__LOCAL_PLANNING__DWA_BACKEND_HPP_
#define AD_PLANNER__LOCAL_PLANNING__DWA_BACKEND_HPP_

#include "ad_planner/local_planning/dwa/dwa.hpp"
#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

class DwaBackend final : public LocalMotionBackend
{
public:
  explicit DwaBackend(DwaConfig config);

  LocalPlanningResult plan(const LocalPlanningRequest & request) override;

private:
  DwaController controller_;
  double dt_s_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__DWA_BACKEND_HPP_
