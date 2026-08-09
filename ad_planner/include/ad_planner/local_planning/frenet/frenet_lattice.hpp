#ifndef AD_PLANNER__LOCAL_PLANNING__FRENET_LATTICE_HPP_
#define AD_PLANNER__LOCAL_PLANNING__FRENET_LATTICE_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

struct FrenetLatticeConfig
{
  std::vector<double> lateral_targets_m;
  std::vector<double> target_speeds_mps;
  std::vector<double> durations_s;
  double sample_dt_s{0.1};
  double maximum_curvature_inv_m{0.0};
  double maximum_acceleration_mps2{0.0};
  double maximum_lateral_acceleration_mps2{0.0};
  double maximum_jerk_mps3{0.0};
  double maximum_lateral_transition_m{5.0};
  double footprint_clearance_m{0.0};
  std::int8_t occupied_threshold{100};
  std::size_t maximum_cells_to_check{4096U};
  double progress_weight{1.0};
  double clearance_weight{1.0};
  double jerk_weight{1.0};
  double lateral_offset_weight{1.0};
  double lane_change_weight{1.0};
  double continuity_weight{1.0};
  std::size_t maximum_candidates{512U};
};

class FrenetLatticeBackend final : public LocalMotionBackend
{
public:
  explicit FrenetLatticeBackend(FrenetLatticeConfig config);

  LocalPlanningResult plan(const LocalPlanningRequest & request) override;

private:
  FrenetLatticeConfig config_;
};

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__FRENET_LATTICE_HPP_
