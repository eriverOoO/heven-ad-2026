#ifndef AD_PLANNER__LOCAL_PLANNING__FUTURE_ROAD_RISK_HPP_
#define AD_PLANNER__LOCAL_PLANNING__FUTURE_ROAD_RISK_HPP_

#include <cstddef>
#include <optional>

#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner
{

struct FutureRoadRiskLimits
{
  double maximum_horizon_s{6.0};
  std::size_t maximum_objects{128U};
  std::size_t maximum_footprints{2048U};
  std::size_t maximum_corridor_segments{4096U};
  double covariance_sigma{2.0};
  double minimum_margin_m{0.20};
};

struct FutureRoadRiskState
{
  bool risk{false};
  std::size_t risky_object_count{0U};
  std::size_t evaluated_object_count{0U};
  std::size_t evaluated_footprint_count{0U};
  // Conservative lower bound for the first risky prediction interval.
  std::optional<double> earliest_risk_time_s;
};

// Evaluates only t > 0 prediction geometry. The current footprint remains an
// OGM concern and is never copied into the planning grid by this function.
// Inputs must already share a frame and the corridor should already be cropped
// around the ego. Invalid geometry or work above the configured bounds throws,
// allowing the caller to fail closed instead of silently dropping risk.
FutureRoadRiskState evaluate_future_road_risk(
  const ReferenceCorridor & corridor,
  const PredictedObjectSet & objects,
  const FutureRoadRiskLimits & limits = {});

}  // namespace ad_planner

#endif  // AD_PLANNER__LOCAL_PLANNING__FUTURE_ROAD_RISK_HPP_
