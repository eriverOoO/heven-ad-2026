#ifndef AD_PLANNER__IO__ROUNDABOUT_CONFLICT_LOADER_HPP_
#define AD_PLANNER__IO__ROUNDABOUT_CONFLICT_LOADER_HPP_

#include <filesystem>
#include <string>

#include "ad_planner/planning/roundabout_gap_risk.hpp"

namespace ad_planner
{

struct LoadedRoundaboutConflict
{
  int schema_version{0};
  std::string frame_id;
  std::string route_corridor_sha256;
  RoundaboutConflictZone zone;
};

// Reads roundabout_conflicts.json and returns the single conflict zone whose
// conflict_zone_id matches zone_id. Throws std::runtime_error on a missing
// file, malformed structure, unknown zone_id, or a zone that fails
// RoundaboutConflictZone::validated().
LoadedRoundaboutConflict load_roundabout_conflict(
  const std::filesystem::path & path, const std::string & zone_id);

}  // namespace ad_planner

#endif  // AD_PLANNER__IO__ROUNDABOUT_CONFLICT_LOADER_HPP_
