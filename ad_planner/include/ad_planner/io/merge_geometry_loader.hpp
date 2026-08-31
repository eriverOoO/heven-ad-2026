#ifndef AD_PLANNER__IO__MERGE_GEOMETRY_LOADER_HPP_
#define AD_PLANNER__IO__MERGE_GEOMETRY_LOADER_HPP_

#include <filesystem>
#include <string>

#include "ad_planner/planning/highway_merge_gap_risk.hpp"

namespace ad_planner
{

struct LoadedMergeGeometry
{
  int schema_version{0};
  std::string frame_id;
  std::string route_corridor_link_set_sha256;
  MergeZone zone;
};

// Reads highway_merge.json and returns the single merge zone whose
// merge_zone_id matches zone_id. The zone's route_s_zone_entry_m /
// route_s_merge_complete_m are declarative provenance values; the caller is
// expected to cross-check them against the source lane in the checksum-verified
// route corridor. Throws std::runtime_error on a missing file, malformed
// structure, unknown zone_id, or a zone that fails MergeZone::validated().
LoadedMergeGeometry load_merge_geometry(
  const std::filesystem::path & path, const std::string & zone_id);

}  // namespace ad_planner

#endif  // AD_PLANNER__IO__MERGE_GEOMETRY_LOADER_HPP_
