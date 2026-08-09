#ifndef AD_PLANNER__IO__ROUTE_CORRIDOR_LOADER_HPP_
#define AD_PLANNER__IO__ROUTE_CORRIDOR_LOADER_HPP_

#include <filesystem>
#include <map>
#include <string>

#include "ad_planner/local_planning/common/local_motion.hpp"

namespace ad_planner {

struct RouteCorridorMetadata {
  int schema_version{0};
  std::map<std::string, std::string> source_sha256;
};

struct LoadedRouteCorridor {
  RouteCorridorMetadata metadata;
  ReferenceCorridor corridor;
};

LoadedRouteCorridor load_route_corridor(
    const std::filesystem::path &path,
    const std::map<std::string, std::string> &expected_sha256 = {});

} // namespace ad_planner

#endif // AD_PLANNER__IO__ROUTE_CORRIDOR_LOADER_HPP_
