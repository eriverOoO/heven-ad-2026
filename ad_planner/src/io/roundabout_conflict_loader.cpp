#include "ad_planner/io/roundabout_conflict_loader.hpp"

#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

namespace ad_planner
{
namespace
{

[[noreturn]] void invalid(
  const std::filesystem::path & path, const std::string & message)
{
  throw std::runtime_error(path.string() + ": " + message);
}

YAML::Node required(
  const YAML::Node & mapping, const std::string & key,
  const std::filesystem::path & path)
{
  if (!mapping.IsMap()) {
    invalid(path, "expected a mapping while reading '" + key + "'");
  }
  const YAML::Node value = mapping[key];
  if (!value || !value.IsDefined()) {
    invalid(path, "missing required field '" + key + "'");
  }
  return value;
}

}  // namespace

LoadedRoundaboutConflict load_roundabout_conflict(
  const std::filesystem::path & path, const std::string & zone_id)
{
  YAML::Node document;
  try {
    document = YAML::LoadFile(path.string());
  } catch (const YAML::Exception & error) {
    invalid(path, std::string("cannot parse roundabout conflict file: ") + error.what());
  }
  if (!document.IsMap()) {
    invalid(path, "roundabout conflict file must be a mapping");
  }

  LoadedRoundaboutConflict loaded;
  try {
    loaded.schema_version = required(document, "schema_version", path).as<int>();
    loaded.frame_id = required(document, "frame_id", path).as<std::string>();
    const YAML::Node sha = document["route_corridor_sha256"];
    if (sha && sha.IsDefined()) {
      loaded.route_corridor_sha256 = sha.as<std::string>();
    }
  } catch (const YAML::Exception & error) {
    invalid(path, std::string("invalid roundabout conflict metadata: ") + error.what());
  }
  if (loaded.schema_version != 1) {
    invalid(path, "unsupported schema_version");
  }
  if (loaded.frame_id != "map") {
    invalid(path, "roundabout conflict frame_id must be 'map'");
  }

  const YAML::Node zones = required(document, "conflict_zones", path);
  if (!zones.IsSequence() || zones.size() == 0U) {
    invalid(path, "conflict_zones must be a non-empty sequence");
  }

  bool found = false;
  try {
    for (const auto & entry : zones) {
      if (!entry.IsMap()) {
        invalid(path, "each conflict zone must be a mapping");
      }
      const std::string id = required(entry, "conflict_zone_id", path).as<std::string>();
      if (id != zone_id) {
        continue;
      }
      if (found) {
        invalid(path, "duplicate conflict_zone_id '" + zone_id + "'");
      }
      found = true;
      loaded.zone.id = id;
      loaded.zone.route_s_enter_m =
        required(entry, "route_s_enter_m", path).as<double>();
      loaded.zone.route_s_exit_m =
        required(entry, "route_s_exit_m", path).as<double>();
      const YAML::Node polygon = required(entry, "polygon_m", path);
      if (!polygon.IsSequence() || polygon.size() < 3U) {
        invalid(path, "polygon_m must be a sequence of >= 3 vertices");
      }
      loaded.zone.polygon_m.reserve(polygon.size());
      for (const auto & vertex : polygon) {
        if (!vertex.IsSequence() || vertex.size() != 2U) {
          invalid(path, "each polygon_m vertex must be an [x, y] pair");
        }
        loaded.zone.polygon_m.push_back(
          ConflictPoint2{vertex[0].as<double>(), vertex[1].as<double>()});
      }
    }
  } catch (const YAML::Exception & error) {
    invalid(path, std::string("invalid conflict zone: ") + error.what());
  }
  if (!found) {
    invalid(path, "conflict_zone_id '" + zone_id + "' not found");
  }

  try {
    loaded.zone = loaded.zone.validated();
  } catch (const std::exception & error) {
    invalid(path, std::string("conflict zone geometry is invalid: ") + error.what());
  }
  return loaded;
}

}  // namespace ad_planner
