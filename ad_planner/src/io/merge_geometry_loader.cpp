#include "ad_planner/io/merge_geometry_loader.hpp"

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

LoadedMergeGeometry load_merge_geometry(
  const std::filesystem::path & path, const std::string & zone_id)
{
  YAML::Node document;
  try {
    document = YAML::LoadFile(path.string());
  } catch (const YAML::Exception & error) {
    invalid(path, std::string("cannot parse merge geometry file: ") + error.what());
  }
  if (!document.IsMap()) {
    invalid(path, "merge geometry file must be a mapping");
  }

  LoadedMergeGeometry loaded;
  try {
    loaded.schema_version = required(document, "schema_version", path).as<int>();
    loaded.frame_id = required(document, "frame_id", path).as<std::string>();
    const YAML::Node sha = document["route_corridor_link_set_sha256"];
    if (sha && sha.IsDefined()) {
      loaded.route_corridor_link_set_sha256 = sha.as<std::string>();
    }
  } catch (const YAML::Exception & error) {
    invalid(path, std::string("invalid merge geometry metadata: ") + error.what());
  }
  if (loaded.schema_version != 1) {
    invalid(path, "unsupported schema_version");
  }
  if (loaded.frame_id != "map") {
    invalid(path, "merge geometry frame_id must be 'map'");
  }

  const YAML::Node zones = required(document, "merge_zones", path);
  if (!zones.IsSequence() || zones.size() == 0U) {
    invalid(path, "merge_zones must be a non-empty sequence");
  }

  bool found = false;
  try {
    for (const auto & entry : zones) {
      if (!entry.IsMap()) {
        invalid(path, "each merge zone must be a mapping");
      }
      const std::string id = required(entry, "merge_zone_id", path).as<std::string>();
      if (id != zone_id) {
        continue;
      }
      if (found) {
        invalid(path, "duplicate merge_zone_id '" + zone_id + "'");
      }
      found = true;
      loaded.zone.id = id;
      loaded.zone.source_lane_id =
        required(entry, "source_lane_sequence_id", path).as<std::string>();
      loaded.zone.target_lane_id =
        required(entry, "target_lane_sequence_id", path).as<std::string>();
      loaded.zone.route_s_zone_entry_m =
        required(entry, "route_s_zone_entry_m", path).as<double>();
      loaded.zone.route_s_merge_complete_m =
        required(entry, "route_s_merge_complete_m", path).as<double>();
    }
  } catch (const YAML::Exception & error) {
    invalid(path, std::string("invalid merge zone: ") + error.what());
  }
  if (!found) {
    invalid(path, "merge_zone_id '" + zone_id + "' not found");
  }
  if (loaded.zone.source_lane_id.empty() || loaded.zone.target_lane_id.empty()) {
    invalid(path, "merge zone lane ids must be non-empty");
  }

  try {
    loaded.zone = loaded.zone.validated();
  } catch (const std::exception & error) {
    invalid(path, std::string("merge zone geometry is invalid: ") + error.what());
  }
  return loaded;
}

}  // namespace ad_planner
