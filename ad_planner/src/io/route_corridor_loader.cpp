#include "ad_planner/io/route_corridor_loader.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace ad_planner
{
namespace
{

[[noreturn]] void invalid(const std::filesystem::path & path, const std::string & message)
{
  throw std::runtime_error(path.string() + ": " + message);
}

YAML::Node required_member(
  const YAML::Node & mapping, const std::string & key, const std::filesystem::path & path)
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

std::string string_value(
  const YAML::Node & node, const std::filesystem::path & path, const std::string & context)
{
  if (!node.IsScalar()) {
    invalid(path, context + " must be a nonempty string");
  }
  try {
    const auto value = node.as<std::string>();
    if (value.empty()) {
      invalid(path, context + " must be a nonempty string");
    }
    return value;
  } catch (const YAML::Exception &) {
    invalid(path, context + " must be a nonempty string");
  }
}

double finite_number(
  const YAML::Node & node, const std::filesystem::path & path, const std::string & context)
{
  if (!node.IsScalar()) {
    invalid(path, context + " must be a finite number");
  }
  try {
    const double value = node.as<double>();
    if (!std::isfinite(value)) {
      invalid(path, context + " must be a finite number");
    }
    return value;
  } catch (const YAML::Exception &) {
    invalid(path, context + " must be a finite number");
  }
}

double positive_number(
  const YAML::Node & node, const std::filesystem::path & path, const std::string & context)
{
  const double value = finite_number(node, path, context);
  if (value <= 0.0) {
    invalid(path, context + " must be greater than zero");
  }
  return value;
}

double nonnegative_number(
  const YAML::Node & node, const std::filesystem::path & path, const std::string & context)
{
  const double value = finite_number(node, path, context);
  if (value < 0.0) {
    invalid(path, context + " must not be negative");
  }
  return value;
}

int schema_version(const YAML::Node & node, const std::filesystem::path & path)
{
  if (!node.IsScalar()) {
    invalid(path, "schema_version must be an integer");
  }
  try {
    const long long value = node.as<long long>();
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
      invalid(path, "schema_version is outside the supported range");
    }
    return static_cast<int>(value);
  } catch (const YAML::Exception &) {
    invalid(path, "schema_version must be an integer");
  }
}

bool sha256(const std::string & value)
{
  if (value.size() != 64U) {
    return false;
  }
  for (const unsigned char character : value) {
    if (!(character >= '0' && character <= '9') &&
      !(character >= 'a' && character <= 'f') &&
      !(character >= 'A' && character <= 'F'))
    {
      return false;
    }
  }
  return true;
}

std::map<std::string, std::string> source_digests(
  const YAML::Node & node, const std::filesystem::path & path)
{
  if (!node.IsMap() || node.size() == 0U) {
    invalid(path, "source_sha256 must be a nonempty mapping");
  }
  std::map<std::string, std::string> output;
  for (const auto & entry : node) {
    const std::string name = string_value(entry.first, path, "source_sha256 key");
    const std::string digest = string_value(entry.second, path, "source_sha256 '" + name + "'");
    if (!sha256(digest)) {
      invalid(path, "source_sha256 '" + name + "' must be a SHA-256 hex digest");
    }
    if (!output.emplace(name, digest).second) {
      invalid(path, "duplicate source_sha256 key '" + name + "'");
    }
  }
  return output;
}

std::vector<std::string> string_sequence(
  const YAML::Node & node, const std::filesystem::path & path, const std::string & context)
{
  if (!node.IsSequence()) {
    invalid(path, context + " must be an array");
  }
  std::vector<std::string> output;
  output.reserve(node.size());
  for (const auto & value : node) {
    output.push_back(string_value(value, path, context + " entry"));
  }
  return output;
}

ReferencePoint reference_point(
  const YAML::Node & node, const std::filesystem::path & path, std::size_t lane_index,
  std::size_t point_index)
{
  const std::string context = "lane " + std::to_string(lane_index) + " point " +
    std::to_string(point_index);
  if (!node.IsMap()) {
    invalid(path, context + " must be a mapping");
  }
  const double route_s_m = finite_number(
    required_member(node, "route_s_m", path), path, context + " route_s_m");
  if (route_s_m < 0.0) {
    invalid(path, context + " route_s_m must not be negative");
  }
  static_cast<void>(finite_number(
    required_member(node, "z_m", path), path, context + " z_m"));
  return ReferencePoint{
    Pose2{
      finite_number(required_member(node, "x_m", path), path, context + " x_m"),
      finite_number(required_member(node, "y_m", path), path, context + " y_m"),
      finite_number(required_member(node, "yaw_rad", path), path, context + " yaw_rad")},
    route_s_m,
    finite_number(
      required_member(node, "curvature_inv_m", path), path, context + " curvature_inv_m"),
    positive_number(
      required_member(node, "left_width_m", path), path, context + " left_width_m"),
    positive_number(
      required_member(node, "right_width_m", path), path, context + " right_width_m"),
    nonnegative_number(
      required_member(node, "speed_limit_mps", path), path, context + " speed_limit_mps")};
}

struct UnresolvedLane
{
  ReferenceLane lane;
  std::map<std::string, std::vector<std::string>> adjacent_ids;
};

UnresolvedLane lane(
  const YAML::Node & node, const std::filesystem::path & path, std::size_t lane_index)
{
  if (!node.IsMap()) {
    invalid(path, "lane " + std::to_string(lane_index) + " must be a mapping");
  }
  UnresolvedLane output;
  output.lane.lane_sequence_id = string_value(
    required_member(node, "lane_sequence_id", path), path, "lane_sequence_id");
  output.lane.source_link_ids = string_sequence(
    required_member(node, "source_link_ids", path), path, "source_link_ids");
  if (output.lane.source_link_ids.empty()) {
    invalid(path, "source_link_ids must not be empty");
  }

  const YAML::Node points = required_member(node, "points", path);
  if (!points.IsSequence() || points.size() < 2U) {
    invalid(path, "lane points must contain at least two entries");
  }
  output.lane.points.reserve(points.size());
  for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
    auto point = reference_point(points[point_index], path, lane_index, point_index);
    if (!output.lane.points.empty() && point.route_s_m <= output.lane.points.back().route_s_m) {
      invalid(path, "route_s_m must be strictly increasing in each lane");
    }
    output.lane.points.push_back(std::move(point));
  }

  const YAML::Node adjacency = required_member(node, "adjacent_lane_sequence_ids", path);
  if (!adjacency.IsMap()) {
    invalid(path, "adjacent_lane_sequence_ids must be a mapping");
  }
  for (const auto & entry : adjacency) {
    const std::string side = string_value(entry.first, path, "adjacent_lane_sequence_ids key");
    if (side != "left" && side != "right") {
      invalid(path, "adjacent_lane_sequence_ids has invalid side '" + side + "'");
    }
    if (output.adjacent_ids.count(side) != 0U) {
      invalid(path, "duplicate adjacent_lane_sequence_ids side '" + side + "'");
    }
    const auto ids = string_sequence(entry.second, path, "adjacent_lane_sequence_ids " + side);
    std::set<std::string> unique_ids;
    for (const auto & id : ids) {
      if (!unique_ids.insert(id).second) {
        invalid(path, "duplicate adjacent lane ID '" + id + "'");
      }
    }
    output.adjacent_ids.emplace(side, ids);
  }
  return output;
}

void verify_expected_digests(
  const std::filesystem::path & path, const std::map<std::string, std::string> & actual,
  const std::map<std::string, std::string> & expected)
{
  for (const auto & [name, digest] : expected) {
    if (!sha256(digest)) {
      invalid(path, "expected SHA-256 for '" + name + "' is malformed");
    }
    const auto actual_digest = actual.find(name);
    if (actual_digest == actual.end() || actual_digest->second != digest) {
      invalid(path, "source SHA-256 mismatch for '" + name + "'");
    }
  }
}

}  // namespace

LoadedRouteCorridor load_route_corridor(
  const std::filesystem::path & path, const std::map<std::string, std::string> & expected_sha256)
{
  YAML::Node document;
  try {
    document = YAML::LoadFile(path.string());
  } catch (const YAML::Exception & error) {
    invalid(path, "cannot parse route corridor cache: " + std::string(error.what()));
  }
  if (!document.IsMap()) {
    invalid(path, "route corridor cache root must be a mapping");
  }

  LoadedRouteCorridor output;
  output.metadata.schema_version = schema_version(
    required_member(document, "schema_version", path), path);
  if (output.metadata.schema_version != 1) {
    invalid(path, "unsupported schema_version " + std::to_string(output.metadata.schema_version));
  }
  output.metadata.source_sha256 = source_digests(
    required_member(document, "source_sha256", path), path);
  verify_expected_digests(path, output.metadata.source_sha256, expected_sha256);
  output.corridor.frame_id = string_value(
    required_member(document, "frame_id", path), path, "frame_id");
  const std::string primary_lane_id = string_value(
    required_member(document, "primary_lane_sequence_id", path), path, "primary_lane_sequence_id");

  const YAML::Node lanes = required_member(document, "lanes", path);
  if (!lanes.IsSequence() || lanes.size() == 0U) {
    invalid(path, "lanes must be a nonempty array");
  }
  std::vector<UnresolvedLane> unresolved;
  unresolved.reserve(lanes.size());
  std::map<std::string, std::size_t> indices_by_id;
  for (std::size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
    auto parsed_lane = lane(lanes[lane_index], path, lane_index);
    if (!indices_by_id.emplace(parsed_lane.lane.lane_sequence_id, lane_index).second) {
      invalid(path, "duplicate lane_sequence_id '" + parsed_lane.lane.lane_sequence_id + "'");
    }
    unresolved.push_back(std::move(parsed_lane));
  }

  const auto primary = indices_by_id.find(primary_lane_id);
  if (primary == indices_by_id.end()) {
    invalid(path, "primary_lane_sequence_id references an unknown lane");
  }
  output.corridor.primary_lane_index = primary->second;
  output.corridor.lanes.reserve(unresolved.size());
  for (std::size_t lane_index = 0; lane_index < unresolved.size(); ++lane_index) {
    auto & parsed_lane = unresolved[lane_index];
    for (const auto & [side, ids] : parsed_lane.adjacent_ids) {
      auto & indices = side == "left" ?
        parsed_lane.lane.left_lane_indices : parsed_lane.lane.right_lane_indices;
      indices.reserve(ids.size());
      for (const auto & id : ids) {
        const auto target = indices_by_id.find(id);
        if (target == indices_by_id.end()) {
          invalid(path, "unknown adjacent lane ID '" + id + "'");
        }
        if (target->second == lane_index) {
          invalid(path, "adjacent lane ID must not self-reference '" + id + "'");
        }
        indices.push_back(target->second);
      }
    }
    output.corridor.lanes.push_back(std::move(parsed_lane.lane));
  }
  return output;
}

}  // namespace ad_planner
