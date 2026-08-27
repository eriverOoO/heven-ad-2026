#pragma once

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/tracked_objects.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/header.hpp>
#include <unique_identifier_msgs/msg/uuid.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <string>
#include <utility>
#include <vector>

namespace ad_viz::perception
{

struct ObjectMarkerConfig
{
  double marker_lifetime_sec{0.5};
  double box_line_width_m{0.06};
  double label_height_m{0.45};
  double label_z_offset_m{0.25};
  double velocity_scale_sec{1.0};
  double minimum_velocity_mps{0.01};
  // Prepended to the track-ID label text (e.g. "A-" / "B-") so two
  // trackers displayed at once never rely on marker color alone to be
  // told apart. Empty by default (single-tracker use is unaffected).
  std::string id_prefix{};
};

// Hex-encodes a UUID exactly as used for both the "tracked/<hex>" marker
// namespace and the on-screen ID suffix -- exported so callers (e.g. the
// trajectory-history accumulator) key their own per-track state
// identically, without re-deriving the encoding.
std::string uuid_hex(const unique_identifier_msgs::msg::UUID & uuid);

visualization_msgs::msg::MarkerArray build_detection_markers(
  const autoware_perception_msgs::msg::DetectedObjects & input,
  const ObjectMarkerConfig & config = {});

visualization_msgs::msg::MarkerArray build_tracked_markers(
  const autoware_perception_msgs::msg::TrackedObjects & input,
  const ObjectMarkerConfig & config = {});

// Renders one LINE_STRIP per (track_key, points) entry -- points already
// bounded/pruned by the caller (see TrajectoryHistory). Entries with fewer
// than 2 points are skipped (nothing meaningful to draw). Does not emit a
// DELETEALL marker of its own; callers append this array's markers onto
// the same MarkerArray/topic as `build_tracked_markers` so a single
// DELETEALL each frame keeps both in sync (no orphaned stale markers).
visualization_msgs::msg::MarkerArray build_trajectory_markers(
  const std::vector<std::pair<std::string, std::vector<geometry_msgs::msg::Point>>> & histories,
  const std_msgs::msg::Header & header,
  const ObjectMarkerConfig & config = {});

}  // namespace ad_viz::perception
