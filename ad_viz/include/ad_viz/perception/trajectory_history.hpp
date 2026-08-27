#pragma once

#include <geometry_msgs/msg/point.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ad_viz::perception
{

// Bounded, per-track recent-position history for RViz trajectory markers.
// Not a predictor: only ever stores positions already observed in past
// TrackedObjects messages. Memory is bounded two ways: each track keeps at
// most `max_points_per_track` points (oldest dropped first), and any track
// not updated within `stale_timeout_sec` is dropped entirely by
// `prune_stale` -- so a track that stops being published (deleted by its
// tracker) does not accumulate forever.
class TrajectoryHistory
{
public:
  TrajectoryHistory(std::size_t max_points_per_track, double stale_timeout_sec);

  void update(
    const std::string & track_key, const geometry_msgs::msg::Point & position,
    std::int64_t stamp_ns);
  void prune_stale(std::int64_t stamp_ns);

  std::vector<std::pair<std::string, std::vector<geometry_msgs::msg::Point>>> entries() const;
  std::size_t track_count() const;

private:
  struct Entry
  {
    std::deque<geometry_msgs::msg::Point> points;
    std::int64_t last_seen_ns{0};
  };

  std::size_t max_points_per_track_;
  std::int64_t stale_timeout_ns_;
  std::unordered_map<std::string, Entry> tracks_;
};

}  // namespace ad_viz::perception
