#include "ad_viz/perception/trajectory_history.hpp"

#include <cmath>
#include <stdexcept>

namespace ad_viz::perception
{
namespace
{
constexpr std::int64_t kNanosecondsPerSecond = 1000000000LL;
}  // namespace

TrajectoryHistory::TrajectoryHistory(
  const std::size_t max_points_per_track, const double stale_timeout_sec)
: max_points_per_track_(max_points_per_track)
{
  if (max_points_per_track_ < 2U) {
    throw std::invalid_argument("trajectory history needs at least 2 points per track");
  }
  if (!std::isfinite(stale_timeout_sec) || stale_timeout_sec <= 0.0) {
    throw std::invalid_argument("trajectory stale timeout must be finite and positive");
  }
  stale_timeout_ns_ = static_cast<std::int64_t>(
    stale_timeout_sec * static_cast<double>(kNanosecondsPerSecond));
}

void TrajectoryHistory::update(
  const std::string & track_key, const geometry_msgs::msg::Point & position,
  const std::int64_t stamp_ns)
{
  auto & entry = tracks_[track_key];
  entry.points.push_back(position);
  while (entry.points.size() > max_points_per_track_) {
    entry.points.pop_front();
  }
  entry.last_seen_ns = stamp_ns;
}

void TrajectoryHistory::prune_stale(const std::int64_t stamp_ns)
{
  for (auto it = tracks_.begin(); it != tracks_.end(); ) {
    if (stamp_ns - it->second.last_seen_ns > stale_timeout_ns_) {
      it = tracks_.erase(it);
    } else {
      ++it;
    }
  }
}

std::vector<std::pair<std::string, std::vector<geometry_msgs::msg::Point>>>
TrajectoryHistory::entries() const
{
  std::vector<std::pair<std::string, std::vector<geometry_msgs::msg::Point>>> result;
  result.reserve(tracks_.size());
  for (const auto & [key, entry] : tracks_) {
    result.emplace_back(
      key, std::vector<geometry_msgs::msg::Point>(entry.points.begin(), entry.points.end()));
  }
  return result;
}

std::size_t TrajectoryHistory::track_count() const
{
  return tracks_.size();
}

}  // namespace ad_viz::perception
