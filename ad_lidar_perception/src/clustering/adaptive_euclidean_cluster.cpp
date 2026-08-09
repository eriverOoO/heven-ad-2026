#include "ad_lidar_perception/clustering/adaptive_euclidean_cluster.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ad_lidar_perception::clustering
{
namespace
{

struct Cell
{
  std::int64_t x;
  std::int64_t y;
  std::int64_t z;

  bool operator==(const Cell & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct CellHash
{
  std::size_t operator()(const Cell & cell) const
  {
    std::size_t seed = std::hash<std::int64_t>{}(cell.x);
    seed ^= std::hash<std::int64_t>{}(cell.y) + 0x9e3779b9U + (seed << 6U) +
    (seed >> 2U);
    seed ^= std::hash<std::int64_t>{}(cell.z) + 0x9e3779b9U + (seed << 6U) +
    (seed >> 2U);
    return seed;
  }
};

bool finite(const Point3 & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y) &&
         std::isfinite(point.z);
}

double range_xy(const Point3 & point) {return std::hypot(point.x, point.y);}

Cell cell_for(const Point3 & point, const double size, const bool use_height)
{
  return Cell{static_cast<std::int64_t>(std::floor(point.x / size)),
    static_cast<std::int64_t>(std::floor(point.y / size)),
    use_height ? static_cast<std::int64_t>(std::floor(point.z / size)) :
    0LL};
}

} // namespace

bool is_dynamic_component(
  const double extent_x_m, const double extent_y_m,
  const double maximum_diagonal_m)
{
  if (!std::isfinite(maximum_diagonal_m) || maximum_diagonal_m <= 0.0) {
    throw std::invalid_argument(
            "maximum dynamic object diagonal must be finite and positive");
  }
  return std::hypot(extent_x_m, extent_y_m) <= maximum_diagonal_m;
}

AdaptiveEuclideanCluster::AdaptiveEuclideanCluster(AdaptiveClusterConfig config)
: config_(std::move(config))
{
  if (!std::isfinite(config_.near_tolerance_m) ||
    !std::isfinite(config_.far_tolerance_m) ||
    !std::isfinite(config_.far_range_m) ||
    !std::isfinite(config_.minimum_points_far_range_m) ||
    config_.near_tolerance_m <= 0.0 ||
    config_.far_tolerance_m < config_.near_tolerance_m ||
    config_.far_range_m <= 0.0 || config_.minimum_points_far_range_m <= 0.0 ||
    config_.near_minimum_points == 0U ||
    config_.far_minimum_points == 0U ||
    config_.far_minimum_points > config_.near_minimum_points ||
    config_.maximum_points < config_.near_minimum_points)
  {
    throw std::invalid_argument("invalid adaptive-cluster configuration");
  }
}

double AdaptiveEuclideanCluster::tolerance_at(const double range_m) const
{
  if (!std::isfinite(range_m) || range_m < 0.0) {
    throw std::invalid_argument("cluster range must be finite and nonnegative");
  }
  const double ratio = std::clamp(range_m / config_.far_range_m, 0.0, 1.0);
  return config_.near_tolerance_m +
         ratio * (config_.far_tolerance_m - config_.near_tolerance_m);
}

std::size_t
AdaptiveEuclideanCluster::minimum_points_at(const double range_m) const
{
  if (!std::isfinite(range_m) || range_m < 0.0) {
    throw std::invalid_argument("cluster range must be finite and nonnegative");
  }
  const double ratio =
    std::clamp(range_m / config_.minimum_points_far_range_m, 0.0, 1.0);
  const double count =
    static_cast<double>(config_.near_minimum_points) +
    ratio * (static_cast<double>(config_.far_minimum_points) -
    static_cast<double>(config_.near_minimum_points));
  return std::max<std::size_t>(
    config_.far_minimum_points,
    static_cast<std::size_t>(std::ceil(count)));
}

std::vector<std::vector<std::size_t>>
AdaptiveEuclideanCluster::cluster(const std::vector<Point3> & points) const
{
  using BucketMap =
    std::unordered_map<Cell, std::vector<std::size_t>, CellHash>;
  BucketMap buckets;
  buckets.reserve(points.size());
  for (std::size_t index = 0U; index < points.size(); ++index) {
    if (finite(points[index])) {
      buckets[cell_for(
          points[index], config_.far_tolerance_m,
          config_.use_height)]
      .push_back(index);
    }
  }

  std::vector<bool> visited(points.size(), false);
  std::vector<std::vector<std::size_t>> result;
  for (std::size_t seed = 0U; seed < points.size(); ++seed) {
    if (visited[seed] || !finite(points[seed])) {
      continue;
    }
    visited[seed] = true;
    std::deque<std::size_t> frontier{seed};
    std::vector<std::size_t> component;
    double range_sum = 0.0;

    while (!frontier.empty()) {
      const std::size_t current = frontier.front();
      frontier.pop_front();
      component.push_back(current);
      const Point3 & point = points[current];
      const double point_range = range_xy(point);
      range_sum += point_range;
      const Cell center =
        cell_for(point, config_.far_tolerance_m, config_.use_height);
      const std::int64_t z_begin = config_.use_height ? -1LL : 0LL;
      const std::int64_t z_end = config_.use_height ? 1LL : 0LL;
      for (std::int64_t dz = z_begin; dz <= z_end; ++dz) {
        for (std::int64_t dy = -1LL; dy <= 1LL; ++dy) {
          for (std::int64_t dx = -1LL; dx <= 1LL; ++dx) {
            const auto bucket =
              buckets.find(Cell{center.x + dx, center.y + dy, center.z + dz});
            if (bucket == buckets.end()) {
              continue;
            }
            for (const std::size_t candidate : bucket->second) {
              if (visited[candidate]) {
                continue;
              }
              const Point3 & other = points[candidate];
              const double x = point.x - other.x;
              const double y = point.y - other.y;
              const double z = config_.use_height ? point.z - other.z : 0.0;
              const double pair_tolerance =
                0.5 *
                (tolerance_at(point_range) + tolerance_at(range_xy(other)));
              if (x * x + y * y + z * z <= pair_tolerance * pair_tolerance) {
                visited[candidate] = true;
                frontier.push_back(candidate);
              }
            }
          }
        }
      }
    }

    std::sort(component.begin(), component.end());
    const double mean_range = range_sum / static_cast<double>(component.size());
    if (component.size() >= minimum_points_at(mean_range) &&
      component.size() <= config_.maximum_points)
    {
      result.push_back(std::move(component));
    }
  }
  std::sort(
    result.begin(), result.end(), [](const auto & lhs, const auto & rhs) {
      return lhs.front() < rhs.front();
    });
  return result;
}

const AdaptiveClusterConfig & AdaptiveEuclideanCluster::config() const
{
  return config_;
}

} // namespace ad_lidar_perception::clustering
