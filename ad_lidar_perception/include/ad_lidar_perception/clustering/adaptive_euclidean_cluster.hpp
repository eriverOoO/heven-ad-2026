#ifndef AD_LIDAR_PERCEPTION__CLUSTERING__ADAPTIVE_EUCLIDEAN_CLUSTER_HPP_
#define AD_LIDAR_PERCEPTION__CLUSTERING__ADAPTIVE_EUCLIDEAN_CLUSTER_HPP_

#include <cstddef>
#include <vector>

namespace ad_lidar_perception::clustering
{

struct Point3
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct AdaptiveClusterConfig
{
  double near_tolerance_m{0.45};
  double far_tolerance_m{1.60};
  double far_range_m{45.0};
  double minimum_points_far_range_m{45.0};
  std::size_t near_minimum_points{5U};
  std::size_t far_minimum_points{2U};
  std::size_t maximum_points{20000U};
  bool use_height{false};
};

bool is_dynamic_component(
  double extent_x_m, double extent_y_m, double maximum_diagonal_m);

class AdaptiveEuclideanCluster
{
public:
  explicit AdaptiveEuclideanCluster(AdaptiveClusterConfig config);

  double tolerance_at(double range_m) const;
  std::size_t minimum_points_at(double range_m) const;
  std::vector<std::vector<std::size_t>>
  cluster(const std::vector<Point3> & points) const;
  const AdaptiveClusterConfig & config() const;

private:
  AdaptiveClusterConfig config_;
};

} // namespace ad_lidar_perception::clustering

#endif // AD_LIDAR_PERCEPTION__CLUSTERING__ADAPTIVE_EUCLIDEAN_CLUSTER_HPP_
