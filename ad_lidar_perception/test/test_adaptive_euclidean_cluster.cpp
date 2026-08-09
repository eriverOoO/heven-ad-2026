#include <ad_lidar_perception/clustering/adaptive_euclidean_cluster.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

using ad_lidar_perception::clustering::AdaptiveClusterConfig;
using ad_lidar_perception::clustering::AdaptiveEuclideanCluster;
using ad_lidar_perception::clustering::Point3;
using ad_lidar_perception::clustering::is_dynamic_component;

AdaptiveClusterConfig test_config()
{
  AdaptiveClusterConfig config;
  config.near_tolerance_m = 0.30;
  config.far_tolerance_m = 1.20;
  config.far_range_m = 60.0;
  config.minimum_points_far_range_m = 60.0;
  config.near_minimum_points = 2U;
  config.far_minimum_points = 2U;
  config.maximum_points = 100U;
  config.use_height = false;
  return config;
}

TEST(AdaptiveEuclideanCluster, GrowsToleranceWithRangeAndCapsIt) {
  const AdaptiveEuclideanCluster clusterer(test_config());

  EXPECT_DOUBLE_EQ(clusterer.tolerance_at(0.0), 0.30);
  EXPECT_NEAR(clusterer.tolerance_at(30.0), 0.75, 1.0e-12);
  EXPECT_DOUBLE_EQ(clusterer.tolerance_at(60.0), 1.20);
  EXPECT_DOUBLE_EQ(clusterer.tolerance_at(100.0), 1.20);
}

TEST(
  AdaptiveEuclideanCluster,
  ConnectsSparseFarReturnsWithoutMergingNearObjects) {
  const AdaptiveEuclideanCluster clusterer(test_config());
  const std::vector<Point3> points{{2.0, 0.0, 0.0}, {2.0, 0.25, 0.0},
    {2.0, 0.75, 0.0}, {2.0, 1.00, 0.0},
    {50.0, 0.0, 0.0}, {50.0, 0.70, 0.0}};

  const auto clusters = clusterer.cluster(points);

  ASSERT_EQ(clusters.size(), 3U);
  EXPECT_EQ(clusters[0], (std::vector<std::size_t>{0U, 1U}));
  EXPECT_EQ(clusters[1], (std::vector<std::size_t>{2U, 3U}));
  EXPECT_EQ(clusters[2], (std::vector<std::size_t>{4U, 5U}));
}

TEST(AdaptiveEuclideanCluster, KeepsSeparatedFarObjectsSeparate) {
  const AdaptiveEuclideanCluster clusterer(test_config());
  const std::vector<Point3> points{
    {50.0, 0.0, 0.0}, {50.0, 0.7, 0.0}, {50.0, 2.1, 0.0}, {50.0, 2.8, 0.0}};

  const auto clusters = clusterer.cluster(points);

  ASSERT_EQ(clusters.size(), 2U);
  EXPECT_EQ(clusters[0], (std::vector<std::size_t>{0U, 1U}));
  EXPECT_EQ(clusters[1], (std::vector<std::size_t>{2U, 3U}));
}

TEST(AdaptiveEuclideanCluster, RelaxesMinimumPointCountOnlyAtLongRange) {
  auto config = test_config();
  config.near_minimum_points = 4U;
  config.far_minimum_points = 2U;
  const AdaptiveEuclideanCluster clusterer(config);
  const std::vector<Point3> points{
    {3.0, 0.0, 0.0}, {3.0, 0.2, 0.0}, {60.0, 0.0, 0.0}, {60.0, 0.8, 0.0}};

  const auto clusters = clusterer.cluster(points);

  ASSERT_EQ(clusters.size(), 1U);
  EXPECT_EQ(clusters.front(), (std::vector<std::size_t>{2U, 3U}));
}

TEST(AdaptiveEuclideanCluster, UsesIndependentToleranceAndMinimumPointRanges) {
  auto config = test_config();
  config.far_range_m = 60.0;
  config.minimum_points_far_range_m = 30.0;
  config.near_minimum_points = 6U;
  config.far_minimum_points = 2U;
  const AdaptiveEuclideanCluster clusterer(config);

  EXPECT_NEAR(clusterer.tolerance_at(30.0), 0.75, 1.0e-12);
  EXPECT_EQ(clusterer.minimum_points_at(15.0), 4U);
  EXPECT_EQ(clusterer.minimum_points_at(30.0), 2U);
  EXPECT_EQ(clusterer.minimum_points_at(45.0), 2U);
  EXPECT_NEAR(clusterer.tolerance_at(45.0), 0.975, 1.0e-12);
}

TEST(AdaptiveEuclideanCluster, KeepsLongConnectedComponentUnderMaximumSize) {
  auto config = test_config();
  config.near_tolerance_m = 0.50;
  config.far_tolerance_m = 0.50;
  config.maximum_points = 100U;
  const AdaptiveEuclideanCluster clusterer(config);
  std::vector<Point3> points;
  for (std::size_t index = 0U; index <= 80U; ++index) {
    points.push_back({10.0 + 0.25 * static_cast<double>(index), 0.0, 0.0});
  }

  const auto clusters = clusterer.cluster(points);

  ASSERT_EQ(clusters.size(), 1U);
  EXPECT_EQ(clusters.front().size(), 81U);
  EXPECT_EQ(clusters.front().front(), 0U);
  EXPECT_EQ(clusters.front().back(), 80U);
}

TEST(AdaptiveEuclideanCluster, DynamicComponentGateIsInclusiveAndValidated) {
  EXPECT_TRUE(is_dynamic_component(9.6, 0.0, 12.0));
  EXPECT_TRUE(is_dynamic_component(12.0, 0.0, 12.0));
  EXPECT_FALSE(is_dynamic_component(20.0, 0.0, 12.0));
  EXPECT_THROW(
    (void)is_dynamic_component(1.0, 1.0, 0.0), std::invalid_argument);
  EXPECT_THROW(
    (void)is_dynamic_component(
      1.0, 1.0, std::numeric_limits<double>::infinity()),
    std::invalid_argument);
}

TEST(AdaptiveEuclideanCluster, RejectsInvalidConfiguration) {
  auto config = test_config();
  config.far_tolerance_m = config.near_tolerance_m - 0.01;
  EXPECT_THROW((void)AdaptiveEuclideanCluster(config), std::invalid_argument);

  config = test_config();
  config.far_minimum_points = config.near_minimum_points + 1U;
  EXPECT_THROW((void)AdaptiveEuclideanCluster(config), std::invalid_argument);

  config = test_config();
  config.minimum_points_far_range_m = 0.0;
  EXPECT_THROW((void)AdaptiveEuclideanCluster(config), std::invalid_argument);

  config = test_config();
  config.minimum_points_far_range_m =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)AdaptiveEuclideanCluster(config), std::invalid_argument);
}

} // namespace
