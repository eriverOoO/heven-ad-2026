#include <atomic>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <pcl/point_types.h>

#include "ikd-Tree/ikd_Tree.h"

namespace
{

using Point = pcl::PointXYZINormal;
using Tree = KD_TREE<Point>;

Point make_point(const float x, const float y, const float z)
{
  Point point;
  point.x = x;
  point.y = y;
  point.z = z;
  point.intensity = x;
  return point;
}

TEST(IkdTreeSnapshot, ReturnsAllValidPoints)
{
  auto tree = std::make_unique<Tree>();
  Tree::PointVector points;
  for (int i = 0; i < 2000; ++i) {
    points.push_back(make_point(
      static_cast<float>(i % 100),
      static_cast<float>((i / 100) % 20),
      static_cast<float>(i) * 0.001F));
  }
  tree->Build(points);

  const auto snapshot = tree->Snapshot();

  ASSERT_EQ(snapshot.size(), points.size());
  for (const auto & point : snapshot) {
    EXPECT_TRUE(std::isfinite(point.x));
    EXPECT_TRUE(std::isfinite(point.y));
    EXPECT_TRUE(std::isfinite(point.z));
  }
}

TEST(IkdTreeSnapshot, RemainsValidWhileBackgroundRebuildRuns)
{
  auto tree = std::make_unique<Tree>();
  Tree::PointVector points;
  for (int i = 0; i < 6000; ++i) {
    points.push_back(make_point(
      static_cast<float>(i % 200),
      static_cast<float>((i / 200) % 30),
      static_cast<float>(i) * 0.0001F));
  }
  tree->Build(points);

  std::vector<BoxPointType> boxes(1);
  boxes[0].vertex_min[0] = -1.0F;
  boxes[0].vertex_min[1] = -1.0F;
  boxes[0].vertex_min[2] = -1.0F;
  boxes[0].vertex_max[0] = 90.0F;
  boxes[0].vertex_max[1] = 40.0F;
  boxes[0].vertex_max[2] = 2.0F;
  tree->Delete_Point_Boxes(boxes);

  std::atomic<bool> failed{false};
  std::thread reader([&tree, &failed]() {
    for (int iteration = 0; iteration < 200; ++iteration) {
      const auto snapshot = tree->Snapshot();
      if (snapshot.empty() || snapshot.size() > 6000) {
        failed.store(true);
        return;
      }
      for (const auto & point : snapshot) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
          !std::isfinite(point.z))
        {
          failed.store(true);
          return;
        }
      }
    }
  });

  for (int i = 0; i < 200; ++i) {
    Tree::PointVector nearest;
    std::vector<float> distances;
    tree->Nearest_Search(make_point(120.0F, 10.0F, 0.0F), 5, nearest, distances);
  }
  reader.join();

  EXPECT_FALSE(failed.load());
}

}  // namespace
