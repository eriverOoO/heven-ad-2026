#include "ad_viz/perception/trajectory_history.hpp"

#include <gtest/gtest.h>

#include <algorithm>

namespace ad_viz::perception
{
namespace
{

geometry_msgs::msg::Point make_point(const double x)
{
  geometry_msgs::msg::Point point;
  point.x = x;
  return point;
}

TEST(TrajectoryHistoryTest, RejectsInvalidConstruction)
{
  EXPECT_THROW(TrajectoryHistory(1U, 1.0), std::invalid_argument);
  EXPECT_THROW(TrajectoryHistory(2U, 0.0), std::invalid_argument);
  EXPECT_THROW(TrajectoryHistory(2U, -1.0), std::invalid_argument);
}

TEST(TrajectoryHistoryTest, AccumulatesPointsPerTrack)
{
  TrajectoryHistory history(10U, 5.0);
  history.update("a", make_point(1.0), 0);
  history.update("a", make_point(2.0), 1000000000LL);
  history.update("b", make_point(5.0), 1000000000LL);

  EXPECT_EQ(history.track_count(), 2U);
  const auto entries = history.entries();
  const auto track_a = std::find_if(
    entries.begin(), entries.end(), [](const auto & entry) {return entry.first == "a";});
  ASSERT_NE(track_a, entries.end());
  EXPECT_EQ(track_a->second.size(), 2U);
}

TEST(TrajectoryHistoryTest, BoundsPointsPerTrack)
{
  TrajectoryHistory history(3U, 5.0);
  for (int i = 0; i < 10; ++i) {
    history.update("a", make_point(static_cast<double>(i)), static_cast<std::int64_t>(i));
  }
  const auto entries = history.entries();
  ASSERT_EQ(entries.size(), 1U);
  EXPECT_EQ(entries.front().second.size(), 3U);
  // Oldest points dropped first: the last 3 updates (x=7,8,9) survive.
  EXPECT_DOUBLE_EQ(entries.front().second.front().x, 7.0);
  EXPECT_DOUBLE_EQ(entries.front().second.back().x, 9.0);
}

TEST(TrajectoryHistoryTest, PrunesStaleTracks)
{
  TrajectoryHistory history(10U, 2.0);
  history.update("a", make_point(1.0), 0);
  history.update("b", make_point(2.0), 1000000000LL);

  // t=1.5s: neither track exceeds the 2s timeout yet.
  history.prune_stale(1500000000LL);
  EXPECT_EQ(history.track_count(), 2U);

  // t=2.5s: "a" (age 2.5s) is stale, "b" (age 1.5s) is not.
  history.prune_stale(2500000000LL);
  EXPECT_EQ(history.track_count(), 1U);
  EXPECT_EQ(history.entries().front().first, "b");

  // t=6s: "b" (age 5s) is now stale too.
  history.prune_stale(6000000000LL);
  EXPECT_EQ(history.track_count(), 0U);
}

}  // namespace
}  // namespace ad_viz::perception

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
