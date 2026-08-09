#include "ad_lidar_perception/occupancy_grid/exact_stamp_pairer.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace
{

using Pairer =
  ad_lidar_perception::occupancy_grid::ExactStampPairer<std::string, int>;

TEST(ExactStampPairer, MatchesIdenticalStampsAcrossOutOfOrderArrivals)
{
  Pairer pairer(3U);

  EXPECT_FALSE(pairer.add_right(20, 200).has_value());
  EXPECT_FALSE(pairer.add_left(10, "ten").has_value());
  const auto ten = pairer.add_right(10, 100);
  ASSERT_TRUE(ten.has_value());
  EXPECT_EQ(ten->stamp_ns, 10);
  EXPECT_EQ(ten->left, "ten");
  EXPECT_EQ(ten->right, 100);

  const auto twenty = pairer.add_left(20, "twenty");
  ASSERT_TRUE(twenty.has_value());
  EXPECT_EQ(twenty->stamp_ns, 20);
  EXPECT_EQ(twenty->left, "twenty");
  EXPECT_EQ(twenty->right, 200);
}

TEST(ExactStampPairer, BoundsEachPendingSideByArrivalOrder)
{
  Pairer pairer(2U);

  EXPECT_FALSE(pairer.add_left(1, "one").has_value());
  EXPECT_FALSE(pairer.add_left(2, "two").has_value());
  EXPECT_FALSE(pairer.add_left(3, "three").has_value());
  EXPECT_EQ(pairer.left_pending(), 2U);
  EXPECT_FALSE(pairer.add_right(1, 10).has_value());

  const auto two = pairer.add_right(2, 20);
  ASSERT_TRUE(two.has_value());
  EXPECT_EQ(two->left, "two");
}

TEST(ExactStampPairer, ClearsPreviousEpochWhenACompletedPairJumpsBackward)
{
  Pairer pairer(3U);

  EXPECT_FALSE(pairer.add_left(100, "old").has_value());
  ASSERT_TRUE(pairer.add_right(100, 1000).has_value());
  EXPECT_FALSE(pairer.add_left(110, "stale future").has_value());
  EXPECT_FALSE(pairer.add_right(5, 50).has_value());

  const auto reset_pair = pairer.add_left(5, "new epoch");
  ASSERT_TRUE(reset_pair.has_value());
  EXPECT_EQ(reset_pair->stamp_ns, 5);
  EXPECT_EQ(reset_pair->left, "new epoch");
  EXPECT_EQ(pairer.left_pending(), 0U);
  EXPECT_EQ(pairer.right_pending(), 0U);
}

TEST(ExactStampPairer, RejectsInvalidCapacityAndNonpositiveStamps)
{
  EXPECT_THROW((void)Pairer(0U), std::invalid_argument);

  Pairer pairer(1U);
  EXPECT_THROW((void)pairer.add_left(0, "zero"), std::invalid_argument);
  EXPECT_THROW((void)pairer.add_right(-1, -1), std::invalid_argument);
}

}  // namespace
