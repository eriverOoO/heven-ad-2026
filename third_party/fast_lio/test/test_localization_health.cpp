#include <gtest/gtest.h>

#include <fast_lio/localization_health.hpp>

TEST(LocalizationHealth, RequiresConsecutiveHealthyMatchesBeforePublishing) {
  fast_lio::LocalizationHealth health({20, 0.5, 3, 2});

  EXPECT_EQ(health.state(), fast_lio::LocalizationHealthState::kInitializing);
  EXPECT_FALSE(health.observe(50, 0.1));
  EXPECT_FALSE(health.observe(50, 0.1));
  EXPECT_TRUE(health.observe(50, 0.1));
  EXPECT_EQ(health.state(), fast_lio::LocalizationHealthState::kHealthy);
}

TEST(LocalizationHealth, SuppressesOutputAfterConsecutiveInvalidMatches) {
  fast_lio::LocalizationHealth health({20, 0.5, 2, 3});
  EXPECT_FALSE(health.observe(50, 0.1));
  EXPECT_TRUE(health.observe(50, 0.1));

  EXPECT_TRUE(health.observe(19, 0.1));
  EXPECT_TRUE(health.observe(50, 0.6));
  EXPECT_FALSE(health.observe(0, 0.0));
  EXPECT_EQ(health.state(), fast_lio::LocalizationHealthState::kLost);
}

TEST(LocalizationHealth, RecoversOnlyAfterACompleteHealthyWindow) {
  fast_lio::LocalizationHealth health({20, 0.5, 2, 1});
  EXPECT_FALSE(health.observe(0, 0.0));
  EXPECT_EQ(health.state(), fast_lio::LocalizationHealthState::kLost);
  EXPECT_FALSE(health.observe(40, 0.2));
  EXPECT_TRUE(health.observe(40, 0.2));
  EXPECT_EQ(health.state(), fast_lio::LocalizationHealthState::kHealthy);
}

TEST(LocalizationHealth, SensorWatchdogImmediatelyMarksAHealthyFilterLost) {
  fast_lio::LocalizationHealth health({20, 0.5, 2, 3});
  EXPECT_FALSE(health.observe(50, 0.1));
  EXPECT_TRUE(health.observe(50, 0.1));

  health.mark_stale();

  EXPECT_EQ(health.state(), fast_lio::LocalizationHealthState::kLost);
  EXPECT_FALSE(health.observe(50, 0.1));
  EXPECT_TRUE(health.observe(50, 0.1));
}

TEST(LocalizationHealth, RejectsInvalidConfiguration) {
  EXPECT_THROW(
      fast_lio::LocalizationHealth({0, 0.5, 2, 1}),
      std::invalid_argument);
  EXPECT_THROW(
      fast_lio::LocalizationHealth({20, 0.0, 2, 1}),
      std::invalid_argument);
  EXPECT_THROW(
      fast_lio::LocalizationHealth({20, 0.5, 0, 1}),
      std::invalid_argument);
  EXPECT_THROW(
      fast_lio::LocalizationHealth({20, 0.5, 2, 0}),
      std::invalid_argument);
}
