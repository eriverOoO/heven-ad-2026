#include <gtest/gtest.h>

#include <fast_lio/mode_policy.hpp>

TEST(ModePolicy, LocalizationRequiresMapAndCannotMutateOrSaveIt) {
  EXPECT_TRUE(fast_lio::ModePolicy::requires_map(fast_lio::Mode::kLocalization));
  EXPECT_FALSE(fast_lio::ModePolicy::may_mutate_map(fast_lio::Mode::kLocalization));
  EXPECT_FALSE(fast_lio::ModePolicy::may_save_map(fast_lio::Mode::kLocalization));
  EXPECT_FALSE(fast_lio::ModePolicy::accepts_sensor_data(
      fast_lio::Mode::kLocalization, true, false));
  EXPECT_TRUE(fast_lio::ModePolicy::accepts_sensor_data(
      fast_lio::Mode::kLocalization, true, true));
}

TEST(ModePolicy, MappingRequiresInitialPoseButNotAPreloadedMap) {
  EXPECT_FALSE(fast_lio::ModePolicy::requires_map(fast_lio::Mode::kMapping));
  EXPECT_TRUE(fast_lio::ModePolicy::may_mutate_map(fast_lio::Mode::kMapping));
  EXPECT_TRUE(fast_lio::ModePolicy::may_save_map(fast_lio::Mode::kMapping));
  EXPECT_FALSE(fast_lio::ModePolicy::accepts_sensor_data(
      fast_lio::Mode::kMapping, false, false));
  EXPECT_TRUE(fast_lio::ModePolicy::accepts_sensor_data(
      fast_lio::Mode::kMapping, true, false));
}

TEST(ModePolicy, RefreshesInitialPoseWhileLocalizationHandoffOwnsTheSeed) {
  EXPECT_TRUE(fast_lio::ModePolicy::accepts_initial_pose_update(
      fast_lio::Mode::kLocalization, false));
  EXPECT_TRUE(fast_lio::ModePolicy::accepts_initial_pose_update(
      fast_lio::Mode::kLocalization, true));
  EXPECT_FALSE(fast_lio::ModePolicy::accepts_initial_pose_update(
      fast_lio::Mode::kMapping, true));
}

TEST(ScanTimingPolicy, MoraiInstantaneousScanHasNoSyntheticDurationOrDeskew) {
  const auto mode = fast_lio::parse_scan_timing_mode("instantaneous");
  EXPECT_EQ(mode, fast_lio::ScanTimingMode::kInstantaneous);
  EXPECT_DOUBLE_EQ(
      fast_lio::ScanTimingPolicy::effective_duration(mode, 0.099, 0.1), 0.0);
  EXPECT_FALSE(fast_lio::ScanTimingPolicy::applies_point_undistortion(mode));
}

TEST(ScanTimingPolicy, RealRollingScanKeepsMeasuredDurationAndDeskew) {
  const auto mode = fast_lio::parse_scan_timing_mode("rolling");
  EXPECT_EQ(mode, fast_lio::ScanTimingMode::kRolling);
  EXPECT_DOUBLE_EQ(
      fast_lio::ScanTimingPolicy::effective_duration(mode, 0.099, 0.1),
      0.099);
  EXPECT_DOUBLE_EQ(
      fast_lio::ScanTimingPolicy::effective_duration(mode, 0.001, 0.1), 0.1);
  EXPECT_TRUE(fast_lio::ScanTimingPolicy::applies_point_undistortion(mode));
}

TEST(ScanTimingPolicy, RejectsUnknownMode) {
  EXPECT_THROW(
      fast_lio::parse_scan_timing_mode("synthetic"), std::invalid_argument);
}
