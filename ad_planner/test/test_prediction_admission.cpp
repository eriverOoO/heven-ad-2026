#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/common/prediction_admission.hpp"

namespace {

using ad_planner::admit_predictions;
using ad_planner::parse_prediction_mode;
using ad_planner::PredictedObject;
using ad_planner::PredictionAdmissionInput;
using ad_planner::PredictionMode;
using ad_planner::PredictionSnapshot;
using ad_planner::PredictionSnapshotHistory;
using ad_planner::PredictionSnapshotInsertResult;

PredictionAdmissionInput valid_input() {
  return PredictionAdmissionInput{true, true, 0.1, 0.1, 0.5, 0.1};
}

TEST(PredictionAdmission, DisabledModeIsExplicitStaticOnlyOperation) {
  const auto result =
      admit_predictions(PredictionMode::kDisabled, PredictionAdmissionInput{});

  EXPECT_TRUE(result.admitted);
  EXPECT_FALSE(result.use_predictions);
  EXPECT_EQ(result.reason, "prediction explicitly disabled");
}

TEST(PredictionAdmission, RequiredModeAdmitsFreshValidInput) {
  const auto result =
      admit_predictions(PredictionMode::kRequired, valid_input());

  EXPECT_TRUE(result.admitted);
  EXPECT_TRUE(result.use_predictions);
}

TEST(PredictionAdmission, RequiredModeFailsClosedForMissingInvalidAndStale) {
  auto input = valid_input();
  input.received = false;
  EXPECT_FALSE(admit_predictions(PredictionMode::kRequired, input).admitted);

  input = valid_input();
  input.valid = false;
  EXPECT_FALSE(admit_predictions(PredictionMode::kRequired, input).admitted);

  input = valid_input();
  input.receipt_age_s = 0.51;
  EXPECT_FALSE(admit_predictions(PredictionMode::kRequired, input).admitted);

  input = valid_input();
  input.stamp_age_s = -0.11;
  EXPECT_FALSE(admit_predictions(PredictionMode::kRequired, input).admitted);

  input = valid_input();
  input.stamp_age_s = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(admit_predictions(PredictionMode::kRequired, input).admitted);
}

TEST(PredictionAdmission, RejectsUnknownMode) {
  EXPECT_EQ(parse_prediction_mode("disabled"), PredictionMode::kDisabled);
  EXPECT_EQ(parse_prediction_mode("required"), PredictionMode::kRequired);
  EXPECT_THROW(parse_prediction_mode("optional"), std::invalid_argument);
}

TEST(PredictionSnapshotHistory, SelectsNewestSnapshotNotAfterOdometry) {
  PredictionSnapshotHistory history(4U, 1'000'000'000LL);
  history.insert(PredictionSnapshot{
      10'000'000'000LL, 20.0, {PredictedObject{"past", {}}}});
  history.insert(PredictionSnapshot{
      10'200'000'000LL, 20.1, {PredictedObject{"future", {}}}});

  const auto selected =
      history.select(PredictionMode::kRequired, 10'100'000'000LL, 20.2, 0.5);

  ASSERT_TRUE(selected.admitted) << selected.reason;
  ASSERT_TRUE(selected.use_predictions);
  ASSERT_EQ(selected.snapshot.objects.size(), 1U);
  EXPECT_EQ(selected.snapshot.objects.front().object_id, "past");
  EXPECT_EQ(selected.snapshot.stamp_ns, 10'000'000'000LL);
  EXPECT_DOUBLE_EQ(selected.age_s, 0.1);
}

TEST(PredictionSnapshotHistory,
     DeduplicatesAndRetainsOnlyNewestBoundedHistory) {
  PredictionSnapshotHistory history(2U, 1'000'000'000LL);
  history.insert(
      PredictionSnapshot{1'000'000'000LL, 10.0, {PredictedObject{"one", {}}}});
  history.insert(
      PredictionSnapshot{2'000'000'000LL, 10.1, {PredictedObject{"two", {}}}});
  EXPECT_EQ(history.insert(PredictionSnapshot{
                2'000'000'000LL, 10.2, {PredictedObject{"two-replaced", {}}}}),
            PredictionSnapshotInsertResult::kReplaced);
  history.insert(PredictionSnapshot{
      3'000'000'000LL, 10.3, {PredictedObject{"three", {}}}});

  EXPECT_EQ(history.size(), 2U);
  const auto evicted =
      history.select(PredictionMode::kRequired, 1'000'000'000LL, 10.4, 20.0);
  EXPECT_FALSE(evicted.admitted);
  const auto replacement =
      history.select(PredictionMode::kRequired, 2'000'000'000LL, 10.4, 20.0);
  ASSERT_TRUE(replacement.admitted) << replacement.reason;
  ASSERT_EQ(replacement.snapshot.objects.size(), 1U);
  EXPECT_EQ(replacement.snapshot.objects.front().object_id, "two-replaced");
}

TEST(PredictionSnapshotHistory,
     ClockRollbackClearsAndDropsFirstNewEpochSample) {
  PredictionSnapshotHistory history(4U, 1'000'000'000LL);
  history.insert(PredictionSnapshot{
      10'000'000'000LL, 20.0, {PredictedObject{"old-epoch", {}}}});

  EXPECT_EQ(history.insert(PredictionSnapshot{
                1'000'000'000LL, 20.1, {PredictedObject{"rollback-edge", {}}}}),
            PredictionSnapshotInsertResult::kClockRollback);
  EXPECT_EQ(history.size(), 0U);
  EXPECT_FALSE(
      history.select(PredictionMode::kRequired, 1'000'000'000LL, 20.2, 0.5)
          .admitted);

  history.insert(PredictionSnapshot{
      1'100'000'000LL, 20.3, {PredictedObject{"new-epoch", {}}}});
  const auto recovered =
      history.select(PredictionMode::kRequired, 1'100'000'000LL, 20.4, 0.5);
  ASSERT_TRUE(recovered.admitted) << recovered.reason;
  ASSERT_EQ(recovered.snapshot.objects.size(), 1U);
  EXPECT_EQ(recovered.snapshot.objects.front().object_id, "new-epoch");
}

TEST(PredictionSnapshotHistory, StaleSelectedSnapshotFailsClosed) {
  PredictionSnapshotHistory history(4U, 1'000'000'000LL);
  history.insert(PredictionSnapshot{
      10'000'000'000LL, 20.0, {PredictedObject{"stale", {}}}});

  const auto selected =
      history.select(PredictionMode::kRequired, 10'600'000'000LL, 20.6, 0.5);

  EXPECT_FALSE(selected.admitted);
  EXPECT_FALSE(selected.use_predictions);
}

} // namespace
