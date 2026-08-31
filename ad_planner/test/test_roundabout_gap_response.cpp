#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "ad_planner/planning/roundabout_gap_response.hpp"
#include "roundabout_gap_response_node.hpp"

namespace ad_planner
{
namespace
{

constexpr std::int64_t kSecond = 1'000'000'000LL;

RoundaboutGapResponseParameters parameters()
{
  return RoundaboutGapResponseParameters{};
}

RoundaboutGapResponseObjectInput object(
  const std::uint8_t id, const double gap, const bool overlap = false,
  const bool coverage = true, const bool gap_valid = true)
{
  RoundaboutGapResponseObjectInput value;
  value.object_id[0] = id;
  value.any_occupancy_overlap = overlap;
  value.minimum_temporal_gap_valid = gap_valid;
  value.minimum_temporal_gap_s = gap;
  value.prediction_covers_ego_exit = coverage;
  return value;
}

RoundaboutGapResponseInput input(
  std::vector<RoundaboutGapResponseObjectInput> objects = {},
  const double distance = 40.0, const double speed = 8.0)
{
  RoundaboutGapResponseInput value;
  value.ego_entry_valid = true;
  value.ego_entry_time_s = 4.0;
  value.ego_exit_valid = true;
  value.ego_exit_time_s = 6.0;
  value.ego_route_distance_to_entry_m = distance;
  value.ego_route_distance_to_exit_m = distance + 24.0;
  value.ego_speed_mps = speed;
  value.relevant_objects = std::move(objects);
  return value;
}

ad_interfaces::msg::RoundaboutGapRisk risk(
  const std::uint8_t id, const double gap, const bool overlap = false,
  const bool coverage = true)
{
  ad_interfaces::msg::RoundaboutGapRisk value;
  value.object_id.uuid[0] = id;
  value.classification_probability = 0.9F;
  value.existence_probability = 0.9F;
  value.relevant_to_conflict = true;
  value.object_map_distance_to_conflict_m = 1.0F;
  value.object_entry_valid = true;
  value.object_entry_time_s = 1.0F;
  value.object_exit_valid = true;
  value.object_exit_time_s = 2.0F;
  value.arrival_delta_valid = true;
  value.arrival_delta_s = -3.0F;
  value.temporal_gap_valid = true;
  value.temporal_gap_s = 2.0F;  // intentionally legacy first-interval data
  value.occupancy_overlap = false;
  value.predicted_conflict_interval_count = overlap ? 2U : 1U;
  value.later_reentry_detected = overlap;
  value.any_occupancy_overlap = overlap;
  value.minimum_temporal_gap_valid = true;
  value.minimum_temporal_gap_s = static_cast<float>(gap);
  value.prediction_horizon_s = 8.0F;
  value.prediction_covers_ego_exit = coverage;
  return value;
}

ad_interfaces::msg::RoundaboutGapRiskArray frame(
  const std::int64_t stamp_ns,
  const std::vector<ad_interfaces::msg::RoundaboutGapRisk> & objects = {})
{
  ad_interfaces::msg::RoundaboutGapRiskArray value;
  value.header.stamp.sec = static_cast<std::int32_t>(stamp_ns / kSecond);
  value.header.stamp.nanosec = static_cast<std::uint32_t>(stamp_ns % kSecond);
  value.header.frame_id = "map";
  value.conflict_zone_id = "kcity_roundabout";
  value.ego_entry_valid = true;
  value.ego_entry_time_s = 4.0F;
  value.ego_exit_valid = true;
  value.ego_exit_time_s = 6.0F;
  value.ego_route_distance_to_entry_m = 32.0F;
  value.ego_route_distance_to_exit_m = 56.0F;
  value.ego_speed_mps = 8.0F;
  value.objects = objects;
  value.relevant_object_count = static_cast<std::uint16_t>(objects.size());
  return value;
}

}  // namespace

TEST(RoundaboutGapResponsePolicy, ZeroRelevantObjectsRelease)
{
  const auto result = compute_roundabout_gap_response(input(), parameters());
  EXPECT_TRUE(result.active);
  EXPECT_EQ(result.action, RoundaboutGapResponseAction::kRelease);
  EXPECT_EQ(result.reason, RoundaboutGapResponseReason::kClearGap);
  EXPECT_TRUE(result.complete_prediction_coverage);
}

TEST(RoundaboutGapResponsePolicy, ClearGapAndBoundaryRelease)
{
  for (const double gap : {2.0, 2.001, 2.5}) {
    const auto result = compute_roundabout_gap_response(input({object(1, gap)}), parameters());
    EXPECT_EQ(result.action, RoundaboutGapResponseAction::kRelease) << gap;
  }
}

TEST(RoundaboutGapResponsePolicy, BelowThresholdAndTouchingDoNotRelease)
{
  for (const double gap : {1.999, 0.5, 0.0}) {
    const auto result = compute_roundabout_gap_response(input({object(1, gap)}), parameters());
    EXPECT_NE(result.action, RoundaboutGapResponseAction::kRelease) << gap;
    EXPECT_EQ(result.reason, RoundaboutGapResponseReason::kGapTooSmall);
  }
}

TEST(RoundaboutGapResponsePolicy, AllIntervalOverlapOverridesClearLegacyInterval)
{
  const auto result = compute_roundabout_gap_response(
    input({object(5, 0.0, true, true)}), parameters());
  EXPECT_NE(result.action, RoundaboutGapResponseAction::kRelease);
  EXPECT_EQ(result.reason, RoundaboutGapResponseReason::kOverlap);
  EXPECT_TRUE(result.conflict_overlap_present);
}

TEST(RoundaboutGapResponsePolicy, IncompletePredictionForbidsRelease)
{
  const auto result = compute_roundabout_gap_response(
    input({object(4, 2.5, false, false)}), parameters());
  EXPECT_NE(result.action, RoundaboutGapResponseAction::kRelease);
  EXPECT_EQ(result.reason, RoundaboutGapResponseReason::kInsufficientPrediction);
  EXPECT_FALSE(result.complete_prediction_coverage);
}

TEST(RoundaboutGapResponsePolicy, ClearBeforeAndClearAfterAreEquivalent)
{
  const auto before = compute_roundabout_gap_response(input({object(1, 2.5)}), parameters());
  const auto after = compute_roundabout_gap_response(input({object(2, 2.5)}), parameters());
  EXPECT_EQ(before.action, RoundaboutGapResponseAction::kRelease);
  EXPECT_EQ(after.action, before.action);
}

TEST(RoundaboutGapResponsePolicy, EveryRelevantObjectMustBeSafe)
{
  EXPECT_EQ(
    compute_roundabout_gap_response(
      input({object(1, 2.5), object(2, 3.0), object(3, 4.0)}), parameters()).action,
    RoundaboutGapResponseAction::kRelease);
  const auto overlap = compute_roundabout_gap_response(
    input({object(1, 3.0), object(2, 0.0, true), object(3, 4.0)}), parameters());
  ASSERT_TRUE(overlap.has_source);
  EXPECT_EQ(overlap.source_object_id[0], 2U);
  const auto coverage = compute_roundabout_gap_response(
    input({object(1, 3.0), object(2, 4.0), object(3, 4.0, false, false)}), parameters());
  EXPECT_EQ(coverage.reason, RoundaboutGapResponseReason::kInsufficientPrediction);
  EXPECT_EQ(coverage.source_object_id[0], 3U);
}

TEST(RoundaboutGapResponsePolicy, ArbitrationPrecedenceAndUuidTieBreakAreDeterministic)
{
  const auto result = compute_roundabout_gap_response(
    input(
    {
      object(9, 0.2), object(8, 0.0, false, false),
      object(7, 0.0, true), object(3, 0.0, true)}), parameters());
  EXPECT_EQ(result.reason, RoundaboutGapResponseReason::kOverlap);
  EXPECT_EQ(result.source_object_id[0], 3U);
  const auto gaps = compute_roundabout_gap_response(
    input({object(1, 1.0), object(2, 0.5), object(3, 0.75)}), parameters());
  EXPECT_EQ(gaps.source_object_id[0], 2U);
  EXPECT_DOUBLE_EQ(gaps.limiting_gap_s, 0.5);
}

TEST(RoundaboutGapResponsePolicy, YieldFarAndHoldNearUseStoppingDistance)
{
  const auto unsafe = std::vector<RoundaboutGapResponseObjectInput>{object(1, 0.5)};
  const auto far = compute_roundabout_gap_response(input(unsafe, 40.0, 8.0), parameters());
  const auto near = compute_roundabout_gap_response(input(unsafe, 20.0, 8.0), parameters());
  EXPECT_EQ(far.action, RoundaboutGapResponseAction::kYield);
  EXPECT_EQ(near.action, RoundaboutGapResponseAction::kHold);
  EXPECT_GT(far.available_distance_m, far.comfortable_stop_distance_m);
  EXPECT_LE(near.available_distance_m, near.comfortable_stop_distance_m);
}

TEST(RoundaboutGapResponsePolicy, StoppedBeforeEntryHoldsWithoutFabricatingEta)
{
  auto value = input({}, 20.0, 0.0);
  value.ego_entry_valid = false;
  value.ego_exit_valid = false;
  const auto result = compute_roundabout_gap_response(value, parameters());
  EXPECT_TRUE(result.active);
  EXPECT_EQ(result.action, RoundaboutGapResponseAction::kHold);
  EXPECT_EQ(result.reason, RoundaboutGapResponseReason::kInvalidEgoState);
}

TEST(RoundaboutGapResponsePolicy, StoppedEgoCannotReleaseEvenWithClaimedTiming)
{
  const auto result = compute_roundabout_gap_response(input({}, 20.0, 0.0), parameters());
  EXPECT_EQ(result.action, RoundaboutGapResponseAction::kHold);
  EXPECT_EQ(result.reason, RoundaboutGapResponseReason::kInvalidEgoState);
}

TEST(RoundaboutGapResponsePolicy, NegativeEgoSpeedIsRejected)
{
  EXPECT_THROW(
    compute_roundabout_gap_response(input({}, 20.0, -1.0), parameters()),
    std::invalid_argument);
}

TEST(RoundaboutGapResponsePolicy, InsideAndPastConflictAreInactiveNonrestrictive)
{
  auto inside = input({object(1, 0.0, true)}, -1.0);
  inside.ego_in_conflict_now = true;
  const auto inside_result = compute_roundabout_gap_response(inside, parameters());
  EXPECT_FALSE(inside_result.active);
  EXPECT_EQ(inside_result.action, RoundaboutGapResponseAction::kRelease);
  auto past = input({object(1, 0.0, true)}, -30.0);
  past.ego_route_distance_to_exit_m = -1.0;
  const auto past_result = compute_roundabout_gap_response(past, parameters());
  EXPECT_FALSE(past_result.active);
  EXPECT_NE(past_result.action, RoundaboutGapResponseAction::kHold);
}

TEST(RoundaboutGapResponsePolicy, GapAndApproachMonotonicity)
{
  int previous = static_cast<int>(RoundaboutGapResponseAction::kHold);
  for (const double gap : {0.0, 0.5, 1.999, 2.0, 3.0}) {
    const auto action = compute_roundabout_gap_response(
      input({object(1, gap)}, 40.0), parameters()).action;
    EXPECT_LE(static_cast<int>(action), previous);
    previous = static_cast<int>(action);
  }
  const auto unsafe = std::vector<RoundaboutGapResponseObjectInput>{object(1, 0.5)};
  EXPECT_EQ(
    compute_roundabout_gap_response(input(unsafe, 40.0), parameters()).action,
    RoundaboutGapResponseAction::kYield);
  EXPECT_EQ(
    compute_roundabout_gap_response(input(unsafe, 20.0), parameters()).action,
    RoundaboutGapResponseAction::kHold);
}

TEST(RoundaboutGapResponseFrame, AcceptsFiniteDeterministicFrame)
{
  const auto value = frame(10 * kSecond, {risk(1, 2.5)});
  const auto a = build_roundabout_gap_response_frame(
    value, 10 * kSecond, std::nullopt, RoundaboutGapResponseFrameConfig{});
  const auto b = build_roundabout_gap_response_frame(
    value, 10 * kSecond, std::nullopt, RoundaboutGapResponseFrameConfig{});
  ASSERT_TRUE(a.published);
  ASSERT_TRUE(b.published);
  EXPECT_EQ(a.output.action, b.output.action);
  EXPECT_TRUE(std::isfinite(a.output.available_distance_m));
  EXPECT_TRUE(std::isfinite(a.output.comfortable_stop_distance_m));
}

TEST(RoundaboutGapResponseFrame, RejectsStaleFutureDuplicateBackwardAndContextErrors)
{
  const auto config = RoundaboutGapResponseFrameConfig{};
  auto value = frame(10 * kSecond);
  EXPECT_FALSE(
    build_roundabout_gap_response_frame(
      value, 11 * kSecond, std::nullopt, config).published);
  EXPECT_FALSE(
    build_roundabout_gap_response_frame(
      value, 9 * kSecond, std::nullopt, config).published);
  EXPECT_FALSE(
    build_roundabout_gap_response_frame(
      value, 10 * kSecond, 10 * kSecond, config).published);
  EXPECT_FALSE(
    build_roundabout_gap_response_frame(
      value, 10 * kSecond, 11 * kSecond, config).published);
  value.header.frame_id = "base_link";
  EXPECT_FALSE(
    build_roundabout_gap_response_frame(
      value, 10 * kSecond, std::nullopt, config).published);
  value.header.frame_id = "map";
  value.conflict_zone_id.clear();
  EXPECT_FALSE(
    build_roundabout_gap_response_frame(
      value, 10 * kSecond, std::nullopt, config).published);
}

TEST(RoundaboutGapResponseFrame, RejectsMalformedOrInconsistentObject)
{
  auto value = frame(10 * kSecond, {risk(1, 2.5)});
  value.objects[0].minimum_temporal_gap_s = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(
    build_roundabout_gap_response_frame(
      value, 10 * kSecond, std::nullopt, RoundaboutGapResponseFrameConfig{}).published);
  value = frame(10 * kSecond, {risk(1, 2.5)});
  value.relevant_object_count = 0U;
  EXPECT_FALSE(
    build_roundabout_gap_response_frame(
      value, 10 * kSecond, std::nullopt, RoundaboutGapResponseFrameConfig{}).published);
}

TEST(RoundaboutGapResponseFrame, LaterReentryAndShortPredictionRegressions)
{
  const auto reentry = build_roundabout_gap_response_frame(
    frame(10 * kSecond, {risk(5, 0.0, true, true)}),
    10 * kSecond, std::nullopt, RoundaboutGapResponseFrameConfig{});
  ASSERT_TRUE(reentry.published);
  EXPECT_NE(
    reentry.output.action,
    ad_interfaces::msg::RoundaboutGapResponse::ACTION_RELEASE);
  EXPECT_EQ(
    reentry.output.reason,
    ad_interfaces::msg::RoundaboutGapResponse::REASON_OVERLAP);
  auto short_risk = risk(4, 2.5, false, false);
  short_risk.prediction_horizon_s = 4.0F;
  const auto short_prediction = build_roundabout_gap_response_frame(
    frame(10 * kSecond, {short_risk}), 10 * kSecond, std::nullopt,
    RoundaboutGapResponseFrameConfig{});
  ASSERT_TRUE(short_prediction.published);
  EXPECT_NE(
    short_prediction.output.action,
    ad_interfaces::msg::RoundaboutGapResponse::ACTION_RELEASE);
  EXPECT_EQ(
    short_prediction.output.reason,
    ad_interfaces::msg::RoundaboutGapResponse::REASON_INSUFFICIENT_PREDICTION);
}

}  // namespace ad_planner
