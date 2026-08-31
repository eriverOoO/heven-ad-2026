#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "ad_planner/planning/highway_merge_gap_response.hpp"
#include "highway_merge_gap_response_node.hpp"

namespace ad_planner
{
namespace
{

constexpr std::int64_t kSecond = 1'000'000'000LL;

HighwayMergeGapResponseParameters parameters()
{
  return HighwayMergeGapResponseParameters{};
}

// A relevant object comfortably ahead at the merge time (no violation).
HighwayMergeGapResponseObjectInput safe_front(
  const std::uint8_t id, const double delta_s_at_merge = 30.0)
{
  HighwayMergeGapResponseObjectInput value;
  value.object_id[0] = id;
  value.delta_s_now_m = delta_s_at_merge - 5.0;
  value.object_longitudinal_speed_mps = 8.0;
  value.delta_s_at_merge_valid = true;
  value.delta_s_at_merge_m = delta_s_at_merge;
  value.is_ahead_at_merge = true;
  value.predicted_min_route_gap_valid = true;
  value.predicted_min_route_gap_m = 20.0;
  value.prediction_covers_merge_time = true;
  return value;
}

// A relevant object comfortably behind at the merge time (no violation).
HighwayMergeGapResponseObjectInput safe_rear(
  const std::uint8_t id, const double delta_s_at_merge = -60.0,
  const double object_speed = 10.0)
{
  HighwayMergeGapResponseObjectInput value;
  value.object_id[0] = id;
  value.delta_s_now_m = delta_s_at_merge + 5.0;
  value.object_longitudinal_speed_mps = object_speed;
  value.delta_s_at_merge_valid = true;
  value.delta_s_at_merge_m = delta_s_at_merge;
  value.is_behind_at_merge = true;
  value.predicted_min_route_gap_valid = true;
  value.predicted_min_route_gap_m = 20.0;
  value.prediction_covers_merge_time = true;
  return value;
}

HighwayMergeGapResponseInput input(
  std::vector<HighwayMergeGapResponseObjectInput> objects = {},
  const double distance_to_merge = 150.0, const double speed = 8.0)
{
  HighwayMergeGapResponseInput value;
  value.applicable = true;
  value.ego_merge_timing_valid = true;
  value.ego_merge_time_s = 20.0;
  value.ego_speed_mps = speed;
  value.ego_route_distance_to_merge_m = distance_to_merge;
  value.relevant_objects = std::move(objects);
  return value;
}

ad_interfaces::msg::HighwayMergeGapRisk risk_object(
  const std::uint8_t id, const bool relevant = true)
{
  ad_interfaces::msg::HighwayMergeGapRisk value;
  value.object_id.uuid[0] = id;
  value.classification_probability = 0.9F;
  value.existence_probability = 0.9F;
  value.relevant_to_merge = relevant;
  value.object_route_s_m = 1200.0F;
  value.delta_s_now_m = 25.0F;
  value.object_longitudinal_speed_mps = 8.0F;
  value.relative_longitudinal_speed_mps = 0.0F;
  value.delta_s_at_merge_valid = true;
  value.delta_s_at_merge_m = 30.0F;
  value.is_ahead_at_merge = true;
  value.predicted_min_route_gap_valid = true;
  value.predicted_min_route_gap_m = 20.0F;
  value.prediction_horizon_s = 24.0F;
  value.prediction_covers_merge_time = true;
  return value;
}

ad_interfaces::msg::HighwayMergeGapRiskArray frame(
  const std::int64_t stamp_ns,
  std::vector<ad_interfaces::msg::HighwayMergeGapRisk> objects = {})
{
  ad_interfaces::msg::HighwayMergeGapRiskArray value;
  value.header.stamp.sec = static_cast<std::int32_t>(stamp_ns / kSecond);
  value.header.stamp.nanosec = static_cast<std::uint32_t>(stamp_ns % kSecond);
  value.header.frame_id = "map";
  value.merge_zone_id = "kcity_highway_onramp";
  value.target_lane_sequence_id = "route:0";
  value.source_lane_sequence_id = "route:0:left:1";
  value.merge_zone_entry_route_s_m = 1118.74F;
  value.merge_reference_route_s_m = 1286.15F;
  value.ego_route_s_m = 1130.0F;
  value.ego_longitudinal_speed_mps = 8.0F;
  value.ego_route_distance_to_zone_entry_m = -11.26F;
  value.ego_route_distance_to_merge_m = 156.15F;
  value.ego_in_merge_zone_now = true;
  value.ego_merge_timing_valid = true;
  value.ego_merge_time_s = 19.5F;
  value.relevant_object_count = static_cast<std::uint16_t>(std::count_if(
      objects.begin(), objects.end(), [](const auto & o) {return o.relevant_to_merge;}));
  value.objects = std::move(objects);
  return value;
}

HighwayMergeGapResponseFrameConfig frame_config()
{
  return HighwayMergeGapResponseFrameConfig{};
}

}  // namespace

// 1
TEST(HighwayMergeGapResponsePolicy, ZeroRelevantObjectsMergeReady)
{
  const auto result = compute_highway_merge_gap_response(input(), parameters());
  EXPECT_TRUE(result.active);
  EXPECT_EQ(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kClearGap);
  EXPECT_TRUE(result.complete_prediction_coverage);
}

// 2, 3, 4
TEST(HighwayMergeGapResponsePolicy, SafeFrontOrRearOrBothAreMergeReady)
{
  EXPECT_EQ(
    compute_highway_merge_gap_response(input({safe_front(1)}), parameters()).action,
    HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(
    compute_highway_merge_gap_response(input({safe_rear(2)}), parameters()).action,
    HighwayMergeGapResponseAction::kMergeReady);
  const auto both = compute_highway_merge_gap_response(
    input({safe_front(1), safe_rear(2)}), parameters());
  EXPECT_EQ(both.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_TRUE(both.front_object_valid);
  EXPECT_TRUE(both.rear_object_valid);
  EXPECT_NEAR(both.front_gap_m, 30.0, 1e-9);
  EXPECT_NEAR(both.rear_gap_m, 60.0, 1e-9);
}

// 5
TEST(HighwayMergeGapResponsePolicy, FrontGapTooSmallIsNotReady)
{
  auto front = safe_front(1, 6.0);  // headway 6/8 = 0.75 < 1.5
  const auto result = compute_highway_merge_gap_response(input({front}), parameters());
  EXPECT_NE(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kFrontGap);
  EXPECT_TRUE(result.has_source);
  EXPECT_EQ(result.source_object_id[0], 1U);
}

// 6
TEST(HighwayMergeGapResponsePolicy, RearGapTooSmallIsNotReady)
{
  auto rear = safe_rear(2, -10.0, 20.0);  // headway 10/20 = 0.5 < 2.0
  const auto result = compute_highway_merge_gap_response(input({rear}), parameters());
  EXPECT_NE(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kRearGap);
  EXPECT_EQ(result.source_object_id[0], 2U);
}

// 7
TEST(HighwayMergeGapResponsePolicy, RearClosingTooQuicklyIsNotReady)
{
  auto rear = safe_rear(3, -80.0, 12.0);  // rear headway 80/12 = 6.7 ok
  rear.delta_s_now_m = -30.0;
  rear.longitudinal_gap_closing = true;
  rear.longitudinal_closing_speed_mps = 6.0;
  rear.time_to_route_coincidence_valid = true;
  rear.time_to_route_coincidence_s = 2.0;  // < 3.0
  const auto result = compute_highway_merge_gap_response(input({rear}), parameters());
  EXPECT_NE(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kRearClosing);
  EXPECT_TRUE(result.rear_closing_object_valid);
  EXPECT_NEAR(result.rear_closing_time_s, 2.0, 1e-9);
}

// 8
TEST(HighwayMergeGapResponsePolicy, AlongsideObjectIsNotReady)
{
  auto obj = safe_front(4, 2.0);
  obj.is_ahead_at_merge = false;
  obj.is_alongside_at_merge = true;
  const auto result = compute_highway_merge_gap_response(input({obj}), parameters());
  EXPECT_NE(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kAlongside);
}

// 9, 10
TEST(HighwayMergeGapResponsePolicy, InsufficientCoverageOrFallbackExtrapolationIsNotReady)
{
  auto no_coverage = safe_front(5, 30.0);
  no_coverage.prediction_covers_merge_time = false;
  EXPECT_EQ(
    compute_highway_merge_gap_response(input({no_coverage}), parameters()).reason,
    HighwayMergeGapResponseReason::kInsufficientPrediction);

  // Fallback constant-speed delta_s_at_merge looks large / safe, but coverage
  // is false -> must not authorize.
  auto fallback = safe_front(6, 200.0);
  fallback.prediction_covers_merge_time = false;
  const auto result = compute_highway_merge_gap_response(input({fallback}), parameters());
  EXPECT_NE(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kInsufficientPrediction);
  EXPECT_FALSE(result.complete_prediction_coverage);
}

// 11
TEST(HighwayMergeGapResponsePolicy, PredictedRouteGapTooSmallIsNotReady)
{
  auto obj = safe_front(7, 30.0);
  obj.predicted_min_route_gap_m = 3.0;  // < 6.0
  const auto result = compute_highway_merge_gap_response(input({obj}), parameters());
  EXPECT_NE(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kPredictedRouteConflict);
}

// 12
TEST(HighwayMergeGapResponsePolicy, LargeTotalSpanNeverOverridesUnsafeRear)
{
  auto front = safe_front(1, 90.0);   // huge front gap
  auto rear = safe_rear(2, -2.0, 20.0);  // rear headway 2/20 = 0.1 -> unsafe
  // Total front-to-rear span is 92 m, yet the merge is not ready.
  const auto result = compute_highway_merge_gap_response(
    input({front, rear}), parameters());
  EXPECT_NE(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kRearGap);
  EXPECT_EQ(result.source_object_id[0], 2U);
}

// 13
TEST(HighwayMergeGapResponsePolicy, MultipleSafeObjectsAreMergeReady)
{
  const auto result = compute_highway_merge_gap_response(
    input({safe_front(1, 40.0), safe_front(2, 60.0), safe_rear(3, -70.0),
        safe_rear(4, -90.0)}), parameters());
  EXPECT_EQ(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.front_object_id[0], 1U);  // nearest ahead
  EXPECT_EQ(result.rear_object_id[0], 3U);   // nearest behind
}

// 14, 15
TEST(HighwayMergeGapResponsePolicy, OneUnsafeAmongManyBlocksAndLimitingUuidIsDeterministic)
{
  auto bad_a = safe_front(9, 6.0);   // front headway 0.75
  auto bad_b = safe_front(3, 6.0);   // same violation, smaller UUID
  const auto result = compute_highway_merge_gap_response(
    input({safe_front(1, 40.0), bad_a, safe_rear(2, -70.0), bad_b}), parameters());
  EXPECT_NE(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kFrontGap);
  EXPECT_EQ(result.source_object_id[0], 3U);
}

TEST(HighwayMergeGapResponsePolicy, ReasonPrecedenceIsAlongsideThenCoverageThenRoute)
{
  auto alongside = safe_front(2, 3.0);
  alongside.is_ahead_at_merge = false;
  alongside.is_alongside_at_merge = true;
  auto no_cov = safe_front(1, 6.0);
  no_cov.prediction_covers_merge_time = false;
  auto route = safe_rear(3, -70.0);
  route.predicted_min_route_gap_m = 1.0;
  const auto result = compute_highway_merge_gap_response(
    input({no_cov, route, alongside}), parameters());
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kAlongside);
  EXPECT_EQ(result.source_object_id[0], 2U);
}

// 16, 17
TEST(HighwayMergeGapResponsePolicy, FrontThresholdBoundaryIsInclusive)
{
  const double eps = 1e-4;
  // headway = delta / 8 ; threshold 1.5 -> delta 12.0
  EXPECT_EQ(
    compute_highway_merge_gap_response(
      input({safe_front(1, 12.0)}), parameters()).action,
    HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(
    compute_highway_merge_gap_response(
      input({safe_front(1, 12.0 + eps)}), parameters()).action,
    HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_NE(
    compute_highway_merge_gap_response(
      input({safe_front(1, 12.0 - eps)}), parameters()).action,
    HighwayMergeGapResponseAction::kMergeReady);
}

// 18, 19
TEST(HighwayMergeGapResponsePolicy, RearThresholdBoundaryIsInclusive)
{
  const double eps = 1e-4;
  // rear headway = |delta| / speed ; speed 10, threshold 2.0 -> |delta| 20.0
  EXPECT_EQ(
    compute_highway_merge_gap_response(
      input({safe_rear(1, -20.0, 10.0)}), parameters()).action,
    HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_NE(
    compute_highway_merge_gap_response(
      input({safe_rear(1, -(20.0 - eps), 10.0)}), parameters()).action,
    HighwayMergeGapResponseAction::kMergeReady);
}

// 20, 21
TEST(HighwayMergeGapResponsePolicy, RearClosingThresholdBoundaryIsInclusive)
{
  const auto make = [](const double coincidence) {
      auto rear = safe_rear(1, -80.0, 12.0);
      rear.delta_s_now_m = -30.0;
      rear.longitudinal_gap_closing = true;
      rear.longitudinal_closing_speed_mps = 4.0;
      rear.time_to_route_coincidence_valid = true;
      rear.time_to_route_coincidence_s = coincidence;
      return rear;
    };
  EXPECT_EQ(
    compute_highway_merge_gap_response(input({make(3.0)}), parameters()).action,
    HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_NE(
    compute_highway_merge_gap_response(input({make(3.0 - 1e-4)}), parameters()).action,
    HighwayMergeGapResponseAction::kMergeReady);
}

// 22, 23, 34
TEST(HighwayMergeGapResponsePolicy, ApproachProgressesWaitThenHold)
{
  auto rear = safe_rear(1, -10.0, 20.0);  // unsafe rear headway 0.5
  int previous = static_cast<int>(HighwayMergeGapResponseAction::kMergeReady);
  bool saw_wait = false;
  bool saw_hold = false;
  for (const double distance : {200.0, 120.0, 40.0, 20.0, 8.0}) {
    const auto result = compute_highway_merge_gap_response(
      input({rear}, distance, 8.0), parameters());
    const int action = static_cast<int>(result.action);
    EXPECT_GE(action, previous);  // monotonically more restrictive
    previous = action;
    saw_wait = saw_wait || result.action == HighwayMergeGapResponseAction::kWait;
    saw_hold = saw_hold || result.action == HighwayMergeGapResponseAction::kHold;
  }
  EXPECT_TRUE(saw_wait);
  EXPECT_TRUE(saw_hold);
}

// 24
TEST(HighwayMergeGapResponsePolicy, StoppedEgoHoldsAndCannotMergeReady)
{
  auto value = input({}, 40.0, 0.0);
  value.ego_merge_timing_valid = false;
  value.ego_merge_time_s = 0.0;
  const auto result = compute_highway_merge_gap_response(value, parameters());
  EXPECT_TRUE(result.active);
  EXPECT_EQ(result.action, HighwayMergeGapResponseAction::kHold);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kInvalidEgoState);

  // Even a claimed-valid timing cannot rescue a stopped ego.
  const auto claimed = compute_highway_merge_gap_response(input({}, 40.0, 0.0), parameters());
  EXPECT_EQ(claimed.action, HighwayMergeGapResponseAction::kHold);
  EXPECT_EQ(claimed.reason, HighwayMergeGapResponseReason::kInvalidEgoState);
}

TEST(HighwayMergeGapResponsePolicy, MovingEgoWithInvalidTimingUsesMarginRuleNeverReady)
{
  auto far = input({}, 5000.0, 8.0);
  far.ego_merge_timing_valid = false;
  far.ego_merge_time_s = 0.0;
  const auto result = compute_highway_merge_gap_response(far, parameters());
  EXPECT_EQ(result.action, HighwayMergeGapResponseAction::kWait);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kInvalidEgoState);
}

// 25
TEST(HighwayMergeGapResponsePolicy, FarEgoIsInactiveNotHold)
{
  auto value = input({}, 40.0, 8.0);
  value.applicable = false;
  const auto result = compute_highway_merge_gap_response(value, parameters());
  EXPECT_FALSE(result.active);
  EXPECT_NE(result.action, HighwayMergeGapResponseAction::kHold);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kNone);
}

// 26
TEST(HighwayMergeGapResponseFrame, EgoPastMergeIsInactive)
{
  auto f = frame(10 * kSecond);
  f.ego_route_distance_to_merge_m = -20.0F;
  f.ego_route_distance_to_zone_entry_m = -180.0F;
  f.ego_merge_timing_valid = false;
  f.ego_merge_time_s = 0.0F;
  const auto result = build_highway_merge_gap_response_frame(
    f, 10 * kSecond, std::nullopt, frame_config());
  ASSERT_TRUE(result.published);
  EXPECT_FALSE(result.output.active);
  EXPECT_EQ(
    result.output.action,
    ad_interfaces::msg::HighwayMergeGapResponse::ACTION_MERGE_READY);
}

// Phase 6 / config-mismatch guard: an ego far before the zone with the risk
// node reporting invalid timing is inactive, never HOLD.
TEST(HighwayMergeGapResponseFrame, FarEgoBeforeZoneIsInactive)
{
  auto f = frame(10 * kSecond);
  f.ego_route_distance_to_zone_entry_m = 800.0F;
  f.ego_route_distance_to_merge_m = 968.0F;
  f.ego_merge_timing_valid = false;
  f.ego_merge_time_s = 0.0F;
  const auto result = build_highway_merge_gap_response_frame(
    f, 10 * kSecond, std::nullopt, frame_config());
  ASSERT_TRUE(result.published);
  EXPECT_FALSE(result.output.active);
}

// 27
TEST(HighwayMergeGapResponseFrame, RejectsMalformedOrInconsistentObject)
{
  auto value = frame(10 * kSecond, {risk_object(1)});
  value.objects[0].delta_s_at_merge_m = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(
    build_highway_merge_gap_response_frame(
      value, 10 * kSecond, std::nullopt, frame_config()).published);

  value = frame(10 * kSecond, {risk_object(1)});
  value.objects[0].is_behind_at_merge = true;  // both ahead and behind
  EXPECT_FALSE(
    build_highway_merge_gap_response_frame(
      value, 10 * kSecond, std::nullopt, frame_config()).published);

  value = frame(10 * kSecond, {risk_object(1)});
  value.objects[0].delta_s_at_merge_valid = false;  // flag set without valid delta
  EXPECT_FALSE(
    build_highway_merge_gap_response_frame(
      value, 10 * kSecond, std::nullopt, frame_config()).published);

  value = frame(10 * kSecond, {risk_object(1)});
  value.relevant_object_count = 0U;  // count inconsistent with objects
  EXPECT_FALSE(
    build_highway_merge_gap_response_frame(
      value, 10 * kSecond, std::nullopt, frame_config()).published);

  value = frame(10 * kSecond, {risk_object(1)});
  value.objects[0].time_to_route_coincidence_valid = true;  // without gap_closing
  EXPECT_FALSE(
    build_highway_merge_gap_response_frame(
      value, 10 * kSecond, std::nullopt, frame_config()).published);
}

TEST(HighwayMergeGapResponseFrame, RejectsCoverageClaimedShorterThanEgoMergeTime)
{
  auto value = frame(10 * kSecond, {risk_object(1)});
  value.objects[0].prediction_covers_merge_time = true;
  value.objects[0].prediction_horizon_s = 5.0F;  // < ego_merge_time_s 19.5
  EXPECT_FALSE(
    build_highway_merge_gap_response_frame(
      value, 10 * kSecond, std::nullopt, frame_config()).published);
}

// Phase 16: predicted route-gap threshold boundary is inclusive.
TEST(HighwayMergeGapResponsePolicy, PredictedRouteGapThresholdBoundaryIsInclusive)
{
  const double eps = 1e-4;
  auto at = safe_front(1, 40.0);
  at.predicted_min_route_gap_m = 6.0;
  EXPECT_EQ(
    compute_highway_merge_gap_response(input({at}), parameters()).action,
    HighwayMergeGapResponseAction::kMergeReady);
  auto below = safe_front(1, 40.0);
  below.predicted_min_route_gap_m = 6.0 - eps;
  const auto result = compute_highway_merge_gap_response(input({below}), parameters());
  EXPECT_NE(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kPredictedRouteConflict);
}

TEST(HighwayMergeGapResponseFrame, PermitsFallbackDeltaWithoutCoverage)
{
  auto value = frame(10 * kSecond, {risk_object(1)});
  value.objects[0].delta_s_at_merge_valid = true;
  value.objects[0].delta_s_at_merge_m = 250.0F;
  value.objects[0].prediction_covers_merge_time = false;
  const auto result = build_highway_merge_gap_response_frame(
    value, 10 * kSecond, std::nullopt, frame_config());
  ASSERT_TRUE(result.published);
  EXPECT_NE(
    result.output.action,
    ad_interfaces::msg::HighwayMergeGapResponse::ACTION_MERGE_READY);
  EXPECT_EQ(
    result.output.reason,
    ad_interfaces::msg::HighwayMergeGapResponse::REASON_INSUFFICIENT_PREDICTION);
}

// 28, 29, 30
TEST(HighwayMergeGapResponseFrame, RejectsStaleFutureDuplicateBackwardAndContextErrors)
{
  const auto config = frame_config();
  auto value = frame(10 * kSecond);
  // stale: now is 1 s past the stamp, default max age 0.5 s
  EXPECT_FALSE(
    build_highway_merge_gap_response_frame(
      value, 11 * kSecond, std::nullopt, config).published);
  // future
  EXPECT_FALSE(
    build_highway_merge_gap_response_frame(
      value, 9 * kSecond, std::nullopt, config).published);
  // duplicate
  EXPECT_FALSE(
    build_highway_merge_gap_response_frame(
      value, 10 * kSecond, 10 * kSecond, config).published);
  // backward (small out-of-order jump, below the rollback threshold)
  EXPECT_FALSE(
    build_highway_merge_gap_response_frame(
      value, 10 * kSecond, 10 * kSecond + kSecond / 5, config).published);
  // wrong frame
  value.header.frame_id = "base_link";
  EXPECT_FALSE(
    build_highway_merge_gap_response_frame(
      value, 10 * kSecond, std::nullopt, config).published);
  value.header.frame_id = "map";
  // wrong merge zone
  value.merge_zone_id = "kcity_roundabout";
  EXPECT_FALSE(
    build_highway_merge_gap_response_frame(
      value, 10 * kSecond, std::nullopt, config).published);
}

TEST(HighwayMergeGapResponseFrame, LargeClockRollbackClearsTheLatch)
{
  const auto config = frame_config();
  const auto value = frame(2 * kSecond, {risk_object(1)});
  // last stamp far in the future -> a sim-time reset, accepted.
  const auto result = build_highway_merge_gap_response_frame(
    value, 2 * kSecond, 100 * kSecond, config);
  EXPECT_TRUE(result.published);
}

// 31, 32
TEST(HighwayMergeGapResponseFrame, DeterministicAndFinite)
{
  const auto value = frame(10 * kSecond, {risk_object(1)});
  const auto a = build_highway_merge_gap_response_frame(
    value, 10 * kSecond, std::nullopt, frame_config());
  const auto b = build_highway_merge_gap_response_frame(
    value, 10 * kSecond, std::nullopt, frame_config());
  ASSERT_TRUE(a.published);
  ASSERT_TRUE(b.published);
  EXPECT_EQ(a.output.action, b.output.action);
  EXPECT_EQ(a.output.reason, b.output.reason);
  for (const float v : {a.output.available_distance_m, a.output.comfortable_stop_distance_m,
      a.output.front_gap_m, a.output.rear_gap_m, a.output.rear_closing_time_s,
      a.output.minimum_predicted_route_gap_m, a.output.front_time_headway_s,
      a.output.rear_time_headway_s, a.output.ego_merge_time_s})
  {
    EXPECT_TRUE(std::isfinite(v));
  }
}

// 33
TEST(HighwayMergeGapResponsePolicy, GapPolicyIsMonotonic)
{
  int previous = static_cast<int>(HighwayMergeGapResponseAction::kMergeReady);
  for (const double delta : {40.0, 12.0, 8.0, 4.0, 1.0}) {
    const auto action = compute_highway_merge_gap_response(
      input({safe_front(1, delta)}, 200.0), parameters()).action;
    EXPECT_GE(static_cast<int>(action), previous);
    previous = static_cast<int>(action);
  }
}

// Phase 26 mandatory regression restated at the pure-core level.
TEST(HighwayMergeGapResponsePolicy, FallbackExtrapolationLooksSafeButCoverageFalseNotReady)
{
  auto obj = safe_front(1, 500.0);
  obj.prediction_covers_merge_time = false;
  const auto result = compute_highway_merge_gap_response(input({obj}), parameters());
  EXPECT_NE(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kInsufficientPrediction);
}

// Phase 27: curved geometry -- an object ahead in route-s with a negative
// base_link x_rel is treated as front purely from the risk relation.
TEST(HighwayMergeGapResponsePolicy, IsAheadAtMergeIsTreatedAsFrontRegardlessOfCartesian)
{
  auto obj = safe_front(1, 25.0);
  obj.delta_s_now_m = 20.0;  // the risk layer already resolved ordering on the route
  const auto result = compute_highway_merge_gap_response(input({obj}), parameters());
  EXPECT_EQ(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_TRUE(result.front_object_valid);
  EXPECT_FALSE(result.rear_object_valid);
}

// Phase 19C: slow front object the ego is overtaking never trips REAR_CLOSING;
// it is caught (if at all) by the predicted route-gap criterion.
TEST(HighwayMergeGapResponsePolicy, SlowFrontObjectEgoClosingIsNotRearClosing)
{
  auto obj = safe_front(1, 20.0);
  obj.delta_s_now_m = 30.0;  // currently ahead
  obj.longitudinal_gap_closing = true;  // symmetric flag: ego catching it
  obj.longitudinal_closing_speed_mps = 5.0;
  obj.predicted_min_route_gap_m = 2.0;  // ego rollout comes close later
  const auto result = compute_highway_merge_gap_response(input({obj}), parameters());
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kPredictedRouteConflict);
  EXPECT_FALSE(result.rear_closing_object_valid);
}

TEST(HighwayMergeGapResponsePolicy, OverBudgetForbidsMergeReady)
{
  auto config = parameters();
  config.maximum_relevant_objects = 1U;
  const auto result = compute_highway_merge_gap_response(
    input({safe_front(1, 40.0), safe_rear(2, -70.0)}), config);
  EXPECT_NE(result.action, HighwayMergeGapResponseAction::kMergeReady);
  EXPECT_EQ(result.reason, HighwayMergeGapResponseReason::kInsufficientPrediction);
  EXPECT_EQ(result.rejected_over_budget, 1U);
}

TEST(HighwayMergeGapResponsePolicy, NegativeEgoSpeedIsRejected)
{
  EXPECT_THROW(
    compute_highway_merge_gap_response(input({}, 40.0, -1.0), parameters()),
    std::invalid_argument);
}

}  // namespace ad_planner
