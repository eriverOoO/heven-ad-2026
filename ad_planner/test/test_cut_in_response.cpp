#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "ad_planner/planning/cut_in_response.hpp"
#include "cut_in_response_node.hpp"

namespace ad_planner
{
namespace
{

constexpr std::int64_t kSecond = 1'000'000'000LL;

CutInResponseParameters default_parameters()
{
  CutInResponseParameters parameters;
  parameters.longitudinal_standoff_m = 6.0;
  parameters.comfortable_deceleration_mps2 = 1.8;
  parameters.maximum_deceleration_mps2 = 3.0;
  parameters.minimum_response_speed_mps = 1.0;
  parameters.maximum_risks = 256U;
  return parameters;
}

CutInResponseObjectInput candidate(
  const std::uint8_t first_byte,
  const double route_s_rel_m,
  const std::uint8_t side = 1U)
{
  CutInResponseObjectInput object;
  object.object_id[0] = first_byte;
  object.side = side;
  object.route_s_rel_m = route_s_rel_m;
  object.lateral_velocity_toward_corridor_mps = 1.0;
  object.predicted_entry_valid = true;
  object.predicted_entry_time_s = 1.5;
  object.predicted_entry_route_s_rel_m = route_s_rel_m;
  return object;
}

ad_interfaces::msg::CutInRisk risk_message(
  const std::uint8_t first_byte, const double route_s_rel_m, const bool is_candidate)
{
  ad_interfaces::msg::CutInRisk risk;
  risk.object_id.uuid[0] = first_byte;
  risk.side = ad_interfaces::msg::CutInRisk::SIDE_RIGHT;
  risk.route_s_rel_m = static_cast<float>(route_s_rel_m);
  risk.lateral_velocity_toward_corridor_mps = 1.0F;
  risk.approaching_corridor = true;
  risk.adjacent_region = true;
  risk.predicted_entry_valid = true;
  risk.predicted_entry_time_s = 1.5F;
  risk.predicted_entry_route_s_rel_m = static_cast<float>(route_s_rel_m);
  risk.predicted_entry_sustained = true;
  risk.cut_in_candidate = is_candidate;
  return risk;
}

ad_interfaces::msg::CutInRiskArray risk_array(
  const std::int64_t stamp_ns,
  const std::vector<ad_interfaces::msg::CutInRisk> & risks)
{
  ad_interfaces::msg::CutInRiskArray array;
  array.header.frame_id = "map";
  array.header.stamp.sec = static_cast<std::int32_t>(stamp_ns / kSecond);
  array.header.stamp.nanosec = static_cast<std::uint32_t>(stamp_ns % kSecond);
  array.risks = risks;
  return array;
}

CutInResponseFrameConfig frame_config()
{
  CutInResponseFrameConfig config;
  config.policy = default_parameters();
  return config;
}

std::optional<EgoSpeedSample> ego_at(const double speed_mps, const std::int64_t stamp_ns)
{
  return EgoSpeedSample{speed_mps, stamp_ns};
}

}  // namespace

// --- Phase 16: core policy states -------------------------------------------

TEST(CutInResponseCore, NoCandidatesIsNone)
{
  const auto result = compute_cut_in_response(8.0, {}, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kNone);
  EXPECT_FALSE(result.active);
  EXPECT_EQ(result.candidate_count, 0U);
  EXPECT_FALSE(result.requested_max_speed_valid);
}

TEST(CutInResponseCore, FarFutureCutInIsNone)
{
  const auto object = candidate(1U, 60.0);
  const auto result = evaluate_cut_in_response_object(5.0, object, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kNone);
  EXPECT_LT(result.required_deceleration_mps2, 1.8);
}

TEST(CutInResponseCore, ModerateCutInIsSlowdownBelowEgoSpeed)
{
  const auto object = candidate(1U, 20.0);
  const auto result = evaluate_cut_in_response_object(8.0, object, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kSlowdown);
  EXPECT_EQ(result.reason, CutInResponseReason::kApproachingEntry);
  EXPECT_TRUE(result.requested_max_speed_valid);
  EXPECT_GT(result.requested_max_speed_mps, 0.0);
  EXPECT_LT(result.requested_max_speed_mps, 8.0);
  EXPECT_GE(result.requested_max_speed_mps, default_parameters().minimum_response_speed_mps);
  EXPECT_TRUE(std::isfinite(result.requested_max_speed_mps));
  EXPECT_GT(result.required_deceleration_mps2, 1.8);
  EXPECT_LE(result.required_deceleration_mps2, 3.0);
}

TEST(CutInResponseCore, ImminentCutInIsHoldWithZeroSpeed)
{
  const auto object = candidate(1U, 8.0);
  const auto result = evaluate_cut_in_response_object(10.0, object, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kHold);
  EXPECT_EQ(result.reason, CutInResponseReason::kCollisionConflict);
  EXPECT_TRUE(result.requested_max_speed_valid);
  EXPECT_DOUBLE_EQ(result.requested_max_speed_mps, 0.0);
}

TEST(CutInResponseCore, TtcInvalidButPredictedEntryNearStillResponds)
{
  auto object = candidate(1U, 40.0);
  object.predicted_entry_route_s_rel_m = 7.0;  // enters just ahead of the standoff
  object.ttc_valid = false;
  const auto result = evaluate_cut_in_response_object(6.0, object, default_parameters());
  EXPECT_NE(result.action, CutInResponseAction::kNone);
}

TEST(CutInResponseCore, LargeCpaDistanceDoesNotEscalate)
{
  auto object = candidate(1U, 30.0);
  object.cpa_valid = true;
  object.cpa_time_s = 3.0;
  object.cpa_distance_m = 50.0;
  const auto result = evaluate_cut_in_response_object(6.0, object, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kNone);
}

TEST(CutInResponseCore, SmallPredictedMinSeparationDoesNotByItselfEscalate)
{
  // A completed cut-in always ends near zero centroid separation, so the copied
  // predicted minimum separation never drives the action on its own.
  auto object = candidate(1U, 40.0);  // geometry alone is NONE
  object.predicted_min_separation_valid = true;
  object.predicted_min_separation_m = 0.5;
  object.predicted_min_separation_time_s = 0.4;
  const auto result = evaluate_cut_in_response_object(4.0, object, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kNone);
}

TEST(CutInResponseCore, SmallTtcEscalatesToHold)
{
  auto object = candidate(1U, 40.0);
  object.ttc_valid = true;
  object.ttc_s = 0.5;  // < ego_speed / maximum_deceleration = 8 / 3
  const auto result = evaluate_cut_in_response_object(8.0, object, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kHold);
  EXPECT_EQ(result.reason, CutInResponseReason::kCollisionConflict);
}

TEST(CutInResponseCore, ObjectAlongsideOrBehindIsNone)
{
  auto object = candidate(1U, -3.0);
  object.predicted_entry_route_s_rel_m = -2.0;
  const auto result = evaluate_cut_in_response_object(10.0, object, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kNone);
}

TEST(CutInResponseCore, ObjectAheadEnteringResponds)
{
  const auto object = candidate(1U, 15.0);
  const auto result = evaluate_cut_in_response_object(7.0, object, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kSlowdown);
}

TEST(CutInResponseCore, ZeroEgoSpeedFarObjectIsNone)
{
  const auto object = candidate(1U, 20.0);
  const auto result = evaluate_cut_in_response_object(0.0, object, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kNone);
}

TEST(CutInResponseCore, LowEgoSpeedIsNone)
{
  const auto object = candidate(1U, 12.0);
  const auto result = evaluate_cut_in_response_object(2.0, object, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kNone);
}

TEST(CutInResponseCore, HighEgoSpeedIsHold)
{
  const auto object = candidate(1U, 20.0);
  const auto result = evaluate_cut_in_response_object(15.0, object, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kHold);
}

TEST(CutInResponseCore, NegativeEgoSpeedIsClampedToZero)
{
  const auto object = candidate(1U, 20.0);
  const auto result = evaluate_cut_in_response_object(-5.0, object, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kNone);
}

// --- Phase 17: left / right symmetry ---------------------------------------

TEST(CutInResponseCore, MirroredLeftRightGiveIdenticalLongitudinalResponse)
{
  auto left = candidate(1U, 18.0, ad_interfaces::msg::CutInRisk::SIDE_LEFT);
  auto right = candidate(2U, 18.0, ad_interfaces::msg::CutInRisk::SIDE_RIGHT);
  left.lateral_velocity_toward_corridor_mps = 1.25;
  right.lateral_velocity_toward_corridor_mps = 1.25;

  const auto left_result = evaluate_cut_in_response_object(9.0, left, default_parameters());
  const auto right_result = evaluate_cut_in_response_object(9.0, right, default_parameters());
  EXPECT_EQ(left_result.action, right_result.action);
  EXPECT_DOUBLE_EQ(left_result.requested_max_speed_mps, right_result.requested_max_speed_mps);
  EXPECT_DOUBLE_EQ(
    left_result.required_deceleration_mps2, right_result.required_deceleration_mps2);

  const auto aggregate =
    compute_cut_in_response(9.0, {left, right}, default_parameters());
  EXPECT_EQ(aggregate.candidate_count, 2U);
  EXPECT_TRUE(aggregate.active);
}

// --- Phase 12 / 15: multi-object arbitration ------------------------------

TEST(CutInResponseCore, MostRestrictiveCandidateWins)
{
  const auto mild = candidate(1U, 30.0);
  const auto severe = candidate(2U, 10.0);
  const auto result = compute_cut_in_response(6.0, {mild, severe}, default_parameters());
  EXPECT_EQ(result.action, CutInResponseAction::kHold);
  ASSERT_TRUE(result.has_source);
  EXPECT_EQ(result.source.input.object_id[0], 2U);
}

TEST(CutInResponseCore, ThreeCandidatesDeterministicWinner)
{
  const auto a = candidate(3U, 40.0);
  const auto b = candidate(1U, 18.0);   // slowdown
  const auto c = candidate(2U, 22.0);   // milder slowdown or none
  const auto result = compute_cut_in_response(8.0, {a, b, c}, default_parameters());
  ASSERT_TRUE(result.has_source);
  // b is at the smallest station among the slowdown candidates -> lowest speed.
  EXPECT_EQ(result.source.input.object_id[0], 1U);
  EXPECT_EQ(result.action, CutInResponseAction::kSlowdown);
}

TEST(CutInResponseCore, EqualCandidatesBreakTieOnSmallestUuid)
{
  const auto high_uuid = candidate(9U, 18.0);
  const auto low_uuid = candidate(2U, 18.0);
  const auto result =
    compute_cut_in_response(8.0, {high_uuid, low_uuid}, default_parameters());
  ASSERT_TRUE(result.has_source);
  EXPECT_EQ(result.source.input.object_id[0], 2U);
}

// --- Phase 16: malformed / determinism ----------------------------------

TEST(CutInResponseCore, NonFiniteFactThrows)
{
  auto object = candidate(1U, 20.0);
  object.route_s_rel_m = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
    evaluate_cut_in_response_object(8.0, object, default_parameters()),
    std::invalid_argument);
}

TEST(CutInResponseCore, DeterministicRepeatedInput)
{
  const auto object = candidate(1U, 18.0);
  const auto first = evaluate_cut_in_response_object(8.0, object, default_parameters());
  const auto second = evaluate_cut_in_response_object(8.0, object, default_parameters());
  EXPECT_EQ(first.action, second.action);
  EXPECT_DOUBLE_EQ(first.requested_max_speed_mps, second.requested_max_speed_mps);
  EXPECT_DOUBLE_EQ(first.required_deceleration_mps2, second.required_deceleration_mps2);
}

TEST(CutInResponseCore, InvalidParametersThrow)
{
  auto parameters = default_parameters();
  parameters.maximum_deceleration_mps2 = 1.0;  // below comfortable
  EXPECT_THROW(parameters.validated(), std::invalid_argument);
}

// --- Phase 10: monotonicity ---------------------------------------------

TEST(CutInResponseCore, MonotonicInEntryStationRequestedSpeedNeverIncreases)
{
  double previous_speed = std::numeric_limits<double>::infinity();
  int previous_rank = -1;
  for (const double station : {40.0, 30.0, 22.0, 16.0, 11.0, 8.0}) {
    const auto object = candidate(1U, station);
    const auto result = evaluate_cut_in_response_object(9.0, object, default_parameters());
    const int rank = static_cast<int>(result.action);
    EXPECT_GE(rank, previous_rank) << "station " << station;
    const double speed = result.requested_max_speed_valid ?
      result.requested_max_speed_mps : 0.0;
    if (result.action == CutInResponseAction::kSlowdown) {
      EXPECT_LE(speed, previous_speed + 1e-9) << "station " << station;
      previous_speed = speed;
    }
    previous_rank = rank;
  }
}

TEST(CutInResponseCore, MonotonicInEntryTimeIsNonIncreasingResponse)
{
  auto object = candidate(1U, 20.0);
  object.predicted_entry_route_s_rel_m = 20.0;  // fixed station
  double previous_speed = 0.0;
  int previous_rank = -1;
  bool first = true;
  for (const double entry_time : {3.0, 2.0, 1.0, 0.5}) {
    object.predicted_entry_time_s = entry_time;
    const auto result = evaluate_cut_in_response_object(8.0, object, default_parameters());
    const int rank = static_cast<int>(result.action);
    if (!first) {
      EXPECT_GE(rank, previous_rank);
      EXPECT_DOUBLE_EQ(result.requested_max_speed_mps, previous_speed);
    }
    previous_rank = rank;
    previous_speed = result.requested_max_speed_mps;
    first = false;
  }
}

TEST(CutInResponseCore, PredictedMinSeparationNeverWeakensOrChangesAction)
{
  const auto baseline =
    evaluate_cut_in_response_object(9.0, candidate(1U, 12.0), default_parameters());
  for (const double separation : {10.0, 6.0, 2.0, 0.5}) {
    auto object = candidate(1U, 12.0);
    object.predicted_min_separation_valid = true;
    object.predicted_min_separation_m = separation;
    object.predicted_min_separation_time_s = 0.5;
    const auto result = evaluate_cut_in_response_object(9.0, object, default_parameters());
    EXPECT_EQ(result.action, baseline.action) << "sep " << separation;
  }
}

TEST(CutInResponseCore, MonotonicInTtc)
{
  int previous_rank = -1;
  for (const double ttc : {10.0, 4.0, 1.0, 0.2}) {
    auto object = candidate(1U, 40.0);
    object.ttc_valid = true;
    object.ttc_s = ttc;
    const auto result = evaluate_cut_in_response_object(8.0, object, default_parameters());
    EXPECT_GE(static_cast<int>(result.action), previous_rank) << "ttc " << ttc;
    previous_rank = static_cast<int>(result.action);
  }
}

// --- Node-level frame behavior -----------------------------------------

TEST(CutInResponseFrame, NonCandidateRiskProducesNone)
{
  const auto risks = risk_array(
    10 * kSecond, {risk_message(1U, 12.0, false)});
  const auto result = build_cut_in_response_frame(
    risks, ego_at(9.0, 10 * kSecond), 10 * kSecond, std::nullopt, frame_config());
  ASSERT_TRUE(result.published);
  EXPECT_EQ(result.output.action, ad_interfaces::msg::CutInResponse::ACTION_NONE);
  EXPECT_FALSE(result.output.active);
  EXPECT_EQ(result.output.candidate_count, 0U);
}

TEST(CutInResponseFrame, ValidTtcOnNonCandidateStaysNone)
{
  auto risk = risk_message(1U, 8.0, false);
  risk.ttc_valid = true;
  risk.ttc_s = 0.4F;
  const auto risks = risk_array(10 * kSecond, {risk});
  const auto result = build_cut_in_response_frame(
    risks, ego_at(9.0, 10 * kSecond), 10 * kSecond, std::nullopt, frame_config());
  ASSERT_TRUE(result.published);
  EXPECT_EQ(result.output.action, ad_interfaces::msg::CutInResponse::ACTION_NONE);
}

TEST(CutInResponseFrame, CandidateRiskProducesActiveResponseWithSource)
{
  const auto risks = risk_array(10 * kSecond, {risk_message(7U, 10.0, true)});
  const auto result = build_cut_in_response_frame(
    risks, ego_at(9.0, 10 * kSecond), 10 * kSecond, std::nullopt, frame_config());
  ASSERT_TRUE(result.published);
  EXPECT_TRUE(result.output.active);
  EXPECT_EQ(result.output.action, ad_interfaces::msg::CutInResponse::ACTION_HOLD);
  EXPECT_EQ(result.output.source_object_id.uuid[0], 7U);
  EXPECT_EQ(result.output.header.frame_id, "map");
  EXPECT_GT(result.output.required_deceleration_mps2, 0.0F);
}

TEST(CutInResponseFrame, EmptyArrayPublishesNone)
{
  const auto risks = risk_array(10 * kSecond, {});
  const auto result = build_cut_in_response_frame(
    risks, ego_at(9.0, 10 * kSecond), 10 * kSecond, std::nullopt, frame_config());
  ASSERT_TRUE(result.published);
  EXPECT_FALSE(result.output.active);
}

TEST(CutInResponseFrame, StaleInputIsNotPublished)
{
  const auto risks = risk_array(10 * kSecond, {risk_message(1U, 10.0, true)});
  const auto result = build_cut_in_response_frame(
    risks, ego_at(9.0, 10 * kSecond), 11 * kSecond, std::nullopt, frame_config());
  EXPECT_FALSE(result.published);
}

TEST(CutInResponseFrame, BackwardTimestampIsNotPublished)
{
  const auto risks = risk_array(10 * kSecond, {risk_message(1U, 10.0, true)});
  const auto result = build_cut_in_response_frame(
    risks, ego_at(9.0, 10 * kSecond), 10 * kSecond, 10 * kSecond, frame_config());
  EXPECT_FALSE(result.published);
}

TEST(CutInResponseFrame, MalformedStampIsNotPublished)
{
  auto risks = risk_array(10 * kSecond, {risk_message(1U, 10.0, true)});
  risks.header.stamp.sec = -1;
  const auto result = build_cut_in_response_frame(
    risks, ego_at(9.0, 10 * kSecond), 10 * kSecond, std::nullopt, frame_config());
  EXPECT_FALSE(result.published);
}

TEST(CutInResponseFrame, MissingOdometryIsNotPublished)
{
  const auto risks = risk_array(10 * kSecond, {risk_message(1U, 10.0, true)});
  const auto result = build_cut_in_response_frame(
    risks, std::nullopt, 10 * kSecond, std::nullopt, frame_config());
  EXPECT_FALSE(result.published);
}

TEST(CutInResponseFrame, NonFiniteRiskFieldIsSkippedAndCounted)
{
  auto risk = risk_message(1U, 10.0, true);
  risk.route_s_rel_m = std::numeric_limits<float>::infinity();
  const auto risks = risk_array(10 * kSecond, {risk});
  const auto result = build_cut_in_response_frame(
    risks, ego_at(9.0, 10 * kSecond), 10 * kSecond, std::nullopt, frame_config());
  ASSERT_TRUE(result.published);
  EXPECT_EQ(result.rejected_malformed, 1U);
  EXPECT_FALSE(result.output.active);
}

TEST(CutInResponseFrame, DeterministicRepeatedFrame)
{
  const auto risks = risk_array(10 * kSecond, {risk_message(1U, 18.0, true)});
  const auto first = build_cut_in_response_frame(
    risks, ego_at(8.0, 10 * kSecond), 10 * kSecond, std::nullopt, frame_config());
  const auto second = build_cut_in_response_frame(
    risks, ego_at(8.0, 10 * kSecond), 10 * kSecond, std::nullopt, frame_config());
  EXPECT_EQ(first.output.action, second.output.action);
  EXPECT_FLOAT_EQ(
    first.output.requested_max_speed_mps, second.output.requested_max_speed_mps);
}

}  // namespace ad_planner
