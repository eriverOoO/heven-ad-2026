#include "ad_planner/planning/cut_in_risk.hpp"
#include "cut_in_risk_node.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace
{

using ad_planner::CutInEgoState;
using ad_planner::CutInFrameConfig;
using ad_planner::CutInObjectInput;
using ad_planner::CutInParameters;
using ad_planner::CutInPredictedStateInput;
using ad_planner::CutInSide;
using ad_planner::Pose2;
using ad_planner::ReferenceCorridor;
using ad_planner::ReferenceLane;
using ad_planner::ReferencePoint;
using ad_planner::RouteEgoSample;

constexpr double kPi = 3.14159265358979323846;

ReferencePoint point(
  const double x, const double y, const double yaw, const double s)
{
  return ReferencePoint{Pose2{x, y, yaw}, s, 0.0, 1.75, 1.75, 13.0};
}

ReferenceLane horizontal_lane()
{
  return ReferenceLane{
    "primary", {},
    {point(-100.0, 0.0, 0.0, 0.0), point(0.0, 0.0, 0.0, 100.0),
      point(100.0, 0.0, 0.0, 200.0)}, {}, {}};
}

ReferenceLane vertical_lane()
{
  return ReferenceLane{
    "primary", {},
    {point(0.0, -100.0, kPi / 2.0, 0.0),
      point(0.0, 0.0, kPi / 2.0, 100.0),
      point(0.0, 100.0, kPi / 2.0, 200.0)}, {}, {}};
}

ReferenceLane curved_lane()
{
  ReferenceLane lane;
  lane.lane_sequence_id = "curve";
  constexpr double radius = 40.0;
  for (int index = 0; index <= 20; ++index) {
    const double angle = -0.5 + 0.05 * index;
    lane.points.push_back(point(
      radius * std::sin(angle), radius * (1.0 - std::cos(angle)),
      angle, radius * (angle + 0.5)));
    lane.points.back().curvature_inv_m = 1.0 / radius;
  }
  return lane;
}

CutInEgoState ego(const double yaw = 0.0)
{
  return CutInEgoState{Pose2{0.0, 0.0, yaw}, 5.0};
}

CutInObjectInput object(
  const double x, const double y, const double vx, const double vy,
  std::vector<CutInPredictedStateInput> states)
{
  CutInObjectInput result;
  result.object_id[15] = 1U;
  result.classification = 1U;
  result.classification_probability = 0.9F;
  result.existence_probability = 0.95F;
  result.x_rel_m = x;
  result.y_rel_m = y;
  result.vx_rel_mps = vx;
  result.vy_rel_mps = vy;
  result.predicted_states = std::move(states);
  result.cpa_valid = true;
  result.cpa_time_s = 2.0;
  result.cpa_distance_m = 1.0;
  result.predicted_min_separation_valid = true;
  result.predicted_min_separation_m = 0.5;
  result.predicted_min_separation_time_s = 2.0;
  return result;
}

CutInObjectInput right_cut_in(const double x = 15.0)
{
  return object(x, -4.0, 0.0, 1.0,
    {{1.0, x, -3.0}, {2.0, x, -1.5}, {3.0, x, -0.5}});
}

CutInObjectInput left_cut_in(const double x = 15.0)
{
  return object(x, 4.0, 0.0, -1.0,
    {{1.0, x, 3.0}, {2.0, x, 1.5}, {3.0, x, 0.5}});
}

TEST(CutInRisk, RightSideMovingTowardCorridorIsCandidate)
{
  const auto risk = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), right_cut_in(), CutInParameters{});
  EXPECT_EQ(risk.side, CutInSide::kRight);
  EXPECT_GT(risk.lateral_velocity_toward_corridor_mps, 0.0);
  EXPECT_TRUE(risk.approaching_corridor);
  EXPECT_TRUE(risk.predicted_entry_valid);
  EXPECT_TRUE(risk.predicted_entry_sustained);
  EXPECT_TRUE(risk.cut_in_candidate);
  EXPECT_NEAR(risk.predicted_entry_time_s, 2.0, 1.0e-9);
}

TEST(CutInRisk, LeftSideMovingTowardCorridorIsCandidate)
{
  const auto risk = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), left_cut_in(), CutInParameters{});
  EXPECT_EQ(risk.side, CutInSide::kLeft);
  EXPECT_GT(risk.lateral_velocity_toward_corridor_mps, 0.0);
  EXPECT_TRUE(risk.cut_in_candidate);
}

TEST(CutInRisk, MirroredLeftAndRightEvidenceIsSymmetric)
{
  const auto right = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), right_cut_in(), CutInParameters{});
  const auto left = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), left_cut_in(), CutInParameters{});
  EXPECT_EQ(left.cut_in_candidate, right.cut_in_candidate);
  EXPECT_NEAR(left.route_s_rel_m, right.route_s_rel_m, 1.0e-9);
  EXPECT_NEAR(left.lateral_offset_m, -right.lateral_offset_m, 1.0e-9);
  EXPECT_NEAR(
    left.lateral_velocity_toward_corridor_mps,
    right.lateral_velocity_toward_corridor_mps, 1.0e-9);
  EXPECT_NEAR(left.predicted_entry_time_s, right.predicted_entry_time_s, 1.0e-9);
}

TEST(CutInRisk, RightAndLeftMovingAwayAreNotCandidates)
{
  auto right = right_cut_in();
  right.vy_rel_mps = -1.0;
  right.predicted_states = {{1.0, 15.0, -5.0}, {2.0, 15.0, -6.0}};
  auto left = left_cut_in();
  left.vy_rel_mps = 1.0;
  left.predicted_states = {{1.0, 15.0, 5.0}, {2.0, 15.0, 6.0}};
  for (const auto & input : {right, left}) {
    const auto risk = ad_planner::compute_cut_in_risk(
      horizontal_lane(), ego(), input, CutInParameters{});
    EXPECT_LT(risk.lateral_velocity_toward_corridor_mps, 0.0);
    EXPECT_FALSE(risk.approaching_corridor);
    EXPECT_FALSE(risk.cut_in_candidate);
  }
}

TEST(CutInRisk, AlreadyInsideCorridorIsNotCandidate)
{
  const auto risk = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), object(10.0, 0.5, 0.0, -1.0,
      {{1.0, 10.0, 0.0}, {2.0, 10.0, -0.5}}), CutInParameters{});
  EXPECT_TRUE(risk.inside_corridor);
  EXPECT_EQ(risk.side, CutInSide::kNone);
  EXPECT_FALSE(risk.cut_in_candidate);
}

TEST(CutInRisk, StationaryAndParallelAdjacentObjectsAreNotCandidates)
{
  for (const double vx : {-5.0, 0.0, 3.0}) {
    const auto risk = ad_planner::compute_cut_in_risk(
      horizontal_lane(), ego(), object(10.0, -4.0, vx, 0.0,
        {{1.0, 10.0 + vx, -4.0}, {2.0, 10.0 + 2.0 * vx, -4.0}}),
      CutInParameters{});
    EXPECT_FALSE(risk.approaching_corridor);
    EXPECT_FALSE(risk.predicted_entry_valid);
    EXPECT_FALSE(risk.cut_in_candidate);
  }
}

TEST(CutInRisk, FarAheadAndFarBehindAreLongitudinallyIrrelevant)
{
  const auto ahead = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), right_cut_in(90.0), CutInParameters{});
  const auto behind = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), right_cut_in(-10.0), CutInParameters{});
  EXPECT_FALSE(ahead.longitudinally_relevant);
  EXPECT_FALSE(behind.longitudinally_relevant);
  EXPECT_FALSE(ahead.cut_in_candidate);
  EXPECT_FALSE(behind.cut_in_candidate);
}

TEST(CutInRisk, CrossingRoadWithoutSustainedEntryIsNotCutIn)
{
  const auto crossing = object(12.0, -4.0, 0.0, 4.0,
    {{0.5, 12.0, -2.0}, {1.0, 12.0, 0.0}, {1.5, 12.0, 2.0},
      {2.0, 12.0, 4.0}});
  const auto risk = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), crossing, CutInParameters{});
  EXPECT_TRUE(risk.predicted_entry_valid);
  EXPECT_FALSE(risk.predicted_entry_sustained);
  EXPECT_FALSE(risk.cut_in_candidate);
}

TEST(CutInRisk, PredictedStopBeforeCorridorHasNoEntry)
{
  const auto stopping = object(12.0, -4.0, 0.0, 0.6,
    {{1.0, 12.0, -3.4}, {2.0, 12.0, -2.8}, {3.0, 12.0, -2.4},
      {4.0, 12.0, -2.4}});
  const auto risk = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), stopping, CutInParameters{});
  EXPECT_FALSE(risk.predicted_entry_valid);
  EXPECT_FALSE(risk.cut_in_candidate);
}

TEST(CutInRisk, PredictedEntryUsesFirstDiscreteInsideStateOnly)
{
  const auto risk = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), right_cut_in(), CutInParameters{});
  EXPECT_TRUE(risk.predicted_entry_valid);
  EXPECT_DOUBLE_EQ(risk.predicted_entry_time_s, 2.0);
  EXPECT_NEAR(risk.predicted_entry_lateral_offset_m, -1.5, 1.0e-9);
}

TEST(CutInRisk, GrazingBoundaryUsesInclusiveCentroidSemantics)
{
  const auto on_boundary = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), object(10.0, -1.75, 0.0, 0.5,
      {{1.0, 10.0, -1.25}, {2.0, 10.0, -0.75}}), CutInParameters{});
  EXPECT_TRUE(on_boundary.inside_corridor);
  EXPECT_TRUE(on_boundary.near_boundary);
  EXPECT_FALSE(on_boundary.cut_in_candidate);

  const auto just_outside = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), object(10.0, -1.75001, 0.0, 0.5,
      {{1.0, 10.0, -1.25}, {2.0, 10.0, -0.75}}), CutInParameters{});
  EXPECT_EQ(just_outside.side, CutInSide::kRight);
  EXPECT_TRUE(just_outside.near_boundary);
  EXPECT_TRUE(just_outside.cut_in_candidate);
}

TEST(CutInRisk, MultipleObjectsFromBothSidesAndEmptyArrayAreDeterministic)
{
  auto left = left_cut_in();
  left.object_id[15] = 2U;
  const auto computation = ad_planner::compute_cut_in_risks(
    horizontal_lane(), ego(), {right_cut_in(), left}, CutInParameters{});
  ASSERT_EQ(computation.risks.size(), 2U);
  EXPECT_TRUE(computation.risks[0].cut_in_candidate);
  EXPECT_TRUE(computation.risks[1].cut_in_candidate);

  const auto empty = ad_planner::compute_cut_in_risks(
    horizontal_lane(), ego(), {}, CutInParameters{});
  EXPECT_TRUE(empty.risks.empty());
}

TEST(CutInRisk, MalformedNonFiniteObjectIsSkipped)
{
  auto malformed = right_cut_in();
  malformed.y_rel_m = std::numeric_limits<double>::quiet_NaN();
  const auto computation = ad_planner::compute_cut_in_risks(
    horizontal_lane(), ego(), {malformed, right_cut_in()}, CutInParameters{});
  EXPECT_EQ(computation.rejected_malformed_objects, 1U);
  ASSERT_EQ(computation.risks.size(), 1U);
  EXPECT_TRUE(computation.risks.front().cut_in_candidate);
}

TEST(CutInRisk, CurvedRouteUsesFrenetGeometryInsteadOfRawBaseY)
{
  const auto lane = curved_lane();
  const CutInEgoState curved_ego{Pose2{0.0, 0.0, 0.0}, 5.0};
  const auto relative_state = [](const double time_s, const double route_s_m,
      const double lateral_m) {
      constexpr double radius = 40.0;
      const double angle = route_s_m / radius - 0.5;
      const double center_x = radius * std::sin(angle);
      const double center_y = radius * (1.0 - std::cos(angle));
      const double world_x = center_x - std::sin(angle) * lateral_m;
      const double world_y = center_y + std::cos(angle) * lateral_m;
      return CutInPredictedStateInput{
        time_s, world_x - 5.0 * time_s, world_y};
    };
  constexpr double object_route_s_m = 30.0;
  constexpr double object_angle = object_route_s_m / 40.0 - 0.5;
  constexpr double object_lateral_m = -4.0;
  const double object_x =
    40.0 * std::sin(object_angle) - std::sin(object_angle) * object_lateral_m;
  const double object_y = 40.0 * (1.0 - std::cos(object_angle)) +
    std::cos(object_angle) * object_lateral_m;
  const double absolute_vx =
    5.0 * std::cos(object_angle) - std::sin(object_angle);
  const double absolute_vy =
    5.0 * std::sin(object_angle) + std::cos(object_angle);
  const auto curved_cut_in = object(
    object_x, object_y, absolute_vx - 5.0, absolute_vy,
    {relative_state(1.0, 35.0, -3.0),
      relative_state(2.0, 40.0, -1.5),
      relative_state(3.0, 45.0, -0.5)});
  const auto risk = ad_planner::compute_cut_in_risk(
    lane, curved_ego, curved_cut_in, CutInParameters{});
  EXPECT_EQ(risk.side, CutInSide::kRight);
  EXPECT_TRUE(risk.cut_in_candidate);
  EXPECT_NE(risk.lateral_offset_m, object_y);
  EXPECT_TRUE(std::isfinite(risk.route_s_rel_m));
}

TEST(CutInRisk, NonZeroEgoAndPathYawPreservesRightSideRule)
{
  const auto risk = ad_planner::compute_cut_in_risk(
    vertical_lane(), ego(kPi / 2.0), right_cut_in(), CutInParameters{});
  EXPECT_EQ(risk.side, CutInSide::kRight);
  EXPECT_GT(risk.lateral_velocity_toward_corridor_mps, 0.0);
  EXPECT_TRUE(risk.cut_in_candidate);
}

TEST(CutInRisk, RepeatedEquivalentInputIsBitwiseStableInFacts)
{
  const auto first = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), right_cut_in(), CutInParameters{});
  const auto second = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), right_cut_in(), CutInParameters{});
  EXPECT_EQ(first.object_id, second.object_id);
  EXPECT_DOUBLE_EQ(first.route_s_rel_m, second.route_s_rel_m);
  EXPECT_DOUBLE_EQ(first.lateral_offset_m, second.lateral_offset_m);
  EXPECT_DOUBLE_EQ(
    first.lateral_velocity_toward_corridor_mps,
    second.lateral_velocity_toward_corridor_mps);
  EXPECT_EQ(first.cut_in_candidate, second.cut_in_candidate);
}

TEST(CutInRisk, GenericInputHasNoBackendDependentBranch)
{
  auto autoware_equivalent = right_cut_in();
  auto ab3dmot_equivalent = right_cut_in();
  const auto first = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), autoware_equivalent, CutInParameters{});
  const auto second = ad_planner::compute_cut_in_risk(
    horizontal_lane(), ego(), ab3dmot_equivalent, CutInParameters{});
  EXPECT_EQ(first.cut_in_candidate, second.cut_in_candidate);
  EXPECT_DOUBLE_EQ(first.predicted_entry_time_s, second.predicted_entry_time_s);
}

ReferenceCorridor corridor()
{
  return ReferenceCorridor{"map", {horizontal_lane()}, 0U};
}

nav_msgs::msg::OccupancyGrid mask(const std::int32_t stamp_sec = 10)
{
  nav_msgs::msg::OccupancyGrid result;
  result.header.frame_id = "base_link";
  result.header.stamp.sec = stamp_sec;
  result.info.width = 1040U;
  result.info.height = 200U;
  result.info.resolution = 0.1F;
  result.info.origin.position.x = -4.0;
  result.info.origin.position.y = -10.0;
  result.info.origin.orientation.w = 1.0;
  result.data.assign(208000U, 0);
  return result;
}

ad_interfaces::msg::DynamicObjectRiskArray risk_array(
  const std::int32_t stamp_sec = 10)
{
  ad_interfaces::msg::DynamicObjectRiskArray result;
  result.header.frame_id = "base_link";
  result.header.stamp.sec = stamp_sec;
  return result;
}

RouteEgoSample route_ego(const std::int64_t stamp_ns = 10'000'000'000LL)
{
  return RouteEgoSample{ego(), stamp_ns};
}

TEST(CutInFrame, ValidZeroObjectArrayPublishesEmptyOutput)
{
  const auto result = ad_planner::build_cut_in_frame(
    risk_array(), mask(), route_ego(), 10'100'000'000LL, std::nullopt,
    corridor(), CutInFrameConfig{});
  EXPECT_TRUE(result.published);
  EXPECT_TRUE(result.output.risks.empty());
  EXPECT_EQ(result.output.header.frame_id, "map");
}

TEST(CutInFrame, RejectsWrongFrameStaleDuplicateBackwardAndMissingContext)
{
  auto risks = risk_array();
  risks.header.frame_id = "odom";
  EXPECT_FALSE(ad_planner::build_cut_in_frame(
    risks, mask(), route_ego(), 10'100'000'000LL, std::nullopt,
    corridor(), CutInFrameConfig{}).published);

  EXPECT_FALSE(ad_planner::build_cut_in_frame(
    risk_array(), mask(), route_ego(), 11'000'000'000LL, std::nullopt,
    corridor(), CutInFrameConfig{}).published);
  EXPECT_FALSE(ad_planner::build_cut_in_frame(
    risk_array(), mask(), route_ego(), 10'100'000'000LL,
    10'000'000'000LL, corridor(), CutInFrameConfig{}).published);
  EXPECT_FALSE(ad_planner::build_cut_in_frame(
    risk_array(), mask(), route_ego(), 10'100'000'000LL,
    11'000'000'000LL, corridor(), CutInFrameConfig{}).published);
  EXPECT_FALSE(ad_planner::build_cut_in_frame(
    risk_array(), mask(), std::nullopt, 10'100'000'000LL, std::nullopt,
    corridor(), CutInFrameConfig{}).published);
}

TEST(CutInFrame, SkipsMalformedObjectAndRejectsMalformedRouteContext)
{
  auto malformed_objects = risk_array();
  malformed_objects.objects.emplace_back();
  malformed_objects.objects.front().predicted_states.emplace_back();
  malformed_objects.objects.front().predicted_states.front().time_from_start.sec = -1;
  const auto skipped = ad_planner::build_cut_in_frame(
    malformed_objects, mask(), route_ego(), 10'100'000'000LL, std::nullopt,
    corridor(), CutInFrameConfig{});
  EXPECT_TRUE(skipped.published);
  EXPECT_TRUE(skipped.output.risks.empty());
  EXPECT_EQ(skipped.rejected_malformed_objects, 1U);

  auto bad_mask = mask();
  bad_mask.data.front() = -1;
  EXPECT_FALSE(ad_planner::build_cut_in_frame(
    risk_array(), bad_mask, route_ego(), 10'100'000'000LL, std::nullopt,
    corridor(), CutInFrameConfig{}).published);
  auto skewed_mask = mask(11);
  EXPECT_FALSE(ad_planner::build_cut_in_frame(
    risk_array(), skewed_mask, route_ego(), 11'100'000'000LL, std::nullopt,
    corridor(), CutInFrameConfig{}).published);
}

}  // namespace
