#include "ad_planner/planning/highway_merge_gap_risk.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/common/local_motion.hpp"
#include "highway_merge_gap_risk_node.hpp"

namespace
{

using ad_planner::compute_highway_merge_gap_risk;
using ad_planner::compute_highway_merge_gap_risks;
using ad_planner::compute_merge_ego_timing;
using ad_planner::HighwayMergeGapRiskResult;
using ad_planner::HighwayMergeParameters;
using ad_planner::MergeEgoState;
using ad_planner::MergeEgoTiming;
using ad_planner::MergeObjectInput;
using ad_planner::MergePredictedStateInput;
using ad_planner::MergeZone;
using ad_planner::Pose2;
using ad_planner::ReferenceLane;
using ad_planner::ReferencePoint;

constexpr double kEgoSpeed = 10.0;

// A straight target corridor along +x from route_s 0 .. length, route_s == x,
// constant half-widths.
ReferenceLane straight_lane(double length = 400.0, double half_width = 1.75)
{
  ReferenceLane lane;
  lane.lane_sequence_id = "route:0";
  lane.source_link_ids = {"L0"};
  for (double x = 0.0; x <= length + 1e-6; x += 2.0) {
    ReferencePoint point;
    point.pose = Pose2{x, 0.0, 0.0};
    point.route_s_m = x;
    point.curvature_inv_m = 0.0;
    point.left_width_m = half_width;
    point.right_width_m = half_width;
    point.speed_limit_mps = 33.3;
    lane.points.push_back(point);
  }
  return lane;
}

// A left-turning arc of radius R and total sweep max_angle starting at the
// origin heading +x, route_s measured along the arc.
ReferenceLane curved_lane(
  double radius = 80.0, double max_angle = M_PI / 2.0, double half_width = 1.75)
{
  ReferenceLane lane;
  lane.lane_sequence_id = "route:0";
  lane.source_link_ids = {"L0"};
  const int steps = 360;
  for (int i = 0; i <= steps; ++i) {
    const double a = max_angle * static_cast<double>(i) / steps;
    ReferencePoint point;
    point.pose = Pose2{radius * std::sin(a), radius * (1.0 - std::cos(a)), a};
    point.route_s_m = radius * a;
    point.curvature_inv_m = 1.0 / radius;
    point.left_width_m = half_width;
    point.right_width_m = half_width;
    point.speed_limit_mps = 20.0;
    lane.points.push_back(point);
  }
  return lane;
}

MergeZone zone(double entry = 150.0, double complete = 250.0)
{
  MergeZone z;
  z.id = "merge";
  z.source_lane_id = "src";
  z.target_lane_id = "route:0";
  z.route_s_zone_entry_m = entry;
  z.route_s_merge_complete_m = complete;
  return z;
}

HighwayMergeParameters params()
{
  return HighwayMergeParameters{};
}

MergeEgoState straight_ego(double route_s_m, double speed = kEgoSpeed)
{
  MergeEgoState ego;
  ego.pose = Pose2{route_s_m, 0.0, 0.0};
  ego.longitudinal_speed_mps = speed;
  ego.route_s_m = route_s_m;
  ego.route_longitudinal_speed_mps = speed;
  return ego;
}

std::array<std::uint8_t, 16U> uuid(std::uint8_t first)
{
  std::array<std::uint8_t, 16U> u{};
  u[0] = first;
  return u;
}

// Object on the straight lane. cur_* is the current map position; each sample is
// (t, map_x, map_y). vx/vy are ego-relative. Ego assumed at (ego_x, 0, yaw 0).
MergeObjectInput straight_object(
  std::uint8_t id, double cur_x, double cur_y, double vx_rel, double vy_rel,
  const std::vector<std::array<double, 3>> & samples, double ego_x,
  double ego_speed = kEgoSpeed)
{
  MergeObjectInput object;
  object.object_id = uuid(id);
  object.classification = 1U;
  object.classification_probability = 0.9F;
  object.existence_probability = 0.9F;
  object.x_rel_m = cur_x - ego_x;
  object.y_rel_m = cur_y;
  object.vx_rel_mps = vx_rel;
  object.vy_rel_mps = vy_rel;
  for (const auto & sample : samples) {
    MergePredictedStateInput state;
    state.time_s = sample[0];
    state.x_rel_m = sample[1] - ego_x - ego_speed * sample[0];
    state.y_rel_m = sample[2];
    object.predicted_states.push_back(state);
  }
  return object;
}

std::vector<std::array<double, 3>> constant_speed_samples(
  double x0, double y, double speed_mps, double dt = 3.0, double horizon = 18.0)
{
  std::vector<std::array<double, 3>> out;
  for (double t = dt; t <= horizon + 1e-6; t += dt) {
    out.push_back({t, x0 + speed_mps * t, y});
  }
  return out;
}

HighwayMergeGapRiskResult one(
  const MergeZone & z, const ReferenceLane & lane, const MergeEgoState & ego,
  const MergeObjectInput & object, const HighwayMergeParameters & p)
{
  const MergeEgoTiming timing = compute_merge_ego_timing(z, ego, p);
  return compute_highway_merge_gap_risk(z, lane, ego, timing, object, p);
}

// --- ego timing -------------------------------------------------------------

TEST(HighwayMergeEgoTiming, ApproachingIsValid)
{
  const auto timing = compute_merge_ego_timing(zone(), straight_ego(100.0), params());
  EXPECT_TRUE(timing.timing_valid);
  EXPECT_FALSE(timing.in_merge_zone_now);
  EXPECT_NEAR(timing.merge_time_s, 150.0 / kEgoSpeed, 1e-6);
  EXPECT_NEAR(timing.route_distance_to_merge_m, 150.0, 1e-6);
  EXPECT_NEAR(timing.route_distance_to_zone_entry_m, 50.0, 1e-6);
}

TEST(HighwayMergeEgoTiming, InsideZoneIsValidAndFlagged)
{
  const auto timing = compute_merge_ego_timing(zone(), straight_ego(200.0), params());
  EXPECT_TRUE(timing.timing_valid);
  EXPECT_TRUE(timing.in_merge_zone_now);
  EXPECT_NEAR(timing.merge_time_s, 50.0 / kEgoSpeed, 1e-6);
}

TEST(HighwayMergeEgoTiming, PastMergeIsInactive)
{
  const auto timing = compute_merge_ego_timing(zone(), straight_ego(260.0), params());
  EXPECT_FALSE(timing.timing_valid);
  EXPECT_EQ(timing.merge_time_s, 0.0);
}

TEST(HighwayMergeEgoTiming, StoppedEgoHoldsInvalid)
{
  auto ego = straight_ego(100.0, 0.0);
  ego.route_longitudinal_speed_mps = 0.0;
  const auto timing = compute_merge_ego_timing(zone(), ego, params());
  EXPECT_FALSE(timing.timing_valid);
}

TEST(HighwayMergeEgoTiming, TooFarIsInactive)
{
  auto p = params();
  p.maximum_ego_approach_distance_m = 30.0;
  const auto timing = compute_merge_ego_timing(zone(), straight_ego(100.0), p);
  EXPECT_FALSE(timing.timing_valid);
}

// --- relevance / seam ------------------------------------------------------

TEST(HighwayMergeRelevance, ObjectOutsideLongitudinalWindowIsNotRelevant)
{
  // In-corridor laterally, but past the front window (250 + 120 = 370).
  const auto lane = straight_lane(500.0);
  const auto object = straight_object(1, 390.0, 0.0, 0.0, 0.0, {}, 100.0);
  const auto risk = one(zone(), lane, straight_ego(100.0), object, params());
  EXPECT_FALSE(risk.relevant_to_merge);
  EXPECT_FALSE(risk.delta_s_at_merge_valid);
  EXPECT_FALSE(risk.longitudinal_gap_closing);
  EXPECT_FALSE(risk.predicted_min_route_gap_valid);
  // A non-relevant object carries only relevance flags + generic facts; the
  // numeric route/speed context is zeroed (it would be window-edge clamped).
  EXPECT_EQ(risk.object_route_s_m, 0.0);
  EXPECT_EQ(risk.delta_s_now_m, 0.0);
  EXPECT_EQ(risk.object_longitudinal_speed_mps, 0.0);
}

TEST(HighwayMergeRelevance, FarObjectClampedToWindowEndIsNotRelevant)
{
  // The production node projects objects onto a station-bounded window of the
  // route. Emulate it: a lane covering only the merge-corridor window, an
  // object well past its end -> clamps to the window end, never relevant, no
  // throw.
  ReferenceLane window;
  window.lane_sequence_id = "route:0";
  window.source_link_ids = {"L0"};
  for (double x = 0.0; x <= 420.0 + 1e-6; x += 2.0) {
    ReferencePoint point;
    point.pose = Pose2{x, 0.0, 0.0};
    point.route_s_m = x;
    point.left_width_m = 1.75;
    point.right_width_m = 1.75;
    window.points.push_back(point);
  }
  // Merge zone at s [150, 250]; object at map x = 900, far beyond the window.
  const auto object = straight_object(1, 900.0, 0.0, 5.0, 0.0, {}, 100.0);
  HighwayMergeGapRiskResult risk;
  EXPECT_NO_THROW(
    risk = one(zone(), window, straight_ego(100.0), object, params()));
  EXPECT_FALSE(risk.relevant_to_merge);
  EXPECT_FALSE(risk.delta_s_at_merge_valid);
  EXPECT_EQ(risk.object_route_s_m, 0.0);
}

TEST(HighwayMergeRelevance, LaterallyOffCorridorNeverPredictedToEnterIsNotRelevant)
{
  const auto lane = straight_lane();
  // y = 6 m -> outside the 1.75 + 0.5 corridor; no predicted samples.
  const auto object = straight_object(1, 220.0, 6.0, 0.0, 0.0, {}, 100.0);
  const auto risk = one(zone(), lane, straight_ego(100.0), object, params());
  EXPECT_FALSE(risk.relevant_to_merge);
  EXPECT_FALSE(risk.object_in_target_corridor_now);
}

TEST(HighwayMergeRelevance, OffCorridorNowButPredictedToMergeInIsRelevant)
{
  const auto lane = straight_lane();
  // Currently 3.9 m left (a tapering ramp), predicted to reach the corridor.
  std::vector<std::array<double, 3>> samples;
  for (double t = 3.0; t <= 18.0 + 1e-6; t += 3.0) {
    const double frac = t / 18.0;
    samples.push_back({t, 200.0 + 10.0 * t, 3.9 * (1.0 - frac)});
  }
  const auto object = straight_object(1, 200.0, 3.9, 0.0, 0.0, samples, 100.0);
  const auto risk = one(zone(), lane, straight_ego(100.0), object, params());
  EXPECT_FALSE(risk.object_in_target_corridor_now);
  EXPECT_TRUE(risk.predicted_to_enter_target_corridor);
  EXPECT_TRUE(risk.predicted_corridor_entry_valid);
  EXPECT_GT(risk.predicted_corridor_entry_time_s, 0.0);
  EXPECT_TRUE(risk.relevant_to_merge);
}

// --- longitudinal ordering at merge --------------------------------------

TEST(HighwayMergeOrdering, ObjectAheadAtMerge)
{
  const auto lane = straight_lane();
  const auto object = straight_object(
    1, 250.0, 0.0, 2.0, 0.0, constant_speed_samples(250.0, 0.0, 12.0), 100.0);
  const auto risk = one(zone(), lane, straight_ego(100.0), object, params());
  ASSERT_TRUE(risk.relevant_to_merge);
  ASSERT_TRUE(risk.delta_s_at_merge_valid);
  EXPECT_TRUE(risk.is_ahead_at_merge);
  EXPECT_FALSE(risk.is_behind_at_merge);
  EXPECT_FALSE(risk.is_alongside_at_merge);
  EXPECT_GT(risk.delta_s_at_merge_m, 0.0);
  EXPECT_TRUE(risk.prediction_covers_merge_time);
}

TEST(HighwayMergeOrdering, ObjectBehindAtMerge)
{
  const auto lane = straight_lane();
  // 30 m ahead now but much slower -> the ego overtakes it before the merge.
  const auto object = straight_object(
    1, 130.0, 0.0, -7.0, 0.0, constant_speed_samples(130.0, 0.0, 3.0), 100.0);
  const auto risk = one(zone(), lane, straight_ego(100.0), object, params());
  ASSERT_TRUE(risk.delta_s_at_merge_valid);
  EXPECT_TRUE(risk.is_behind_at_merge);
  EXPECT_LT(risk.delta_s_at_merge_m, 0.0);
}

TEST(HighwayMergeOrdering, ObjectAlongsideAtMerge)
{
  const auto lane = straight_lane();
  // Ego reaches s=250 in 15 s at 10 m/s; object at 10 m/s from s=100 -> also 250.
  const auto object = straight_object(
    1, 100.0, 0.0, 0.0, 0.0, constant_speed_samples(100.0, 0.0, 10.0), 100.0);
  const auto risk = one(zone(), lane, straight_ego(100.0), object, params());
  ASSERT_TRUE(risk.delta_s_at_merge_valid);
  EXPECT_TRUE(risk.is_alongside_at_merge);
  EXPECT_NEAR(risk.delta_s_at_merge_m, 0.0, params().alongside_longitudinal_band_m);
}

// --- relative speed / closing -------------------------------------------

TEST(HighwayMergeClosing, FasterObjectHasPositiveRelativeSpeed)
{
  const auto lane = straight_lane();
  const auto object = straight_object(1, 200.0, 0.0, 4.0, 0.0, {}, 100.0);
  const auto risk = one(zone(), lane, straight_ego(100.0), object, params());
  EXPECT_NEAR(risk.object_longitudinal_speed_mps, 14.0, 0.3);
  EXPECT_NEAR(risk.relative_longitudinal_speed_mps, 4.0, 0.3);
}

TEST(HighwayMergeClosing, TrailingFasterObjectIsClosingWithCoincidenceTime)
{
  const auto lane = straight_lane();
  // 40 m behind ego (still inside the rear relevance window), closing at 2 m/s.
  const auto object = straight_object(1, 60.0, 0.0, 2.0, 0.0, {}, 100.0);
  const auto risk = one(zone(), lane, straight_ego(100.0), object, params());
  ASSERT_TRUE(risk.relevant_to_merge);
  EXPECT_TRUE(risk.longitudinal_gap_closing);
  EXPECT_NEAR(risk.longitudinal_closing_speed_mps, 2.0, 0.3);
  ASSERT_TRUE(risk.time_to_route_coincidence_valid);
  EXPECT_NEAR(risk.time_to_route_coincidence_s, 20.0, 3.0);
}

TEST(HighwayMergeClosing, TrailingSlowerObjectIsNotClosing)
{
  const auto lane = straight_lane();
  const auto object = straight_object(1, 80.0, 0.0, -2.0, 0.0, {}, 100.0);
  const auto risk = one(zone(), lane, straight_ego(100.0), object, params());
  EXPECT_FALSE(risk.longitudinal_gap_closing);
  EXPECT_FALSE(risk.time_to_route_coincidence_valid);
}

// --- predicted minimum route gap ---------------------------------------

TEST(HighwayMergePredictedGap, MinimumRouteGapCapturesOvertake)
{
  const auto lane = straight_lane();
  // Object behind, faster: passes the ego route station mid-horizon.
  const auto object = straight_object(
    1, 60.0, 0.0, 6.0, 0.0, constant_speed_samples(60.0, 0.0, 16.0), 100.0);
  const auto risk = one(zone(), lane, straight_ego(100.0), object, params());
  ASSERT_TRUE(risk.predicted_min_route_gap_valid);
  EXPECT_LT(risk.predicted_min_route_gap_m, 5.0);
  EXPECT_GT(risk.predicted_min_route_gap_time_s, 0.0);
}

// --- prediction coverage --------------------------------------------------

TEST(HighwayMergeCoverage, ShortPredictionDoesNotCoverMergeTime)
{
  const auto lane = straight_lane();
  // Ego merge time = 15 s; horizon only 6 s.
  const auto object = straight_object(
    1, 250.0, 0.0, 0.0, 0.0, constant_speed_samples(250.0, 0.0, 10.0, 2.0, 6.0),
    100.0);
  const auto risk = one(zone(), lane, straight_ego(100.0), object, params());
  ASSERT_TRUE(risk.relevant_to_merge);
  EXPECT_FALSE(risk.prediction_covers_merge_time);
  EXPECT_TRUE(risk.delta_s_at_merge_valid);  // constant-speed extrapolation
  EXPECT_NEAR(risk.prediction_horizon_s, 6.0, 1e-6);
}

TEST(HighwayMergeCoverage, NoPredictionStillExtrapolatesDeltaSAtMerge)
{
  const auto lane = straight_lane();
  const auto object = straight_object(1, 250.0, 0.0, 0.0, 0.0, {}, 100.0);
  const auto risk = one(zone(), lane, straight_ego(100.0), object, params());
  EXPECT_FALSE(risk.prediction_covers_merge_time);
  EXPECT_TRUE(risk.delta_s_at_merge_valid);
  EXPECT_FALSE(risk.predicted_min_route_gap_valid);
}

// --- curved geometry: base_link x misleads, route-s is correct ----------

TEST(HighwayMergeCurved, RouteStationOrderingBeatsBaseLinkX)
{
  // A sharply curving corridor: an object well ahead ALONG THE ROUTE ends up
  // behind the ego in base_link (negative x_rel). base_link x would misclassify
  // it as a rear object; the route-station projection gets it right.
  const double radius = 30.0;
  const double total = 240.0 * M_PI / 180.0;
  const auto lane = curved_lane(radius, total);
  const auto pose_at = [&](double s) {
      const double a = s / radius;
      return Pose2{radius * std::sin(a), radius * (1.0 - std::cos(a)), a};
    };
  const double ego_s = 12.0;
  const Pose2 ego_pose = pose_at(ego_s);
  MergeEgoState ego;
  ego.pose = ego_pose;
  ego.longitudinal_speed_mps = kEgoSpeed;
  ego.route_s_m = ego_s;
  ego.route_longitudinal_speed_mps = kEgoSpeed;

  const Pose2 ahead = pose_at(108.0);   // far ahead along the route
  const double c = std::cos(ego_pose.yaw_rad);
  const double s = std::sin(ego_pose.yaw_rad);
  const double dx = ahead.x - ego_pose.x;
  const double dy = ahead.y - ego_pose.y;
  const double x_rel = c * dx + s * dy;
  const double y_rel = -s * dx + c * dy;

  MergeObjectInput object;
  object.object_id = uuid(1);
  object.classification = 1U;
  object.classification_probability = 0.9F;
  object.existence_probability = 0.9F;
  object.x_rel_m = x_rel;
  object.y_rel_m = y_rel;

  MergeZone z = zone(5.0, 115.0);
  const auto p = params();
  const MergeEgoTiming timing = compute_merge_ego_timing(z, ego, p);
  const auto risk = compute_highway_merge_gap_risk(z, lane, ego, timing, object, p);

  // base_link x says "behind", route station says "ahead".
  EXPECT_LT(x_rel, 0.0);
  EXPECT_TRUE(risk.relevant_to_merge);
  EXPECT_GT(risk.delta_s_now_m, 0.0);
  EXPECT_NEAR(risk.object_route_s_m, 108.0, 3.0);
}

// --- array summary / arbitration --------------------------------------

TEST(HighwayMergeArray, NearestTrailingIsClosestBehindNotFarthest)
{
  const auto lane = straight_lane();
  const auto z = zone();
  const auto ego = straight_ego(100.0);
  // Ego reaches s=250 at t=15. Object 2 ends near s=235 (delta -15); object 1
  // ends near s=195 (delta -55). Nearest-behind must be object 2.
  std::vector<MergeObjectInput> objects{
    straight_object(2, 145.0, 0.0, -4.0, 0.0,
      constant_speed_samples(145.0, 0.0, 6.0), 100.0),
    straight_object(1, 120.0, 0.0, -5.0, 0.0,
      constant_speed_samples(120.0, 0.0, 5.0), 100.0)};
  const auto out = compute_highway_merge_gap_risks(z, lane, ego, objects, params());
  ASSERT_TRUE(out.nearest_trailing_valid);
  EXPECT_EQ(out.nearest_trailing_object_id[0], 2U);
  EXPECT_GT(out.nearest_trailing_delta_s_at_merge_m, -25.0);
}

TEST(HighwayMergeArray, MergeGapIsLeadingMinusTrailing)
{
  const auto lane = straight_lane();
  const auto z = zone();
  const auto ego = straight_ego(100.0);
  std::vector<MergeObjectInput> objects{
    // Near-stationary vehicle just past the merge point -> ends near s=285.
    straight_object(1, 270.0, 0.0, -9.0, 0.0,
      constant_speed_samples(270.0, 0.0, 1.0), 100.0),
    // Trailing vehicle -> ends near s=225.
    straight_object(2, 150.0, 0.0, -5.0, 0.0,
      constant_speed_samples(150.0, 0.0, 5.0), 100.0)};
  const auto out = compute_highway_merge_gap_risks(z, lane, ego, objects, params());
  ASSERT_TRUE(out.nearest_leading_valid);
  ASSERT_TRUE(out.nearest_trailing_valid);
  ASSERT_TRUE(out.merge_gap_valid);
  EXPECT_NEAR(
    out.merge_gap_m,
    out.nearest_leading_delta_s_at_merge_m - out.nearest_trailing_delta_s_at_merge_m,
    1e-4);
  EXPECT_GT(out.merge_gap_m, 0.0);
}

TEST(HighwayMergeArray, ZeroObjectsStillComputesEgoTiming)
{
  const auto lane = straight_lane();
  const auto out = compute_highway_merge_gap_risks(
    zone(), lane, straight_ego(100.0), {}, params());
  EXPECT_TRUE(out.ego.timing_valid);
  EXPECT_EQ(out.relevant_object_count, 0U);
  EXPECT_FALSE(out.merge_gap_valid);
  EXPECT_FALSE(out.nearest_leading_valid);
}

// --- robustness -------------------------------------------------------

TEST(HighwayMergeRobustness, MalformedObjectIsRejectedOthersSurvive)
{
  const auto lane = straight_lane();
  MergeObjectInput bad = straight_object(9, 200.0, 0.0, 0.0, 0.0, {}, 100.0);
  bad.vx_rel_mps = std::numeric_limits<double>::quiet_NaN();
  std::vector<MergeObjectInput> objects{
    bad,
    straight_object(1, 200.0, 0.0, 0.0, 0.0, {}, 100.0)};
  const auto out = compute_highway_merge_gap_risks(
    zone(), lane, straight_ego(100.0), objects, params());
  EXPECT_EQ(out.rejected_malformed_objects, 1U);
  EXPECT_EQ(out.objects.size(), 1U);
}

TEST(HighwayMergeRobustness, MaximumObjectsBudget)
{
  const auto lane = straight_lane();
  auto p = params();
  p.maximum_objects = 2U;
  std::vector<MergeObjectInput> objects;
  for (std::uint8_t i = 1; i <= 5; ++i) {
    objects.push_back(straight_object(i, 200.0, 0.0, 0.0, 0.0, {}, 100.0));
  }
  const auto out = compute_highway_merge_gap_risks(
    zone(), lane, straight_ego(100.0), objects, p);
  EXPECT_EQ(out.objects.size(), 2U);
  EXPECT_EQ(out.rejected_over_budget, 3U);
}

TEST(HighwayMergeRobustness, DeterministicAndFinite)
{
  const auto lane = straight_lane();
  const auto z = zone();
  const auto ego = straight_ego(100.0);
  std::vector<MergeObjectInput> objects{
    straight_object(1, 220.0, 0.5, 3.0, -0.2,
      constant_speed_samples(220.0, 0.5, 13.0), 100.0),
    straight_object(2, 80.0, -0.4, 1.0, 0.1,
      constant_speed_samples(80.0, -0.4, 11.0), 100.0)};
  const auto a = compute_highway_merge_gap_risks(z, lane, ego, objects, params());
  const auto b = compute_highway_merge_gap_risks(z, lane, ego, objects, params());
  ASSERT_EQ(a.objects.size(), b.objects.size());
  for (std::size_t i = 0; i < a.objects.size(); ++i) {
    EXPECT_EQ(a.objects[i].delta_s_at_merge_m, b.objects[i].delta_s_at_merge_m);
    EXPECT_TRUE(std::isfinite(a.objects[i].delta_s_at_merge_m));
    EXPECT_TRUE(std::isfinite(a.objects[i].delta_s_now_m));
    EXPECT_TRUE(std::isfinite(a.objects[i].object_longitudinal_speed_mps));
    EXPECT_TRUE(std::isfinite(a.objects[i].longitudinal_closing_speed_mps));
    EXPECT_TRUE(std::isfinite(a.objects[i].predicted_min_route_gap_m));
    EXPECT_GE(a.objects[i].predicted_min_route_gap_m, 0.0);
  }
}

TEST(HighwayMergeRobustness, InvalidZoneThrows)
{
  MergeZone z = zone();
  z.route_s_merge_complete_m = z.route_s_zone_entry_m - 1.0;
  EXPECT_THROW(z.validated(), std::invalid_argument);
}

// --- backend independence: velocity direction, never orientation --------

TEST(HighwayMergeBackend, ObjectOrientationFieldNeverConsulted)
{
  // The core reads only x_rel/y_rel/vx_rel/vy_rel + predicted centroids; there
  // is no orientation input at all. Two objects with identical kinematics
  // produce identical facts regardless of any notional heading.
  const auto lane = straight_lane();
  const auto z = zone();
  const auto ego = straight_ego(100.0);
  const auto object = straight_object(
    1, 210.0, 0.0, 2.0, 0.0, constant_speed_samples(210.0, 0.0, 12.0), 100.0);
  const auto r1 = one(z, lane, ego, object, params());
  const auto r2 = one(z, lane, ego, object, params());
  EXPECT_EQ(r1.delta_s_at_merge_m, r2.delta_s_at_merge_m);
  EXPECT_EQ(r1.object_longitudinal_speed_mps, r2.object_longitudinal_speed_mps);
}

// --- node frame contract ------------------------------------------------

using ad_planner::build_highway_merge_frame;
using ad_planner::MergeEgoSample;
using ad_planner::MergeFrameConfig;

ad_interfaces::msg::DynamicObjectRiskArray risk_array(
  std::int64_t stamp_ns, const std::string & frame = "base_link")
{
  ad_interfaces::msg::DynamicObjectRiskArray array;
  array.header.stamp.sec = static_cast<std::int32_t>(stamp_ns / 1'000'000'000LL);
  array.header.stamp.nanosec = static_cast<std::uint32_t>(stamp_ns % 1'000'000'000LL);
  array.header.frame_id = frame;
  return array;
}

MergeEgoSample ego_sample(std::int64_t stamp_ns)
{
  MergeEgoSample s;
  s.map_pose = Pose2{100.0, 0.0, 0.0};
  s.longitudinal_speed_mps = kEgoSpeed;
  s.stamp_ns = stamp_ns;
  return s;
}

TEST(HighwayMergeFrame, ValidEmptyFramePublishes)
{
  const std::int64_t stamp = 1'000'000'000LL;
  const auto result = build_highway_merge_frame(
    risk_array(stamp), ego_sample(stamp), stamp, std::nullopt, "map",
    straight_lane(), straight_lane(), zone(), MergeFrameConfig{});
  EXPECT_TRUE(result.published);
  EXPECT_EQ(result.output.objects.size(), 0U);
  EXPECT_EQ(result.output.header.frame_id, "map");
  EXPECT_EQ(result.output.merge_zone_id, "merge");
}

TEST(HighwayMergeFrame, WrongFrameIsRejected)
{
  const std::int64_t stamp = 1'000'000'000LL;
  const auto result = build_highway_merge_frame(
    risk_array(stamp, "odom"), ego_sample(stamp), stamp, std::nullopt,
    "map", straight_lane(), straight_lane(), zone(), MergeFrameConfig{});
  EXPECT_FALSE(result.published);
}

TEST(HighwayMergeFrame, MissingOdometryIsRejected)
{
  const std::int64_t stamp = 1'000'000'000LL;
  const auto result = build_highway_merge_frame(
    risk_array(stamp), std::nullopt, stamp, std::nullopt, "map",
    straight_lane(), straight_lane(), zone(), MergeFrameConfig{});
  EXPECT_FALSE(result.published);
}

TEST(HighwayMergeFrame, FutureStampIsRejected)
{
  const std::int64_t stamp = 5'000'000'000LL;
  const auto result = build_highway_merge_frame(
    risk_array(stamp), ego_sample(stamp), 1'000'000'000LL, std::nullopt,
    "map", straight_lane(), straight_lane(), zone(), MergeFrameConfig{});
  EXPECT_FALSE(result.published);
}

TEST(HighwayMergeFrame, StaleStampIsRejected)
{
  const std::int64_t stamp = 1'000'000'000LL;
  const auto result = build_highway_merge_frame(
    risk_array(stamp), ego_sample(stamp), 3'000'000'000LL, std::nullopt,
    "map", straight_lane(), straight_lane(), zone(), MergeFrameConfig{});
  EXPECT_FALSE(result.published);
}

TEST(HighwayMergeFrame, DuplicateOrBackwardStampIsRejected)
{
  const std::int64_t stamp = 2'000'000'000LL;
  const auto result = build_highway_merge_frame(
    risk_array(stamp), ego_sample(stamp), stamp, stamp, "map",
    straight_lane(), straight_lane(), zone(), MergeFrameConfig{});
  EXPECT_FALSE(result.published);
}

TEST(HighwayMergeFrame, InvalidCorridorIsRejected)
{
  const std::int64_t stamp = 1'000'000'000LL;
  ReferenceLane degenerate;
  const auto result = build_highway_merge_frame(
    risk_array(stamp), ego_sample(stamp), stamp, std::nullopt, "map",
    degenerate, degenerate, zone(), MergeFrameConfig{});
  EXPECT_FALSE(result.published);
}

}  // namespace
