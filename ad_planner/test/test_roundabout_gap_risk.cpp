#include "ad_planner/planning/roundabout_gap_risk.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace
{

using ad_planner::compute_roundabout_ego_timing;
using ad_planner::compute_roundabout_gap_risk;
using ad_planner::compute_roundabout_gap_risks;
using ad_planner::ConflictPoint2;
using ad_planner::distance_to_conflict_polygon;
using ad_planner::point_in_conflict_polygon;
using ad_planner::Pose2;
using ad_planner::RoundaboutConflictZone;
using ad_planner::RoundaboutEgoState;
using ad_planner::RoundaboutEgoTiming;
using ad_planner::RoundaboutGapParameters;
using ad_planner::RoundaboutObjectInput;
using ad_planner::RoundaboutPredictedStateInput;

// A 20 x 20 conflict polygon at [10, 30] x [10, 30]. All test points are kept
// strictly interior or exterior (never on an edge or vertex).
RoundaboutConflictZone square_zone(double s0 = 100.0, double s1 = 120.0)
{
  RoundaboutConflictZone zone;
  zone.id = "square";
  zone.polygon_m = {
    ConflictPoint2{10.0, 10.0}, ConflictPoint2{30.0, 10.0},
    ConflictPoint2{30.0, 30.0}, ConflictPoint2{10.0, 30.0}};
  zone.route_s_enter_m = s0;
  zone.route_s_exit_m = s1;
  return zone;
}

RoundaboutGapParameters params()
{
  return RoundaboutGapParameters{};
}

// Ego at the map origin, heading +x: an object's body-relative coordinates plus
// ego_speed * time equal its absolute map coordinates.
RoundaboutEgoState origin_ego(double route_s_m, double speed_mps)
{
  RoundaboutEgoState ego;
  ego.pose = Pose2{0.0, 0.0, 0.0};
  ego.longitudinal_speed_mps = speed_mps;
  ego.route_s_m = route_s_m;
  return ego;
}

RoundaboutEgoTiming ego_interval(double e0, double e1)
{
  RoundaboutEgoTiming timing;
  timing.entry_valid = true;
  timing.entry_time_s = e0;
  timing.exit_valid = true;
  timing.exit_time_s = e1;
  return timing;
}

// Builds an object whose discrete predicted centroids are given directly as
// absolute map positions (t, x_map, y_map), converted to the CV-ego-relative
// form the core expects.
RoundaboutObjectInput map_object(
  std::uint8_t id, double x_now_map, double y_now_map, double ego_speed_mps,
  std::vector<std::array<double, 3>> map_samples)
{
  RoundaboutObjectInput object;
  object.object_id = {id};
  object.classification = 1U;
  object.classification_probability = 0.9F;
  object.existence_probability = 0.9F;
  object.x_rel_m = x_now_map;   // ego at origin heading +x
  object.y_rel_m = y_now_map;
  for (const auto & sample : map_samples) {
    object.predicted_states.push_back(RoundaboutPredictedStateInput{
      sample[0], sample[1] - ego_speed_mps * sample[0], sample[2]});
  }
  return object;
}

// ---- point-in-polygon / distance --------------------------------------

TEST(RoundaboutGeometry, PointInsideAndOutsidePolygon)
{
  const auto zone = square_zone();
  EXPECT_TRUE(point_in_conflict_polygon(zone.polygon_m, 20.0, 20.0));
  EXPECT_FALSE(point_in_conflict_polygon(zone.polygon_m, 50.0, 20.0));
  EXPECT_FALSE(point_in_conflict_polygon(zone.polygon_m, 0.0, 20.0));
}

TEST(RoundaboutGeometry, DistanceIsZeroInsideAndPositiveOutside)
{
  const auto zone = square_zone();
  EXPECT_DOUBLE_EQ(distance_to_conflict_polygon(zone.polygon_m, 20.0, 20.0), 0.0);
  EXPECT_NEAR(distance_to_conflict_polygon(zone.polygon_m, 35.0, 20.0), 5.0, 1e-9);
  EXPECT_NEAR(
    distance_to_conflict_polygon(zone.polygon_m, 6.0, 7.0), 5.0, 1e-9);
}

TEST(RoundaboutGeometry, NonFinitePointIsOutside)
{
  const auto zone = square_zone();
  EXPECT_FALSE(point_in_conflict_polygon(
      zone.polygon_m, std::numeric_limits<double>::quiet_NaN(), 1.0));
}

// ---- ego timing (Phase 8/9) -----------------------------------------

TEST(RoundaboutEgoTimingTest, ApproachingEgoHasConstantSpeedEta)
{
  const auto ego = origin_ego(90.0, 10.0);
  const auto timing =
    compute_roundabout_ego_timing(square_zone(100.0, 120.0), ego, params());
  EXPECT_TRUE(timing.entry_valid);
  EXPECT_TRUE(timing.exit_valid);
  EXPECT_NEAR(timing.entry_time_s, 1.0, 1e-9);
  EXPECT_NEAR(timing.exit_time_s, 3.0, 1e-9);
  EXPECT_NEAR(timing.route_distance_to_entry_m, 10.0, 1e-9);
  EXPECT_FALSE(timing.in_conflict_now);
}

TEST(RoundaboutEgoTimingTest, StoppedEgoHasNoArrivalEta)
{
  const auto ego = origin_ego(90.0, 0.1);   // below the 0.5 epsilon
  const auto timing = compute_roundabout_ego_timing(square_zone(), ego, params());
  EXPECT_FALSE(timing.entry_valid);
  EXPECT_FALSE(timing.exit_valid);
}

TEST(RoundaboutEgoTimingTest, EgoInsideZoneReportsZeroEntryTime)
{
  RoundaboutEgoState ego;
  ego.pose = Pose2{20.0, 20.0, 0.0};   // inside the polygon
  ego.longitudinal_speed_mps = 8.0;
  ego.route_s_m = 110.0;
  const auto timing =
    compute_roundabout_ego_timing(square_zone(100.0, 120.0), ego, params());
  EXPECT_TRUE(timing.in_conflict_now);
  EXPECT_TRUE(timing.entry_valid);
  EXPECT_NEAR(timing.entry_time_s, 0.0, 1e-9);
  EXPECT_NEAR(timing.exit_time_s, (120.0 - 110.0) / 8.0, 1e-9);
}

TEST(RoundaboutEgoTimingTest, EgoPastConflictExitIsInvalid)
{
  const auto ego = origin_ego(130.0, 8.0);
  const auto timing =
    compute_roundabout_ego_timing(square_zone(100.0, 120.0), ego, params());
  EXPECT_FALSE(timing.entry_valid);
}

TEST(RoundaboutEgoTimingTest, EgoFarBeyondApproachBoundIsInvalid)
{
  const auto ego = origin_ego(-500.0, 8.0);
  const auto timing =
    compute_roundabout_ego_timing(square_zone(100.0, 120.0), ego, params());
  EXPECT_FALSE(timing.entry_valid);
}

// ---- object conflict entry / exit (Phase 10/11/12) ------------------

TEST(RoundaboutObjectTiming, ObjectEntersConflictInFuture)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 100.0, 100.0, 0.0, {
    {{1.0, 45.0, 20.0}}, {{2.0, 20.0, 20.0}}, {{3.0, 22.0, 20.0}},
    {{4.0, 0.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(10.0, 11.0), object, params());
  EXPECT_TRUE(risk.relevant_to_conflict);
  EXPECT_FALSE(risk.object_in_conflict_now);
  EXPECT_TRUE(risk.object_entry_valid);
  EXPECT_NEAR(risk.object_entry_time_s, 2.0, 1e-9);
  EXPECT_TRUE(risk.object_exit_valid);
  EXPECT_NEAR(risk.object_exit_time_s, 4.0, 1e-9);
}

TEST(RoundaboutObjectTiming, ObjectAlreadyInsideConflict)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 20.0, 20.0, 0.0, {
    {{1.0, 22.0, 20.0}}, {{2.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(10.0, 11.0), object, params());
  EXPECT_TRUE(risk.object_in_conflict_now);
  EXPECT_TRUE(risk.object_entry_valid);
  EXPECT_NEAR(risk.object_entry_time_s, 0.0, 1e-9);
  EXPECT_TRUE(risk.object_exit_valid);
  EXPECT_NEAR(risk.object_exit_time_s, 2.0, 1e-9);
}

TEST(RoundaboutObjectTiming, ObjectNeverEntersConflict)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 100.0, 100.0, 0.0, {
    {{1.0, 100.0, 100.0}}, {{2.0, 120.0, 100.0}}, {{3.0, 140.0, 100.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(1.0, 2.0), object, params());
  EXPECT_FALSE(risk.relevant_to_conflict);
  EXPECT_FALSE(risk.object_entry_valid);
  EXPECT_FALSE(risk.temporal_gap_valid);
  EXPECT_GT(risk.object_map_distance_to_conflict_m, 0.0);
}

TEST(RoundaboutObjectTiming, OnlyFirstContiguousOccupancyIntervalIsReported)
{
  // Inside now, leaves at t = 2, re-enters at t = 4. The first-interval fields
  // report only [0, 2]; the re-entry is represented only in the all-interval
  // summary (later_reentry_detected / predicted_conflict_interval_count).
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 20.0, 20.0, 0.0, {
    {{1.0, 22.0, 20.0}}, {{2.0, 45.0, 20.0}}, {{3.0, 46.0, 20.0}},
    {{4.0, 20.0, 20.0}}, {{5.0, 21.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(10.0, 11.0), object, params());
  EXPECT_TRUE(risk.object_in_conflict_now);
  EXPECT_TRUE(risk.object_entry_valid);
  EXPECT_NEAR(risk.object_entry_time_s, 0.0, 1e-9);
  EXPECT_TRUE(risk.object_exit_valid);
  EXPECT_NEAR(risk.object_exit_time_s, 2.0, 1e-9);
  // First-interval fields unchanged, but the summary still sees the re-entry.
  EXPECT_EQ(risk.predicted_conflict_interval_count, 2U);
  EXPECT_TRUE(risk.later_reentry_detected);
}

// ---- all-interval summary (multi-interval, policy-consumer contract) ----

TEST(RoundaboutMultiInterval, SingleIntervalMatchesFirstIntervalFacts)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{1.0, 60.0, 20.0}}, {{2.0, 20.0, 20.0}}, {{3.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(6.0, 7.0), object, params());
  EXPECT_EQ(risk.predicted_conflict_interval_count, 1U);
  EXPECT_FALSE(risk.later_reentry_detected);
  EXPECT_FALSE(risk.any_occupancy_overlap);
  EXPECT_TRUE(risk.minimum_temporal_gap_valid);
  EXPECT_NEAR(risk.minimum_temporal_gap_s, risk.temporal_gap_s, 1e-9);
  EXPECT_NEAR(risk.minimum_temporal_gap_s, 3.0, 1e-9);  // ego 6 - object exit 3
}

TEST(RoundaboutMultiInterval, FirstIntervalClearsButSecondOverlapsEgo)
{
  // The central counterexample. ego [4, 6]; object occupies [1, 2] then [5, 7].
  // First-interval facts say "clear, gap 2" -- the summary must say "overlap".
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}, {{3.0, 46.0, 20.0}},
    {{4.0, 47.0, 20.0}}, {{5.0, 20.0, 20.0}}, {{6.0, 21.0, 20.0}},
    {{7.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(4.0, 6.0), object, params());
  // First-interval fields (legacy diagnostics): clear, gap 2, no overlap.
  EXPECT_NEAR(risk.object_entry_time_s, 1.0, 1e-9);
  EXPECT_NEAR(risk.object_exit_time_s, 2.0, 1e-9);
  EXPECT_FALSE(risk.occupancy_overlap);
  EXPECT_NEAR(risk.temporal_gap_s, 2.0, 1e-9);
  // All-interval summary: the re-entry overlaps the ego window.
  EXPECT_EQ(risk.predicted_conflict_interval_count, 2U);
  EXPECT_TRUE(risk.later_reentry_detected);
  EXPECT_TRUE(risk.any_occupancy_overlap);
  EXPECT_TRUE(risk.minimum_temporal_gap_valid);
  EXPECT_NEAR(risk.minimum_temporal_gap_s, 0.0, 1e-9);
  EXPECT_TRUE(risk.prediction_covers_ego_exit);
}

TEST(RoundaboutMultiInterval, TwoSeparatedIntervalsReportMinimumGap)
{
  // ego [4, 5]; object [1, 2] (gap 2) and [8, 9] (gap 3) -> minimum 2.
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}, {{3.0, 46.0, 20.0}},
    {{7.0, 47.0, 20.0}}, {{8.0, 20.0, 20.0}}, {{9.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(4.0, 5.0), object, params());
  EXPECT_EQ(risk.predicted_conflict_interval_count, 2U);
  EXPECT_TRUE(risk.later_reentry_detected);
  EXPECT_FALSE(risk.any_occupancy_overlap);
  EXPECT_TRUE(risk.minimum_temporal_gap_valid);
  EXPECT_NEAR(risk.minimum_temporal_gap_s, 2.0, 1e-9);
}

TEST(RoundaboutMultiInterval, IntervalsOnBothSidesOfEgoNeverOverlap)
{
  // ego [4, 5]; object [1, 2] before and [7, 8] after -> gaps 2 and 2, no
  // overlap even though a later interval exists.
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}, {{6.0, 46.0, 20.0}},
    {{7.0, 20.0, 20.0}}, {{8.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(4.0, 5.0), object, params());
  EXPECT_EQ(risk.predicted_conflict_interval_count, 2U);
  EXPECT_TRUE(risk.later_reentry_detected);
  EXPECT_FALSE(risk.any_occupancy_overlap);
  EXPECT_NEAR(risk.minimum_temporal_gap_s, 2.0, 1e-9);
}

TEST(RoundaboutMultiInterval, FirstIntervalOverlapsSecondIrrelevant)
{
  // ego [1, 3]; object [2, 4] (overlap) then [8, 9].
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{2.0, 20.0, 20.0}}, {{4.0, 45.0, 20.0}}, {{7.0, 46.0, 20.0}},
    {{8.0, 20.0, 20.0}}, {{9.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(1.0, 3.0), object, params());
  EXPECT_TRUE(risk.occupancy_overlap);
  EXPECT_TRUE(risk.any_occupancy_overlap);
  EXPECT_NEAR(risk.minimum_temporal_gap_s, 0.0, 1e-9);
  EXPECT_EQ(risk.predicted_conflict_interval_count, 2U);
}

TEST(RoundaboutMultiInterval, BothIntervalsBeforeEgo)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}, {{3.0, 20.0, 20.0}},
    {{4.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(6.0, 7.0), object, params());
  EXPECT_EQ(risk.predicted_conflict_interval_count, 2U);
  EXPECT_FALSE(risk.any_occupancy_overlap);
  EXPECT_NEAR(risk.minimum_temporal_gap_s, 2.0, 1e-9);  // 6 - 4
}

TEST(RoundaboutMultiInterval, BothIntervalsAfterEgo)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{4.0, 20.0, 20.0}}, {{5.0, 45.0, 20.0}}, {{7.0, 20.0, 20.0}},
    {{8.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(1.0, 2.0), object, params());
  EXPECT_EQ(risk.predicted_conflict_interval_count, 2U);
  EXPECT_FALSE(risk.any_occupancy_overlap);
  EXPECT_NEAR(risk.minimum_temporal_gap_s, 2.0, 1e-9);  // 4 - 2
}

TEST(RoundaboutMultiInterval, TouchingBoundaryIsNotOverlapInSummary)
{
  // object [1, 2], ego [2, 3]: touch at t = 2 -> not overlap, min gap 0.
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(2.0, 3.0), object, params());
  EXPECT_FALSE(risk.any_occupancy_overlap);
  EXPECT_TRUE(risk.minimum_temporal_gap_valid);
  EXPECT_NEAR(risk.minimum_temporal_gap_s, 0.0, 1e-9);
}

TEST(RoundaboutMultiInterval, CurrentInsideExitsThenReenters)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 20.0, 20.0, 0.0, {
    {{1.0, 22.0, 20.0}}, {{2.0, 45.0, 20.0}}, {{3.0, 46.0, 20.0}},
    {{4.0, 20.0, 20.0}}, {{5.0, 21.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(3.5, 6.0), object, params());
  EXPECT_TRUE(risk.object_in_conflict_now);
  EXPECT_EQ(risk.predicted_conflict_interval_count, 2U);
  EXPECT_TRUE(risk.later_reentry_detected);
  // First interval [0, 2] clears before ego (3.5); the re-entry [4, open]
  // overlaps the ego window.
  EXPECT_FALSE(risk.occupancy_overlap);
  EXPECT_TRUE(risk.any_occupancy_overlap);
  EXPECT_NEAR(risk.minimum_temporal_gap_s, 0.0, 1e-9);
}

TEST(RoundaboutMultiInterval, OpenFinalIntervalAfterEgoUsesEntryGap)
{
  // Object enters at 3 and is still inside at the last sample (open interval).
  // ego [1, 2] -> the open interval is entirely after ego, gap = 3 - 2 = 1.
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{1.0, 60.0, 20.0}}, {{3.0, 20.0, 20.0}}, {{4.0, 21.0, 20.0}},
    {{5.0, 22.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(1.0, 2.0), object, params());
  EXPECT_FALSE(risk.object_exit_valid);
  EXPECT_EQ(risk.predicted_conflict_interval_count, 1U);
  EXPECT_FALSE(risk.any_occupancy_overlap);
  EXPECT_TRUE(risk.minimum_temporal_gap_valid);
  EXPECT_NEAR(risk.minimum_temporal_gap_s, 1.0, 1e-9);
}

TEST(RoundaboutMultiInterval, NoConflictIntervalHasNoSummary)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 100.0, 100.0, 0.0, {
    {{1.0, 100.0, 100.0}}, {{2.0, 120.0, 100.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(1.0, 2.0), object, params());
  EXPECT_EQ(risk.predicted_conflict_interval_count, 0U);
  EXPECT_FALSE(risk.later_reentry_detected);
  EXPECT_FALSE(risk.any_occupancy_overlap);
  EXPECT_FALSE(risk.minimum_temporal_gap_valid);
  EXPECT_DOUBLE_EQ(risk.minimum_temporal_gap_s, 0.0);
}

TEST(RoundaboutMultiInterval, PredictionHorizonShorterThanEgoExitIsNotCovered)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}, {{4.0, 46.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(4.0, 5.5), object, params());
  EXPECT_NEAR(risk.prediction_horizon_s, 4.0, 1e-9);
  EXPECT_FALSE(risk.prediction_covers_ego_exit);
}

TEST(RoundaboutMultiInterval, PredictionHorizonExactlyReachesEgoExitIsCovered)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}, {{5.5, 46.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(4.0, 5.5), object, params());
  EXPECT_NEAR(risk.prediction_horizon_s, 5.5, 1e-9);
  EXPECT_TRUE(risk.prediction_covers_ego_exit);
}

TEST(RoundaboutMultiInterval, PredictionHorizonExceedsEgoExitIsCovered)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}, {{9.0, 46.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(4.0, 5.5), object, params());
  EXPECT_NEAR(risk.prediction_horizon_s, 9.0, 1e-9);
  EXPECT_TRUE(risk.prediction_covers_ego_exit);
}

TEST(RoundaboutMultiInterval, StoppedEgoLeavesAggregatesInvalidButIntervalsCounted)
{
  const auto ego = origin_ego(90.0, 0.0);
  RoundaboutEgoTiming no_ego;   // entry_valid / exit_valid false
  const auto object = map_object(1U, 20.0, 20.0, 0.0, {
    {{1.0, 22.0, 20.0}}, {{2.0, 45.0, 20.0}}, {{3.0, 20.0, 20.0}},
    {{4.0, 45.0, 20.0}}});
  const auto risk =
    compute_roundabout_gap_risk(square_zone(), ego, no_ego, object, params());
  EXPECT_EQ(risk.predicted_conflict_interval_count, 2U);
  EXPECT_TRUE(risk.later_reentry_detected);
  EXPECT_FALSE(risk.any_occupancy_overlap);
  EXPECT_FALSE(risk.minimum_temporal_gap_valid);
  EXPECT_FALSE(risk.prediction_covers_ego_exit);
  EXPECT_NEAR(risk.prediction_horizon_s, 4.0, 1e-9);
}

TEST(RoundaboutMultiInterval, DeterministicSummaryOnRepeatedInput)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}, {{3.0, 46.0, 20.0}},
    {{5.0, 20.0, 20.0}}, {{6.0, 21.0, 20.0}}, {{7.0, 45.0, 20.0}}});
  const auto a = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(4.0, 6.0), object, params());
  const auto b = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(4.0, 6.0), object, params());
  EXPECT_EQ(a.predicted_conflict_interval_count, b.predicted_conflict_interval_count);
  EXPECT_EQ(a.any_occupancy_overlap, b.any_occupancy_overlap);
  EXPECT_EQ(a.minimum_temporal_gap_s, b.minimum_temporal_gap_s);
  EXPECT_EQ(a.prediction_horizon_s, b.prediction_horizon_s);
  EXPECT_EQ(a.prediction_covers_ego_exit, b.prediction_covers_ego_exit);
}

TEST(RoundaboutMultiInterval, AllSummaryOutputsFinite)
{
  const auto ego = origin_ego(90.0, 0.0);
  const std::vector<RoundaboutEgoTiming> egos = {
    ego_interval(4.0, 6.0), RoundaboutEgoTiming{}};
  const std::vector<RoundaboutObjectInput> objects = {
    map_object(1U, 60.0, 20.0, 0.0, {
      {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}, {{5.0, 20.0, 20.0}},
      {{7.0, 45.0, 20.0}}}),
    map_object(2U, 20.0, 20.0, 0.0, {{{1.0, 21.0, 20.0}}}),
    map_object(3U, 100.0, 100.0, 0.0, {{{1.0, 100.0, 100.0}}})};
  for (const auto & timing : egos) {
    for (const auto & object : objects) {
      const auto risk = compute_roundabout_gap_risk(
        square_zone(), ego, timing, object, params());
      EXPECT_TRUE(std::isfinite(risk.minimum_temporal_gap_s));
      EXPECT_TRUE(std::isfinite(risk.prediction_horizon_s));
      EXPECT_GE(risk.minimum_temporal_gap_s, 0.0);
    }
  }
}

TEST(RoundaboutObjectTiming, ObjectStillInsideAtHorizonEndHasNoExit)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 20.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 22.0, 20.0}}, {{3.0, 18.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(10.0, 11.0), object, params());
  EXPECT_TRUE(risk.object_entry_valid);
  EXPECT_FALSE(risk.object_exit_valid);
}

// ---- temporal interval logic (Phase 7/19) -------------------------

TEST(RoundaboutInterval, ObjectClearsBeforeEgoEnters)
{
  // Phase 19 CASE A: object [1, 2], ego [4, 5] -> no overlap, gap 2.
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 45.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(4.0, 5.0), object, params());
  EXPECT_TRUE(risk.temporal_gap_valid);
  EXPECT_FALSE(risk.occupancy_overlap);
  EXPECT_NEAR(risk.temporal_gap_s, 2.0, 1e-9);
}

TEST(RoundaboutInterval, EgoClearsBeforeObjectEnters)
{
  // Phase 19 CASE B: ego [1, 2], object [4, 5] -> no overlap, gap 2.
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 45.0, 20.0, 0.0, {
    {{3.0, 45.0, 20.0}}, {{4.0, 20.0, 20.0}}, {{5.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(1.0, 2.0), object, params());
  EXPECT_TRUE(risk.temporal_gap_valid);
  EXPECT_FALSE(risk.occupancy_overlap);
  EXPECT_NEAR(risk.temporal_gap_s, 2.0, 1e-9);
}

TEST(RoundaboutInterval, OverlappingOccupancyHasZeroGap)
{
  // Phase 19 CASE C: ego [2, 4], object [3, 5] -> overlap, gap 0.
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 45.0, 20.0, 0.0, {
    {{1.0, 45.0, 20.0}}, {{3.0, 20.0, 20.0}}, {{5.0, 22.0, 20.0}},
    {{6.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(2.0, 4.0), object, params());
  EXPECT_TRUE(risk.occupancy_overlap);
  EXPECT_NEAR(risk.temporal_gap_s, 0.0, 1e-9);
}

TEST(RoundaboutInterval, TouchingBoundariesAreNotOverlap)
{
  // object [1, 2], ego [2, 3]: touch at t = 2 -> not overlap, gap 0.
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 45.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(2.0, 3.0), object, params());
  EXPECT_FALSE(risk.occupancy_overlap);
  EXPECT_NEAR(risk.temporal_gap_s, 0.0, 1e-9);
}

TEST(RoundaboutInterval, UnboundedObjectExitOverlapsWhenEgoEntersFirst)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 45.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{3.0, 21.0, 20.0}}, {{5.0, 22.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(2.0, 5.0), object, params());
  EXPECT_FALSE(risk.object_exit_valid);
  EXPECT_TRUE(risk.occupancy_overlap);
  EXPECT_NEAR(risk.temporal_gap_s, 0.0, 1e-9);
}

// ---- arrival delta sign (Phase 6/20) -----------------------------

TEST(RoundaboutArrivalDelta, ObjectBeforeEgoIsNegative)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 45.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(3.0, 4.0), object, params());
  EXPECT_TRUE(risk.arrival_delta_valid);
  EXPECT_NEAR(risk.arrival_delta_s, 1.0 - 3.0, 1e-9);
}

TEST(RoundaboutArrivalDelta, ObjectAfterEgoIsPositive)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 45.0, 20.0, 0.0, {
    {{4.0, 20.0, 20.0}}, {{5.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(1.0, 2.0), object, params());
  EXPECT_TRUE(risk.arrival_delta_valid);
  EXPECT_NEAR(risk.arrival_delta_s, 4.0 - 1.0, 1e-9);
}

TEST(RoundaboutArrivalDelta, SimultaneousArrivalIsApproximatelyZero)
{
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 45.0, 20.0, 0.0, {
    {{2.0, 20.0, 20.0}}, {{3.0, 45.0, 20.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(2.0, 4.0), object, params());
  EXPECT_TRUE(risk.arrival_delta_valid);
  EXPECT_NEAR(risk.arrival_delta_s, 0.0, 1e-9);
}

TEST(RoundaboutArrivalDelta, InvalidWhenEgoTimingInvalid)
{
  const auto ego = origin_ego(90.0, 0.0);
  RoundaboutEgoTiming no_ego;
  const auto object = map_object(1U, 45.0, 20.0, 0.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}});
  const auto risk =
    compute_roundabout_gap_risk(square_zone(), ego, no_ego, object, params());
  EXPECT_TRUE(risk.object_entry_valid);
  EXPECT_FALSE(risk.arrival_delta_valid);
  EXPECT_FALSE(risk.temporal_gap_valid);
}

// ---- plural / frame level (Phase 15) ----------------------------

TEST(RoundaboutGapRisks, MultipleObjectsEachGetOneRecord)
{
  const auto ego = origin_ego(90.0, 10.0);
  std::vector<RoundaboutObjectInput> objects;
  objects.push_back(map_object(1U, 45.0, 20.0, 10.0, {
    {{1.0, 20.0, 20.0}}, {{2.0, 45.0, 20.0}}}));
  objects.push_back(map_object(2U, 200.0, 200.0, 10.0, {
    {{1.0, 200.0, 200.0}}}));
  const auto computation = compute_roundabout_gap_risks(
    square_zone(100.0, 120.0), ego, objects, params());
  ASSERT_EQ(computation.objects.size(), 2U);
  EXPECT_EQ(computation.relevant_object_count, 1U);
  EXPECT_TRUE(computation.objects[0].relevant_to_conflict);
  EXPECT_FALSE(computation.objects[1].relevant_to_conflict);
}

TEST(RoundaboutGapRisks, ZeroObjectsIsValidWithEgoTiming)
{
  const auto ego = origin_ego(90.0, 10.0);
  const auto computation = compute_roundabout_gap_risks(
    square_zone(100.0, 120.0), ego, {}, params());
  EXPECT_TRUE(computation.objects.empty());
  EXPECT_EQ(computation.relevant_object_count, 0U);
  EXPECT_TRUE(computation.ego.entry_valid);
}

TEST(RoundaboutGapRisks, MalformedObjectIsSkippedAndCounted)
{
  const auto ego = origin_ego(90.0, 0.0);
  std::vector<RoundaboutObjectInput> objects;
  objects.push_back(map_object(1U, 20.0, 20.0, 0.0, {{{1.0, 20.0, 20.0}}}));
  RoundaboutObjectInput bad = map_object(2U, 0.0, 0.0, 0.0, {});
  bad.x_rel_m = std::numeric_limits<double>::quiet_NaN();
  objects.push_back(bad);
  const auto computation =
    compute_roundabout_gap_risks(square_zone(), ego, objects, params());
  EXPECT_EQ(computation.objects.size(), 1U);
  EXPECT_EQ(computation.rejected_malformed_objects, 1U);
}

TEST(RoundaboutGapRisks, ObjectBudgetIsEnforced)
{
  const auto ego = origin_ego(90.0, 0.0);
  auto config = params();
  config.maximum_objects = 2U;
  std::vector<RoundaboutObjectInput> objects(
    5, map_object(1U, 100.0, 100.0, 0.0, {{{1.0, 100.0, 100.0}}}));
  const auto computation =
    compute_roundabout_gap_risks(square_zone(), ego, objects, config);
  EXPECT_EQ(computation.objects.size(), 2U);
  EXPECT_EQ(computation.rejected_over_budget, 3U);
}

TEST(RoundaboutGapRisks, DegeneratePolygonIsRejected)
{
  RoundaboutConflictZone zone;
  zone.id = "bad";
  zone.polygon_m = {ConflictPoint2{0.0, 0.0}, ConflictPoint2{1.0, 1.0}};
  zone.route_s_enter_m = 0.0;
  zone.route_s_exit_m = 10.0;
  EXPECT_THROW(zone.validated(), std::invalid_argument);
}

TEST(RoundaboutGapRisks, NonIncreasingRouteSpanIsRejected)
{
  auto zone = square_zone(120.0, 100.0);
  EXPECT_THROW(zone.validated(), std::invalid_argument);
}

TEST(RoundaboutGapRisks, DeterministicOnRepeatedInput)
{
  const auto ego = origin_ego(90.0, 9.0);
  std::vector<RoundaboutObjectInput> objects;
  objects.push_back(map_object(1U, 45.0, 20.0, 9.0, {
    {{1.0, 45.0, 20.0}}, {{2.0, 20.0, 20.0}}, {{3.0, 45.0, 20.0}}}));
  const auto a =
    compute_roundabout_gap_risks(square_zone(100.0, 120.0), ego, objects, params());
  const auto b =
    compute_roundabout_gap_risks(square_zone(100.0, 120.0), ego, objects, params());
  ASSERT_EQ(a.objects.size(), b.objects.size());
  EXPECT_EQ(a.objects[0].object_entry_time_s, b.objects[0].object_entry_time_s);
  EXPECT_EQ(a.objects[0].temporal_gap_s, b.objects[0].temporal_gap_s);
  EXPECT_EQ(a.ego.entry_time_s, b.ego.entry_time_s);
}

TEST(RoundaboutGapRisks, CurvedTrajectoryUsesDiscretePredictionNotCurrentVelocity)
{
  // Currently far outside; a straight extrapolation of the current position
  // would miss the zone, but the discrete predicted centroids curve into it.
  const auto ego = origin_ego(90.0, 0.0);
  const auto object = map_object(1U, 60.0, 60.0, 0.0, {
    {{1.0, 45.0, 45.0}}, {{2.0, 35.0, 35.0}}, {{3.0, 20.0, 20.0}},
    {{4.0, 15.0, 45.0}}});
  const auto risk = compute_roundabout_gap_risk(
    square_zone(), ego, ego_interval(10.0, 11.0), object, params());
  EXPECT_TRUE(risk.relevant_to_conflict);
  EXPECT_NEAR(risk.object_entry_time_s, 3.0, 1e-9);
  EXPECT_NEAR(risk.object_exit_time_s, 4.0, 1e-9);
}

}  // namespace
