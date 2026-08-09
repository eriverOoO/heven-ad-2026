#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/common/future_road_risk.hpp"

namespace {

using ad_planner::FutureRoadRiskLimits;
using ad_planner::Pose2;
using ad_planner::PredictedFootprint;
using ad_planner::PredictedObject;
using ad_planner::ReferenceCorridor;
using ad_planner::ReferenceLane;
using ad_planner::ReferencePoint;

ReferenceCorridor straight_corridor() {
  ReferenceLane lane;
  lane.lane_sequence_id = "primary";
  lane.points = {
      ReferencePoint{Pose2{0.0, 0.0, 0.0}, 0.0, 0.0, 1.0, 1.0, 16.0},
      ReferencePoint{Pose2{30.0, 0.0, 0.0}, 30.0, 0.0, 1.0, 1.0, 16.0},
  };
  ReferenceCorridor corridor;
  corridor.frame_id = "odom";
  corridor.lanes = {lane};
  corridor.primary_lane_index = 0U;
  return corridor;
}

PredictedFootprint footprint(const double time_s, const double x,
                             const double y, const double length = 0.2,
                             const double width = 0.2) {
  return PredictedFootprint{time_s, Pose2{x, y, 0.0}, length, width, 0.0, 0.0,
                            0.0};
}

PredictedObject object(const std::string &id,
                       std::vector<PredictedFootprint> footprints) {
  return PredictedObject{id, std::move(footprints)};
}

TEST(FutureRoadRisk, IgnoresCurrentFootprintWithoutAFutureRoadEntry) {
  const auto result = ad_planner::evaluate_future_road_risk(
      straight_corridor(), {object("current", {footprint(0.0, 5.0, 0.0)})},
      FutureRoadRiskLimits{6.0, 16U, 128U});

  EXPECT_FALSE(result.risk);
  EXPECT_FALSE(result.earliest_risk_time_s.has_value());
  EXPECT_EQ(result.evaluated_object_count, 1U);
  EXPECT_EQ(result.evaluated_footprint_count, 1U);
}

TEST(FutureRoadRisk, DetectsOffRoadObjectWhoseFutureFootprintEntersRoad) {
  const auto result = ad_planner::evaluate_future_road_risk(
      straight_corridor(),
      {object("pedestrian",
              {footprint(0.0, 10.0, -4.0), footprint(1.0, 10.0, -2.5),
               footprint(2.0, 10.0, 0.0)})},
      FutureRoadRiskLimits{6.0, 16U, 128U});

  EXPECT_TRUE(result.risk);
  ASSERT_TRUE(result.earliest_risk_time_s.has_value());
  EXPECT_GE(*result.earliest_risk_time_s, 0.0);
  EXPECT_LE(*result.earliest_risk_time_s, 2.0);
  EXPECT_EQ(result.risky_object_count, 1U);
}

TEST(FutureRoadRisk, DetectsRoadCrossingBetweenDiscretePredictionStates) {
  const auto result = ad_planner::evaluate_future_road_risk(
      straight_corridor(),
      {object("coarse_crossing",
              {footprint(0.0, 15.0, -4.0), footprint(1.0, 15.0, -3.0),
               footprint(2.0, 15.0, 3.0)})},
      FutureRoadRiskLimits{6.0, 16U, 128U});

  EXPECT_TRUE(result.risk);
  ASSERT_TRUE(result.earliest_risk_time_s.has_value());
  EXPECT_GE(*result.earliest_risk_time_s, 1.0);
  EXPECT_LE(*result.earliest_risk_time_s, 2.0);
}

TEST(FutureRoadRisk, IgnoresPredictionsBeyondTheConfiguredHorizon) {
  const auto result = ad_planner::evaluate_future_road_risk(
      straight_corridor(),
      {object("late_entry",
              {footprint(0.0, 20.0, -4.0), footprint(2.0, 20.0, -3.0),
               footprint(3.0, 20.0, 0.0)})},
      FutureRoadRiskLimits{1.5, 16U, 128U});

  EXPECT_FALSE(result.risk);
}

TEST(FutureRoadRisk, RejectsMalformedOrUnboundedPredictionWork) {
  const auto corridor = straight_corridor();
  const auto valid =
      object("valid", {footprint(0.0, 5.0, -4.0), footprint(1.0, 5.0, -3.0)});

  EXPECT_THROW(
      ad_planner::evaluate_future_road_risk(
          corridor, {valid, valid}, FutureRoadRiskLimits{6.0, 1U, 128U}),
      std::invalid_argument);

  EXPECT_THROW(ad_planner::evaluate_future_road_risk(
                   corridor, {valid}, FutureRoadRiskLimits{6.0, 16U, 1U}),
               std::invalid_argument);

  auto malformed = valid;
  malformed.footprints.back().pose.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(ad_planner::evaluate_future_road_risk(
                   corridor, {malformed}, FutureRoadRiskLimits{6.0, 16U, 128U}),
               std::invalid_argument);
}

} // namespace
