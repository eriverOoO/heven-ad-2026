#include "ad_lidar_perception/planning/dynamic_object_risk.hpp"

#include "dynamic_object_risk_node.hpp"

#include <gtest/gtest.h>

#include <ad_interfaces/msg/predicted_object_array.hpp>

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

namespace
{

using ad_lidar_perception::planning::build_risk_frame;
using ad_lidar_perception::planning::compute_dynamic_object_risks;
using ad_lidar_perception::planning::compute_object_risk;
using ad_lidar_perception::planning::DynamicObjectRiskResult;
using ad_lidar_perception::planning::EgoSample;
using ad_lidar_perception::planning::EgoState;
using ad_lidar_perception::planning::PredictedObjectInput;
using ad_lidar_perception::planning::PredictedPoint;
using ad_lidar_perception::planning::RiskNodeConfig;
using ad_lidar_perception::planning::RiskParameters;

RiskParameters params()
{
  return RiskParameters{}.validated();
}

// Ego at the origin, heading +x (world), moving forward at `speed`.
EgoState ego_forward(const double speed, const double yaw = 0.0)
{
  EgoState ego;
  ego.x_m = 0.0;
  ego.y_m = 0.0;
  ego.yaw_rad = yaw;
  ego.longitudinal_speed_mps = speed;
  return ego;
}

PredictedObjectInput object_at(
  const double x, const double y, const double vx_world, const double vy_world,
  const double length = 4.0, const double width = 2.0)
{
  PredictedObjectInput object;
  object.x_m = x;
  object.y_m = y;
  object.vx_world_mps = vx_world;
  object.vy_world_mps = vy_world;
  object.length_m = length;
  object.width_m = width;
  return object;
}

bool all_finite(const DynamicObjectRiskResult & r)
{
  return std::isfinite(r.x_rel_m) && std::isfinite(r.y_rel_m) &&
    std::isfinite(r.distance_m) && std::isfinite(r.vx_rel_mps) &&
    std::isfinite(r.vy_rel_mps) && std::isfinite(r.relative_speed_mps) &&
    std::isfinite(r.range_rate_mps) && std::isfinite(r.longitudinal_closing_mps) &&
    std::isfinite(r.ttc_s) && std::isfinite(r.cpa_time_s) &&
    std::isfinite(r.cpa_distance_m) && std::isfinite(r.predicted_min_separation_m) &&
    std::isfinite(r.predicted_min_separation_time_s) &&
    std::isfinite(r.position_uncertainty_m);
}

// ---------------------------------------------------------------------------
// Phase 12: geometry
// ---------------------------------------------------------------------------

// 1. stationary object 20 m ahead, ego approaching at 10 m/s.
TEST(DynamicObjectRisk, StationaryLeadEgoApproaching) {
  const auto r = compute_object_risk(ego_forward(10.0), object_at(20.0, 0.0, 0.0, 0.0), params());
  EXPECT_NEAR(r.x_rel_m, 20.0, 1e-9);
  EXPECT_NEAR(r.y_rel_m, 0.0, 1e-9);
  EXPECT_NEAR(r.vx_rel_mps, -10.0, 1e-9);   // object still, ego forward
  EXPECT_GT(r.range_rate_mps, 0.0);          // range decreasing
  EXPECT_GT(r.longitudinal_closing_mps, 0.0);
  ASSERT_TRUE(r.ttc_valid);
  EXPECT_GT(r.ttc_s, 0.0);
  EXPECT_LT(r.ttc_s, 20.0 / 10.0);           // shortened by the collision radii
  EXPECT_TRUE(r.cpa_valid);
  EXPECT_TRUE(all_finite(r));
}

// 2. lead object ahead, slower than ego -> still closing.
TEST(DynamicObjectRisk, SlowerLeadCloses) {
  const auto r = compute_object_risk(ego_forward(12.0), object_at(30.0, 0.0, 5.0, 0.0), params());
  EXPECT_NEAR(r.vx_rel_mps, -7.0, 1e-9);
  EXPECT_GT(r.range_rate_mps, 0.0);
  EXPECT_GT(r.longitudinal_closing_mps, 0.0);
  EXPECT_TRUE(r.ttc_valid);
}

// 3. lead object at the same speed -> no relative motion.
TEST(DynamicObjectRisk, SameSpeedLeadHasNoClosing) {
  const auto r = compute_object_risk(ego_forward(9.0), object_at(15.0, 0.0, 9.0, 0.0), params());
  EXPECT_NEAR(r.relative_speed_mps, 0.0, 1e-9);
  EXPECT_NEAR(r.range_rate_mps, 0.0, 1e-9);
  EXPECT_FALSE(r.ttc_valid);
  EXPECT_FALSE(r.cpa_valid);
}

// 4. lead object faster than ego -> separating.
TEST(DynamicObjectRisk, FasterLeadSeparates) {
  const auto r = compute_object_risk(ego_forward(8.0), object_at(15.0, 0.0, 14.0, 0.0), params());
  EXPECT_NEAR(r.vx_rel_mps, 6.0, 1e-9);
  EXPECT_LT(r.range_rate_mps, 0.0);
  EXPECT_LT(r.longitudinal_closing_mps, 0.0);
  EXPECT_FALSE(r.ttc_valid);
  ASSERT_TRUE(r.cpa_valid);
  EXPECT_NEAR(r.cpa_time_s, 0.0, 1e-9);           // closest approach is now
  EXPECT_NEAR(r.cpa_distance_m, 15.0, 1e-6);
}

// 5. oncoming object in the ego lane.
TEST(DynamicObjectRisk, OncomingObjectClosesFast) {
  const auto r = compute_object_risk(ego_forward(10.0), object_at(50.0, 0.0, -12.0, 0.0), params());
  EXPECT_NEAR(r.vx_rel_mps, -22.0, 1e-9);
  EXPECT_GT(r.range_rate_mps, 20.0);
  ASSERT_TRUE(r.ttc_valid);
  EXPECT_LT(r.ttc_s, 50.0 / 22.0);
}

// 6 / 7. crossing object, both directions.
TEST(DynamicObjectRisk, CrossingObjectLeftToRight) {
  const auto r = compute_object_risk(
    ego_forward(0.0), object_at(10.0, 8.0, 0.0, -6.0), params());
  EXPECT_NEAR(r.vy_rel_mps, -6.0, 1e-9);
  EXPECT_TRUE(r.cpa_valid);
  EXPECT_LT(r.cpa_distance_m, r.distance_m);  // it passes closer than it starts
  EXPECT_TRUE(all_finite(r));
}
TEST(DynamicObjectRisk, CrossingObjectRightToLeft) {
  const auto r = compute_object_risk(
    ego_forward(0.0), object_at(10.0, -8.0, 0.0, 6.0), params());
  EXPECT_NEAR(r.vy_rel_mps, 6.0, 1e-9);
  EXPECT_TRUE(r.cpa_valid);
  EXPECT_LT(r.cpa_distance_m, r.distance_m);
}

// 8. object behind: separating vs catching up.
TEST(DynamicObjectRisk, ObjectBehindSeparatingAndCatchingUp) {
  const auto separating = compute_object_risk(
    ego_forward(10.0), object_at(-12.0, 0.0, 0.0, 0.0), params());
  EXPECT_LT(separating.x_rel_m, 0.0);
  EXPECT_LT(separating.range_rate_mps, 0.0);
  EXPECT_LT(separating.longitudinal_closing_mps, 0.0);
  EXPECT_FALSE(separating.ttc_valid);

  const auto catching = compute_object_risk(
    ego_forward(10.0), object_at(-12.0, 0.0, 16.0, 0.0), params());
  EXPECT_GT(catching.range_rate_mps, 0.0);
  EXPECT_GT(catching.longitudinal_closing_mps, 0.0);
  EXPECT_TRUE(catching.ttc_valid);
}

// 9. zero relative velocity.
TEST(DynamicObjectRisk, ZeroRelativeVelocity) {
  const auto r = compute_object_risk(
    ego_forward(0.0), object_at(10.0, 3.0, 0.0, 0.0), params());
  EXPECT_NEAR(r.relative_speed_mps, 0.0, 1e-12);
  EXPECT_FALSE(r.ttc_valid);
  EXPECT_FALSE(r.cpa_valid);
  EXPECT_NEAR(r.range_rate_mps, 0.0, 1e-12);
}

// 10. zero objects.
TEST(DynamicObjectRisk, ZeroObjectsProducesEmptyResult) {
  const auto computation =
    compute_dynamic_object_risks(ego_forward(5.0), {}, params());
  EXPECT_TRUE(computation.objects.empty());
  EXPECT_EQ(computation.rejected_non_finite_state, 0U);
}

// 11. multiple objects handled independently.
TEST(DynamicObjectRisk, MultipleObjectsAreIndependent) {
  std::vector<PredictedObjectInput> objects{
    object_at(20.0, 0.0, 0.0, 0.0),
    object_at(-5.0, 0.0, 30.0, 0.0),
    object_at(0.0, 40.0, 0.0, -3.0)};
  const auto computation =
    compute_dynamic_object_risks(ego_forward(10.0), objects, params());
  ASSERT_EQ(computation.objects.size(), 3U);
  EXPECT_TRUE(computation.objects[0].ttc_valid);
  EXPECT_TRUE(computation.objects[1].ttc_valid);   // fast rear catch-up
  EXPECT_FALSE(computation.objects[2].ttc_valid);  // far to the side
}

// 12. exact head-on collision course.
TEST(DynamicObjectRisk, ExactCollisionCourse) {
  const auto r = compute_object_risk(
    ego_forward(10.0), object_at(40.0, 0.0, -10.0, 0.0), params());
  ASSERT_TRUE(r.ttc_valid);
  EXPECT_NEAR(r.ttc_s, (40.0 - std::hypot(2.5, 1.1) - std::hypot(2.0, 1.0)) / 20.0, 1e-6);
  ASSERT_TRUE(r.cpa_valid);
  EXPECT_LT(r.cpa_distance_m, std::hypot(2.5, 1.1) + std::hypot(2.0, 1.0));
}

// 13. near miss: lateral offset just outside the contact radius.
TEST(DynamicObjectRisk, NearMissHasCpaButNoTtc) {
  const double clearance = std::hypot(2.5, 1.1) + std::hypot(2.0, 1.0) + 0.5;
  const auto r = compute_object_risk(
    ego_forward(0.0), object_at(0.0, clearance, 8.0, 0.0), params());
  EXPECT_FALSE(r.ttc_valid);
  ASSERT_TRUE(r.cpa_valid);
  EXPECT_NEAR(r.cpa_distance_m, clearance, 1e-6);  // passes abeam
}

// 14. large lateral separation.
TEST(DynamicObjectRisk, LargeLateralSeparation) {
  const auto r = compute_object_risk(
    ego_forward(10.0), object_at(5.0, 100.0, 0.0, 0.0), params());
  EXPECT_FALSE(r.ttc_valid);
  EXPECT_GT(r.cpa_distance_m, 90.0);
  EXPECT_TRUE(all_finite(r));
}

// 15. non-finite input.
TEST(DynamicObjectRisk, NonFiniteEgoThrows) {
  EgoState bad = ego_forward(std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(
    compute_object_risk(bad, object_at(10.0, 0.0, 0.0, 0.0), params()),
    std::invalid_argument);
}
TEST(DynamicObjectRisk, NonFiniteObjectIsSkippedAndCounted) {
  std::vector<PredictedObjectInput> objects{
    object_at(std::numeric_limits<double>::infinity(), 0.0, 0.0, 0.0),
    object_at(10.0, 0.0, 0.0, 0.0, 0.0, 2.0),  // zero length
    object_at(20.0, 0.0, 0.0, 0.0)};
  const auto computation =
    compute_dynamic_object_risks(ego_forward(5.0), objects, params());
  ASSERT_EQ(computation.objects.size(), 1U);
  EXPECT_EQ(computation.rejected_non_finite_state, 1U);
  EXPECT_EQ(computation.rejected_non_finite_dimensions, 1U);
}

// 19. rotation with a non-zero ego yaw.
TEST(DynamicObjectRisk, NonZeroEgoYawRotatesIntoBaseLink) {
  EgoState ego = ego_forward(4.0, M_PI / 2.0);  // ego faces +y world
  const auto ahead = compute_object_risk(ego, object_at(0.0, 10.0, 0.0, 0.0), params());
  EXPECT_NEAR(ahead.x_rel_m, 10.0, 1e-6);   // 10 m in front
  EXPECT_NEAR(ahead.y_rel_m, 0.0, 1e-6);
  EXPECT_NEAR(ahead.vx_rel_mps, -4.0, 1e-6);  // closing at ego speed

  const auto left = compute_object_risk(ego, object_at(-10.0, 0.0, 0.0, 0.0), params());
  EXPECT_NEAR(left.x_rel_m, 0.0, 1e-6);
  EXPECT_NEAR(left.y_rel_m, 10.0, 1e-6);   // -x world is ego-left when facing +y
}

// 20. determinism.
TEST(DynamicObjectRisk, DeterministicForIdenticalInput) {
  const auto ego = ego_forward(7.5, 0.3);
  const auto object = object_at(12.0, -3.0, 2.0, 1.0);
  const auto a = compute_object_risk(ego, object, params());
  const auto b = compute_object_risk(ego, object, params());
  EXPECT_EQ(a.x_rel_m, b.x_rel_m);
  EXPECT_EQ(a.ttc_s, b.ttc_s);
  EXPECT_EQ(a.cpa_distance_m, b.cpa_distance_m);
  EXPECT_EQ(a.ttc_valid, b.ttc_valid);
}

// Radial closing speed sign is stable in every quadrant.
TEST(DynamicObjectRisk, RangeRateSignIsStableInAllQuadrants) {
  const auto params_ = params();
  for (const double angle : {0.0, M_PI_2, M_PI, -M_PI_2, 0.9, -2.3}) {
    const double px = 20.0 * std::cos(angle);
    const double py = 20.0 * std::sin(angle);
    // object moving straight toward the ego origin.
    const auto approaching = compute_object_risk(
      ego_forward(0.0), object_at(px, py, -px / 20.0 * 5.0, -py / 20.0 * 5.0), params_);
    EXPECT_GT(approaching.range_rate_mps, 0.0) << "angle=" << angle;
    const auto receding = compute_object_risk(
      ego_forward(0.0), object_at(px, py, px / 20.0 * 5.0, py / 20.0 * 5.0), params_);
    EXPECT_LT(receding.range_rate_mps, 0.0) << "angle=" << angle;
  }
}

// predicted-state trajectory drives the horizon-min-separation metric.
TEST(DynamicObjectRisk, PredictedMinSeparationUsesDiscreteStates) {
  auto object = object_at(30.0, 0.0, -10.0, 0.0);
  object.predicted_points = {
    PredictedPoint{1.0, 20.0, 0.0},
    PredictedPoint{2.0, 10.0, 0.0},
    PredictedPoint{3.0, 2.0, 0.0},
    PredictedPoint{4.0, 6.0, 0.0}};
  const auto r = compute_object_risk(ego_forward(0.0), object, params());
  ASSERT_TRUE(r.predicted_min_separation_valid);
  EXPECT_NEAR(r.predicted_min_separation_m, 2.0, 1e-9);
  EXPECT_NEAR(r.predicted_min_separation_time_s, 3.0, 1e-9);
}
TEST(DynamicObjectRisk, PredictedMinSeparationInvalidWithoutStates) {
  const auto r = compute_object_risk(
    ego_forward(0.0), object_at(10.0, 0.0, 1.0, 0.0), params());
  EXPECT_FALSE(r.predicted_min_separation_valid);
  EXPECT_EQ(r.predicted_min_separation_m, 0.0);
}

// position uncertainty is the larger covariance eigenvalue's sqrt.
TEST(DynamicObjectRisk, PositionUncertaintyFromCovarianceEigenvalue) {
  auto object = object_at(10.0, 0.0, 0.0, 0.0);
  object.position_covariance_xy = {9.0, 0.0, 0.0, 4.0};
  const auto r = compute_object_risk(ego_forward(0.0), object, params());
  EXPECT_NEAR(r.position_uncertainty_m, 3.0, 1e-9);

  object.position_covariance_xy = {0.0, 0.0, 0.0, 0.0};
  EXPECT_EQ(compute_object_risk(ego_forward(0.0), object, params()).position_uncertainty_m, 0.0);
}

// overlapping footprints -> TTC 0, still valid, never negative.
TEST(DynamicObjectRisk, OverlappingFootprintGivesZeroTtc) {
  const auto r = compute_object_risk(
    ego_forward(1.0), object_at(1.0, 0.0, 0.0, 0.0), params());
  EXPECT_TRUE(r.ttc_valid);
  EXPECT_EQ(r.ttc_s, 0.0);
}

// contact beyond the horizon is reported invalid, not as a huge number.
TEST(DynamicObjectRisk, ContactBeyondHorizonIsInvalid) {
  auto p = params();
  p.ttc_horizon_s = 3.0;
  const auto r = compute_object_risk(ego_forward(1.0), object_at(100.0, 0.0, 0.0, 0.0), p);
  EXPECT_FALSE(r.ttc_valid);
  EXPECT_EQ(r.ttc_s, 0.0);
}

TEST(DynamicObjectRisk, MaximumObjectBudgetIsEnforced) {
  auto p = params();
  p.maximum_objects = 2;
  std::vector<PredictedObjectInput> objects(5, object_at(10.0, 0.0, 0.0, 0.0));
  const auto computation = compute_dynamic_object_risks(ego_forward(1.0), objects, p);
  EXPECT_EQ(computation.objects.size(), 2U);
  EXPECT_EQ(computation.rejected_over_budget, 3U);
}

// ---------------------------------------------------------------------------
// Phase 13: backend-agnostic equivalence
// ---------------------------------------------------------------------------

// The pure core has no orientation input at all -- backend independence is
// structural there. It is exercised for real at the message boundary, where
// orientation exists, in DynamicObjectRiskFrame.ObjectOrientationNeverAffectsRiskOutput.

// An object coincident with the ego origin: the radial rate is undefined and
// reported 0 (distance guard is a length, not a speed).
TEST(DynamicObjectRisk, ObjectAtEgoOriginHasZeroRangeRate) {
  auto object = object_at(0.0, 0.0, 5.0, 5.0);
  const auto r = compute_object_risk(ego_forward(0.0), object, params());
  EXPECT_EQ(r.distance_m, 0.0);
  EXPECT_EQ(r.range_rate_mps, 0.0);
  EXPECT_TRUE(all_finite(r));
  EXPECT_TRUE(r.ttc_valid);       // footprints overlap
  EXPECT_EQ(r.ttc_s, 0.0);
}

// ---------------------------------------------------------------------------
// Phase 9: node-level invalid / stale data contract (build_risk_frame)
// ---------------------------------------------------------------------------

ad_interfaces::msg::PredictedObjectArray prediction_at(
  const std::int64_t stamp_ns, const std::string & frame = "odom")
{
  ad_interfaces::msg::PredictedObjectArray message;
  message.header.frame_id = frame;
  message.header.stamp.sec = static_cast<std::int32_t>(stamp_ns / 1'000'000'000LL);
  message.header.stamp.nanosec = static_cast<std::uint32_t>(stamp_ns % 1'000'000'000LL);
  return message;
}

ad_interfaces::msg::PredictedObject one_object(const double x, const double y)
{
  ad_interfaces::msg::PredictedObject object;
  object.initial_pose.pose.position.x = x;
  object.initial_pose.pose.position.y = y;
  object.dimensions.x = 4.0;
  object.dimensions.y = 2.0;
  object.initial_twist.twist.linear.x = 0.0;
  object.initial_twist.twist.linear.y = 0.0;
  return object;
}

std::optional<EgoSample> ego_sample(const std::int64_t stamp_ns, const double speed = 5.0)
{
  EgoSample sample;
  sample.stamp_ns = stamp_ns;
  sample.state = ego_forward(speed);
  return sample;
}

TEST(DynamicObjectRiskFrame, ValidFramePublishesBaseLinkArray) {
  const std::int64_t stamp = 1'000'000'000LL;
  auto prediction = prediction_at(stamp);
  prediction.objects.push_back(one_object(20.0, 0.0));
  const auto result = build_risk_frame(
    prediction, ego_sample(stamp), stamp + 10'000'000LL, std::nullopt, RiskNodeConfig{});
  ASSERT_TRUE(result.published);
  EXPECT_EQ(result.risks.header.frame_id, "base_link");
  EXPECT_EQ(result.risks.header.stamp.sec, prediction.header.stamp.sec);
  ASSERT_EQ(result.risks.objects.size(), 1U);
  EXPECT_EQ(result.objects_in, 1U);
  EXPECT_EQ(result.objects_out, 1U);
}

TEST(DynamicObjectRiskFrame, EmptyObjectsIsValidEmptyArray) {
  const std::int64_t stamp = 1'000'000'000LL;
  const auto result = build_risk_frame(
    prediction_at(stamp), ego_sample(stamp), stamp + 1'000'000LL, std::nullopt,
    RiskNodeConfig{});
  ASSERT_TRUE(result.published);
  EXPECT_TRUE(result.risks.objects.empty());
  EXPECT_TRUE(result.reason.empty());
}

TEST(DynamicObjectRiskFrame, NonPositiveStampRejected) {
  auto prediction = prediction_at(1'000'000'000LL);
  prediction.header.stamp.sec = 0;
  prediction.header.stamp.nanosec = 0;
  const auto result = build_risk_frame(
    prediction, ego_sample(1), 1'000'000'000LL, std::nullopt, RiskNodeConfig{});
  EXPECT_FALSE(result.published);
  EXPECT_NE(result.reason.find("strictly positive"), std::string::npos);
}

TEST(DynamicObjectRiskFrame, WrongFrameRejected) {
  const std::int64_t stamp = 1'000'000'000LL;
  const auto result = build_risk_frame(
    prediction_at(stamp, "map"), ego_sample(stamp), stamp, std::nullopt, RiskNodeConfig{});
  EXPECT_FALSE(result.published);
  EXPECT_NE(result.reason.find("frame_id"), std::string::npos);
}

TEST(DynamicObjectRiskFrame, DuplicateOrBackwardStampRejected) {
  const std::int64_t stamp = 5'000'000'000LL;
  const auto duplicate = build_risk_frame(
    prediction_at(stamp), ego_sample(stamp), stamp, stamp, RiskNodeConfig{});
  EXPECT_FALSE(duplicate.published);
  EXPECT_NE(duplicate.reason.find("duplicate or backward"), std::string::npos);
  const auto backward = build_risk_frame(
    prediction_at(stamp - 10'000'000LL), ego_sample(stamp), stamp, stamp, RiskNodeConfig{});
  EXPECT_FALSE(backward.published);
}

TEST(DynamicObjectRiskFrame, StalePredictionRejected) {
  const std::int64_t stamp = 10'000'000'000LL;
  const auto result = build_risk_frame(
    prediction_at(stamp), ego_sample(stamp), stamp + 900'000'000LL, std::nullopt,
    RiskNodeConfig{});
  EXPECT_FALSE(result.published);
  EXPECT_NE(result.reason.find("stale"), std::string::npos);
}

TEST(DynamicObjectRiskFrame, FutureStampRejected) {
  const std::int64_t stamp = 10'000'000'000LL;
  const auto result = build_risk_frame(
    prediction_at(stamp), ego_sample(stamp), stamp - 500'000'000LL, std::nullopt,
    RiskNodeConfig{});
  EXPECT_FALSE(result.published);
  EXPECT_NE(result.reason.find("future"), std::string::npos);
}

TEST(DynamicObjectRiskFrame, MissingEgoRejected) {
  const std::int64_t stamp = 1'000'000'000LL;
  const auto result = build_risk_frame(
    prediction_at(stamp), std::nullopt, stamp, std::nullopt, RiskNodeConfig{});
  EXPECT_FALSE(result.published);
  EXPECT_NE(result.reason.find("ego state is unavailable"), std::string::npos);
}

TEST(DynamicObjectRiskFrame, StaleEgoRejected) {
  const std::int64_t stamp = 5'000'000'000LL;
  const auto result = build_risk_frame(
    prediction_at(stamp), ego_sample(stamp - 900'000'000LL), stamp, std::nullopt,
    RiskNodeConfig{});
  EXPECT_FALSE(result.published);
  EXPECT_NE(result.reason.find("ego state is stale"), std::string::npos);
}

TEST(DynamicObjectRiskFrame, NonFiniteObjectSkippedFrameStillPublishes) {
  const std::int64_t stamp = 1'000'000'000LL;
  auto prediction = prediction_at(stamp);
  auto bad = one_object(10.0, 0.0);
  bad.initial_pose.pose.position.x = std::numeric_limits<double>::quiet_NaN();
  prediction.objects.push_back(bad);
  prediction.objects.push_back(one_object(20.0, 0.0));
  const auto result = build_risk_frame(
    prediction, ego_sample(stamp), stamp, std::nullopt, RiskNodeConfig{});
  ASSERT_TRUE(result.published);
  EXPECT_EQ(result.objects_in, 2U);
  EXPECT_EQ(result.objects_out, 1U);
  EXPECT_EQ(result.rejected_non_finite_state, 1U);
}

// Backend independence, tested where the two backends actually differ:
// AB3DMOT publishes an identity (unavailable) quaternion, Autoware a real yaw.
// Same physical centroid / world velocity / dimensions -> identical risk.
TEST(DynamicObjectRiskFrame, ObjectOrientationNeverAffectsRiskOutput) {
  const std::int64_t stamp = 1'000'000'000LL;
  auto ab3dmot = prediction_at(stamp);
  auto autoware = prediction_at(stamp);

  auto a = one_object(25.0, 1.0);          // AB3DMOT: yaw unavailable
  a.initial_pose.pose.orientation.x = 0.0;
  a.initial_pose.pose.orientation.y = 0.0;
  a.initial_pose.pose.orientation.z = 0.0;
  a.initial_pose.pose.orientation.w = 1.0;
  a.initial_twist.twist.linear.x = 3.0;
  a.initial_twist.twist.linear.y = -1.0;

  auto b = a;                              // Autoware: same state, real yaw 0.9 rad
  b.initial_pose.pose.orientation.z = std::sin(0.9 / 2.0);
  b.initial_pose.pose.orientation.w = std::cos(0.9 / 2.0);

  ab3dmot.objects.push_back(a);
  autoware.objects.push_back(b);
  const auto ra = build_risk_frame(
    ab3dmot, ego_sample(stamp), stamp, std::nullopt, RiskNodeConfig{});
  const auto rb = build_risk_frame(
    autoware, ego_sample(stamp), stamp, std::nullopt, RiskNodeConfig{});
  ASSERT_EQ(ra.risks.objects.size(), 1U);
  ASSERT_EQ(rb.risks.objects.size(), 1U);
  const auto & x = ra.risks.objects[0];
  const auto & y = rb.risks.objects[0];
  EXPECT_EQ(x.x_rel_m, y.x_rel_m);
  EXPECT_EQ(x.y_rel_m, y.y_rel_m);
  EXPECT_EQ(x.vx_rel_mps, y.vx_rel_mps);
  EXPECT_EQ(x.range_rate_mps, y.range_rate_mps);
  EXPECT_EQ(x.ttc_valid, y.ttc_valid);
  EXPECT_EQ(x.ttc_s, y.ttc_s);
  EXPECT_EQ(x.cpa_distance_m, y.cpa_distance_m);
  EXPECT_EQ(x.predicted_min_separation_m, y.predicted_min_separation_m);
}

TEST(DynamicObjectRiskFrame, DeterministicFrameOutput) {
  const std::int64_t stamp = 2'000'000'000LL;
  auto prediction = prediction_at(stamp);
  prediction.objects.push_back(one_object(18.0, -2.0));
  const auto a = build_risk_frame(
    prediction, ego_sample(stamp), stamp, std::nullopt, RiskNodeConfig{});
  const auto b = build_risk_frame(
    prediction, ego_sample(stamp), stamp, std::nullopt, RiskNodeConfig{});
  ASSERT_EQ(a.risks.objects.size(), b.risks.objects.size());
  EXPECT_EQ(a.risks.objects[0].ttc_s, b.risks.objects[0].ttc_s);
  EXPECT_EQ(a.risks.objects[0].cpa_distance_m, b.risks.objects[0].cpa_distance_m);
}

}  // namespace
