#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ad_planner/local_planning/frenet/frenet_geometry.hpp"

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

using ad_planner::EgoState;
using ad_planner::frenet_to_cartesian;
using ad_planner::FrenetState;
using ad_planner::Pose2;
using ad_planner::project_to_frenet;
using ad_planner::QuarticPolynomial;
using ad_planner::QuinticPolynomial;
using ad_planner::ReferenceLane;
using ad_planner::ReferencePoint;

ReferencePoint reference_point(const double x, const double y, const double yaw,
                               const double s, const double curvature = 0.0) {
  ReferencePoint point;
  point.pose = Pose2{x, y, yaw};
  point.route_s_m = s;
  point.curvature_inv_m = curvature;
  point.left_width_m = 2.0;
  point.right_width_m = 2.0;
  point.speed_limit_mps = 10.0;
  return point;
}

ReferenceLane straight_lane() {
  ReferenceLane lane;
  lane.lane_sequence_id = "straight";
  lane.points = {reference_point(0.0, 0.0, 0.0, 0.0),
                 reference_point(10.0, 0.0, 0.0, 10.0)};
  return lane;
}

ReferenceLane circular_lane() {
  ReferenceLane lane;
  lane.lane_sequence_id = "radius-ten";
  constexpr double kRadiusM = 10.0;
  constexpr double kAngleStepRad = 0.005;
  constexpr int kSampleCount = 201;
  lane.points.reserve(kSampleCount);
  for (int index = 0; index < kSampleCount; ++index) {
    const double angle = kAngleStepRad * static_cast<double>(index);
    lane.points.push_back(reference_point(
        kRadiusM * std::sin(angle), kRadiusM * (1.0 - std::cos(angle)), angle,
        kRadiusM * angle, 1.0 / kRadiusM));
  }
  return lane;
}

TEST(FrenetGeometry, QuinticMeetsAllSixBoundaryConditions) {
  constexpr double kDurationS = 2.5;
  const QuinticPolynomial polynomial(1.2, -0.4, 0.3, 6.2, 1.1, -0.2,
                                     kDurationS);

  EXPECT_NEAR(polynomial.position(0.0), 1.2, 1e-9);
  EXPECT_NEAR(polynomial.velocity(0.0), -0.4, 1e-9);
  EXPECT_NEAR(polynomial.acceleration(0.0), 0.3, 1e-9);
  EXPECT_NEAR(polynomial.position(kDurationS), 6.2, 1e-9);
  EXPECT_NEAR(polynomial.velocity(kDurationS), 1.1, 1e-9);
  EXPECT_NEAR(polynomial.acceleration(kDurationS), -0.2, 1e-9);
}

TEST(FrenetGeometry, QuarticMeetsFiveBoundaryConditions) {
  constexpr double kDurationS = 3.25;
  const QuarticPolynomial polynomial(-2.0, 0.7, -0.1, 4.2, 0.35, kDurationS);

  EXPECT_NEAR(polynomial.position(0.0), -2.0, 1e-9);
  EXPECT_NEAR(polynomial.velocity(0.0), 0.7, 1e-9);
  EXPECT_NEAR(polynomial.acceleration(0.0), -0.1, 1e-9);
  EXPECT_NEAR(polynomial.velocity(kDurationS), 4.2, 1e-9);
  EXPECT_NEAR(polynomial.acceleration(kDurationS), 0.35, 1e-9);
}

TEST(FrenetGeometry, StraightLaneRoundTripPreservesPositionAndHeading) {
  const auto lane = straight_lane();
  const EgoState ego{Pose2{4.25, -1.2, 0.25}, 3.2, 0.0};

  const FrenetState projected = project_to_frenet(lane, ego);
  const auto reconstructed = frenet_to_cartesian(lane, projected, 0.4);

  EXPECT_NEAR(reconstructed.pose.x, ego.pose.x, 1e-6);
  EXPECT_NEAR(reconstructed.pose.y, ego.pose.y, 1e-6);
  EXPECT_NEAR(
      std::remainder(reconstructed.pose.yaw_rad - ego.pose.yaw_rad, kTwoPi),
      0.0, 1e-6);
  EXPECT_NEAR(reconstructed.speed_mps, ego.speed_mps, 1e-6);
  EXPECT_DOUBLE_EQ(reconstructed.time_from_start_s, 0.4);
}

TEST(FrenetGeometry, CurvedLaneRoundTripStaysWithinOneCentimeter) {
  const auto lane = circular_lane();
  const FrenetState original{5.123, 3.0, 0.1, 0.7, 0.2, -0.05};
  const auto cartesian = frenet_to_cartesian(lane, original, 0.0);
  const EgoState ego{cartesian.pose, cartesian.speed_mps, 0.0};

  const FrenetState projected = project_to_frenet(lane, ego);
  const auto reconstructed = frenet_to_cartesian(lane, projected, 0.0);

  EXPECT_LT(std::hypot(reconstructed.pose.x - cartesian.pose.x,
                       reconstructed.pose.y - cartesian.pose.y),
            0.01);
  EXPECT_NEAR(std::remainder(
                  reconstructed.pose.yaw_rad - cartesian.pose.yaw_rad, kTwoPi),
              0.0, 0.01);
}

TEST(FrenetGeometry, ProjectionUsesSignedLateralOffset) {
  const auto lane = straight_lane();

  const auto left =
      project_to_frenet(lane, EgoState{Pose2{3.0, 1.5, 0.0}, 2.0, 0.0});
  const auto right =
      project_to_frenet(lane, EgoState{Pose2{3.0, -1.5, 0.0}, 2.0, 0.0});

  EXPECT_NEAR(left.d_m, 1.5, 1e-9);
  EXPECT_NEAR(right.d_m, -1.5, 1e-9);
  EXPECT_NEAR(left.s_m, 3.0, 1e-9);
  EXPECT_NEAR(right.s_m, 3.0, 1e-9);
}

TEST(FrenetGeometry, RejectsDegenerateLaneAndNonpositiveDuration) {
  EXPECT_THROW((void)QuinticPolynomial(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
               std::invalid_argument);
  EXPECT_THROW((void)QuarticPolynomial(0.0, 0.0, 0.0, 1.0, 0.0, -1.0),
               std::invalid_argument);

  ReferenceLane too_short;
  too_short.points.push_back(reference_point(0.0, 0.0, 0.0, 0.0));
  EXPECT_THROW(project_to_frenet(too_short, EgoState{}), std::invalid_argument);
  EXPECT_THROW(frenet_to_cartesian(too_short, FrenetState{}, 0.0),
               std::invalid_argument);

  ReferenceLane zero_length;
  zero_length.points = {reference_point(0.0, 0.0, 0.0, 0.0),
                        reference_point(0.0, 0.0, 0.0, 1.0)};
  EXPECT_THROW(project_to_frenet(zero_length, EgoState{}),
               std::invalid_argument);

  auto nonmonotonic = straight_lane();
  nonmonotonic.points.back().route_s_m = 0.0;
  EXPECT_THROW(frenet_to_cartesian(nonmonotonic, FrenetState{}, 0.0),
               std::invalid_argument);
}

TEST(FrenetGeometry, RejectsNonfiniteAndOutOfRangeInputs) {
  const auto lane = straight_lane();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)QuinticPolynomial(0.0, 0.0, 0.0, 1.0, 0.0, nan, 1.0),
               std::invalid_argument);
  EXPECT_THROW((void)QuarticPolynomial(0.0, 0.0, 0.0, 1.0, 0.0, 1.0).jerk(nan),
               std::invalid_argument);
  EXPECT_THROW(
      project_to_frenet(lane, EgoState{Pose2{nan, 0.0, 0.0}, 0.0, 0.0}),
      std::invalid_argument);
  EXPECT_THROW(frenet_to_cartesian(
                   lane, FrenetState{-0.01, 0.0, 0.0, 0.0, 0.0, 0.0}, 0.0),
               std::out_of_range);
  EXPECT_THROW(frenet_to_cartesian(
                   lane, FrenetState{10.01, 0.0, 0.0, 0.0, 0.0, 0.0}, 0.0),
               std::out_of_range);
  EXPECT_THROW(frenet_to_cartesian(lane, FrenetState{}, -0.1),
               std::invalid_argument);
}

TEST(FrenetGeometry, WrapsReferenceYawAcrossPiContinuously) {
  ReferenceLane lane;
  lane.points = {reference_point(0.0, 0.0, 3.12413936106985, 0.0),
                 reference_point(-10.0, 0.0, -3.12413936106985, 10.0)};
  const auto midpoint =
      frenet_to_cartesian(lane, FrenetState{5.0, 2.0, 0.0, 0.0, 0.0, 0.0}, 0.0);

  EXPECT_NEAR(midpoint.pose.x, -5.0, 1e-9);
  EXPECT_NEAR(midpoint.pose.y, 0.0, 1e-9);
  EXPECT_NEAR(std::abs(midpoint.pose.yaw_rad), 3.14159265358979323846, 1e-9);
}

TEST(FrenetGeometry, PreservesYawRateThroughFrenetAcceleration) {
  const auto lane = straight_lane();
  const EgoState ego{Pose2{4.0, 0.5, 0.2}, 4.0, 0.3};

  const auto state = project_to_frenet(lane, ego);
  const auto cartesian = frenet_to_cartesian(lane, state, 0.0);

  EXPECT_NEAR(cartesian.curvature_inv_m, ego.yaw_rate_radps / ego.speed_mps,
              1e-9);
}

TEST(FrenetGeometry, RejectsSingularTransformAndInvalidReferenceSamples) {
  auto curved = straight_lane();
  for (auto &point : curved.points) {
    point.curvature_inv_m = 1.0;
  }
  EXPECT_THROW(frenet_to_cartesian(
                   curved, FrenetState{5.0, 1.0, 0.0, 1.0, 0.0, 0.0}, 0.0),
               std::domain_error);

  auto nonfinite_reference = straight_lane();
  nonfinite_reference.points.front().curvature_inv_m =
      std::numeric_limits<double>::infinity();
  EXPECT_THROW(project_to_frenet(nonfinite_reference, EgoState{}),
               std::invalid_argument);

  auto zero_length = straight_lane();
  zero_length.points.back().pose = zero_length.points.front().pose;
  EXPECT_THROW(frenet_to_cartesian(zero_length, FrenetState{}, 0.0),
               std::invalid_argument);

  auto nonmonotonic = straight_lane();
  nonmonotonic.points.back().route_s_m = -1.0;
  EXPECT_THROW(project_to_frenet(nonmonotonic, EgoState{}),
               std::invalid_argument);
}

TEST(FrenetGeometry, RejectsFiniteValuesWhoseDerivedLaneDeltasOverflow) {
  const double largest = std::numeric_limits<double>::max();

  ReferenceLane coordinate_overflow;
  coordinate_overflow.points = {reference_point(largest, 0.0, 0.0, 0.0),
                                reference_point(-largest, 0.0, 0.0, 1.0)};
  EXPECT_THROW(project_to_frenet(coordinate_overflow, EgoState{}),
               std::invalid_argument);
  EXPECT_THROW(frenet_to_cartesian(coordinate_overflow, FrenetState{}, 0.0),
               std::invalid_argument);

  ReferenceLane progress_overflow;
  progress_overflow.points = {reference_point(0.0, 0.0, 0.0, -largest),
                              reference_point(1.0, 0.0, 0.0, largest)};
  EXPECT_THROW(project_to_frenet(progress_overflow, EgoState{}),
               std::invalid_argument);
  EXPECT_THROW(
      frenet_to_cartesian(progress_overflow,
                          FrenetState{-largest, 0.0, 0.0, 0.0, 0.0, 0.0}, 0.0),
      std::invalid_argument);
}

TEST(FrenetGeometry,
     ProjectsExtremeFiniteSegmentWithoutSquaredDistanceOverflow) {
  ReferenceLane lane;
  lane.points = {reference_point(0.0, 0.0, 0.7853981633974483, 0.0),
                 reference_point(1e200, 1e200, 0.7853981633974483, 10.0)};
  const EgoState ego{Pose2{5e199, 5e199, 0.7853981633974483}, 3.0, 0.0};

  const auto state = project_to_frenet(lane, ego);

  EXPECT_NEAR(state.s_m, 5.0, 1e-12);
  EXPECT_NEAR(state.d_m, 0.0, 1e185);
  EXPECT_NEAR(state.s_dot_mps, 3.0, 1e-12);
}

TEST(FrenetGeometry, RejectsFiniteEgoCoordinatesWhenRelativeVectorOverflows) {
  const double largest = std::numeric_limits<double>::max();
  const double previous = std::nextafter(largest, 0.0);
  ReferenceLane lane;
  lane.points = {reference_point(previous, 0.0, 0.0, 0.0),
                 reference_point(largest, 0.0, 0.0, 1.0)};

  EXPECT_THROW(
      project_to_frenet(lane, EgoState{Pose2{-largest, 0.0, 0.0}, 0.0, 0.0}),
      std::invalid_argument);
}

TEST(FrenetGeometry, CurvedNonzeroOffsetRoundTripPreservesKinematics) {
  const auto lane = circular_lane();
  const FrenetState original{5.123, 3.0, 0.15, 0.7, 0.2, -0.05};
  const auto cartesian = frenet_to_cartesian(lane, original, 0.0);
  const EgoState ego{cartesian.pose, cartesian.speed_mps,
                     cartesian.speed_mps * cartesian.curvature_inv_m};

  const auto projected = project_to_frenet(lane, ego);
  const auto reconstructed = frenet_to_cartesian(lane, projected, 0.0);

  EXPECT_LT(std::hypot(reconstructed.pose.x - cartesian.pose.x,
                       reconstructed.pose.y - cartesian.pose.y),
            0.01);
  EXPECT_NEAR(std::remainder(
                  reconstructed.pose.yaw_rad - cartesian.pose.yaw_rad, kTwoPi),
              0.0, 0.01);
  EXPECT_NEAR(reconstructed.speed_mps, cartesian.speed_mps, 0.01);
  EXPECT_NEAR(reconstructed.curvature_inv_m, cartesian.curvature_inv_m, 1e-3);
}

} // namespace
