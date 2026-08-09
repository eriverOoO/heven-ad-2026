#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ad_control/path/route_speed_profile.hpp"

namespace
{
using namespace ad_control;

constexpr double kTerminalDecelerationMps2 = 2.5;
constexpr double kPointSpacingM = 0.5;

Route straight_route(double length_m, bool closed = false)
{
  Route route;
  const auto segment_count = static_cast<std::size_t>(length_m / kPointSpacingM);
  for (std::size_t index = 0; index <= segment_count; ++index) {
    route.points.push_back(
      Point3{static_cast<double>(index) * kPointSpacingM, 0.0, 0.0});
  }
  route.closed = closed;
  return route;
}

RouteSpeedProfileConfig rich_config()
{
  return RouteSpeedProfileConfig{
    0.1,
    100.0,
    1.0,
    100.0,
    2.0,
    1};
}

TEST(RouteSpeedProfile, ReturnsCurvatureAndSpeedWithMatchingRouteSize)
{
  const Route route{{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0},
    {2.0, 1.0, 0.0}, {2.0, 2.0, 0.0}}, false};

  const RouteSpeedProfile profile = build_route_speed_profile(route, rich_config());

  ASSERT_EQ(profile.speed_mps.size(), route.points.size());
  ASSERT_EQ(profile.curvature_inv_m.size(), route.points.size());
  EXPECT_NEAR(profile.curvature_inv_m[2], std::sqrt(2.0), 1.0e-12);
}

TEST(RouteSpeedProfile, ConfiguredDecelerationChangesReachableStopSpeed)
{
  const Route route = straight_route(3.0);
  RouteSpeedProfileConfig config = rich_config();
  config.deceleration_mps2 = 2.0;

  const RouteSpeedProfile slower = build_route_speed_profile(route, config);
  config.deceleration_mps2 = 8.0;
  const RouteSpeedProfile faster = build_route_speed_profile(route, config);

  const double remaining_distance = 3.0;
  EXPECT_NEAR(
    slower.speed_mps.front(),
    std::sqrt(2.0 * 2.0 * remaining_distance), 1.0e-12);
  EXPECT_NEAR(
    faster.speed_mps.front(),
    std::sqrt(2.0 * 8.0 * remaining_distance), 1.0e-12);
}

TEST(RouteSpeedProfile, MeasuredDecelerationTightensTheReachableSpeedEnvelope)
{
  const Route route = straight_route(10.0);
  RouteSpeedProfileConfig config = rich_config();
  config.maximum_speed_mps = 20.0;
  config.deceleration_mps2 = 10.0;
  const auto scalar = build_route_speed_profile(route, config);

  config.longitudinal_profile = LongitudinalProfile{
    {0.0, 20.0}, {10.0, 10.0}, {1.0, 1.0}, 0.0};
  const auto measured = build_route_speed_profile(route, config);

  EXPECT_FALSE(scalar.uses_longitudinal_profile);
  EXPECT_TRUE(measured.uses_longitudinal_profile);
  EXPECT_NEAR(measured.speed_mps.front(), std::sqrt(20.0), 1.0e-12);
  EXPECT_LT(measured.speed_mps.front(), scalar.speed_mps.front());
}

TEST(RouteSpeedProfile, MeasuredBrakeDelayMovesTheSlowdownEarlier)
{
  const Route route = straight_route(10.0);
  RouteSpeedProfileConfig config = rich_config();
  config.maximum_speed_mps = 10.0;
  config.deceleration_mps2 = 1.0;
  config.longitudinal_profile = LongitudinalProfile{
    {0.0, 10.0}, {10.0, 10.0}, {1.0, 1.0}, 0.0};
  const auto immediate = build_route_speed_profile(route, config);

  config.longitudinal_profile.braking_delay_s = 0.1;
  const auto delayed = build_route_speed_profile(route, config);

  EXPECT_LT(delayed.speed_mps.front(), immediate.speed_mps.front());
  EXPECT_NEAR(delayed.speed_mps.front(), std::sqrt(18.0), 1.0e-12);
}

TEST(RouteSpeedProfile, SpeedZoneCapsOnlyConfiguredForwardInterval)
{
  const Route route = straight_route(10.0, true);
  RouteSpeedProfileConfig config = rich_config();
  config.maximum_speed_mps = 10.0;
  config.lateral_acceleration_mps2 = 1000.0;
  config.acceleration_mps2 = 1000.0;
  config.deceleration_mps2 = 1000.0;
  config.speed_zones = {
    RouteSpeedZone{{4.0, 2.0, 0.0}, {6.0, -3.0, 0.0}, 3.0}};

  const auto profile = build_route_speed_profile(route, config);

  EXPECT_DOUBLE_EQ(profile.speed_mps[6], 10.0);
  for (std::size_t index = 8; index <= 12; ++index) {
    EXPECT_DOUBLE_EQ(profile.speed_mps[index], 3.0);
  }
  EXPECT_DOUBLE_EQ(profile.speed_mps[14], 10.0);
}

TEST(RouteSpeedProfile, SpeedZoneUsesMeasuredEnvelopeBeforeEntry)
{
  const Route route = straight_route(10.0, true);
  RouteSpeedProfileConfig config = rich_config();
  config.maximum_speed_mps = 10.0;
  config.lateral_acceleration_mps2 = 1000.0;
  config.acceleration_mps2 = 100.0;
  config.deceleration_mps2 = 100.0;
  config.longitudinal_profile = LongitudinalProfile{
    {0.0, 10.0}, {100.0, 100.0}, {1.0, 1.0}, 0.0};
  config.speed_zones = {
    RouteSpeedZone{{4.0, 0.0, 0.0}, {6.0, 0.0, 0.0}, 2.0}};

  const auto profile = build_route_speed_profile(route, config);

  EXPECT_NEAR(profile.speed_mps[7], std::sqrt(5.0), 1.0e-12);
  EXPECT_DOUBLE_EQ(profile.speed_mps[8], 2.0);
}

TEST(RouteSpeedProfile, DistantSpeedZoneDoesNotAffectAnUnrelatedRoute)
{
  const Route route = straight_route(10.0);
  RouteSpeedProfileConfig config = rich_config();
  config.speed_zones = {RouteSpeedZone{
    {38.868875371112615, -480.68740975673563, 0.0},
    {-81.83284234744308, -547.3316347631321, 0.0}, 8.3333333333}};

  const auto baseline = build_route_speed_profile(route, rich_config());
  const auto with_distant_zone = build_route_speed_profile(route, config);

  EXPECT_EQ(with_distant_zone.speed_mps, baseline.speed_mps);
}

TEST(RouteSpeedProfile, RejectsMalformedLongitudinalProfiles)
{
  const Route route = straight_route(3.0);
  RouteSpeedProfileConfig config = rich_config();
  config.longitudinal_profile = LongitudinalProfile{
    {0.0, 10.0}, {1.0}, {1.0, 1.0}, 0.0};
  EXPECT_THROW(build_route_speed_profile(route, config), std::invalid_argument);

  config.longitudinal_profile = LongitudinalProfile{
    {10.0, 5.0}, {1.0, 1.0}, {1.0, 1.0}, 0.0};
  EXPECT_THROW(build_route_speed_profile(route, config), std::invalid_argument);

  config.longitudinal_profile = LongitudinalProfile{
    {0.0, 10.0}, {1.0, 1.0}, {1.0, 1.0}, -0.1};
  EXPECT_THROW(build_route_speed_profile(route, config), std::invalid_argument);
}

TEST(RouteSpeedProfile, RejectsInvalidLimitsAndZeroWindow)
{
  const Route route = straight_route(3.0);
  RouteSpeedProfileConfig config = rich_config();

  config.minimum_speed_mps = -0.1;
  EXPECT_THROW(build_route_speed_profile(route, config), std::invalid_argument);
  config = rich_config();
  config.maximum_speed_mps = config.minimum_speed_mps - 0.01;
  EXPECT_THROW(build_route_speed_profile(route, config), std::invalid_argument);
  config = rich_config();
  config.lateral_acceleration_mps2 = 0.0;
  EXPECT_THROW(build_route_speed_profile(route, config), std::invalid_argument);
  config = rich_config();
  config.acceleration_mps2 = 0.0;
  EXPECT_THROW(build_route_speed_profile(route, config), std::invalid_argument);
  config = rich_config();
  config.deceleration_mps2 = 0.0;
  EXPECT_THROW(build_route_speed_profile(route, config), std::invalid_argument);
  config = rich_config();
  config.curvature_window_radius = 0;
  EXPECT_THROW(build_route_speed_profile(route, config), std::invalid_argument);
}

TEST(RouteSpeedProfile, HugeClosedWindowReturnsGlobalCurvatureWithoutRepeatingRoute)
{
  Route route;
  constexpr std::size_t kPointCount = 5;
  for (std::size_t index = 0; index < kPointCount; ++index) {
    const double angle =
      2.0 * std::acos(-1.0) * static_cast<double>(index) /
      static_cast<double>(kPointCount);
    route.points.push_back(Point3{std::cos(angle), std::sin(angle), 0.0});
  }
  route.closed = true;
  RouteSpeedProfileConfig config = rich_config();
  config.curvature_window_radius = std::numeric_limits<std::size_t>::max();

  const RouteSpeedProfile profile = build_route_speed_profile(route, config);

  ASSERT_EQ(profile.curvature_inv_m.size(), route.points.size());
  ASSERT_EQ(profile.speed_mps.size(), route.points.size());
  for (std::size_t index = 0; index < route.points.size(); ++index) {
    EXPECT_NEAR(profile.curvature_inv_m[index], 1.0, 1.0e-12);
    EXPECT_NEAR(profile.speed_mps[index], 1.0, 1.0e-12);
  }
}

TEST(RouteSpeedProfile, OpenRouteBackPropagatesAZeroTerminalSpeed)
{
  const Route route = straight_route(5.0);

  const auto profile = build_route_speed_profile(route, 1000.0);

  ASSERT_EQ(profile.size(), route.points.size());
  EXPECT_DOUBLE_EQ(profile.back(), 0.0);
  for (std::size_t index = 0; index < profile.size(); ++index) {
    const double remaining_distance =
      route.points.back().x - route.points[index].x;
    const double expected_speed = std::sqrt(
      2.0 * kTerminalDecelerationMps2 * remaining_distance);
    EXPECT_NEAR(profile[index], expected_speed, 1.0e-12);
  }
}

TEST(RouteSpeedProfile, FourMetersPerSecondIsGovernedByRemainingStopDistance)
{
  const Route route = straight_route(10.0);
  const auto profile = build_route_speed_profile(route, 1000.0);

  const std::size_t three_and_a_half_meters_remaining = profile.size() - 8;
  const std::size_t three_meters_remaining = profile.size() - 7;
  EXPECT_GT(profile[three_and_a_half_meters_remaining], 4.0);
  EXPECT_LT(profile[three_meters_remaining], 4.0);
}

TEST(RouteSpeedProfile, ClosedRouteKeepsAContinuousNonzeroProfile)
{
  const auto profile = build_route_speed_profile(straight_route(5.0, true), 1000.0);

  ASSERT_FALSE(profile.empty());
  for (const double speed : profile) {
    EXPECT_GT(speed, 0.0);
  }
}
}  // namespace
