#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include "ad_morai_interfaces/msg/ego_vehicle_status.hpp"
#include "ad_morai_interfaces/msg/gps_rmc.hpp"
#include "ad_localization/adapter/localization_adapter.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "sensor_msgs/msg/nav_sat_status.hpp"

namespace
{

using ad_morai_interfaces::msg::EgoVehicleStatus;
using ad_morai_interfaces::msg::GpsRmc;
using ad_localization::AdapterConfig;
using ad_localization::LocalizationAdapter;
using geometry_msgs::msg::PoseStamped;
using nav_msgs::msg::Odometry;
using sensor_msgs::msg::Imu;
using sensor_msgs::msg::NavSatFix;
using sensor_msgs::msg::NavSatStatus;

constexpr double kPi = 3.14159265358979323846;
constexpr double kGridYawCorrectionRad = -0.02350724531030645;

TEST(AdapterConfig, DefaultsMatchFinalMoraiGpsMount)
{
  const AdapterConfig config;
  EXPECT_EQ(config.gnss_lever_arm_m, (std::array<double, 3>{0.0, 0.0, 1.5685}));
  EXPECT_EQ(config.initial_orientation_source, "imu");
  EXPECT_DOUBLE_EQ(config.initial_orientation_yaw_offset_rad, 0.0);
  EXPECT_EQ(config.initial_position_sample_count, 1);
  EXPECT_FALSE(config.initial_position_override_xy_m.has_value());
  EXPECT_FALSE(config.gps_course_enabled);
  EXPECT_DOUBLE_EQ(config.gps_course_maximum_age_sec, 0.25);
  EXPECT_DOUBLE_EQ(config.gps_course_yaw_offset_rad, 0.0);
  EXPECT_DOUBLE_EQ(config.wheel_lateral_speed_variance, 0.04);
  EXPECT_FALSE(config.wheel_use_device_timestamp);
  EXPECT_DOUBLE_EQ(config.maximum_abs_sideslip_rad, 0.35);
  EXPECT_DOUBLE_EQ(config.maximum_map_radius_m, 100000.0);
}

builtin_interfaces::msg::Time stamp(std::int32_t seconds, std::uint32_t nanoseconds = 0)
{
  builtin_interfaces::msg::Time result;
  result.sec = seconds;
  result.nanosec = nanoseconds;
  return result;
}

NavSatFix valid_fix(std::int32_t seconds = 10)
{
  NavSatFix fix;
  fix.header.stamp = stamp(seconds);
  fix.header.frame_id = "gps_link";
  fix.status.status = NavSatStatus::STATUS_FIX;
  fix.latitude = 37.2390904486269;
  fix.longitude = 126.773066537479;
  fix.altitude = 29.5769634246826;
  return fix;
}

EgoVehicleStatus moving_status(double velocity_x, std::int8_t gear, std::int32_t seconds = 10)
{
  EgoVehicleStatus status;
  status.header.stamp = stamp(seconds);
  status.velocity.x = velocity_x;
  status.gear = gear;
  return status;
}

GpsRmc valid_rmc(
  double track_degrees, std::int32_t seconds = 10, std::uint32_t nanoseconds = 0)
{
  GpsRmc rmc;
  rmc.header.stamp = stamp(seconds, nanoseconds);
  rmc.header.frame_id = "gps_link";
  rmc.valid = true;
  rmc.has_track = true;
  rmc.track_degrees = track_degrees;
  return rmc;
}

geometry_msgs::msg::Quaternion yaw_quaternion(double yaw, double scale = 1.0)
{
  geometry_msgs::msg::Quaternion quaternion;
  quaternion.z = std::sin(yaw * 0.5) * scale;
  quaternion.w = std::cos(yaw * 0.5) * scale;
  return quaternion;
}

TEST(ImuConvention, LeftMultipliesWorldYawAndPreservesEveryOtherField)
{
  Imu imu;
  imu.header.stamp = stamp(42, 123456789);
  imu.header.frame_id = "imu_link";
  imu.orientation.x = std::sqrt(0.5);
  imu.orientation.w = std::sqrt(0.5);
  imu.angular_velocity.x = 1.25;
  imu.angular_velocity.y = -2.5;
  imu.angular_velocity.z = 3.75;
  imu.linear_acceleration.x = -4.25;
  imu.linear_acceleration.y = 5.5;
  imu.linear_acceleration.z = 9.80665;
  for (std::size_t index = 0; index < 9; ++index) {
    imu.orientation_covariance[index] = 10.0 + static_cast<double>(index);
    imu.angular_velocity_covariance[index] = 20.0 + static_cast<double>(index);
    imu.linear_acceleration_covariance[index] = 30.0 + static_cast<double>(index);
  }

  const auto corrected =
    ad_localization::apply_world_yaw_correction(imu, -0.2);

  ASSERT_TRUE(corrected.has_value());
  EXPECT_NEAR(corrected->orientation.x, 0.7035741925769523, 1.0e-15);
  EXPECT_NEAR(corrected->orientation.y, -0.07059288589999414, 1.0e-15);
  EXPECT_NEAR(corrected->orientation.z, -0.07059288589999414, 1.0e-15);
  EXPECT_NEAR(corrected->orientation.w, 0.7035741925769523, 1.0e-15);
  EXPECT_EQ(corrected->header, imu.header);
  EXPECT_EQ(corrected->angular_velocity, imu.angular_velocity);
  EXPECT_EQ(corrected->linear_acceleration, imu.linear_acceleration);
  EXPECT_EQ(corrected->orientation_covariance, imu.orientation_covariance);
  EXPECT_EQ(corrected->angular_velocity_covariance, imu.angular_velocity_covariance);
  EXPECT_EQ(corrected->linear_acceleration_covariance, imu.linear_acceleration_covariance);
}

TEST(ImuConvention, RejectsInvalidOrientationOrYaw)
{
  Imu imu;
  imu.orientation.w = 1.0;
  EXPECT_FALSE(ad_localization::apply_world_yaw_correction(
      imu, std::numeric_limits<double>::quiet_NaN()));

  imu.orientation.w = 0.0;
  EXPECT_FALSE(ad_localization::apply_world_yaw_correction(
      imu, kGridYawCorrectionRad));

  imu.orientation.x = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(ad_localization::apply_world_yaw_correction(
      imu, kGridYawCorrectionRad));
}

TEST(ImuConvention, CorrectedImuInitializesWithTheSameWorldYaw)
{
  AdapterConfig config;
  config.gnss_lever_arm_m = {0.0, 0.0, 0.0};
  LocalizationAdapter adapter(config);

  PoseStamped gnss_pose;
  gnss_pose.header.stamp = stamp(10);
  gnss_pose.header.frame_id = "odom";
  gnss_pose.pose.orientation.w = 1.0;
  ASSERT_FALSE(adapter.observe_gnss_for_initialization(gnss_pose));

  Imu raw_imu;
  raw_imu.header.stamp = stamp(10, 100000000);
  raw_imu.header.frame_id = "imu_link";
  raw_imu.orientation.w = 1.0;
  const auto corrected = ad_localization::apply_world_yaw_correction(
    raw_imu, kGridYawCorrectionRad);
  ASSERT_TRUE(corrected.has_value());

  const auto initial = adapter.observe_imu_for_initialization(*corrected);

  ASSERT_TRUE(initial.has_value());
  EXPECT_DOUBLE_EQ(initial->pose.orientation.x, corrected->orientation.x);
  EXPECT_DOUBLE_EQ(initial->pose.orientation.y, corrected->orientation.y);
  EXPECT_DOUBLE_EQ(initial->pose.orientation.z, corrected->orientation.z);
  EXPECT_DOUBLE_EQ(initial->pose.orientation.w, corrected->orientation.w);
  EXPECT_NEAR(initial->pose.orientation.z, -0.011753352034473157, 1.0e-15);
  EXPECT_NEAR(initial->pose.orientation.w, 0.9999309269724354, 1.0e-15);
}

TEST(LocalizationAdapter, ConvertsKnownWgs84FixToOffsetUtm52N)
{
  LocalizationAdapter adapter(AdapterConfig{});

  const auto converted = adapter.convert_gnss(valid_fix());

  ASSERT_TRUE(converted.has_value());
  EXPECT_EQ(converted->header.stamp, stamp(10));
  EXPECT_EQ(converted->header.frame_id, "odom");
  EXPECT_NEAR(converted->pose.position.x, -130.088119408, 1.0e-3);
  EXPECT_NEAR(converted->pose.position.y, -425.305177602, 1.0e-3);
  EXPECT_NEAR(converted->pose.position.z, 29.5769634246826, 1.0e-9);
  EXPECT_DOUBLE_EQ(converted->pose.orientation.w, 1.0);
}

TEST(LocalizationAdapter, RejectsInvalidAndNonMonotonicGnssWithoutAdvancingState)
{
  LocalizationAdapter adapter(AdapterConfig{});
  auto no_fix = valid_fix(1);
  no_fix.status.status = NavSatStatus::STATUS_NO_FIX;
  EXPECT_FALSE(adapter.convert_gnss(no_fix));

  auto zero_stamp = valid_fix(1);
  zero_stamp.header.stamp = stamp(0);
  EXPECT_FALSE(adapter.convert_gnss(zero_stamp));

  auto bad_altitude = valid_fix(1);
  bad_altitude.altitude = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(adapter.convert_gnss(bad_altitude));

  auto bad_latitude = valid_fix(1);
  bad_latitude.latitude = 90.1;
  EXPECT_FALSE(adapter.convert_gnss(bad_latitude));

  auto bad_longitude = valid_fix(1);
  bad_longitude.longitude = -180.1;
  EXPECT_FALSE(adapter.convert_gnss(bad_longitude));

  auto morai_gps_shadow = valid_fix(1);
  morai_gps_shadow.latitude = 0.0;
  morai_gps_shadow.longitude = 0.0;
  EXPECT_FALSE(adapter.convert_gnss(morai_gps_shadow));

  ASSERT_TRUE(adapter.convert_gnss(valid_fix(2)));
  EXPECT_FALSE(adapter.convert_gnss(valid_fix(2)));
  EXPECT_FALSE(adapter.convert_gnss(valid_fix(1)));
  EXPECT_TRUE(adapter.convert_gnss(valid_fix(3)));
}

TEST(LocalizationAdapter, RejectsProjectedFixOutsideConfiguredMapRadius)
{
  AdapterConfig config;
  config.maximum_map_radius_m = 10.0;
  LocalizationAdapter adapter(config);

  EXPECT_FALSE(adapter.convert_gnss(valid_fix()));
}

TEST(LocalizationAdapter, AppliesGearDirectionToMetresPerSecondMagnitude)
{
  struct Case
  {
    double input;
    std::int8_t gear;
    double expected;
  };
  const Case cases[] = {
    {-7.2, 4, 7.2},
    {7.2, 2, -7.2},
    {-4.5, 5, 4.5},
    {36.0, 4, 36.0},
  };

  std::int32_t seconds = 1;
  for (const auto & test_case : cases) {
    LocalizationAdapter adapter(AdapterConfig{});
    const auto converted = adapter.convert_wheel_speed(
      moving_status(test_case.input, test_case.gear, seconds++));
    ASSERT_TRUE(converted.has_value());
    EXPECT_EQ(converted->header.frame_id, "base_link");
    EXPECT_DOUBLE_EQ(converted->twist.twist.linear.x, test_case.expected);
    EXPECT_DOUBLE_EQ(converted->twist.twist.linear.y, 0.0);
    EXPECT_DOUBLE_EQ(converted->twist.twist.linear.z, 0.0);
    EXPECT_DOUBLE_EQ(converted->twist.covariance[0], 0.04);
  }
}

TEST(LocalizationAdapter, PublishesZeroAtStandstillAndDropsAmbiguousMovingGears)
{
  for (const auto gear : {0, 1, 2, 3, 4, 5, 6, 127}) {
    LocalizationAdapter adapter(AdapterConfig{});
    const auto stopped = adapter.convert_wheel_speed(moving_status(0.05, gear));
    ASSERT_TRUE(stopped.has_value());
    EXPECT_DOUBLE_EQ(stopped->twist.twist.linear.x, 0.0);
  }

  for (const auto gear : {0, 1, 3, 6, 127}) {
    LocalizationAdapter adapter(AdapterConfig{});
    EXPECT_FALSE(adapter.convert_wheel_speed(moving_status(1.0, gear)));
  }
}

TEST(LocalizationAdapter, RejectsInvalidWheelSamples)
{
  LocalizationAdapter adapter(AdapterConfig{});
  auto zero_stamp = moving_status(1.0, 4);
  zero_stamp.header.stamp = stamp(0);
  EXPECT_FALSE(adapter.convert_wheel_speed(zero_stamp));

  auto not_finite = moving_status(std::numeric_limits<double>::infinity(), 4);
  EXPECT_FALSE(adapter.convert_wheel_speed(not_finite));

  EXPECT_TRUE(adapter.convert_wheel_speed(moving_status(1.0, 4, 2)));
  EXPECT_FALSE(adapter.convert_wheel_speed(moving_status(1.0, 4, 2)));
  EXPECT_FALSE(adapter.convert_wheel_speed(moving_status(1.0, 4, 1)));
  EXPECT_TRUE(adapter.convert_wheel_speed(moving_status(1.0, 4, 3)));
}

TEST(LocalizationAdapter, UsesFreshNmeaCourseForForwardBodyLateralSpeed)
{
  AdapterConfig config;
  config.gps_course_enabled = true;
  LocalizationAdapter adapter(config);

  // NMEA course is clockwise from north. 84.2704 degrees converts to a
  // ROS world yaw of +0.1 rad when the configured offset is zero.
  const double course_yaw = 0.1;
  const double track_degrees = (kPi / 2.0 - course_yaw) * 180.0 / kPi;
  ASSERT_TRUE(adapter.observe_gps_rmc(valid_rmc(track_degrees, 10)));
  auto status = moving_status(5.0, 4, 10);
  status.header.stamp.nanosec = 100000000;
  status.rpy.z = 0.0;

  const auto converted = adapter.convert_wheel_speed(status);

  ASSERT_TRUE(converted.has_value());
  EXPECT_DOUBLE_EQ(converted->twist.twist.linear.x, 5.0);
  EXPECT_NEAR(converted->twist.twist.linear.y, 5.0 * std::sin(course_yaw), 1.0e-12);
  EXPECT_DOUBLE_EQ(converted->twist.covariance[0], 0.04);
  EXPECT_DOUBLE_EQ(converted->twist.covariance[7], 0.04);
}

TEST(LocalizationAdapter, AppliesCourseYawOffsetAndReverseTravelDirection)
{
  AdapterConfig offset_config;
  offset_config.gps_course_enabled = true;
  offset_config.gps_course_yaw_offset_rad = kPi;
  LocalizationAdapter offset_adapter(offset_config);
  ASSERT_TRUE(offset_adapter.observe_gps_rmc(valid_rmc(0.0, 10)));
  auto offset_status = moving_status(2.0, 4, 10);
  offset_status.header.stamp.nanosec = 100000000;
  offset_status.rpy.z = -kPi / 2.0;
  const auto offset_wheel = offset_adapter.convert_wheel_speed(offset_status);
  ASSERT_TRUE(offset_wheel.has_value());
  EXPECT_NEAR(offset_wheel->twist.twist.linear.y, 0.0, 1.0e-12);

  AdapterConfig reverse_config;
  reverse_config.gps_course_enabled = true;
  LocalizationAdapter reverse_adapter(reverse_config);
  const double reverse_course_yaw = -kPi + 0.1;
  const double reverse_track_degrees =
    std::fmod((kPi / 2.0 - reverse_course_yaw) * 180.0 / kPi, 360.0);
  ASSERT_TRUE(reverse_adapter.observe_gps_rmc(valid_rmc(reverse_track_degrees, 20)));
  auto reverse_status = moving_status(5.0, 2, 20);
  reverse_status.header.stamp.nanosec = 100000000;
  reverse_status.rpy.z = 0.0;

  const auto reverse_wheel = reverse_adapter.convert_wheel_speed(reverse_status);

  ASSERT_TRUE(reverse_wheel.has_value());
  EXPECT_DOUBLE_EQ(reverse_wheel->twist.twist.linear.x, -5.0);
  EXPECT_NEAR(
    reverse_wheel->twist.twist.linear.y,
    5.0 * std::sin(reverse_course_yaw), 1.0e-12);
}

TEST(LocalizationAdapter, RejectsInvalidAndNonMonotonicNmeaCourse)
{
  AdapterConfig config;
  config.gps_course_enabled = true;
  LocalizationAdapter adapter(config);

  auto zero_stamp = valid_rmc(90.0, 0);
  EXPECT_FALSE(adapter.observe_gps_rmc(zero_stamp));
  auto invalid_fix = valid_rmc(90.0, 1);
  invalid_fix.valid = false;
  EXPECT_FALSE(adapter.observe_gps_rmc(invalid_fix));
  auto no_track = valid_rmc(90.0, 1);
  no_track.has_track = false;
  EXPECT_FALSE(adapter.observe_gps_rmc(no_track));
  EXPECT_FALSE(adapter.observe_gps_rmc(valid_rmc(-0.01, 1)));
  EXPECT_FALSE(adapter.observe_gps_rmc(valid_rmc(360.0, 1)));
  EXPECT_FALSE(adapter.observe_gps_rmc(
    valid_rmc(std::numeric_limits<double>::quiet_NaN(), 1)));

  EXPECT_TRUE(adapter.observe_gps_rmc(valid_rmc(90.0, 2)));
  EXPECT_FALSE(adapter.observe_gps_rmc(valid_rmc(91.0, 2)));
  EXPECT_FALSE(adapter.observe_gps_rmc(valid_rmc(91.0, 1)));
  EXPECT_TRUE(adapter.observe_gps_rmc(valid_rmc(91.0, 3)));
}

TEST(LocalizationAdapter, CourseEnabledMovingWheelFailsClosedWithoutSafeFreshCourse)
{
  AdapterConfig config;
  config.gps_course_enabled = true;

  LocalizationAdapter missing_adapter(config);
  EXPECT_FALSE(missing_adapter.convert_wheel_speed(moving_status(2.0, 4, 10)));

  LocalizationAdapter stale_adapter(config);
  ASSERT_TRUE(stale_adapter.observe_gps_rmc(valid_rmc(90.0, 10)));
  auto stale_status = moving_status(2.0, 4, 10);
  stale_status.header.stamp.nanosec = 250000001;
  stale_status.rpy.z = 0.0;
  EXPECT_FALSE(stale_adapter.convert_wheel_speed(stale_status));

  LocalizationAdapter future_adapter(config);
  ASSERT_TRUE(future_adapter.observe_gps_rmc(valid_rmc(90.0, 10, 1)));
  auto earlier_status = moving_status(2.0, 4, 10);
  earlier_status.rpy.z = 0.0;
  EXPECT_FALSE(future_adapter.convert_wheel_speed(earlier_status));

  LocalizationAdapter sideslip_adapter(config);
  ASSERT_TRUE(sideslip_adapter.observe_gps_rmc(valid_rmc(60.0, 10)));
  auto excessive_sideslip = moving_status(2.0, 4, 10);
  excessive_sideslip.header.stamp.nanosec = 100000000;
  excessive_sideslip.rpy.z = 0.0;
  EXPECT_FALSE(sideslip_adapter.convert_wheel_speed(excessive_sideslip));

  LocalizationAdapter invalid_heading_adapter(config);
  ASSERT_TRUE(invalid_heading_adapter.observe_gps_rmc(valid_rmc(90.0, 10)));
  auto invalid_heading = moving_status(2.0, 4, 10);
  invalid_heading.header.stamp.nanosec = 100000000;
  invalid_heading.rpy.z = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(invalid_heading_adapter.convert_wheel_speed(invalid_heading));
}

TEST(LocalizationAdapter, CourseEnabledStandstillDoesNotRequireCourseOrHeading)
{
  AdapterConfig config;
  config.gps_course_enabled = true;
  LocalizationAdapter adapter(config);
  auto stopped = moving_status(0.05, 0, 10);
  stopped.rpy.z = std::numeric_limits<double>::quiet_NaN();

  const auto converted = adapter.convert_wheel_speed(stopped);

  ASSERT_TRUE(converted.has_value());
  EXPECT_DOUBLE_EQ(converted->twist.twist.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(converted->twist.twist.linear.y, 0.0);
  EXPECT_DOUBLE_EQ(converted->twist.covariance[7], 0.04);
}

TEST(LocalizationAdapter, RejectsInvalidCourseConfiguration)
{
  AdapterConfig config;
  config.gps_course_maximum_age_sec = 0.0;
  EXPECT_THROW((void)LocalizationAdapter{config}, std::invalid_argument);

  config = AdapterConfig{};
  config.wheel_lateral_speed_variance = 0.0;
  EXPECT_THROW((void)LocalizationAdapter{config}, std::invalid_argument);

  config = AdapterConfig{};
  config.maximum_abs_sideslip_rad = kPi + 0.01;
  EXPECT_THROW((void)LocalizationAdapter{config}, std::invalid_argument);

  config = AdapterConfig{};
  config.gps_course_yaw_offset_rad = std::numeric_limits<double>::infinity();
  EXPECT_THROW((void)LocalizationAdapter{config}, std::invalid_argument);
}

TEST(LocalizationAdapter, BuildsOneBodyInitialPoseFromSynchronizedGnssAndImu)
{
  LocalizationAdapter adapter(AdapterConfig{});
  PoseStamped gnss_pose;
  gnss_pose.header.stamp = stamp(10);
  gnss_pose.header.frame_id = "odom";
  gnss_pose.pose.position.x = 10.0;
  gnss_pose.pose.position.y = 20.0;
  gnss_pose.pose.position.z = 30.0;
  gnss_pose.pose.orientation.w = 1.0;

  Imu imu;
  imu.header.stamp = stamp(10, 100000000);
  imu.header.frame_id = "imu_link";
  imu.orientation = yaw_quaternion(kPi / 2.0, 2.0);

  EXPECT_FALSE(adapter.observe_gnss_for_initialization(gnss_pose));
  const auto initial = adapter.observe_imu_for_initialization(imu);
  ASSERT_TRUE(initial.has_value());
  EXPECT_EQ(initial->header.frame_id, "odom");
  EXPECT_NEAR(initial->pose.position.x, 10.0, 1.0e-12);
  EXPECT_NEAR(initial->pose.position.y, 20.0, 1.0e-12);
  EXPECT_NEAR(initial->pose.position.z, 28.4315, 1.0e-12);
  EXPECT_NEAR(initial->pose.orientation.z, std::sqrt(0.5), 1.0e-12);
  EXPECT_NEAR(initial->pose.orientation.w, std::sqrt(0.5), 1.0e-12);

  ASSERT_TRUE(adapter.pending_initial_pose());
  EXPECT_EQ(adapter.pending_initial_pose()->header.stamp, initial->header.stamp);
  adapter.acknowledge_filtered_odometry();
  EXPECT_FALSE(adapter.pending_initial_pose());
  EXPECT_FALSE(adapter.observe_gnss_for_initialization(gnss_pose));
  EXPECT_FALSE(adapter.observe_imu_for_initialization(imu));
}

TEST(LocalizationAdapter, RejectsInvalidOrUnsynchronizedInitialOrientation)
{
  PoseStamped gnss_pose;
  gnss_pose.header.stamp = stamp(10);
  gnss_pose.header.frame_id = "odom";
  gnss_pose.pose.orientation.w = 1.0;

  Imu zero_quaternion;
  zero_quaternion.header.stamp = stamp(10, 100000000);
  zero_quaternion.orientation.w = 0.0;
  LocalizationAdapter invalid_adapter(AdapterConfig{});
  EXPECT_FALSE(invalid_adapter.observe_gnss_for_initialization(gnss_pose));
  EXPECT_FALSE(invalid_adapter.observe_imu_for_initialization(zero_quaternion));

  Imu late_imu;
  late_imu.header.stamp = stamp(11);
  late_imu.orientation.w = 1.0;
  LocalizationAdapter unsynchronized_adapter(AdapterConfig{});
  EXPECT_FALSE(unsynchronized_adapter.observe_gnss_for_initialization(gnss_pose));
  EXPECT_FALSE(unsynchronized_adapter.observe_imu_for_initialization(late_imu));
  EXPECT_FALSE(unsynchronized_adapter.pending_initial_pose());
}

TEST(LocalizationAdapter, UsesVehicleStatusRpyForFastlioInitialOrientation)
{
  AdapterConfig config;
  config.initial_orientation_source = "vehicle_status";
  config.gnss_lever_arm_m = {1.0, 0.0, 0.0};
  LocalizationAdapter adapter(config);

  PoseStamped gnss_pose;
  gnss_pose.header.stamp = stamp(10);
  gnss_pose.header.frame_id = "odom";
  gnss_pose.pose.position.x = 10.0;
  gnss_pose.pose.position.y = 20.0;
  gnss_pose.pose.position.z = 30.0;

  Imu misleading_imu;
  misleading_imu.header.stamp = stamp(10, 50000000);
  misleading_imu.orientation = yaw_quaternion(1.1);

  EgoVehicleStatus status;
  status.header.stamp = stamp(10, 100000000);
  status.rpy.z = -2.4;

  EXPECT_FALSE(adapter.observe_gnss_for_initialization(gnss_pose));
  EXPECT_FALSE(adapter.observe_imu_for_initialization(misleading_imu));
  const auto initial = adapter.observe_vehicle_status_for_initialization(status);

  ASSERT_TRUE(initial.has_value());
  EXPECT_NEAR(initial->pose.orientation.z, std::sin(-1.2), 1.0e-12);
  EXPECT_NEAR(initial->pose.orientation.w, std::cos(-1.2), 1.0e-12);
  EXPECT_NEAR(initial->pose.position.x, 10.0 - std::cos(-2.4), 1.0e-12);
  EXPECT_NEAR(initial->pose.position.y, 20.0 - std::sin(-2.4), 1.0e-12);
  EXPECT_NEAR(initial->pose.position.z, 30.0, 1.0e-12);
}

TEST(LocalizationAdapter, RotatesInitialOrientationIntoTheReferenceGrid)
{
  AdapterConfig config;
  config.initial_orientation_source = "vehicle_status";
  config.initial_orientation_yaw_offset_rad = -0.1;
  config.gnss_lever_arm_m = {1.0, 0.0, 0.0};
  LocalizationAdapter adapter(config);

  PoseStamped gnss_pose;
  gnss_pose.header.stamp = stamp(10);
  gnss_pose.header.frame_id = "odom";
  gnss_pose.pose.position.x = 10.0;
  gnss_pose.pose.position.y = 20.0;

  EgoVehicleStatus status;
  status.header.stamp = stamp(10, 100000000);
  status.rpy.z = 0.5;

  EXPECT_FALSE(adapter.observe_gnss_for_initialization(gnss_pose));
  const auto initial = adapter.observe_vehicle_status_for_initialization(status);

  ASSERT_TRUE(initial.has_value());
  EXPECT_NEAR(initial->pose.orientation.z, std::sin(0.2), 1.0e-12);
  EXPECT_NEAR(initial->pose.orientation.w, std::cos(0.2), 1.0e-12);
  EXPECT_NEAR(initial->pose.position.x, 10.0 - std::cos(0.4), 1.0e-12);
  EXPECT_NEAR(initial->pose.position.y, 20.0 - std::sin(0.4), 1.0e-12);
}

TEST(LocalizationAdapter, UsesKnownCheckpointBodyXYWithoutReplacingGnssAltitude)
{
  AdapterConfig config;
  config.initial_orientation_source = "vehicle_status";
  config.gnss_lever_arm_m = {1.0, 0.0, 1.2};
  config.initial_position_override_xy_m =
    std::array<double, 2>{38.868875371112615, -480.68740975673563};
  LocalizationAdapter adapter(config);

  PoseStamped gnss_pose;
  gnss_pose.header.stamp = stamp(10);
  gnss_pose.header.frame_id = "odom";
  gnss_pose.pose.position.x = 100.0;
  gnss_pose.pose.position.y = 200.0;
  gnss_pose.pose.position.z = 30.0;

  EgoVehicleStatus status;
  status.header.stamp = stamp(10, 100000000);
  status.rpy.z = kPi / 2.0;

  EXPECT_FALSE(adapter.observe_gnss_for_initialization(gnss_pose));
  const auto initial = adapter.observe_vehicle_status_for_initialization(status);

  ASSERT_TRUE(initial.has_value());
  EXPECT_NEAR(initial->pose.position.x, 38.868875371112615, 1.0e-12);
  EXPECT_NEAR(initial->pose.position.y, -480.68740975673563, 1.0e-12);
  EXPECT_NEAR(initial->pose.position.z, 28.8, 1.0e-12);
  EXPECT_NEAR(initial->pose.orientation.z, std::sqrt(0.5), 1.0e-12);
  EXPECT_NEAR(initial->pose.orientation.w, std::sqrt(0.5), 1.0e-12);
}

TEST(LocalizationAdapter, AveragesConfiguredGnssSamplesBeforeInitialization)
{
  AdapterConfig config;
  config.gnss_lever_arm_m = {0.0, 0.0, 0.0};
  config.initial_position_sample_count = 3;
  LocalizationAdapter adapter(config);

  Imu imu;
  imu.header.stamp = stamp(10, 100000000);
  imu.orientation.w = 1.0;

  PoseStamped pose;
  pose.header.frame_id = "odom";
  pose.pose.position.z = 30.0;
  pose.header.stamp = stamp(10);
  pose.pose.position.x = 9.0;
  pose.pose.position.y = 18.0;
  EXPECT_FALSE(adapter.observe_gnss_for_initialization(pose));
  EXPECT_FALSE(adapter.observe_imu_for_initialization(imu));

  pose.header.stamp = stamp(10, 50000000);
  pose.pose.position.x = 12.0;
  pose.pose.position.y = 21.0;
  EXPECT_FALSE(adapter.observe_gnss_for_initialization(pose));

  pose.header.stamp = stamp(10, 100000000);
  pose.pose.position.x = 15.0;
  pose.pose.position.y = 24.0;
  const auto initial = adapter.observe_gnss_for_initialization(pose);
  ASSERT_TRUE(initial.has_value());
  EXPECT_NEAR(initial->pose.position.x, 12.0, 1.0e-12);
  EXPECT_NEAR(initial->pose.position.y, 21.0, 1.0e-12);
  EXPECT_NEAR(initial->pose.position.z, 30.0, 1.0e-12);
}

TEST(LocalizationAdapter, RejectsInvalidVehicleStatusInitialOrientation)
{
  AdapterConfig config;
  config.initial_orientation_source = "vehicle_status";

  PoseStamped gnss_pose;
  gnss_pose.header.stamp = stamp(10);
  gnss_pose.header.frame_id = "odom";

  LocalizationAdapter invalid_adapter(config);
  ASSERT_FALSE(invalid_adapter.observe_gnss_for_initialization(gnss_pose));
  auto invalid_status = moving_status(0.0, 1, 10);
  invalid_status.rpy.z = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(invalid_adapter.observe_vehicle_status_for_initialization(invalid_status));

  LocalizationAdapter unsynchronized_adapter(config);
  ASSERT_FALSE(unsynchronized_adapter.observe_gnss_for_initialization(gnss_pose));
  auto late_status = moving_status(0.0, 1, 11);
  late_status.rpy.z = 0.0;
  EXPECT_FALSE(unsynchronized_adapter.observe_vehicle_status_for_initialization(late_status));
  EXPECT_FALSE(unsynchronized_adapter.pending_initial_pose());
}

TEST(LocalizationAdapter, UsesDeviceTimeForWheelDistanceAndRejectsDuplicates)
{
  AdapterConfig config;
  config.wheel_use_device_timestamp = true;
  LocalizationAdapter adapter(config);

  auto status = moving_status(2.0, 4, 10);
  status.has_device_stamp = true;
  status.device_stamp = stamp(100, 200000000);
  const auto wheel = adapter.convert_wheel_speed(status);

  ASSERT_TRUE(wheel.has_value());
  EXPECT_EQ(wheel->header.stamp, status.device_stamp);
  status.header.stamp = stamp(11);
  EXPECT_FALSE(adapter.convert_wheel_speed(status));

  LocalizationAdapter missing_device_adapter(config);
  status.has_device_stamp = false;
  status.header.stamp = stamp(12);
  EXPECT_FALSE(missing_device_adapter.convert_wheel_speed(status));
}

TEST(LocalizationAdapter, RejectsUnknownInitialOrientationSource)
{
  AdapterConfig config;
  config.initial_orientation_source = "competition_magic";
  EXPECT_THROW((void)LocalizationAdapter{config}, std::invalid_argument);
}

TEST(LocalizationAdapter, RejectsNonFiniteInitialOrientationYawOffset)
{
  AdapterConfig config;
  config.initial_orientation_yaw_offset_rad =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW((void)LocalizationAdapter{config}, std::invalid_argument);
}

TEST(LocalizationAdapter, RejectsNonPositiveInitialPositionSampleCount)
{
  AdapterConfig config;
  config.initial_position_sample_count = 0;
  EXPECT_THROW((void)LocalizationAdapter{config}, std::invalid_argument);
}

TEST(LocalizationAdapter, RejectsNonFiniteKnownCheckpointXY)
{
  AdapterConfig config;
  config.initial_position_override_xy_m =
    std::array<double, 2>{std::numeric_limits<double>::quiet_NaN(), 0.0};
  EXPECT_THROW((void)LocalizationAdapter{config}, std::invalid_argument);
}

TEST(LocalizationAdapter, ConvertsMonotonicFiniteOdometryToNormalizedPose2d)
{
  LocalizationAdapter adapter(AdapterConfig{});
  Odometry odometry;
  odometry.header.stamp = stamp(1);
  odometry.header.frame_id = "odom";
  odometry.child_frame_id = "base_link";
  odometry.pose.pose.position.x = 12.0;
  odometry.pose.pose.position.y = -4.0;
  odometry.pose.pose.orientation = yaw_quaternion(4.0);

  const auto pose = adapter.convert_filtered_odometry(odometry);
  ASSERT_TRUE(pose.has_value());
  EXPECT_DOUBLE_EQ(pose->x, 12.0);
  EXPECT_DOUBLE_EQ(pose->y, -4.0);
  EXPECT_NEAR(pose->theta, 4.0 - 2.0 * kPi, 1.0e-12);
  EXPECT_FALSE(adapter.convert_filtered_odometry(odometry));

  odometry.header.stamp = stamp(2);
  odometry.pose.pose.position.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(adapter.convert_filtered_odometry(odometry));

  odometry.header.stamp = stamp(3);
  odometry.pose.pose.position.x = 13.0;
  odometry.pose.pose.orientation.w = 0.0;
  odometry.pose.pose.orientation.z = 0.0;
  EXPECT_FALSE(adapter.convert_filtered_odometry(odometry));

  odometry.header.stamp = stamp(4);
  odometry.pose.pose.orientation.w = 1.0;
  ASSERT_TRUE(adapter.convert_filtered_odometry(odometry));
  EXPECT_FALSE(adapter.convert_filtered_odometry(odometry));
}

TEST(LocalizationAdapter, ResetAcceptsARegressedSimulatorTimestamp)
{
  LocalizationAdapter adapter(AdapterConfig{});
  Odometry odometry;
  odometry.header.stamp = stamp(30);
  odometry.header.frame_id = "odom";
  odometry.child_frame_id = "base_link";
  odometry.pose.pose.orientation.w = 1.0;
  ASSERT_TRUE(adapter.convert_filtered_odometry(odometry));

  adapter.reset();
  odometry.header.stamp = stamp(1);
  EXPECT_TRUE(adapter.convert_filtered_odometry(odometry));
}

}  // namespace
