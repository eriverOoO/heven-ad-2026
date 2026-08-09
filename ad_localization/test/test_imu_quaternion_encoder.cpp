#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include "ad_localization/imu_quaternion_encoder/imu_quaternion_encoder.hpp"

namespace
{

constexpr double kPi = 3.14159265358979323846;

ad_morai_interfaces::msg::EgoVehicleStatus status(
  int sec, double speed, double roll = 0.0, double pitch = 0.0,
  double yaw = 0.0, std::uint32_t nanosec = 0)
{
  ad_morai_interfaces::msg::EgoVehicleStatus message;
  message.header.stamp.sec = sec;
  message.header.stamp.nanosec = nanosec;
  message.header.frame_id = "map";
  message.signed_velocity = speed;
  message.position.x = 10.0;
  message.position.y = 20.0;
  message.position.z = 3.0;
  message.rpy.x = roll;
  message.rpy.y = pitch;
  message.rpy.z = yaw;
  return message;
}

sensor_msgs::msg::Imu imu_rpy(
  int sec, double roll, double pitch, double yaw,
  std::uint32_t nanosec = 0)
{
  sensor_msgs::msg::Imu message;
  message.header.stamp.sec = sec;
  message.header.stamp.nanosec = nanosec;
  message.header.frame_id = "imu_link";
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);
  message.orientation.x = sr * cp * cy - cr * sp * sy;
  message.orientation.y = cr * sp * cy + sr * cp * sy;
  message.orientation.z = cr * cp * sy - sr * sp * cy;
  message.orientation.w = cr * cp * cy + sr * sp * sy;
  return message;
}

sensor_msgs::msg::Imu imu(
  int sec, double yaw, std::uint32_t nanosec = 0)
{
  return imu_rpy(sec, 0.0, 0.0, yaw, nanosec);
}

geometry_msgs::msg::PoseStamped seed(
  int sec, double x, double y, double z, std::uint32_t nanosec = 0)
{
  geometry_msgs::msg::PoseStamped message;
  message.header.stamp.sec = sec;
  message.header.stamp.nanosec = nanosec;
  message.header.frame_id = "odom";
  message.pose.position.x = x;
  message.pose.position.y = y;
  message.pose.position.z = z;
  message.pose.orientation.w = 1.0;
  return message;
}

TEST(ImuQuaternionEncoder, StatusPoseUsesFullRpyAndBodyFrameOriginOffset)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kStatusPose;
  config.status_origin_to_base_m = {1.0, 0.0, 0.5};
  ad_localization::ImuQuaternionEncoder encoder(config);

  const auto output = encoder.observe_status(
    status(1, 4.0, 0.1, -0.2, kPi / 2.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_NEAR(output->pose.pose.position.x, 10.049916708323414, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.y, 20.8812281720142, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.z, 3.686254494395969, 1.0e-12);
  EXPECT_NEAR(output->twist.twist.linear.x, 4.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.orientation.x, 0.1056687, 1.0e-6);
  EXPECT_NEAR(output->pose.pose.orientation.y, -0.0353406, 1.0e-6);
  EXPECT_NEAR(output->pose.pose.orientation.z, 0.7062231, 1.0e-6);
  EXPECT_NEAR(output->pose.pose.orientation.w, 0.6991667, 1.0e-6);
  EXPECT_EQ(output->header.frame_id, "odom");
  EXPECT_EQ(output->child_frame_id, "base_link");
}

TEST(ImuQuaternionEncoder, StatusPoseRejectsExactZeroPositionUnlessExplicitlyAllowed)
{
  auto zero_position = status(1, 0.0);
  zero_position.position.x = 0.0;
  zero_position.position.y = 0.0;
  zero_position.position.z = 0.0;

  ad_localization::ImuQuaternionEncoder rejected({});
  EXPECT_FALSE(rejected.observe_status(zero_position).has_value());

  ad_localization::ImuQuaternionEncoderConfig allow_origin_config;
  allow_origin_config.reject_zero_status_position = false;
  ad_localization::ImuQuaternionEncoder allowed(allow_origin_config);
  const auto output = allowed.observe_status(zero_position);

  ASSERT_TRUE(output.has_value());
  EXPECT_DOUBLE_EQ(output->pose.pose.position.x, 0.0);
  EXPECT_DOUBLE_EQ(output->pose.pose.position.y, 0.0);
  EXPECT_DOUBLE_EQ(output->pose.pose.position.z, 0.0);
}

TEST(ImuQuaternionEncoder, DeadReckoningDoesNotConsumeStatusPosition)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.maximum_imu_age_sec = 0.2;
  ad_localization::ImuQuaternionEncoder encoder(config);

  ASSERT_TRUE(encoder.observe_imu(imu(1, 0.0)));
  ASSERT_TRUE(encoder.observe_gnss_seed(seed(1, 5.0, 6.0, 7.0)));
  auto zero_position = status(1, 0.0);
  zero_position.position.x = 0.0;
  zero_position.position.y = 0.0;
  zero_position.position.z = 0.0;

  const auto output = encoder.observe_status(zero_position);

  ASSERT_TRUE(output.has_value());
  EXPECT_DOUBLE_EQ(output->pose.pose.position.x, 5.0);
  EXPECT_DOUBLE_EQ(output->pose.pose.position.y, 6.0);
  EXPECT_DOUBLE_EQ(output->pose.pose.position.z, 7.0);
}

TEST(ImuQuaternionEncoder, DeadReckoningAveragesConfiguredInitialGnssSamples)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.initial_seed_sample_count = 3;
  config.maximum_imu_age_sec = 0.5;
  ad_localization::ImuQuaternionEncoder encoder(config);

  ASSERT_TRUE(encoder.observe_imu(imu(1, 0.0)));
  EXPECT_FALSE(encoder.observe_gnss_seed(seed(1, 9.0, 18.0, 3.0)));
  EXPECT_FALSE(
    encoder.observe_gnss_seed(
      seed(1, 12.0, 21.0, 6.0, 100'000'000U)));
  ASSERT_TRUE(encoder.observe_imu(imu(1, 0.0, 200'000'000U)));
  EXPECT_TRUE(
    encoder.observe_gnss_seed(
      seed(1, 15.0, 24.0, 9.0, 200'000'000U)));

  const auto output = encoder.observe_status(
    status(1, 0.0, 0.0, 0.0, 0.0, 200'000'000U));
  ASSERT_TRUE(output.has_value());
  EXPECT_NEAR(output->pose.pose.position.x, 12.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.y, 21.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.z, 6.0, 1.0e-12);
}

TEST(ImuQuaternionEncoder, RejectsDuplicateAndRegressingStatusTime)
{
  ad_localization::ImuQuaternionEncoder encoder({});
  ASSERT_TRUE(encoder.observe_status(status(5, 0.0)).has_value());
  EXPECT_FALSE(encoder.observe_status(status(5, 1.0)).has_value());
  EXPECT_FALSE(encoder.observe_status(status(4, 1.0)).has_value());
}

TEST(ImuQuaternionEncoder, DeadReckoningUsesImuYawSignedSpeedAndHoldsSeedZ)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.maximum_imu_age_sec = 0.2;
  config.maximum_integration_dt_sec = 1.1;
  ad_localization::ImuQuaternionEncoder encoder(config);

  ASSERT_TRUE(encoder.observe_gnss_seed(seed(1, 100.0, 200.0, 7.0)));
  ASSERT_TRUE(encoder.observe_imu(imu(1, 0.0)));
  const auto first = encoder.observe_status(status(1, 2.0));
  ASSERT_TRUE(first.has_value());
  EXPECT_DOUBLE_EQ(first->pose.pose.position.x, 100.0);

  ASSERT_TRUE(encoder.observe_imu(imu(2, 0.0)));
  const auto forward = encoder.observe_status(status(2, 2.0));
  ASSERT_TRUE(forward.has_value());
  EXPECT_NEAR(forward->pose.pose.position.x, 102.0, 1.0e-12);
  EXPECT_NEAR(forward->pose.pose.position.y, 200.0, 1.0e-12);

  ASSERT_TRUE(encoder.observe_imu(imu(3, kPi / 2.0)));
  const auto reverse = encoder.observe_status(status(3, -1.0));
  ASSERT_TRUE(reverse.has_value());
  EXPECT_NEAR(reverse->pose.pose.position.x, 102.0, 1.0e-12);
  EXPECT_NEAR(reverse->pose.pose.position.y, 199.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(reverse->pose.pose.position.z, 7.0);
  EXPECT_NEAR(reverse->pose.pose.orientation.z, std::sqrt(0.5), 1.0e-12);
}

TEST(ImuQuaternionEncoder, DeadReckoningSubtractsRotatedThreeDimensionalGnssLeverArm)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.gnss_lever_arm_m = {1.0, 0.0, 2.0};
  config.maximum_imu_age_sec = 0.2;
  ad_localization::ImuQuaternionEncoder encoder(config);

  ASSERT_TRUE(encoder.observe_imu(imu_rpy(1, 0.0, kPi / 2.0, 0.0)));
  ASSERT_TRUE(encoder.observe_gnss_seed(seed(1, 10.0, 20.0, 30.0)));
  const auto output = encoder.observe_status(status(1, 0.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_NEAR(output->pose.pose.position.x, 8.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.y, 20.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.z, 31.0, 1.0e-12);
}

TEST(ImuQuaternionEncoder, DeadReckoningAppliesGridYawCorrectionToSeedAndMotion)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.gnss_lever_arm_m = {1.0, 0.0, 0.0};
  config.world_yaw_offset_rad = kPi / 2.0;
  config.maximum_imu_age_sec = 0.2;
  config.maximum_integration_dt_sec = 1.1;
  ad_localization::ImuQuaternionEncoder encoder(config);

  ASSERT_TRUE(encoder.observe_imu(imu(1, 0.0)));
  ASSERT_TRUE(encoder.observe_gnss_seed(seed(1, 10.0, 20.0, 30.0)));
  const auto initial = encoder.observe_status(status(1, 0.0));
  ASSERT_TRUE(initial.has_value());
  EXPECT_NEAR(initial->pose.pose.position.x, 10.0, 1.0e-12);
  EXPECT_NEAR(initial->pose.pose.position.y, 19.0, 1.0e-12);
  EXPECT_NEAR(initial->pose.pose.orientation.z, std::sqrt(0.5), 1.0e-12);

  ASSERT_TRUE(encoder.observe_imu(imu(2, 0.0)));
  const auto moved = encoder.observe_status(status(2, 2.0));
  ASSERT_TRUE(moved.has_value());
  EXPECT_NEAR(moved->pose.pose.position.x, 10.0, 1.0e-12);
  EXPECT_NEAR(moved->pose.pose.position.y, 21.0, 1.0e-12);
}

TEST(ImuQuaternionEncoder, DeadReckoningRemovesImuMountYawBeforeLeverArmCorrection)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.gnss_lever_arm_m = {1.0, 0.0, 0.0};
  config.base_to_imu_orientation = imu(1, kPi / 2.0).orientation;
  config.maximum_imu_age_sec = 0.2;
  ad_localization::ImuQuaternionEncoder encoder(config);

  ASSERT_TRUE(encoder.observe_imu(imu(1, kPi / 2.0)));
  ASSERT_TRUE(encoder.observe_gnss_seed(seed(1, 10.0, 20.0, 30.0)));
  const auto output = encoder.observe_status(status(1, 0.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_NEAR(output->pose.pose.position.x, 9.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.y, 20.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.orientation.z, 0.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.orientation.w, 1.0, 1.0e-12);
}

TEST(ImuQuaternionEncoder, DeadReckoningRejectsDelayedSeedWithoutRewinding)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.maximum_imu_age_sec = 0.2;
  config.maximum_integration_dt_sec = 1.1;
  ad_localization::ImuQuaternionEncoder encoder(config);

  ASSERT_TRUE(encoder.observe_imu(imu(1, 0.0)));
  ASSERT_TRUE(encoder.observe_gnss_seed(seed(1, 100.0, 200.0, 7.0)));
  ASSERT_TRUE(encoder.observe_status(status(1, 1.0)).has_value());
  ASSERT_TRUE(encoder.observe_imu(imu(2, 0.0)));
  ASSERT_TRUE(encoder.observe_status(status(2, 1.0)).has_value());

  EXPECT_FALSE(
    encoder.observe_gnss_seed(
      seed(2, -300.0, -400.0, -5.0)));
  EXPECT_FALSE(
    encoder.observe_gnss_seed(
      seed(1, -500.0, -600.0, -7.0, 500'000'000U)));
  ASSERT_TRUE(encoder.observe_imu(imu(3, 0.0)));
  const auto output = encoder.observe_status(status(3, 1.0));
  ASSERT_TRUE(output.has_value());
  EXPECT_NEAR(output->pose.pose.position.x, 102.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.y, 200.0, 1.0e-12);
}

TEST(ImuQuaternionEncoder, DeadReckoningIgnoresOrdinaryGnssAfterInitialization)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.maximum_imu_age_sec = 0.5;
  config.maximum_integration_dt_sec = 1.1;
  config.automatic_reseed_enabled = true;
  config.automatic_reseed_distance_m = 8.0;
  config.automatic_reseed_confirmation_samples = 3;
  config.automatic_reseed_candidate_radius_m = 4.0;
  config.automatic_reseed_max_interval_sec = 0.5;
  ad_localization::ImuQuaternionEncoder encoder(config);

  ASSERT_TRUE(encoder.observe_imu(imu(1, 0.0)));
  ASSERT_TRUE(encoder.observe_gnss_seed(seed(1, 100.0, 200.0, 7.0)));
  ASSERT_TRUE(encoder.observe_status(status(1, 1.0)).has_value());

  EXPECT_FALSE(
    encoder.observe_gnss_seed(
      seed(1, 100.8, 199.4, 8.0, 500'000'000U)));
  ASSERT_TRUE(encoder.observe_imu(imu(2, 0.0)));
  const auto output = encoder.observe_status(status(2, 1.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_NEAR(output->pose.pose.position.x, 101.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.y, 200.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.z, 7.0, 1.0e-12);
}

TEST(ImuQuaternionEncoder, DeadReckoningReseedsConfirmedCheckpointTeleport)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.maximum_imu_age_sec = 0.5;
  config.maximum_integration_dt_sec = 1.1;
  config.automatic_reseed_enabled = true;
  config.automatic_reseed_distance_m = 8.0;
  config.automatic_reseed_confirmation_samples = 3;
  config.automatic_reseed_candidate_radius_m = 4.0;
  config.automatic_reseed_max_interval_sec = 0.5;
  ad_localization::ImuQuaternionEncoder encoder(config);

  ASSERT_TRUE(encoder.observe_imu(imu(1, 0.0)));
  ASSERT_TRUE(encoder.observe_gnss_seed(seed(1, 0.0, 0.0, 5.0)));
  ASSERT_TRUE(encoder.observe_status(status(1, 0.0)).has_value());
  ASSERT_TRUE(encoder.observe_imu(imu(2, 0.0)));
  ASSERT_TRUE(encoder.observe_status(status(2, 0.0)).has_value());

  EXPECT_FALSE(
    encoder.observe_gnss_seed(
      seed(2, 100.0, 200.0, 9.0, 100'000'000U)));
  EXPECT_FALSE(
    encoder.observe_gnss_seed(
      seed(2, 100.6, 199.5, 9.2, 200'000'000U)));
  ASSERT_TRUE(encoder.observe_imu(imu(2, 0.0, 250'000'000U)));
  EXPECT_TRUE(
    encoder.observe_gnss_seed(
      seed(2, 99.7, 200.4, 8.8, 300'000'000U)));

  const auto output = encoder.observe_status(
    status(2, 0.0, 0.0, 0.0, 0.0, 400'000'000U));
  ASSERT_TRUE(output.has_value());
  EXPECT_NEAR(output->pose.pose.position.x, 100.1, 1.0e-12);
  EXPECT_NEAR(
    output->pose.pose.position.y, 199.96666666666667, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.z, 9.0, 1.0e-12);
}

TEST(ImuQuaternionEncoder, ResetAllowsAReplacementSeed)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.maximum_imu_age_sec = 0.5;
  ad_localization::ImuQuaternionEncoder encoder(config);

  ASSERT_TRUE(encoder.observe_imu(imu(1, 0.0)));
  ASSERT_TRUE(encoder.observe_gnss_seed(seed(1, 10.0, 20.0, 5.0)));
  ASSERT_TRUE(encoder.observe_status(status(1, 0.0)).has_value());

  encoder.reset();
  ASSERT_TRUE(encoder.observe_imu(imu(2, 0.0)));
  ASSERT_TRUE(encoder.observe_gnss_seed(seed(2, 50.0, 60.0, 8.0)));
  const auto output = encoder.observe_status(status(2, 0.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_NEAR(output->pose.pose.position.x, 50.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.y, 60.0, 1.0e-12);
  EXPECT_NEAR(output->pose.pose.position.z, 8.0, 1.0e-12);
}

TEST(ImuQuaternionEncoder, DeadReckoningNeverUsesFutureImuForAStatusSample)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.maximum_imu_age_sec = 0.2;
  config.maximum_integration_dt_sec = 1.0;
  ad_localization::ImuQuaternionEncoder encoder(config);

  ASSERT_TRUE(encoder.observe_imu(imu(1, 0.0)));
  ASSERT_TRUE(encoder.observe_gnss_seed(seed(1, 0.0, 0.0, 0.0)));
  ASSERT_TRUE(encoder.observe_status(status(1, 1.0)).has_value());
  ASSERT_TRUE(encoder.observe_imu(imu(1, kPi / 2.0, 600'000'000U)));

  EXPECT_FALSE(
    encoder.observe_status(
      status(1, 1.0, 0.0, 0.0, 0.0, 500'000'000U)).has_value());
}

TEST(ImuQuaternionEncoder, DeadReckoningGapInvalidatesStateUntilFreshSeed)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.maximum_imu_age_sec = 0.2;
  config.maximum_integration_dt_sec = 0.5;
  ad_localization::ImuQuaternionEncoder encoder(config);

  ASSERT_TRUE(encoder.observe_imu(imu(1, 0.0)));
  ASSERT_TRUE(encoder.observe_gnss_seed(seed(1, 0.0, 0.0, 5.0)));
  ASSERT_TRUE(encoder.observe_status(status(1, 2.0)).has_value());
  EXPECT_FALSE(encoder.observe_status(status(2, 2.0)).has_value());
  EXPECT_FALSE(
    encoder.observe_gnss_seed(
      seed(1, -100.0, -100.0, -100.0, 500'000'000U)));

  ASSERT_TRUE(encoder.observe_imu(imu(2, 0.0, 100'000'000U)));
  EXPECT_FALSE(
    encoder.observe_status(
      status(2, 2.0, 0.0, 0.0, 0.0, 100'000'000U)).has_value());

  ASSERT_TRUE(
    encoder.observe_gnss_seed(
      seed(2, 50.0, 60.0, 7.0, 100'000'000U)));
  ASSERT_TRUE(encoder.observe_imu(imu(2, 0.0, 200'000'000U)));
  const auto restarted = encoder.observe_status(
    status(2, 2.0, 0.0, 0.0, 0.0, 200'000'000U));
  ASSERT_TRUE(restarted.has_value());
  EXPECT_NEAR(restarted->pose.pose.position.x, 50.2, 1.0e-12);
  EXPECT_NEAR(restarted->pose.pose.position.y, 60.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(restarted->pose.pose.position.z, 7.0);
}

TEST(ImuQuaternionEncoder, RejectsStructurallyInvalidRosTimestamps)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  ad_localization::ImuQuaternionEncoder encoder(config);

  auto invalid_imu = imu(1, 0.0);
  invalid_imu.header.stamp.nanosec = 1'000'000'000U;
  EXPECT_FALSE(encoder.observe_imu(invalid_imu));

  auto invalid_seed = seed(1, 0.0, 0.0, 0.0);
  invalid_seed.header.stamp.sec = -1;
  invalid_seed.header.stamp.nanosec = 999'999'999U;
  EXPECT_FALSE(encoder.observe_gnss_seed(invalid_seed));

  auto invalid_status = status(1, 0.0);
  invalid_status.header.stamp.nanosec = 1'000'000'000U;
  EXPECT_FALSE(encoder.observe_status(invalid_status).has_value());
}

TEST(ImuQuaternionEncoder, DeadReckoningFailsClosedWithoutFreshSeedAndImu)
{
  ad_localization::ImuQuaternionEncoderConfig config;
  config.mode = ad_localization::ImuQuaternionEncoderMode::kDeadReckoning;
  config.maximum_imu_age_sec = 0.1;
  config.maximum_integration_dt_sec = 0.5;
  ad_localization::ImuQuaternionEncoder encoder(config);

  EXPECT_FALSE(encoder.observe_status(status(1, 2.0)).has_value());
  ASSERT_TRUE(encoder.observe_gnss_seed(seed(1, 0.0, 0.0, 0.0)));
  EXPECT_FALSE(encoder.observe_status(status(1, 2.0)).has_value());
  ASSERT_TRUE(encoder.observe_imu(imu(1, 0.0)));
  ASSERT_TRUE(encoder.observe_status(status(1, 2.0)).has_value());
  ASSERT_TRUE(encoder.observe_imu(imu(3, 0.0)));
  EXPECT_FALSE(encoder.observe_status(status(3, 2.0)).has_value());
}

TEST(ImuQuaternionEncoder, ParserUsesInputBasedNamesOnly)
{
  EXPECT_EQ(
    ad_localization::parse_imu_quaternion_encoder_mode("status_pose"),
    ad_localization::ImuQuaternionEncoderMode::kStatusPose);
  EXPECT_EQ(
    ad_localization::parse_imu_quaternion_encoder_mode("dead_reckoning"),
    ad_localization::ImuQuaternionEncoderMode::kDeadReckoning);
  EXPECT_THROW(
    ad_localization::parse_imu_quaternion_encoder_mode("direct_gt"),
    std::invalid_argument);
}

}  // namespace
