#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "ad_localization/quaternion_wheel_gnss_ekf/quaternion_wheel_gnss_ekf.hpp"

namespace
{

constexpr double kPi = 3.14159265358979323846;

sensor_msgs::msg::Imu imu_rpy(
  int sec, double roll, double pitch, double yaw,
  std::uint32_t nanosec = 0U)
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
  message.orientation_covariance.fill(0.0);
  message.orientation_covariance[0] = 0.01;
  message.orientation_covariance[4] = 0.01;
  message.orientation_covariance[8] = 0.01;
  message.angular_velocity_covariance = message.orientation_covariance;
  message.linear_acceleration_covariance = message.orientation_covariance;
  return message;
}

geometry_msgs::msg::PoseStamped gnss(
  int sec, double x, double y, double z = 0.0,
  std::uint32_t nanosec = 0U)
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

geometry_msgs::msg::TwistWithCovarianceStamped wheel(
  int sec, double speed, double variance = 0.01,
  std::uint32_t nanosec = 0U)
{
  geometry_msgs::msg::TwistWithCovarianceStamped message;
  message.header.stamp.sec = sec;
  message.header.stamp.nanosec = nanosec;
  message.header.frame_id = "base_link";
  message.twist.twist.linear.x = speed;
  message.twist.covariance.fill(0.0);
  message.twist.covariance[0] = variance;
  return message;
}

ad_localization::QuaternionWheelGnssEkfConfig base_config()
{
  ad_localization::QuaternionWheelGnssEkfConfig config;
  config.maximum_imu_age_sec = 0.5;
  config.maximum_prediction_dt_sec = 1.1;
  config.initialization_sample_count = 1;
  config.initial_position_variance_m2 = 1.0;
  config.initial_wheel_bias_mps = 0.0;
  config.initial_wheel_bias_variance_m2ps2 = 0.25;
  config.wheel_speed_variance_floor_m2ps2 = 0.001;
  config.wheel_bias_random_walk_variance_m2ps3 = 0.04;
  config.gnss_variance_m2 = 9.0;
  config.gnss_mahalanobis_threshold = 9.21;
  config.teleport_distance_m = 8.0;
  config.teleport_confirmation_samples = 3;
  config.teleport_candidate_radius_m = 2.0;
  config.teleport_max_interval_sec = 0.5;
  config.fixed_output_z_m = 0.0;
  config.unobserved_variance = 1.0e6;
  config.orientation_variance_rad2 = 0.01;
  return config;
}

void initialize(
  ad_localization::QuaternionWheelGnssEkf & filter,
  double x = 0.0, double y = 0.0)
{
  ASSERT_TRUE(filter.observe_imu(imu_rpy(1, 0.0, 0.0, 0.0)));
  ASSERT_TRUE(filter.observe_gnss(gnss(1, x, y)).has_value());
  ASSERT_TRUE(filter.state().initialized);
}

TEST(QuaternionWheelGnssEkf, RejectsInvalidConfiguration)
{
  auto config = base_config();
  config.reference_frame = "/odom";
  EXPECT_THROW(
    ad_localization::QuaternionWheelGnssEkf filter(config),
    std::invalid_argument);

  config = base_config();
  config.gnss_variance_m2 = 0.0;
  EXPECT_THROW(
    ad_localization::QuaternionWheelGnssEkf filter(config),
    std::invalid_argument);

  config = base_config();
  config.gnss_mahalanobis_threshold =
    std::numeric_limits<double>::infinity();
  EXPECT_THROW(
    ad_localization::QuaternionWheelGnssEkf filter(config),
    std::invalid_argument);

  config = base_config();
  config.initialization_sample_count = 0;
  EXPECT_THROW(
    ad_localization::QuaternionWheelGnssEkf filter(config),
    std::invalid_argument);
}

TEST(QuaternionWheelGnssEkf, AveragesLeverArmCorrectedInitialGnssXyOnly)
{
  auto config = base_config();
  config.initialization_sample_count = 3;
  config.gnss_lever_arm_m = {0.0, 0.0, 2.0};
  ad_localization::QuaternionWheelGnssEkf filter(config);

  ASSERT_TRUE(filter.observe_imu(imu_rpy(1, 0.0, kPi / 2.0, 0.0)));
  EXPECT_FALSE(filter.observe_gnss(gnss(1, 10.0, 20.0, 1000.0)).has_value());
  ASSERT_TRUE(filter.observe_imu(imu_rpy(1, 0.0, kPi / 2.0, 0.0, 100'000'000U)));
  EXPECT_FALSE(
    filter.observe_gnss(
      gnss(1, 12.0, 22.0, -1000.0, 100'000'000U)).has_value());
  ASSERT_TRUE(filter.observe_imu(imu_rpy(1, 0.0, kPi / 2.0, 0.0, 200'000'000U)));
  const auto output = filter.observe_gnss(
    gnss(1, 14.0, 24.0, 42.0, 200'000'000U));

  ASSERT_TRUE(output.has_value());
  EXPECT_NEAR(filter.state().value[0], 10.0, 1.0e-12);
  EXPECT_NEAR(filter.state().value[1], 22.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(output->pose.pose.position.z, 0.0);
  EXPECT_DOUBLE_EQ(output->pose.covariance[14], config.unobserved_variance);
}

TEST(QuaternionWheelGnssEkf, PredictsStraightRotatedAndReverseWithAdditiveBias)
{
  auto config = base_config();
  config.initial_wheel_bias_mps = 1.0;
  ad_localization::QuaternionWheelGnssEkf filter(config);
  initialize(filter);

  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0)));
  EXPECT_FALSE(filter.observe_wheel_speed(wheel(2, 3.0)).has_value());
  ASSERT_TRUE(filter.observe_imu(imu_rpy(3, 0.0, 0.0, 0.0)));
  const auto straight = filter.observe_wheel_speed(wheel(3, 3.0));
  ASSERT_TRUE(straight.has_value());
  EXPECT_NEAR(straight->pose.pose.position.x, 2.0, 1.0e-12);
  EXPECT_NEAR(straight->pose.pose.position.y, 0.0, 1.0e-12);

  ASSERT_TRUE(filter.observe_imu(imu_rpy(4, 0.0, 0.0, kPi / 2.0)));
  const auto rotated = filter.observe_wheel_speed(wheel(4, 3.0));
  ASSERT_TRUE(rotated.has_value());
  EXPECT_NEAR(rotated->pose.pose.position.x, 2.0, 1.0e-12);
  EXPECT_NEAR(rotated->pose.pose.position.y, 2.0, 1.0e-12);

  ASSERT_TRUE(filter.observe_imu(imu_rpy(5, 0.0, 0.0, kPi / 2.0)));
  const auto reverse = filter.observe_wheel_speed(wheel(5, -1.0));
  ASSERT_TRUE(reverse.has_value());
  EXPECT_NEAR(reverse->pose.pose.position.x, 2.0, 1.0e-12);
  EXPECT_NEAR(reverse->pose.pose.position.y, 0.0, 1.0e-12);
}

TEST(QuaternionWheelGnssEkf, PublishesFullCorrectedQuaternionAndIgnoresImuVectors)
{
  auto config = base_config();
  config.world_yaw_offset_rad = 0.2;
  config.base_to_imu_orientation = imu_rpy(0, 0.0, 0.0, 0.1).orientation;
  ad_localization::QuaternionWheelGnssEkf first(config);
  ad_localization::QuaternionWheelGnssEkf second(config);
  auto first_imu = imu_rpy(1, 0.3, -0.2, 0.5);
  auto second_imu = first_imu;
  first_imu.angular_velocity.x = 1.0;
  first_imu.linear_acceleration.z = 9.0;
  second_imu.angular_velocity.x = -200.0;
  second_imu.linear_acceleration.z = -500.0;

  ASSERT_TRUE(first.observe_imu(first_imu));
  ASSERT_TRUE(second.observe_imu(second_imu));
  const auto first_output = first.observe_gnss(gnss(1, 0.0, 0.0));
  const auto second_output = second.observe_gnss(gnss(1, 0.0, 0.0));

  ASSERT_TRUE(first_output.has_value());
  ASSERT_TRUE(second_output.has_value());
  EXPECT_NEAR(first_output->pose.pose.orientation.x, 0.1753944, 1.0e-6);
  EXPECT_NEAR(first_output->pose.pose.orientation.y, -0.0330169, 1.0e-6);
  EXPECT_NEAR(first_output->pose.pose.orientation.z, 0.3049946, 1.0e-6);
  EXPECT_NEAR(first_output->pose.pose.orientation.w, 0.9354811, 1.0e-6);
  EXPECT_EQ(first.state().value, second.state().value);
  EXPECT_EQ(first.state().covariance, second.state().covariance);
  EXPECT_DOUBLE_EQ(first_output->twist.twist.angular.x, 0.0);
  EXPECT_DOUBLE_EQ(first_output->twist.twist.angular.y, 0.0);
  EXPECT_DOUBLE_EQ(first_output->twist.twist.angular.z, 0.0);
}

TEST(QuaternionWheelGnssEkf, WheelPredictionGrowsBiasCovariance)
{
  auto config = base_config();
  ad_localization::QuaternionWheelGnssEkf filter(config);
  initialize(filter);
  const double before = filter.state().covariance[8];

  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0)));
  EXPECT_FALSE(filter.observe_wheel_speed(wheel(2, 1.0)).has_value());
  ASSERT_TRUE(filter.observe_imu(imu_rpy(3, 0.0, 0.0, 0.0)));
  ASSERT_TRUE(filter.observe_wheel_speed(wheel(3, 1.0)).has_value());

  EXPECT_GT(filter.state().covariance[8], before);
}

TEST(QuaternionWheelGnssEkf, PredictsToAsynchronousGnssWithoutReintegratingInterval)
{
  auto config = base_config();
  ad_localization::QuaternionWheelGnssEkf filter(config);
  initialize(filter);

  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0)));
  EXPECT_FALSE(filter.observe_wheel_speed(wheel(2, 1.0, 0.04)).has_value());
  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0, 500'000'000U)));
  const auto gnss_output = filter.observe_gnss(
    gnss(2, 0.5, 0.0, 0.0, 500'000'000U));

  ASSERT_TRUE(gnss_output.has_value());
  EXPECT_NEAR(gnss_output->pose.pose.position.x, 0.5, 1.0e-12);
  EXPECT_NEAR(filter.state().value[2], 0.0, 1.0e-12);

  ASSERT_TRUE(filter.observe_imu(imu_rpy(3, 0.0, 0.0, 0.0)));
  const auto wheel_output = filter.observe_wheel_speed(wheel(3, 1.0, 0.04));

  ASSERT_TRUE(wheel_output.has_value());
  EXPECT_NEAR(wheel_output->pose.pose.position.x, 1.0, 1.0e-12);
  EXPECT_NEAR(filter.state().value[2], 0.0, 1.0e-12);
}

TEST(QuaternionWheelGnssEkf, GnssKeepsFreshForwardAndReverseWheelVelocity)
{
  const auto config = base_config();
  for (const double speed : {2.0, -2.0}) {
    ad_localization::QuaternionWheelGnssEkf filter(config);
    initialize(filter);

    ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0)));
    EXPECT_FALSE(filter.observe_wheel_speed(wheel(2, speed, 0.04)).has_value());
    ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0, 500'000'000U)));
    const auto output = filter.observe_gnss(
      gnss(2, speed * 0.5, 0.0, 0.0, 500'000'000U));

    ASSERT_TRUE(output.has_value());
    EXPECT_NEAR(output->twist.twist.linear.x, speed, 1.0e-12);
    EXPECT_NEAR(
      output->twist.covariance[0],
      0.04 + filter.state().covariance[8], 1.0e-12);
    EXPECT_LT(output->twist.covariance[0], config.unobserved_variance);
  }
}

TEST(QuaternionWheelGnssEkf, AcceptsGnssAtCurrentPredictionEpoch)
{
  const auto config = base_config();
  ad_localization::QuaternionWheelGnssEkf filter(config);
  initialize(filter);

  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0)));
  EXPECT_FALSE(filter.observe_wheel_speed(wheel(2, 1.0, 0.04)).has_value());
  const auto output = filter.observe_gnss(gnss(2, 0.0, 0.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_TRUE(filter.state().initialized);
  EXPECT_NEAR(output->pose.pose.position.x, 0.0, 1.0e-12);
  EXPECT_NEAR(output->twist.twist.linear.x, 1.0, 1.0e-12);
}

TEST(QuaternionWheelGnssEkf, InitialGnssMarksVelocityUnobservedWithoutWheel)
{
  const auto config = base_config();
  ad_localization::QuaternionWheelGnssEkf filter(config);

  ASSERT_TRUE(filter.observe_imu(imu_rpy(1, 0.0, 0.0, 0.0)));
  const auto output = filter.observe_gnss(gnss(1, 5.0, 6.0));

  ASSERT_TRUE(output.has_value());
  EXPECT_DOUBLE_EQ(output->twist.twist.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(output->twist.covariance[0], config.unobserved_variance);
}

TEST(QuaternionWheelGnssEkf, AppliesLowGainGnssUpdateAndRejectsMahalanobisOutlier)
{
  auto config = base_config();
  ad_localization::QuaternionWheelGnssEkf filter(config);
  initialize(filter);
  const auto before = filter.state();

  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0)));
  const auto corrected = filter.observe_gnss(gnss(2, 1.0, 0.0, 999.0));
  ASSERT_TRUE(corrected.has_value());
  EXPECT_GT(filter.state().value[0], 0.0);
  EXPECT_LT(filter.state().value[0], 0.2);
  EXPECT_DOUBLE_EQ(corrected->pose.pose.position.z, 0.0);

  const auto accepted_state = filter.state();
  ASSERT_TRUE(filter.observe_imu(imu_rpy(3, 0.0, 0.0, 0.0)));
  EXPECT_FALSE(filter.observe_gnss(gnss(3, 1000.0, -1000.0)).has_value());
  EXPECT_EQ(filter.state().value, accepted_state.value);
  EXPECT_EQ(filter.state().covariance, accepted_state.covariance);
  EXPECT_NE(filter.state().value, before.value);
}

TEST(QuaternionWheelGnssEkf, SingleJumpIsRejectedButConsistentTeleportReseeds)
{
  auto config = base_config();
  ad_localization::QuaternionWheelGnssEkf filter(config);
  initialize(filter);

  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0)));
  EXPECT_FALSE(filter.observe_gnss(gnss(2, 100.0, 200.0)).has_value());
  EXPECT_NEAR(filter.state().value[0], 0.0, 1.0e-12);
  EXPECT_NEAR(filter.state().value[1], 0.0, 1.0e-12);

  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0, 100'000'000U)));
  EXPECT_FALSE(
    filter.observe_gnss(
      gnss(2, 100.5, 199.5, 0.0, 100'000'000U)).has_value());
  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0, 200'000'000U)));
  const auto reseeded = filter.observe_gnss(
    gnss(2, 99.5, 200.5, 0.0, 200'000'000U));

  ASSERT_TRUE(reseeded.has_value());
  EXPECT_NEAR(filter.state().value[0], 100.0, 1.0e-12);
  EXPECT_NEAR(filter.state().value[1], 200.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(filter.state().value[2], config.initial_wheel_bias_mps);
  EXPECT_FALSE(filter.observe_wheel_speed(wheel(3, 1.0)).has_value());
}

TEST(QuaternionWheelGnssEkf, TeleportResetClearsHeldWheelVelocity)
{
  const auto config = base_config();
  ad_localization::QuaternionWheelGnssEkf filter(config);
  initialize(filter);

  ASSERT_TRUE(filter.observe_imu(imu_rpy(1, 0.0, 0.0, 0.0, 100'000'000U)));
  EXPECT_FALSE(
    filter.observe_wheel_speed(
      wheel(1, 1.0, 0.04, 100'000'000U)).has_value());
  ASSERT_TRUE(filter.observe_imu(imu_rpy(1, 0.0, 0.0, 0.0, 500'000'000U)));
  ASSERT_TRUE(
    filter.observe_wheel_speed(
      wheel(1, 1.0, 0.04, 500'000'000U)).has_value());

  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0)));
  EXPECT_FALSE(filter.observe_gnss(gnss(2, 100.0, 200.0)).has_value());
  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0, 100'000'000U)));
  EXPECT_FALSE(
    filter.observe_gnss(
      gnss(2, 100.5, 199.5, 0.0, 100'000'000U)).has_value());
  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0, 200'000'000U)));
  const auto reseeded = filter.observe_gnss(
    gnss(2, 99.5, 200.5, 0.0, 200'000'000U));

  ASSERT_TRUE(reseeded.has_value());
  EXPECT_DOUBLE_EQ(reseeded->twist.twist.linear.x, 0.0);
  EXPECT_DOUBLE_EQ(reseeded->twist.covariance[0], config.unobserved_variance);
}

TEST(QuaternionWheelGnssEkf, ClockRegressionClearsEpochAndRequiresNewSeed)
{
  auto config = base_config();
  ad_localization::QuaternionWheelGnssEkf filter(config);
  initialize(filter, 5.0, 6.0);
  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0)));

  EXPECT_TRUE(filter.observe_imu(imu_rpy(1, 0.0, 0.0, 0.5)));
  EXPECT_FALSE(filter.state().initialized);
  EXPECT_FALSE(filter.observe_wheel_speed(wheel(1, 1.0, 0.01, 500'000'000U)).has_value());
  ASSERT_TRUE(filter.observe_gnss(gnss(1, 50.0, 60.0, 0.0, 500'000'000U)).has_value());
  EXPECT_NEAR(filter.state().value[0], 50.0, 1.0e-12);
  EXPECT_NEAR(filter.state().value[1], 60.0, 1.0e-12);
}

TEST(QuaternionWheelGnssEkf, RejectsInvalidFutureNonmonotonicAndOverGapMeasurements)
{
  auto config = base_config();
  config.maximum_prediction_dt_sec = 0.5;
  ad_localization::QuaternionWheelGnssEkf filter(config);

  auto bad_imu = imu_rpy(1, 0.0, 0.0, 0.0);
  bad_imu.orientation_covariance[5] =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(filter.observe_imu(bad_imu));
  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.0)));
  EXPECT_FALSE(filter.observe_gnss(gnss(1, 0.0, 0.0)).has_value());
  ASSERT_TRUE(filter.observe_gnss(gnss(2, 0.0, 0.0)).has_value());

  auto bad_wheel = wheel(3, 1.0);
  bad_wheel.twist.covariance[0] = 0.0;
  EXPECT_FALSE(filter.observe_wheel_speed(bad_wheel).has_value());
  EXPECT_FALSE(filter.observe_wheel_speed(wheel(4, 1.0)).has_value());
  ASSERT_TRUE(filter.observe_imu(imu_rpy(3, 0.0, 0.0, 0.0)));
  EXPECT_FALSE(filter.observe_wheel_speed(wheel(3, 1.0)).has_value());
  EXPECT_TRUE(filter.state().initialized);
  ASSERT_TRUE(filter.observe_imu(imu_rpy(4, 0.0, 0.0, 0.0)));
  EXPECT_FALSE(filter.observe_wheel_speed(wheel(4, 1.0)).has_value());
  EXPECT_FALSE(filter.state().initialized);
}

TEST(QuaternionWheelGnssEkf, CovarianceStaysFiniteSymmetricAndPositive)
{
  auto config = base_config();
  ad_localization::QuaternionWheelGnssEkf filter(config);
  initialize(filter);
  ASSERT_TRUE(filter.observe_imu(imu_rpy(2, 0.0, 0.0, 0.3)));
  EXPECT_FALSE(filter.observe_wheel_speed(wheel(2, 2.0)).has_value());
  for (int sec = 3; sec <= 8; ++sec) {
    ASSERT_TRUE(filter.observe_imu(imu_rpy(sec, 0.0, 0.0, 0.3)));
    ASSERT_TRUE(filter.observe_wheel_speed(wheel(sec, 2.0)).has_value());
    ASSERT_TRUE(filter.observe_imu(imu_rpy(sec, 0.0, 0.0, 0.3, 100'000'000U)));
    ASSERT_TRUE(
      filter.observe_gnss(
        gnss(
          sec, filter.state().value[0] + 0.1,
          filter.state().value[1] - 0.1, 100.0, 100'000'000U)).has_value());
  }

  const auto covariance = filter.state().covariance;
  for (double value : covariance) {
    EXPECT_TRUE(std::isfinite(value));
  }
  EXPECT_GT(covariance[0], 0.0);
  EXPECT_GT(covariance[4], 0.0);
  EXPECT_GT(covariance[8], 0.0);
  EXPECT_NEAR(covariance[1], covariance[3], 1.0e-12);
  EXPECT_NEAR(covariance[2], covariance[6], 1.0e-12);
  EXPECT_NEAR(covariance[5], covariance[7], 1.0e-12);
}

TEST(QuaternionWheelGnssEkf, ResetClearsAllStateAndAcceptsNewEpoch)
{
  ad_localization::QuaternionWheelGnssEkf filter(base_config());
  initialize(filter, 1.0, 2.0);
  filter.reset();

  EXPECT_FALSE(filter.state().initialized);
  EXPECT_EQ(filter.state().value, (std::array<double, 3>{0.0, 0.0, 0.0}));
  ASSERT_TRUE(filter.observe_imu(imu_rpy(1, 0.0, 0.0, 0.0)));
  ASSERT_TRUE(filter.observe_gnss(gnss(1, 10.0, 20.0)).has_value());
  EXPECT_NEAR(filter.state().value[0], 10.0, 1.0e-12);
  EXPECT_NEAR(filter.state().value[1], 20.0, 1.0e-12);
}

}  // namespace
