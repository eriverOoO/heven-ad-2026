#include "autoware_prediction_node.hpp"

#include <ad_interfaces/msg/predicted_object.hpp>
#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/shape.hpp>

#include <gtest/gtest.h>
#include <rmw/types.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

namespace
{

using ad_lidar_perception::tracking::adapt_tracked_objects;
using ad_lidar_perception::tracking::AutowarePredictionAdapterConfig;
using ad_lidar_perception::tracking::ImmUpdateReason;
using ad_lidar_perception::tracking::prediction_output_qos;
using ad_lidar_perception::tracking::parse_yaw_rate_source;
using ad_lidar_perception::tracking::PredictionAdaptation;
using ad_lidar_perception::tracking::rejected_prediction_diagnostics;
using ad_lidar_perception::tracking::StatefulImmPredictionAdapter;
using ad_lidar_perception::tracking::YawRateSource;
using autoware_perception_msgs::msg::ObjectClassification;
using autoware_perception_msgs::msg::Shape;
using autoware_perception_msgs::msg::TrackedObjects;

constexpr double kPi = 3.14159265358979323846;
constexpr std::int64_t kSecondNs = 1000000000LL;

void set_stamp(TrackedObjects & input, const std::int64_t stamp_ns)
{
  input.header.stamp.sec = static_cast<std::int32_t>(stamp_ns / kSecondNs);
  input.header.stamp.nanosec = static_cast<std::uint32_t>(stamp_ns % kSecondNs);
}

TrackedObjects valid_input(const std::int64_t stamp_ns = 10 * kSecondNs)
{
  TrackedObjects input;
  input.header.frame_id = "odom";
  set_stamp(input, stamp_ns);
  input.objects.emplace_back();
  auto & object = input.objects.back();
  for (std::size_t index = 0; index < object.object_id.uuid.size(); ++index) {
    object.object_id.uuid[index] = static_cast<std::uint8_t>(index + 10U);
  }
  object.existence_probability = 0.9F;
  object.classification.emplace_back();
  object.classification.back().label = ObjectClassification::CAR;
  object.classification.back().probability = 0.8F;

  auto & pose = object.kinematics.pose_with_covariance;
  pose.pose.position.x = 10.0;
  pose.pose.position.y = 20.0;
  pose.pose.position.z = 0.5;
  pose.pose.orientation.z = std::sin(kPi / 4.0);
  pose.pose.orientation.w = std::cos(kPi / 4.0);
  pose.covariance[0] = 1.0;
  pose.covariance[1] = 0.2;
  pose.covariance[6] = 0.2;
  pose.covariance[7] = 2.0;

  auto & twist = object.kinematics.twist_with_covariance;
  twist.twist.linear.x = 3.0;
  twist.twist.linear.y = 2.0;
  twist.twist.linear.z = 99.0;
  twist.twist.angular.z = 88.0;
  twist.covariance[0] = 4.0;
  twist.covariance[1] = 1.0;
  twist.covariance[6] = 1.0;
  twist.covariance[7] = 9.0;
  twist.covariance[35] = 77.0;

  object.shape.type = Shape::BOUNDING_BOX;
  object.shape.dimensions.x = 4.5;
  object.shape.dimensions.y = 1.8;
  object.shape.dimensions.z = 1.6;
  return input;
}

AutowarePredictionAdapterConfig valid_config()
{
  AutowarePredictionAdapterConfig config;
  config.expected_frame_id = "odom";
  config.maximum_input_age_sec = 0.5;
  config.prediction.horizons_s = {0.5, 1.0};
  config.prediction.acceleration_noise_std_mps2 = 1.5;
  return config;
}

void assign_second_uuid(TrackedObjects & input)
{
  input.objects.push_back(input.objects.front());
  input.objects.back().object_id.uuid[0] = 0xfeU;
}

void expect_same_prediction(
  const ad_interfaces::msg::PredictedObject & actual,
  const ad_interfaces::msg::PredictedObject & expected)
{
  EXPECT_EQ(actual.object_id, expected.object_id);
  EXPECT_EQ(actual.initial_pose, expected.initial_pose);
  EXPECT_EQ(actual.initial_twist, expected.initial_twist);
  EXPECT_EQ(actual.states, expected.states);
}

std::map<std::string, std::string> diagnostic_values(
  const diagnostic_msgs::msg::DiagnosticStatus & status)
{
  std::map<std::string, std::string> output;
  for (const auto & value : status.values) {
    output.emplace(value.key, value.value);
  }
  return output;
}

const diagnostic_msgs::msg::DiagnosticStatus & diagnostic_for(
  const diagnostic_msgs::msg::DiagnosticArray & diagnostics,
  const std::string & name)
{
  const auto found = std::find_if(
    diagnostics.status.begin(), diagnostics.status.end(),
    [&name](const auto & status) {return status.name == name;});
  if (found == diagnostics.status.end()) {
    throw std::runtime_error("diagnostic UUID is missing");
  }
  return *found;
}

TEST(AutowarePredictionAdapter, MapsLocalMotionAndCovarianceToWorld) {
  const auto input = valid_input();
  const auto output = adapt_tracked_objects(
    input, 10 * kSecondNs + 100000000LL,
    std::nullopt, valid_config());

  EXPECT_EQ(output.header.frame_id, "odom");
  EXPECT_EQ(output.header.stamp, input.header.stamp);
  ASSERT_EQ(output.objects.size(), 1U);
  const auto & object = output.objects.front();
  EXPECT_EQ(object.object_id, input.objects.front().object_id);
  EXPECT_FLOAT_EQ(object.existence_probability, 0.9F);
  EXPECT_EQ(object.classification, ad_interfaces::msg::PredictedObject::CAR);
  EXPECT_FLOAT_EQ(object.classification_probability, 0.8F);
  EXPECT_DOUBLE_EQ(object.dimensions.x, 4.5);
  EXPECT_DOUBLE_EQ(object.dimensions.y, 1.8);
  EXPECT_DOUBLE_EQ(object.dimensions.z, 1.6);
  EXPECT_DOUBLE_EQ(object.initial_pose.covariance[0], 1.0);
  EXPECT_DOUBLE_EQ(object.initial_pose.covariance[1], 0.2);
  EXPECT_DOUBLE_EQ(object.initial_pose.covariance[6], 0.2);
  EXPECT_DOUBLE_EQ(object.initial_pose.covariance[7], 2.0);
  EXPECT_DOUBLE_EQ(object.initial_pose.covariance[35], 0.04);
  for (std::size_t index = 0; index < object.initial_pose.covariance.size(); ++index) {
    if (index == 0U || index == 1U || index == 6U || index == 7U || index == 35U) {
      continue;
    }
    EXPECT_DOUBLE_EQ(object.initial_pose.covariance[index], 0.0);
  }

  EXPECT_NEAR(object.initial_twist.twist.linear.x, -2.0, 1.0e-12);
  EXPECT_NEAR(object.initial_twist.twist.linear.y, 3.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(object.initial_twist.twist.linear.z, 0.0);
  EXPECT_DOUBLE_EQ(object.initial_twist.twist.angular.z, 0.0);
  EXPECT_NEAR(object.initial_twist.covariance[0], 9.0, 1.0e-12);
  EXPECT_NEAR(object.initial_twist.covariance[1], -1.0, 1.0e-12);
  EXPECT_NEAR(object.initial_twist.covariance[6], -1.0, 1.0e-12);
  EXPECT_NEAR(object.initial_twist.covariance[7], 4.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(object.initial_twist.covariance[35], 0.0);

  ASSERT_EQ(object.states.size(), 2U);
  EXPECT_EQ(object.states[0].time_from_start.sec, 0);
  EXPECT_EQ(object.states[0].time_from_start.nanosec, 500000000U);
  EXPECT_NEAR(object.states[0].pose.pose.position.x, 9.0, 1.0e-12);
  EXPECT_NEAR(object.states[0].pose.pose.position.y, 21.5, 1.0e-12);
  EXPECT_EQ(object.states[1].time_from_start.sec, 1);
  EXPECT_EQ(object.states[1].time_from_start.nanosec, 0U);
}

TEST(AutowarePredictionAdapter, RemovesInvalidNonplanarTrackerCovariance) {
  auto input = valid_input();
  auto & covariance =
    input.objects.front().kinematics.pose_with_covariance.covariance;
  covariance[21] = 0.01;
  covariance[22] = -8.0;
  covariance[23] = 1100.0;
  covariance[27] = -8.0;
  covariance[28] = 0.01;
  covariance[29] = -290.0;
  covariance[33] = 1100.0;
  covariance[34] = -290.0;
  covariance[35] = 0.01;

  const auto output = adapt_tracked_objects(
    input, 10 * kSecondNs + 100000000LL,
    std::nullopt, valid_config());

  ASSERT_EQ(output.objects.size(), 1U);
  const auto & projected = output.objects.front().initial_pose.covariance;
  EXPECT_DOUBLE_EQ(projected[0], 1.0);
  EXPECT_DOUBLE_EQ(projected[1], 0.2);
  EXPECT_DOUBLE_EQ(projected[6], 0.2);
  EXPECT_DOUBLE_EQ(projected[7], 2.0);
  EXPECT_DOUBLE_EQ(projected[35], 0.01);
  EXPECT_DOUBLE_EQ(projected[22], 0.0);
  EXPECT_DOUBLE_EQ(projected[23], 0.0);
  EXPECT_DOUBLE_EQ(projected[29], 0.0);
  EXPECT_DOUBLE_EQ(projected[33], 0.0);
  EXPECT_DOUBLE_EQ(projected[34], 0.0);
}

TEST(AutowarePredictionAdapter, SelectsClassificationDeterministically) {
  auto input = valid_input();
  auto & classifications = input.objects.front().classification;
  classifications.clear();
  classifications.emplace_back();
  classifications.back().label = ObjectClassification::TRUCK;
  classifications.back().probability = 0.6F;
  classifications.emplace_back();
  classifications.back().label = ObjectClassification::CAR;
  classifications.back().probability = 0.6F;

  const auto output = adapt_tracked_objects(
    input, 10 * kSecondNs + 100000000LL,
    std::nullopt, valid_config());

  ASSERT_EQ(output.objects.size(), 1U);
  EXPECT_EQ(
    output.objects.front().classification,
    ad_interfaces::msg::PredictedObject::CAR);
  EXPECT_FLOAT_EQ(output.objects.front().classification_probability, 0.6F);
}

TEST(AutowarePredictionAdapter, RejectsInvalidClassificationAndShape) {
  const auto expect_invalid = [](const TrackedObjects & input) {
      EXPECT_THROW(
        (void)adapt_tracked_objects(
          input,
          10 * kSecondNs + 100000000LL,
          std::nullopt, valid_config()),
        std::invalid_argument);
    };

  auto input = valid_input();
  input.objects.front().classification.clear();
  expect_invalid(input);
  input = valid_input();
  input.objects.front().classification.front().label = 8U;
  expect_invalid(input);
  input = valid_input();
  input.objects.front().classification.front().probability =
    std::numeric_limits<float>::quiet_NaN();
  expect_invalid(input);
  input = valid_input();
  input.objects.front().shape.type = Shape::CYLINDER;
  expect_invalid(input);
  input = valid_input();
  input.objects.front().shape.dimensions.x = 0.0;
  expect_invalid(input);
}

TEST(AutowarePredictionAdapter, RejectsInvalidPoseQuaternion) {
  const auto expect_invalid = [](const TrackedObjects & input) {
      EXPECT_THROW(
        (void)adapt_tracked_objects(
          input,
          10 * kSecondNs + 100000000LL,
          std::nullopt, valid_config()),
        std::invalid_argument);
    };

  auto input = valid_input();
  auto & zero_orientation =
    input.objects.front().kinematics.pose_with_covariance.pose.orientation;
  zero_orientation.x = 0.0;
  zero_orientation.y = 0.0;
  zero_orientation.z = 0.0;
  zero_orientation.w = 0.0;
  expect_invalid(input);
  input = valid_input();
  input.objects.front().kinematics.pose_with_covariance.pose.orientation.w *=
    0.9;
  expect_invalid(input);
  input = valid_input();
  input.objects.front().kinematics.pose_with_covariance.pose.orientation.x =
    0.01;
  expect_invalid(input);
  input = valid_input();
  input.objects.front().kinematics.pose_with_covariance.pose.orientation.z =
    std::numeric_limits<double>::quiet_NaN();
  expect_invalid(input);
}

TEST(AutowarePredictionAdapter, NormalizesNearUnitQuaternion) {
  auto input = valid_input();
  auto & orientation =
    input.objects.front().kinematics.pose_with_covariance.pose.orientation;
  orientation.z *= 1.0 + 5.0e-7;
  orientation.w *= 1.0 + 5.0e-7;

  const auto output = adapt_tracked_objects(
    input, 10 * kSecondNs + 100000000LL,
    std::nullopt, valid_config());

  ASSERT_EQ(output.objects.size(), 1U);
  const auto & normalized = output.objects.front().initial_pose.pose.orientation;
  const double norm =
    std::sqrt(
    normalized.x * normalized.x + normalized.y * normalized.y +
    normalized.z * normalized.z + normalized.w * normalized.w);
  EXPECT_NEAR(norm, 1.0, 1.0e-12);
}

TEST(AutowarePredictionAdapter, ProjectsValidTrackerRollPitchToPlanarYaw) {
  auto input = valid_input();
  constexpr double roll = 0.12;
  constexpr double pitch = -0.08;
  constexpr double yaw = 0.70;
  const double cr = std::cos(0.5 * roll);
  const double sr = std::sin(0.5 * roll);
  const double cp = std::cos(0.5 * pitch);
  const double sp = std::sin(0.5 * pitch);
  const double cy = std::cos(0.5 * yaw);
  const double sy = std::sin(0.5 * yaw);
  auto & orientation =
    input.objects.front().kinematics.pose_with_covariance.pose.orientation;
  orientation.w = cr * cp * cy + sr * sp * sy;
  orientation.x = sr * cp * cy - cr * sp * sy;
  orientation.y = cr * sp * cy + sr * cp * sy;
  orientation.z = cr * cp * sy - sr * sp * cy;

  const auto output = adapt_tracked_objects(
    input, 10 * kSecondNs + 100000000LL,
    std::nullopt, valid_config());

  ASSERT_EQ(output.objects.size(), 1U);
  const auto & projected = output.objects.front().initial_pose.pose.orientation;
  EXPECT_DOUBLE_EQ(projected.x, 0.0);
  EXPECT_DOUBLE_EQ(projected.y, 0.0);
  EXPECT_NEAR(projected.z, std::sin(0.5 * yaw), 1.0e-12);
  EXPECT_NEAR(projected.w, std::cos(0.5 * yaw), 1.0e-12);
  EXPECT_NEAR(
    output.objects.front().initial_twist.twist.linear.x,
    3.0 * std::cos(yaw) - 2.0 * std::sin(yaw), 1.0e-12);
  EXPECT_NEAR(
    output.objects.front().initial_twist.twist.linear.y,
    3.0 * std::sin(yaw) + 2.0 * std::cos(yaw), 1.0e-12);
}

TEST(AutowarePredictionAdapter, RejectsInvalidWorldAndLocalXyCovariance) {
  auto input = valid_input();
  input.objects.front().kinematics.pose_with_covariance.covariance[1] = 0.3;
  EXPECT_THROW(
    (void)adapt_tracked_objects(
      input, 10 * kSecondNs + 100000000LL,
      std::nullopt, valid_config()),
    std::invalid_argument);

  input = valid_input();
  auto & covariance =
    input.objects.front().kinematics.twist_with_covariance.covariance;
  covariance[0] = 1.0;
  covariance[1] = 2.0;
  covariance[6] = 2.0;
  covariance[7] = 1.0;
  EXPECT_THROW(
    (void)adapt_tracked_objects(
      input, 10 * kSecondNs + 100000000LL,
      std::nullopt, valid_config()),
    std::invalid_argument);
}

TEST(AutowarePredictionAdapter, AppliesFrameTimeAndMonotonicAdmission) {
  const auto config = valid_config();
  auto input = valid_input();
  input.header.frame_id = "map";
  EXPECT_THROW(
    (void)adapt_tracked_objects(
      input, 10 * kSecondNs + 100000000LL,
      std::nullopt, config),
    std::invalid_argument);

  input = valid_input();
  set_stamp(input, 0);
  EXPECT_THROW(
    (void)adapt_tracked_objects(input, 10 * kSecondNs, std::nullopt, config),
    std::invalid_argument);

  input = valid_input(10 * kSecondNs);
  EXPECT_THROW(
    (void)adapt_tracked_objects(
      input, 10 * kSecondNs + 100000000LL,
      10 * kSecondNs, config),
    std::invalid_argument);
  EXPECT_THROW(
    (void)adapt_tracked_objects(
      input, 10 * kSecondNs + 100000000LL,
      11 * kSecondNs, config),
    std::invalid_argument);
  EXPECT_THROW(
    (void)adapt_tracked_objects(input, 9 * kSecondNs, std::nullopt, config),
    std::invalid_argument);
  EXPECT_THROW(
    (void)adapt_tracked_objects(input, 11 * kSecondNs, std::nullopt, config),
    std::invalid_argument);
}

TEST(AutowarePredictionAdapter, FailedArrayAllowsCorrectedSameStampRetry) {
  auto malformed = valid_input();
  malformed.objects.front().shape.dimensions.x = 0.0;
  EXPECT_THROW(
    (void)adapt_tracked_objects(
      malformed,
      10 * kSecondNs + 100000000LL,
      9 * kSecondNs, valid_config()),
    std::invalid_argument);

  const auto corrected =
    adapt_tracked_objects(
    valid_input(), 10 * kSecondNs + 100000000LL,
    9 * kSecondNs, valid_config());
  EXPECT_EQ(corrected.objects.size(), 1U);
}

TEST(AutowarePredictionAdapter, RejectsWholeArrayForOneMalformedObject) {
  auto input = valid_input();
  input.objects.push_back(input.objects.front());
  input.objects.back().shape.dimensions.y = -1.0;

  EXPECT_THROW(
    (void)adapt_tracked_objects(
      input, 10 * kSecondNs + 100000000LL,
      std::nullopt, valid_config()),
    std::invalid_argument);
}

TEST(AutowarePredictionAdapter, UsesReliableVolatileKeepLastOneOutput) {
  const auto profile = prediction_output_qos().get_rmw_qos_profile();

  EXPECT_EQ(profile.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  EXPECT_EQ(profile.depth, 1U);
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_VOLATILE);
}

TEST(AutowarePredictionAdapter, StatefulImmCurvesTurningObjectPrediction) {
  auto config = valid_config();
  config.imm_prediction.horizons_s = config.prediction.horizons_s;
  StatefulImmPredictionAdapter adapter(config);

  auto first = valid_input(10 * kSecondNs);
  auto & first_object = first.objects.front();
  first_object.kinematics.pose_with_covariance.pose.orientation.x = 0.0;
  first_object.kinematics.pose_with_covariance.pose.orientation.y = 0.0;
  first_object.kinematics.pose_with_covariance.pose.orientation.z = 0.0;
  first_object.kinematics.pose_with_covariance.pose.orientation.w = 1.0;
  first_object.kinematics.pose_with_covariance.pose.position.x = 0.0;
  first_object.kinematics.pose_with_covariance.pose.position.y = 0.0;
  first_object.kinematics.pose_with_covariance.covariance[35] = 0.01;
  first_object.kinematics.twist_with_covariance.twist.linear.x = 6.0;
  first_object.kinematics.twist_with_covariance.twist.linear.y = 0.0;
  first_object.kinematics.twist_with_covariance.twist.angular.z = 0.3;
  first_object.kinematics.twist_with_covariance.covariance[35] = 0.01;
  (void)adapter.adapt(first, 10 * kSecondNs + 100000000LL, std::nullopt);

  auto second = first;
  set_stamp(second, 11 * kSecondNs);
  constexpr double radius = 20.0;
  constexpr double yaw = 0.3;
  second.objects.front().kinematics.pose_with_covariance.pose.position.x =
    radius * std::sin(yaw);
  second.objects.front().kinematics.pose_with_covariance.pose.position.y =
    radius * (1.0 - std::cos(yaw));
  second.objects.front().kinematics.pose_with_covariance.pose.orientation.z =
    std::sin(0.5 * yaw);
  second.objects.front().kinematics.pose_with_covariance.pose.orientation.w =
    std::cos(0.5 * yaw);

  const auto output =
    adapter.adapt(second, 11 * kSecondNs + 100000000LL, 10 * kSecondNs);

  ASSERT_EQ(output.objects.size(), 1U);
  ASSERT_EQ(output.objects.front().states.size(), 2U);
  EXPECT_GT(
    output.objects.front().states.back().pose.pose.position.y,
    second.objects.front().kinematics.pose_with_covariance.pose.position.y);
  EXPECT_GT(output.objects.front().initial_twist.twist.angular.z, 0.0);
}

TEST(AutowarePredictionAdapter, MovingUnknownReachesImmAndReportsModelState) {
  auto config = valid_config();
  config.imm_prediction.horizons_s = config.prediction.horizons_s;
  StatefulImmPredictionAdapter adapter(config);

  auto input = valid_input();
  input.objects.front().classification.front().label =
    ObjectClassification::UNKNOWN;
  input.objects.front().classification.front().probability = 1.0F;
  input.objects.front().kinematics.pose_with_covariance.pose.orientation.z =
    0.0;
  input.objects.front().kinematics.pose_with_covariance.pose.orientation.w =
    1.0;
  input.objects.front().kinematics.twist_with_covariance.twist.linear.x = 4.0;
  input.objects.front().kinematics.twist_with_covariance.twist.linear.y = 0.0;
  input.objects.front().kinematics.twist_with_covariance.twist.angular.z = 0.0;

  const auto cycle = adapter.adapt_with_diagnostics(
    input, 10 * kSecondNs + 100000000LL, std::nullopt);

  ASSERT_EQ(cycle.predictions.objects.size(), 1U);
  EXPECT_EQ(
    cycle.predictions.objects.front().classification,
    ad_interfaces::msg::PredictedObject::UNKNOWN);
  EXPECT_GT(
    std::hypot(
      cycle.predictions.objects.front().initial_twist.twist.linear.x,
      cycle.predictions.objects.front().initial_twist.twist.linear.y),
    0.1);
  EXPECT_EQ(cycle.diagnostics.header, input.header);
  ASSERT_EQ(cycle.diagnostics.status.size(), 1U);
  const auto & status = cycle.diagnostics.status.front();
  EXPECT_EQ(status.name, "0a0b0c0d0e0f10111213141516171819");
  const auto values = diagnostic_values(status);
  EXPECT_EQ(values.size(), 7U);
  EXPECT_NE(values.at("stationary_probability"), "");
  EXPECT_NE(values.at("constant_velocity_probability"), "");
  EXPECT_NE(values.at("coordinated_turn_probability"), "");
  EXPECT_EQ(values.at("selected_mode"), "constant_velocity");
  EXPECT_EQ(values.at("reset_or_gating_reason"), "track_initialized");
  // Curve-Aware Prediction v1: default config keeps the tracker yaw-rate.
  EXPECT_EQ(values.at("yaw_rate_source_used"), "tracker");
  EXPECT_NE(values.at("yaw_rate_radps"), "");
}

TEST(AutowarePredictionAdapter, ImmDiagnosticsBindReasonsToEachUuid) {
  auto config = valid_config();
  config.imm_prediction.horizons_s = config.prediction.horizons_s;
  StatefulImmPredictionAdapter adapter(config);
  auto first = valid_input(10 * kSecondNs);
  assign_second_uuid(first);

  const auto initialized = adapter.adapt_with_diagnostics(
    first, 10 * kSecondNs + 100000000LL, std::nullopt);
  ASSERT_EQ(initialized.diagnostics.status.size(), 2U);
  EXPECT_EQ(
    diagnostic_values(initialized.diagnostics.status[0]).at(
      "reset_or_gating_reason"),
    "track_initialized");
  EXPECT_EQ(
    diagnostic_values(initialized.diagnostics.status[1]).at(
      "reset_or_gating_reason"),
    "track_initialized");
  EXPECT_NE(
    initialized.diagnostics.status[0].name,
    initialized.diagnostics.status[1].name);

  auto second = first;
  set_stamp(second, 11 * kSecondNs);
  const auto accepted = adapter.adapt_with_diagnostics(
    second, 11 * kSecondNs + 100000000LL, 10 * kSecondNs);
  ASSERT_EQ(accepted.diagnostics.status.size(), 2U);
  for (std::size_t index = 0; index < 2U; ++index) {
    const auto values = diagnostic_values(accepted.diagnostics.status[index]);
    EXPECT_EQ(values.at("reset_or_gating_reason"), "measurement_accepted");
    EXPECT_NE(values.at("stationary_probability"), "");
    EXPECT_NE(values.at("selected_mode"), "");
  }
}

TEST(AutowarePredictionAdapter, ImmReappearanceExpiresBeforeCurrentUpdate) {
  auto config = valid_config();
  config.imm_prediction.horizons_s = config.prediction.horizons_s;
  config.imm_track_retention_sec = 0.5;
  StatefulImmPredictionAdapter adapter(config);
  (void)adapter.adapt_with_diagnostics(
    valid_input(10 * kSecondNs),
    10 * kSecondNs + 100000000LL, std::nullopt);

  const auto cycle = adapter.adapt_with_diagnostics(
    valid_input(11 * kSecondNs),
    11 * kSecondNs + 100000000LL, 10 * kSecondNs);

  ASSERT_EQ(cycle.diagnostics.status.size(), 1U);
  EXPECT_EQ(
    diagnostic_values(cycle.diagnostics.status.front()).at(
      "reset_or_gating_reason"),
    "retention_expired");
}

TEST(AutowarePredictionAdapter, ImmReportsClockRollbackAndIntervalClamp) {
  auto config = valid_config();
  config.imm_prediction.horizons_s = config.prediction.horizons_s;
  config.imm_prediction.maximum_update_interval_s = 0.25;
  config.imm_track_retention_sec = 5.0;
  StatefulImmPredictionAdapter adapter(config);
  (void)adapter.adapt_with_diagnostics(
    valid_input(10 * kSecondNs),
    10 * kSecondNs + 100000000LL, std::nullopt);

  const auto gap = adapter.adapt_with_diagnostics(
    valid_input(11 * kSecondNs),
    11 * kSecondNs + 100000000LL, 10 * kSecondNs);
  ASSERT_EQ(gap.diagnostics.status.size(), 1U);
  EXPECT_EQ(
    diagnostic_values(gap.diagnostics.status.front()).at(
      "reset_or_gating_reason"),
    "update_interval_clamped");

  adapter.reset(ImmUpdateReason::kClockRollback);
  const auto rollback = adapter.adapt_with_diagnostics(
    valid_input(5 * kSecondNs),
    5 * kSecondNs + 100000000LL, std::nullopt);
  ASSERT_EQ(rollback.diagnostics.status.size(), 1U);
  EXPECT_EQ(
    diagnostic_values(rollback.diagnostics.status.front()).at(
      "reset_or_gating_reason"),
    "clock_rollback");
}

TEST(AutowarePredictionAdapter, RejectedArraysPublishTypedPerObjectEvidence) {
  const std::map<std::string, std::string> cases{
    {"tracked objects are not in the odom frame", "rejected_frame_gate"},
    {"tracked-object stamp is not newer", "rejected_monotonic_stamp_gate"},
    {"tracked-object stamp is in the future", "rejected_future_stamp_gate"},
    {"tracked-object array is stale", "rejected_stale_array_gate"},
    {"tracked-object dimensions must be finite and positive",
      "rejected_object_validation"},
    {"IMM measurement variances must be finite and positive",
      "rejected_imm_update"}};
  const auto input = valid_input();

  for (const auto & [message, expected_reason] : cases) {
    const auto diagnostics = rejected_prediction_diagnostics(input, message);
    EXPECT_EQ(diagnostics.header, input.header);
    ASSERT_EQ(diagnostics.status.size(), 1U);
    EXPECT_EQ(diagnostics.status[0].name, "tracked_object_array");
    EXPECT_EQ(
      diagnostic_values(diagnostics.status[0]).at(
        "reset_or_gating_reason"),
      expected_reason);
  }

  auto two_objects = input;
  assign_second_uuid(two_objects);
  const auto culprit = rejected_prediction_diagnostics(
    two_objects,
    "IMM measurements must be finite",
    two_objects.objects[1].object_id.uuid);
  ASSERT_EQ(culprit.status.size(), 2U);
  EXPECT_EQ(culprit.status[0].name, "tracked_object_array");
  EXPECT_EQ(culprit.status[1].name, "fe0b0c0d0e0f10111213141516171819");
}

TEST(AutowarePredictionAdapter, RejectedSecondObjectCannotPoisonSameStampRetry) {
  auto config = valid_config();
  config.imm_prediction.horizons_s = config.prediction.horizons_s;
  StatefulImmPredictionAdapter retried(config);
  StatefulImmPredictionAdapter baseline(config);
  auto first = valid_input(10 * kSecondNs);
  assign_second_uuid(first);
  (void)retried.adapt_with_diagnostics(
    first, 10 * kSecondNs + 100000000LL, std::nullopt);
  (void)baseline.adapt_with_diagnostics(
    first, 10 * kSecondNs + 100000000LL, std::nullopt);

  auto malformed = first;
  set_stamp(malformed, 11 * kSecondNs);
  malformed.objects[1].kinematics.twist_with_covariance.twist.angular.z =
    std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(
    (void)retried.adapt_with_diagnostics(
      malformed, 11 * kSecondNs + 100000000LL, 10 * kSecondNs),
    std::invalid_argument);

  auto corrected = malformed;
  corrected.objects[1].kinematics.twist_with_covariance.twist.angular.z = 0.0;
  const auto retry = retried.adapt_with_diagnostics(
    corrected, 11 * kSecondNs + 100000000LL, 10 * kSecondNs);
  const auto clean = baseline.adapt_with_diagnostics(
    corrected, 11 * kSecondNs + 100000000LL, 10 * kSecondNs);

  ASSERT_EQ(retry.predictions.objects.size(), 2U);
  ASSERT_EQ(clean.predictions.objects.size(), 2U);
  for (std::size_t index = 0; index < retry.predictions.objects.size(); ++index) {
    expect_same_prediction(
      retry.predictions.objects[index], clean.predictions.objects[index]);
  }
  EXPECT_EQ(retry.diagnostics, clean.diagnostics);
}

TEST(AutowarePredictionAdapter, DuplicateUuidRejectionDoesNotMutateHistory) {
  auto config = valid_config();
  config.imm_prediction.horizons_s = config.prediction.horizons_s;
  StatefulImmPredictionAdapter retried(config);
  StatefulImmPredictionAdapter baseline(config);
  const auto first = valid_input(10 * kSecondNs);
  (void)retried.adapt_with_diagnostics(
    first, 10 * kSecondNs + 100000000LL, std::nullopt);
  (void)baseline.adapt_with_diagnostics(
    first, 10 * kSecondNs + 100000000LL, std::nullopt);

  auto duplicate = first;
  set_stamp(duplicate, 11 * kSecondNs);
  duplicate.objects.push_back(duplicate.objects.front());
  EXPECT_THROW(
    (void)retried.adapt_with_diagnostics(
      duplicate, 11 * kSecondNs + 100000000LL, 10 * kSecondNs),
    std::invalid_argument);

  auto corrected = first;
  set_stamp(corrected, 11 * kSecondNs);
  const auto retry = retried.adapt_with_diagnostics(
    corrected, 11 * kSecondNs + 100000000LL, 10 * kSecondNs);
  const auto clean = baseline.adapt_with_diagnostics(
    corrected, 11 * kSecondNs + 100000000LL, 10 * kSecondNs);

  ASSERT_EQ(retry.predictions.objects.size(), 1U);
  ASSERT_EQ(clean.predictions.objects.size(), 1U);
  expect_same_prediction(
    retry.predictions.objects[0], clean.predictions.objects[0]);
  EXPECT_EQ(retry.diagnostics, clean.diagnostics);
}

TEST(AutowarePredictionAdapter, RetentionReasonSurvivesEmptyCallback) {
  auto config = valid_config();
  config.imm_prediction.horizons_s = config.prediction.horizons_s;
  config.imm_track_retention_sec = 0.5;
  StatefulImmPredictionAdapter adapter(config);
  StatefulImmPredictionAdapter fresh(config);
  (void)adapter.adapt_with_diagnostics(
    valid_input(10 * kSecondNs),
    10 * kSecondNs + 100000000LL, std::nullopt);

  auto empty = valid_input(11 * kSecondNs);
  empty.objects.clear();
  (void)adapter.adapt_with_diagnostics(
    empty, 11 * kSecondNs + 100000000LL, 10 * kSecondNs);
  const auto reappeared_input = valid_input(12 * kSecondNs);
  const auto reappeared = adapter.adapt_with_diagnostics(
    reappeared_input,
    12 * kSecondNs + 100000000LL, 11 * kSecondNs);
  const auto baseline = fresh.adapt_with_diagnostics(
    reappeared_input,
    12 * kSecondNs + 100000000LL, std::nullopt);

  ASSERT_EQ(reappeared.diagnostics.status.size(), 1U);
  auto reappeared_values = diagnostic_values(
    reappeared.diagnostics.status.front());
  const auto baseline_values = diagnostic_values(
    baseline.diagnostics.status.front());
  EXPECT_EQ(
    reappeared_values.at("reset_or_gating_reason"),
    "retention_expired");
  reappeared_values["reset_or_gating_reason"] = "track_initialized";
  EXPECT_EQ(reappeared_values, baseline_values);
}

TEST(AutowarePredictionAdapter, TombstonesBindOnlyExpiredUuidInMultiObjectFlow) {
  auto config = valid_config();
  config.imm_prediction.horizons_s = config.prediction.horizons_s;
  config.imm_track_retention_sec = 0.5;
  StatefulImmPredictionAdapter adapter(config);
  auto first = valid_input(10 * kSecondNs);
  assign_second_uuid(first);
  (void)adapter.adapt_with_diagnostics(
    first, 10 * kSecondNs + 100000000LL, std::nullopt);

  auto only_b = first;
  set_stamp(only_b, 10 * kSecondNs + 600000000LL);
  only_b.objects.erase(only_b.objects.begin());
  (void)adapter.adapt_with_diagnostics(
    only_b, 10 * kSecondNs + 700000000LL, 10 * kSecondNs);

  auto both = first;
  set_stamp(both, 10 * kSecondNs + 800000000LL);
  const auto cycle = adapter.adapt_with_diagnostics(
    both, 10 * kSecondNs + 900000000LL,
    10 * kSecondNs + 600000000LL);

  const auto a = diagnostic_values(diagnostic_for(
      cycle.diagnostics, "0a0b0c0d0e0f10111213141516171819"));
  const auto b = diagnostic_values(diagnostic_for(
      cycle.diagnostics, "fe0b0c0d0e0f10111213141516171819"));
  EXPECT_EQ(a.at("reset_or_gating_reason"), "retention_expired");
  EXPECT_EQ(b.at("reset_or_gating_reason"), "measurement_accepted");
}

// --- AB3DMOT Competition MOT baseline -> HEVEN prediction contract ---
// Adaptive Euclidean clustering never observes yaw, so the AB3DMOT baseline
// publishes an identity quaternion with orientation_availability=UNAVAILABLE.
// Prediction must treat that as planar Cartesian motion, never as an
// observed heading, and must not fabricate a turn from the placeholder yaw.

TrackedObjects ab3dmot_style_input(
  const std::int64_t stamp_ns, const double local_vx, const double local_vy)
{
  auto input = valid_input(stamp_ns);
  auto & object = input.objects.back();
  object.classification.back().label = ObjectClassification::UNKNOWN;
  object.classification.back().probability = 1.0F;
  auto & pose = object.kinematics.pose_with_covariance.pose;
  pose.orientation.x = 0.0;
  pose.orientation.y = 0.0;
  pose.orientation.z = 0.0;
  pose.orientation.w = 1.0;
  object.kinematics.orientation_availability = 0U;  // UNAVAILABLE
  auto & twist = object.kinematics.twist_with_covariance.twist;
  twist.linear.x = local_vx;
  twist.linear.y = local_vy;
  twist.linear.z = 0.0;
  twist.angular.z = 0.0;
  return input;
}

TEST(AutowarePredictionAdapter, Ab3dmotOrientationUnavailableYieldsCartesianWorldTwist) {
  const auto input = ab3dmot_style_input(10 * kSecondNs, 3.0, -2.0);
  const auto output = adapt_tracked_objects(
    input, 10 * kSecondNs + 100000000LL, std::nullopt, valid_config());

  ASSERT_EQ(output.objects.size(), 1U);
  const auto & twist = output.objects.front().initial_twist;
  // Latent yaw is zero, so R(+yaw) is identity: world velocity equals the
  // published object-local twist with no rotation introduced.
  EXPECT_NEAR(twist.twist.linear.x, 3.0, 1.0e-12);
  EXPECT_NEAR(twist.twist.linear.y, -2.0, 1.0e-12);
  EXPECT_TRUE(
    std::all_of(
      twist.covariance.begin(), twist.covariance.end(),
      [](const double value) {return std::isfinite(value);}));
  const auto & orientation = output.objects.front().initial_pose.pose.orientation;
  EXPECT_DOUBLE_EQ(orientation.x, 0.0);
  EXPECT_DOUBLE_EQ(orientation.y, 0.0);
  EXPECT_NEAR(orientation.z, 0.0, 1.0e-12);
  EXPECT_NEAR(orientation.w, 1.0, 1.0e-12);
}

TEST(AutowarePredictionAdapter, Ab3dmotOrientationUnavailableDoesNotForceCoordinatedTurn) {
  auto config = valid_config();
  config.imm_prediction.horizons_s = config.prediction.horizons_s;
  StatefulImmPredictionAdapter adapter(config);

  const auto straight_step = [&](const int step) {
    const std::int64_t stamp = 10 * kSecondNs + step * 100000000LL;
    // Straight-line world motion along +x; identity quaternion throughout.
    auto input = ab3dmot_style_input(stamp, 4.0, 0.0);
    input.objects.back().kinematics.pose_with_covariance.pose.position.x =
      10.0 + 0.4 * step;
    input.objects.back().kinematics.pose_with_covariance.pose.position.y = 20.0;
    const std::optional<std::int64_t> last =
      step == 0 ? std::nullopt :
      std::optional<std::int64_t>(10 * kSecondNs + (step - 1) * 100000000LL);
    return adapter.adapt_with_diagnostics(input, stamp + 50000000LL, last);
  };
  (void)straight_step(0);
  (void)straight_step(1);
  (void)straight_step(2);
  const auto cycle = straight_step(3);

  ASSERT_EQ(cycle.diagnostics.status.size(), 1U);
  const auto values = diagnostic_values(cycle.diagnostics.status.front());
  EXPECT_NE(values.at("selected_mode"), "coordinated_turn");
  ASSERT_EQ(cycle.predictions.objects.size(), 1U);
  const auto & twist = cycle.predictions.objects.front().initial_twist;
  EXPECT_TRUE(
    std::all_of(
      twist.covariance.begin(), twist.covariance.end(),
      [](const double value) {return std::isfinite(value);}));
  EXPECT_GT(std::hypot(twist.twist.linear.x, twist.twist.linear.y), 0.1);
}

// --- Curve-Aware Prediction v1: motion-history yaw-rate estimator -------------
// AB3DMOT publishes twist.angular.z == 0 (no angular-velocity state), so the IMM
// coordinated-turn model never sees curvature. In motion_history mode the
// adapter derives a robust turn rate from the recent world-velocity heading and
// overrides the IMM yaw-rate measurement. These tests drive the adapter (the
// estimator itself is file-local) and read the derived rate from the diagnostic
// KeyValue plus the resulting predicted-trajectory curvature.

double quaternion_yaw(const geometry_msgs::msg::Quaternion & q)
{
  return 2.0 * std::atan2(q.z, q.w);
}

AutowarePredictionAdapterConfig motion_history_config()
{
  auto config = valid_config();
  config.imm_prediction.horizons_s = config.prediction.horizons_s;  // {0.5, 1.0}
  config.yaw_rate_source = YawRateSource::kMotionHistory;
  return config;  // motion_history defaults: 4 samples, 2.0 m/s, 1.5 rad/s, 0.10
}

// Identity-quaternion tracked object whose WORLD velocity heading is `heading`
// at `speed`, positioned at (x, y). Mirrors the AB3DMOT contract.
TrackedObjects world_velocity_input(
  const std::int64_t stamp_ns, const double speed, const double heading,
  const double x, const double y)
{
  auto input = ab3dmot_style_input(
    stamp_ns, speed * std::cos(heading), speed * std::sin(heading));
  auto & kinematics = input.objects.back().kinematics;
  kinematics.pose_with_covariance.pose.position.x = x;
  kinematics.pose_with_covariance.pose.position.y = y;
  // Real AB3DMOT leaves the yaw-rate variance slot at 0 (no angular state);
  // valid_input()'s 77.0 sentinel would otherwise dominate the tracker-mode
  // IMM likelihood and mask the motion_history comparison.
  kinematics.twist_with_covariance.covariance[35] = 0.0;
  return input;
}

// Feeds `frames` frames at `dt` spacing, world-velocity heading advancing by
// `omega * dt` per frame from `heading0`, integrating position. Returns the
// final adapt_with_diagnostics cycle.
PredictionAdaptation feed_arc(
  StatefulImmPredictionAdapter & adapter, const int frames, const double speed,
  const double heading0, const double omega, const double dt = 0.1)
{
  const std::int64_t base = 10 * kSecondNs;
  const auto dt_ns = static_cast<std::int64_t>(dt * 1.0e9);
  double x = 0.0;
  double y = 0.0;
  double heading = heading0;
  PredictionAdaptation cycle;
  for (int step = 0; step < frames; ++step) {
    const std::int64_t stamp = base + step * dt_ns;
    const std::optional<std::int64_t> last =
      step == 0 ? std::nullopt
      : std::optional<std::int64_t>(base + (step - 1) * dt_ns);
    cycle = adapter.adapt_with_diagnostics(
      world_velocity_input(stamp, speed, heading, x, y), stamp + 50000000LL,
      last);
    x += speed * std::cos(heading) * dt;
    y += speed * std::sin(heading) * dt;
    heading += omega * dt;
  }
  return cycle;
}

std::map<std::string, std::string> final_diagnostic(
  const PredictionAdaptation & cycle)
{
  EXPECT_EQ(cycle.diagnostics.status.size(), 1U);
  return diagnostic_values(cycle.diagnostics.status.front());
}

double predicted_heading_change(const ad_interfaces::msg::PredictedObject & object)
{
  EXPECT_GE(object.states.size(), 2U);
  return quaternion_yaw(object.states.back().pose.pose.orientation) -
         quaternion_yaw(object.states.front().pose.pose.orientation);
}

double predicted_heading_change(const PredictionAdaptation & cycle)
{
  return predicted_heading_change(cycle.predictions.objects.front());
}

// A. Straight motion -> omega ~ 0, constant-velocity, no curvature.
TEST(CurveAwarePrediction, StraightMotionYieldsZeroRateAndNoCurve) {
  StatefulImmPredictionAdapter adapter(motion_history_config());
  const auto cycle = feed_arc(adapter, 8, 6.0, 0.30, 0.0);

  const auto values = final_diagnostic(cycle);
  EXPECT_EQ(values.at("yaw_rate_source_used"), "motion_history");
  EXPECT_NEAR(std::stod(values.at("yaw_rate_radps")), 0.0, 1.0e-9);
  EXPECT_NE(values.at("selected_mode"), "coordinated_turn");
  EXPECT_NEAR(predicted_heading_change(cycle), 0.0, 1.0e-6);
}

// A'. motion_history must not materially perturb a straight object relative to
// tracker mode. (Not bit-identical: the reported yaw-rate variance changes by
// design, so the IMM likelihood term shifts microscopically.)
TEST(CurveAwarePrediction, StraightMotionTracksTrackerModePrediction) {
  auto tracker_config = valid_config();
  tracker_config.imm_prediction.horizons_s = tracker_config.prediction.horizons_s;
  StatefulImmPredictionAdapter tracker_adapter(tracker_config);
  StatefulImmPredictionAdapter history_adapter(motion_history_config());

  const auto tracker_cycle = feed_arc(tracker_adapter, 8, 5.0, -0.4, 0.0);
  const auto history_cycle = feed_arc(history_adapter, 8, 5.0, -0.4, 0.0);

  const auto & tracker_object = tracker_cycle.predictions.objects.front();
  const auto & history_object = history_cycle.predictions.objects.front();
  ASSERT_EQ(history_object.states.size(), tracker_object.states.size());
  for (std::size_t i = 0; i < history_object.states.size(); ++i) {
    EXPECT_NEAR(
      history_object.states[i].pose.pose.position.x,
      tracker_object.states[i].pose.pose.position.x, 1.0e-3);
    EXPECT_NEAR(
      history_object.states[i].pose.pose.position.y,
      tracker_object.states[i].pose.pose.position.y, 1.0e-3);
  }
  EXPECT_NEAR(predicted_heading_change(history_object), 0.0, 1.0e-6);
}

// B. Constant left turn -> positive derived rate, CT contributes, curve bends
// the predicted trajectory left. Proves the existing IMM logic responds to a
// valid non-zero yaw-rate (no imm_predictor.cpp change needed).
TEST(CurveAwarePrediction, ConstantLeftTurnBendsPredictionAndReachesCt) {
  StatefulImmPredictionAdapter adapter(motion_history_config());
  const auto cycle = feed_arc(adapter, 12, 6.0, 0.0, 0.30);

  const auto values = final_diagnostic(cycle);
  EXPECT_EQ(values.at("yaw_rate_source_used"), "motion_history");
  const double derived = std::stod(values.at("yaw_rate_radps"));
  EXPECT_NEAR(derived, 0.30, 0.05);

  // Magnitude, not just sign: the fused yaw-rate must reach a meaningful
  // fraction of the derived rate and the predicted heading must bend left.
  // (Measured: derived 0.30 -> CT prob ~0.88, fused wz ~0.27.)
  const double fused_rate =
    cycle.predictions.objects.front().initial_twist.twist.angular.z;
  EXPECT_GT(fused_rate, 0.10);
  EXPECT_EQ(values.at("selected_mode"), "coordinated_turn");
  EXPECT_GT(std::stod(values.at("coordinated_turn_probability")), 0.50);
  EXPECT_GT(predicted_heading_change(cycle), 0.10);
}

// C. Constant right turn -> negative derived rate, prediction bends right.
TEST(CurveAwarePrediction, ConstantRightTurnBendsPredictionOppositeWay) {
  StatefulImmPredictionAdapter adapter(motion_history_config());
  const auto cycle = feed_arc(adapter, 12, 6.0, 0.0, -0.30);

  const auto values = final_diagnostic(cycle);
  EXPECT_EQ(values.at("yaw_rate_source_used"), "motion_history");
  EXPECT_NEAR(std::stod(values.at("yaw_rate_radps")), -0.30, 0.05);
  EXPECT_LT(
    cycle.predictions.objects.front().initial_twist.twist.angular.z, -0.10);
  EXPECT_LT(predicted_heading_change(cycle), -0.10);
}

// D. Insufficient history -> tracker fallback, no curvature.
TEST(CurveAwarePrediction, InsufficientHistoryFallsBackToTracker) {
  StatefulImmPredictionAdapter adapter(motion_history_config());
  const auto cycle = feed_arc(adapter, 3, 6.0, 0.0, 0.30);  // < history_samples

  const auto values = final_diagnostic(cycle);
  EXPECT_EQ(values.at("yaw_rate_source_used"), "tracker_fallback");
  EXPECT_NEAR(std::stod(values.at("yaw_rate_radps")), 0.0, 1.0e-12);
  EXPECT_NE(values.at("selected_mode"), "coordinated_turn");
}

// E. Below the speed gate -> tracker fallback even though the heading rotates.
TEST(CurveAwarePrediction, LowSpeedFallsBackDespiteRotatingHeading) {
  StatefulImmPredictionAdapter adapter(motion_history_config());
  const auto cycle = feed_arc(adapter, 8, 1.0, 0.0, 0.60);  // 1.0 < min 2.0

  const auto values = final_diagnostic(cycle);
  EXPECT_EQ(values.at("yaw_rate_source_used"), "tracker_fallback");
  EXPECT_NEAR(std::stod(values.at("yaw_rate_radps")), 0.0, 1.0e-12);
  EXPECT_NEAR(predicted_heading_change(cycle), 0.0, 1.0e-6);
}

// F. Heading wraparound (+/-pi crossing) must not inject a 2*pi/dt spike. A
// median of >= 3 slopes is structurally robust to one bad pair, so this uses
// history_samples == 3 (two slopes) - the tightest guard on the unwrap.
TEST(CurveAwarePrediction, HeadingWraparoundProducesNoSpike) {
  auto config = motion_history_config();
  config.motion_history.history_samples = 3U;
  StatefulImmPredictionAdapter adapter(config);

  const double deg = kPi / 180.0;
  // Velocity headings 175 deg -> -175 deg -> -170 deg at 0.1 s spacing.
  // True rate: +10 deg then +5 deg -> ~ +1.31 rad/s. A missing unwrap turns the
  // 175 -> -175 step into -350 deg and flips the median sign.
  const std::array<double, 3> headings{175.0 * deg, -175.0 * deg, -170.0 * deg};
  PredictionAdaptation cycle;
  for (std::size_t step = 0; step < headings.size(); ++step) {
    const std::int64_t stamp = 10 * kSecondNs + static_cast<std::int64_t>(step) *
      100000000LL;
    const std::optional<std::int64_t> last = step == 0 ? std::nullopt
      : std::optional<std::int64_t>(
      10 * kSecondNs + static_cast<std::int64_t>(step - 1) * 100000000LL);
    cycle = adapter.adapt_with_diagnostics(
      world_velocity_input(
        stamp, 6.0, headings[step], 0.2 * static_cast<double>(step), 0.0),
      stamp + 50000000LL, last);
  }

  const auto values = final_diagnostic(cycle);
  EXPECT_EQ(values.at("yaw_rate_source_used"), "motion_history");
  const double derived = std::stod(values.at("yaw_rate_radps"));
  EXPECT_GT(derived, 0.5);            // correct unwrap: ~ +1.31 rad/s
  EXPECT_LT(derived, 1.5);            // not clamped -> no wrap spike
}

// G. Estimates beyond the clamp are limited to max_yaw_rate_radps.
TEST(CurveAwarePrediction, LargeEstimateIsClampedToConfiguredCeiling) {
  auto config = motion_history_config();
  StatefulImmPredictionAdapter adapter(config);
  // ~12 rad/s heading rate, well past the 1.5 rad/s ceiling.
  const auto cycle = feed_arc(adapter, 8, 6.0, 0.0, 12.0);

  const auto values = final_diagnostic(cycle);
  EXPECT_EQ(values.at("yaw_rate_source_used"), "motion_history");
  EXPECT_NEAR(
    std::stod(values.at("yaw_rate_radps")),
    config.motion_history.max_yaw_rate_radps, 1.0e-9);
}

// H. Tracker mode (the default) is byte-compatible with the previous path: the
// tracker twist.angular.z (0 for AB3DMOT) is used verbatim and turning input
// never bends the prediction.
TEST(CurveAwarePrediction, TrackerModeReproducesPreviousBehaviour) {
  auto config = valid_config();
  config.imm_prediction.horizons_s = config.prediction.horizons_s;
  ASSERT_EQ(config.yaw_rate_source, YawRateSource::kTracker);
  StatefulImmPredictionAdapter adapter(config);

  const auto cycle = feed_arc(adapter, 12, 6.0, 0.0, 0.30);

  const auto values = final_diagnostic(cycle);
  EXPECT_EQ(values.size(), 7U);
  EXPECT_EQ(values.at("yaw_rate_source_used"), "tracker");
  EXPECT_NEAR(std::stod(values.at("yaw_rate_radps")), 0.0, 1.0e-12);
  EXPECT_NE(values.at("selected_mode"), "coordinated_turn");
  EXPECT_NEAR(
    cycle.predictions.objects.front().initial_twist.twist.angular.z, 0.0,
    1.0e-9);
  EXPECT_NEAR(predicted_heading_change(cycle), 0.0, 1.0e-6);
}

TEST(CurveAwarePrediction, ParsesYawRateSourceStrings) {
  EXPECT_EQ(parse_yaw_rate_source("tracker"), YawRateSource::kTracker);
  EXPECT_EQ(
    parse_yaw_rate_source("motion_history"), YawRateSource::kMotionHistory);
  EXPECT_THROW(parse_yaw_rate_source("gyro"), std::invalid_argument);
}

TEST(CurveAwarePrediction, RejectsDegenerateMotionHistoryConfig) {
  auto config = motion_history_config();
  config.motion_history.history_samples = 2U;
  EXPECT_THROW(StatefulImmPredictionAdapter{config}, std::invalid_argument);

  config = motion_history_config();
  config.motion_history.max_yaw_rate_radps = 0.0;
  EXPECT_THROW(StatefulImmPredictionAdapter{config}, std::invalid_argument);
}

} // namespace
