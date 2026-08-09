#include <ad_viz/perception/marker_builder.hpp>
#include <ad_viz/perception/perception_marker_node.hpp>

#include <gtest/gtest.h>

#include <ad_interfaces/msg/predicted_object.hpp>
#include <ad_interfaces/msg/predicted_object_array.hpp>
#include <ad_interfaces/msg/predicted_state.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <geometry_msgs/msg/pose_with_covariance.hpp>
#include <rmw/qos_profiles.h>
#include <visualization_msgs/msg/marker.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

using ad_interfaces::msg::PredictedObject;
using ad_interfaces::msg::PredictedObjectArray;
using ad_interfaces::msg::PredictedState;
using ad_viz::perception::MarkerBuilderConfig;
using ad_viz::perception::MarkerPublicationState;
using ad_viz::perception::build_delete_all;
using ad_viz::perception::build_prediction_markers;
using ad_viz::perception::perception_marker_output_qos;
using ad_viz::perception::predicted_object_input_qos;
using ad_viz::perception::prediction_stamp_ns;
using ad_viz::perception::uuid_namespace;
using visualization_msgs::msg::Marker;

geometry_msgs::msg::PoseWithCovariance pose(
  const double x, const double y, const double z, const double yaw)
{
  geometry_msgs::msg::PoseWithCovariance result;
  result.pose.position.x = x;
  result.pose.position.y = y;
  result.pose.position.z = z;
  result.pose.orientation.z = std::sin(yaw * 0.5);
  result.pose.orientation.w = std::cos(yaw * 0.5);
  for (std::size_t index = 0; index < 6U; ++index) {
    result.covariance[index * 6U + index] = 0.1 + 0.01 * index;
  }
  return result;
}

PredictedState state(
  const std::int32_t sec, const std::uint32_t nanosec,
  const double x, const double y, const double z, const double yaw)
{
  PredictedState result;
  result.time_from_start.sec = sec;
  result.time_from_start.nanosec = nanosec;
  result.pose = pose(x, y, z, yaw);
  return result;
}

PredictedObject object()
{
  PredictedObject result;
  for (std::size_t index = 0; index < result.object_id.uuid.size(); ++index) {
    result.object_id.uuid[index] = static_cast<std::uint8_t>(index + 1U);
  }
  result.existence_probability = 0.9F;
  result.classification = PredictedObject::CAR;
  result.classification_probability = 0.8F;
  result.initial_pose = pose(1.0, 2.0, 0.5, M_PI_2);
  result.initial_twist.twist.linear.x = 2.0;
  result.initial_twist.twist.linear.y = -1.0;
  result.initial_twist.twist.linear.z = 0.25;
  for (std::size_t index = 0; index < 6U; ++index) {
    result.initial_twist.covariance[index * 6U + index] =
      0.2 + 0.01 * index;
  }
  result.dimensions.x = 4.6;
  result.dimensions.y = 1.9;
  result.dimensions.z = 1.6;
  result.states = {
    state(0, 500000000U, 2.0, 1.5, 0.6, 0.25),
    state(1, 0U, 3.0, 1.0, 0.7, -0.5)};
  for (std::int32_t step = 3; step <= 12; ++step) {
    result.states.push_back(
      state(
        step / 2, step % 2 == 0 ? 0U : 500000000U,
        1.0 + static_cast<double>(step),
        2.0 - 0.5 * static_cast<double>(step),
        0.5 + 0.1 * static_cast<double>(step),
        0.05 * static_cast<double>(step)));
  }
  return result;
}

PredictedObjectArray array_with_one_object()
{
  PredictedObjectArray result;
  result.header.frame_id = "odom";
  result.header.stamp.sec = 12;
  result.header.stamp.nanosec = 345U;
  result.objects.push_back(object());
  return result;
}

double duration_seconds(const builtin_interfaces::msg::Duration & duration)
{
  return static_cast<double>(duration.sec) +
         static_cast<double>(duration.nanosec) * 1.0e-9;
}

void expect_same_pose(
  const geometry_msgs::msg::Pose & actual,
  const geometry_msgs::msg::Pose & expected)
{
  EXPECT_DOUBLE_EQ(actual.position.x, expected.position.x);
  EXPECT_DOUBLE_EQ(actual.position.y, expected.position.y);
  EXPECT_DOUBLE_EQ(actual.position.z, expected.position.z);
  EXPECT_DOUBLE_EQ(actual.orientation.x, expected.orientation.x);
  EXPECT_DOUBLE_EQ(actual.orientation.y, expected.orientation.y);
  EXPECT_DOUBLE_EQ(actual.orientation.z, expected.orientation.z);
  EXPECT_DOUBLE_EQ(actual.orientation.w, expected.orientation.w);
}

TEST(MarkerBuilder, BuildsExactCurrentFutureLineAndVelocityGeometry)
{
  const auto input = array_with_one_object();
  const auto output = build_prediction_markers(input, MarkerBuilderConfig{});

  ASSERT_EQ(output.markers.size(), 7U);
  EXPECT_EQ(output.markers[0].action, Marker::DELETEALL);
  const std::string expected_namespace =
    "0102030405060708090a0b0c0d0e0f10";

  const auto & current = output.markers[1];
  EXPECT_EQ(current.header, input.header);
  EXPECT_EQ(current.ns, expected_namespace);
  EXPECT_EQ(current.id, ad_viz::perception::kCurrentBoxMarkerId);
  EXPECT_EQ(current.type, Marker::LINE_LIST);
  EXPECT_EQ(current.action, Marker::ADD);
  expect_same_pose(current.pose, input.objects[0].initial_pose.pose);
  EXPECT_DOUBLE_EQ(current.scale.x, 0.06);
  EXPECT_DOUBLE_EQ(current.scale.y, 0.0);
  EXPECT_DOUBLE_EQ(current.scale.z, 0.0);
  ASSERT_EQ(current.points.size(), 24U);
  EXPECT_DOUBLE_EQ(current.points[0].x, -2.3);
  EXPECT_DOUBLE_EQ(current.points[0].y, -0.95);
  EXPECT_DOUBLE_EQ(current.points[0].z, -0.8);
  EXPECT_DOUBLE_EQ(current.points[1].x, 2.3);
  EXPECT_DOUBLE_EQ(current.points[1].y, -0.95);
  EXPECT_DOUBLE_EQ(current.points[1].z, -0.8);
  EXPECT_FLOAT_EQ(current.color.a, 0.80F);

  const auto & half_second = output.markers[2];
  EXPECT_EQ(half_second.id, ad_viz::perception::kHalfSecondBoxMarkerId);
  EXPECT_EQ(half_second.type, Marker::LINE_LIST);
  expect_same_pose(half_second.pose, input.objects[0].states[0].pose.pose);
  EXPECT_EQ(half_second.scale, current.scale);
  EXPECT_EQ(half_second.points, current.points);
  EXPECT_FLOAT_EQ(half_second.color.r, current.color.r);
  EXPECT_FLOAT_EQ(half_second.color.g, current.color.g);
  EXPECT_FLOAT_EQ(half_second.color.b, current.color.b);
  EXPECT_FLOAT_EQ(half_second.color.a, 0.50F);

  const auto & one_second = output.markers[3];
  EXPECT_EQ(one_second.id, ad_viz::perception::kOneSecondBoxMarkerId);
  EXPECT_EQ(one_second.type, Marker::LINE_LIST);
  expect_same_pose(one_second.pose, input.objects[0].states[1].pose.pose);
  EXPECT_EQ(one_second.scale, current.scale);
  EXPECT_EQ(one_second.points, current.points);
  EXPECT_FLOAT_EQ(one_second.color.r, current.color.r);
  EXPECT_FLOAT_EQ(one_second.color.g, current.color.g);
  EXPECT_FLOAT_EQ(one_second.color.b, current.color.b);
  EXPECT_FLOAT_EQ(one_second.color.a, 0.25F);

  const auto & line = output.markers[4];
  EXPECT_EQ(line.id, ad_viz::perception::kMotionLineMarkerId);
  EXPECT_EQ(line.type, Marker::LINE_STRIP);
  EXPECT_DOUBLE_EQ(line.scale.x, 0.10);
  EXPECT_FLOAT_EQ(line.color.r, current.color.r);
  EXPECT_FLOAT_EQ(line.color.g, current.color.g);
  EXPECT_FLOAT_EQ(line.color.b, current.color.b);
  EXPECT_FLOAT_EQ(line.color.a, 0.90F);
  ASSERT_EQ(line.points.size(), 13U);
  EXPECT_EQ(line.points[0], input.objects[0].initial_pose.pose.position);
  EXPECT_EQ(line.points[1], input.objects[0].states[0].pose.pose.position);
  EXPECT_EQ(line.points[2], input.objects[0].states[1].pose.pose.position);
  EXPECT_EQ(line.points.back(), input.objects[0].states.back().pose.pose.position);

  const auto & arrow = output.markers[5];
  EXPECT_EQ(arrow.id, ad_viz::perception::kVelocityArrowMarkerId);
  EXPECT_EQ(arrow.type, Marker::ARROW);
  EXPECT_DOUBLE_EQ(arrow.scale.x, 0.10);
  EXPECT_DOUBLE_EQ(arrow.scale.y, 0.22);
  EXPECT_DOUBLE_EQ(arrow.scale.z, 0.30);
  EXPECT_FLOAT_EQ(arrow.color.r, current.color.r);
  EXPECT_FLOAT_EQ(arrow.color.g, current.color.g);
  EXPECT_FLOAT_EQ(arrow.color.b, current.color.b);
  EXPECT_FLOAT_EQ(arrow.color.a, 0.95F);
  ASSERT_EQ(arrow.points.size(), 2U);
  EXPECT_EQ(arrow.points[0], input.objects[0].initial_pose.pose.position);
  EXPECT_DOUBLE_EQ(arrow.points[1].x, 3.0);
  EXPECT_DOUBLE_EQ(arrow.points[1].y, 1.0);
  EXPECT_DOUBLE_EQ(arrow.points[1].z, 0.75);

  const auto & label = output.markers[6];
  EXPECT_EQ(label.id, ad_viz::perception::kTrackLabelMarkerId);
  EXPECT_EQ(label.type, Marker::TEXT_VIEW_FACING);
  EXPECT_EQ(label.text, "01020304  2.24 m/s");
  EXPECT_DOUBLE_EQ(label.pose.position.x, 1.0);
  EXPECT_DOUBLE_EQ(label.pose.position.y, 2.0);
  EXPECT_DOUBLE_EQ(label.pose.position.z, 1.55);
  EXPECT_DOUBLE_EQ(label.scale.z, 0.45);
  EXPECT_FLOAT_EQ(label.color.r, current.color.r);
  EXPECT_FLOAT_EQ(label.color.g, current.color.g);
  EXPECT_FLOAT_EQ(label.color.b, current.color.b);
  EXPECT_FLOAT_EQ(label.color.a, 1.0F);

  for (const auto & marker : output.markers) {
    EXPECT_GT(duration_seconds(marker.lifetime), 0.10);
    EXPECT_LT(duration_seconds(marker.lifetime), 1.00);
  }
}

TEST(MarkerBuilder, UsesFullLowercaseUuidNamespacesAndFixedPerObjectIds)
{
  auto input = array_with_one_object();
  auto second = object();
  second.object_id.uuid = input.objects[0].object_id.uuid;
  second.object_id.uuid[15] = 0xFFU;
  input.objects.push_back(second);

  EXPECT_EQ(uuid_namespace(input.objects[0].object_id),
    "0102030405060708090a0b0c0d0e0f10");
  EXPECT_EQ(uuid_namespace(input.objects[1].object_id),
    "0102030405060708090a0b0c0d0e0fff");

  const auto output = build_prediction_markers(input, MarkerBuilderConfig{});
  ASSERT_EQ(output.markers.size(), 13U);
  EXPECT_EQ(output.markers[1].ns, output.markers[2].ns);
  EXPECT_NE(output.markers[1].ns, output.markers[7].ns);
  for (std::size_t index = 0; index < 6U; ++index) {
    EXPECT_EQ(output.markers[1U + index].id, static_cast<std::int32_t>(index));
    EXPECT_EQ(output.markers[7U + index].id, static_cast<std::int32_t>(index));
  }
}

TEST(MarkerBuilder, UsesDeterministicUuidColorIndependentOfClassification)
{
  auto input = array_with_one_object();
  auto same_track = input;
  same_track.objects[0].classification = 255U;
  auto different_track = input;
  different_track.objects[0].object_id.uuid[15] = 0xFFU;

  const auto first = build_prediction_markers(input, MarkerBuilderConfig{});
  const auto second = build_prediction_markers(same_track, MarkerBuilderConfig{});
  const auto third = build_prediction_markers(different_track, MarkerBuilderConfig{});
  ASSERT_GE(first.markers.size(), 2U);
  ASSERT_GE(second.markers.size(), 2U);
  EXPECT_EQ(first.markers[1].color, second.markers[1].color);
  EXPECT_NE(first.markers[1].color, third.markers[1].color);

  auto old_palette_collision_a = input;
  auto old_palette_collision_b = input;
  old_palette_collision_a.objects[0].object_id.uuid[15] = 4U;
  old_palette_collision_b.objects[0].object_id.uuid[15] = 8U;
  const auto collision_a =
    build_prediction_markers(old_palette_collision_a, MarkerBuilderConfig{});
  const auto collision_b =
    build_prediction_markers(old_palette_collision_b, MarkerBuilderConfig{});
  EXPECT_NE(collision_a.markers[1].color, collision_b.markers[1].color);
}

TEST(MarkerBuilder, EmitsDeleteAllFirstForNormalEmptyAndExplicitClear)
{
  const auto normal =
    build_prediction_markers(array_with_one_object(), MarkerBuilderConfig{});
  ASSERT_FALSE(normal.markers.empty());
  EXPECT_EQ(normal.markers.front().action, Marker::DELETEALL);

  auto empty = array_with_one_object();
  empty.objects.clear();
  const auto empty_output =
    build_prediction_markers(empty, MarkerBuilderConfig{});
  ASSERT_EQ(empty_output.markers.size(), 1U);
  EXPECT_EQ(empty_output.markers.front().action, Marker::DELETEALL);

  const auto clear = build_delete_all(empty.header, MarkerBuilderConfig{});
  ASSERT_EQ(clear.markers.size(), 1U);
  EXPECT_EQ(clear.markers.front().action, Marker::DELETEALL);
}

TEST(MarkerBuilder, RejectsMalformedArraysAtomically)
{
  const auto expect_invalid = [](const PredictedObjectArray & input) {
      EXPECT_THROW(
        (void)build_prediction_markers(input, MarkerBuilderConfig{}),
        std::invalid_argument);
    };

  auto input = array_with_one_object();
  input.header.frame_id.clear();
  expect_invalid(input);
  input = array_with_one_object();
  input.header.stamp.sec = 0;
  input.header.stamp.nanosec = 0U;
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].object_id.uuid.fill(0U);
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].existence_probability =
    std::numeric_limits<float>::quiet_NaN();
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].classification_probability = 1.1F;
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].dimensions.x = 0.0;
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].initial_pose.pose.position.y =
    std::numeric_limits<double>::infinity();
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].initial_pose.pose.orientation.w = 2.0;
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].initial_pose.covariance[0] = -1.0;
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].initial_twist.twist.linear.x =
    std::numeric_limits<double>::quiet_NaN();
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].initial_twist.twist.linear.x =
    std::numeric_limits<double>::max();
  input.objects[0].initial_twist.twist.linear.y =
    std::numeric_limits<double>::max();
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].initial_pose.pose.position.z =
    std::numeric_limits<double>::max();
  input.objects[0].dimensions.z = std::numeric_limits<double>::max();
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].states.resize(1U);
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].states[0].time_from_start.nanosec = 400000000U;
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].states[2].time_from_start =
    input.objects[0].states[1].time_from_start;
  expect_invalid(input);
  input = array_with_one_object();
  input.objects[0].states[1].pose.covariance[7] =
    std::numeric_limits<double>::quiet_NaN();
  expect_invalid(input);

  input = array_with_one_object();
  input.objects.push_back(object());
  input.objects[1].dimensions.y = -1.0;
  EXPECT_THROW(
    (void)build_prediction_markers(input, MarkerBuilderConfig{}),
    std::invalid_argument);
}

TEST(MarkerBuilder, RejectsInvalidBuilderConfiguration)
{
  auto config = MarkerBuilderConfig{};
  config.marker_lifetime_sec = 0.10;
  EXPECT_THROW(
    (void)build_prediction_markers(array_with_one_object(), config),
    std::invalid_argument);
  config.marker_lifetime_sec = 1.0;
  EXPECT_THROW(
    (void)build_prediction_markers(array_with_one_object(), config),
    std::invalid_argument);
  config.marker_lifetime_sec =
    static_cast<double>(std::numeric_limits<std::int32_t>::max()) + 1.0;
  config.stale_timeout_sec = config.marker_lifetime_sec + 1.0;
  EXPECT_THROW(
    (void)build_prediction_markers(array_with_one_object(), config),
    std::invalid_argument);
}

TEST(MarkerPublicationState, AdvancesOnlyAfterSuccessfulPublication)
{
  using Clock = std::chrono::steady_clock;
  MarkerPublicationState state(std::chrono::milliseconds(1000));
  const auto receipt = Clock::time_point{} + std::chrono::milliseconds(100);

  EXPECT_TRUE(state.accepts(100));
  EXPECT_TRUE(state.accepts(100));
  state.record_successful_publication(100, receipt);
  EXPECT_FALSE(state.accepts(100));
  EXPECT_FALSE(state.accepts(99));
  EXPECT_TRUE(state.accepts(101));
}

TEST(MarkerPublicationState, ResetsOnlyForLargeClockRollback)
{
  using namespace std::chrono_literals;
  using Clock = std::chrono::steady_clock;
  MarkerPublicationState state(1000ms);
  const auto receipt = Clock::time_point{} + 100ms;
  state.record_successful_publication(100, receipt);

  EXPECT_FALSE(state.clock_rollback_reset_due(80));
  state.record_clock_rollback();
  EXPECT_FALSE(state.clock_rollback_reset_due(100));
  EXPECT_TRUE(state.clock_rollback_reset_due(80));
  state.reset_for_clock_rollback();
  EXPECT_TRUE(state.accepts(80));
  state.record_successful_publication(80, receipt + 1ms);
  EXPECT_FALSE(state.clock_rollback_reset_due(75));
  EXPECT_FALSE(state.accepts(80));
  EXPECT_FALSE(state.accepts(75));
  EXPECT_TRUE(state.accepts(81));
}

TEST(MarkerPublicationState, PublishesOneStaleClearAndRearmsAfterFreshData)
{
  using namespace std::chrono_literals;
  using Clock = std::chrono::steady_clock;
  MarkerPublicationState state(1000ms);
  const auto receipt = Clock::time_point{} + 100ms;
  state.record_successful_publication(100, receipt);

  EXPECT_FALSE(state.stale_clear_due(receipt + 1000ms));
  EXPECT_TRUE(state.stale_clear_due(receipt + 1001ms));
  state.record_stale_clear_publication();
  EXPECT_FALSE(state.stale_clear_due(receipt + 2000ms));

  state.record_successful_publication(101, receipt + 2100ms);
  EXPECT_TRUE(state.stale_clear_due(receipt + 3101ms));
}

TEST(MarkerNodeContract, UsesKeepLastOneReliableVolatileQos)
{
  const auto input = predicted_object_input_qos().get_rmw_qos_profile();
  const auto output = perception_marker_output_qos().get_rmw_qos_profile();

  for (const auto & qos : {input, output}) {
    EXPECT_EQ(qos.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
    EXPECT_EQ(qos.depth, 1U);
    EXPECT_EQ(qos.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    EXPECT_EQ(qos.durability, RMW_QOS_POLICY_DURABILITY_VOLATILE);
  }
}

TEST(MarkerNodeContract, ValidatesStrictlyPositiveRepresentableStamps)
{
  const auto input = array_with_one_object();
  EXPECT_EQ(prediction_stamp_ns(input.header), 12000000345LL);

  auto invalid = input.header;
  invalid.stamp.sec = -1;
  EXPECT_THROW((void)prediction_stamp_ns(invalid), std::invalid_argument);
  invalid = input.header;
  invalid.stamp.sec = 0;
  invalid.stamp.nanosec = 0U;
  EXPECT_THROW((void)prediction_stamp_ns(invalid), std::invalid_argument);
  invalid = input.header;
  invalid.stamp.nanosec = 1000000000U;
  EXPECT_THROW((void)prediction_stamp_ns(invalid), std::invalid_argument);
}

}  // namespace
