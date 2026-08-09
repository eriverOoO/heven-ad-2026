#include "ad_lidar_perception/preprocessing/pointcloud_densifier.hpp"

#include <gtest/gtest.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace
{

using ad_lidar_perception::preprocessing::DensifierStatus;
using ad_lidar_perception::preprocessing::DensifierConfig;
using ad_lidar_perception::preprocessing::DensifierTransform;
using ad_lidar_perception::preprocessing::PointcloudDensifier;

struct TestPoint
{
  float x;
  float y;
  float z;
  std::uint32_t label;
};

sensor_msgs::msg::PointCloud2 make_cloud(
  const std::vector<TestPoint> & points = {{25.0F, 1.0F, 0.5F, 0x12345678U}},
  const std::int32_t stamp_sec = 10, const std::uint32_t stamp_nanosec = 123U,
  const std::string & frame = "lidar_link", const std::uint32_t height = 1U,
  const std::uint32_t row_padding = 3U)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp.sec = stamp_sec;
  cloud.header.stamp.nanosec = stamp_nanosec;
  cloud.header.frame_id = frame;
  cloud.height = height;
  cloud.width = height == 0U ? 0U : static_cast<std::uint32_t>(points.size()) / height;
  cloud.fields = {
    sensor_msgs::msg::PointField().set__name("x").set__offset(0U).set__datatype(
      sensor_msgs::msg::PointField::FLOAT32).set__count(1U),
    sensor_msgs::msg::PointField().set__name("y").set__offset(4U).set__datatype(
      sensor_msgs::msg::PointField::FLOAT32).set__count(1U),
    sensor_msgs::msg::PointField().set__name("z").set__offset(8U).set__datatype(
      sensor_msgs::msg::PointField::FLOAT32).set__count(1U),
    sensor_msgs::msg::PointField().set__name("label").set__offset(12U).set__datatype(
      sensor_msgs::msg::PointField::UINT32).set__count(1U),
  };
  cloud.is_bigendian = false;
  cloud.point_step = 16U;
  cloud.row_step = cloud.width * cloud.point_step + row_padding;
  cloud.data.assign(static_cast<std::size_t>(cloud.row_step) * height, 0xA5U);
  for (std::size_t index = 0U; index < points.size(); ++index) {
    const auto row = index / cloud.width;
    const auto column = index % cloud.width;
    const auto offset = row * cloud.row_step + column * cloud.point_step;
    std::memcpy(cloud.data.data() + offset, &points[index].x, sizeof(float));
    std::memcpy(cloud.data.data() + offset + 4U, &points[index].y, sizeof(float));
    std::memcpy(cloud.data.data() + offset + 8U, &points[index].z, sizeof(float));
    std::memcpy(
      cloud.data.data() + offset + 12U, &points[index].label,
      sizeof(std::uint32_t));
  }
  cloud.is_dense = true;
  return cloud;
}

TestPoint point_at(
  const sensor_msgs::msg::PointCloud2 & cloud, const std::size_t index)
{
  const auto row = index / cloud.width;
  const auto column = index % cloud.width;
  const auto offset = row * cloud.row_step + column * cloud.point_step;
  TestPoint point{};
  std::memcpy(&point.x, cloud.data.data() + offset, sizeof(float));
  std::memcpy(&point.y, cloud.data.data() + offset + 4U, sizeof(float));
  std::memcpy(&point.z, cloud.data.data() + offset + 8U, sizeof(float));
  std::memcpy(
    &point.label, cloud.data.data() + offset + 12U,
    sizeof(std::uint32_t));
  return point;
}

DensifierTransform identity_transform()
{
  return {};
}

void expect_current_only(
  PointcloudDensifier & densifier,
  const sensor_msgs::msg::PointCloud2 & current,
  const std::optional<DensifierTransform> & transform,
  const DensifierStatus status)
{
  const auto result = densifier.process(current, transform);
  EXPECT_EQ(result.status, status);
  EXPECT_EQ(result.historical_points_added, 0U);
  EXPECT_EQ(result.cloud, current);
}

void expect_malformed_schema_clears_history(
  const sensor_msgs::msg::PointCloud2 & malformed)
{
  PointcloudDensifier densifier;
  densifier.process(
    make_cloud({{30.0F, 0.0F, 0.0F, 1U}}, 10, 0U), std::nullopt);

  const auto result = densifier.process(malformed, identity_transform());

  EXPECT_EQ(result.status, DensifierStatus::kMalformedCurrent);
  EXPECT_EQ(result.historical_points_added, 0U);
  EXPECT_EQ(result.cloud, malformed);
  EXPECT_FALSE(densifier.has_history());
}

TEST(PointcloudDensifier, FirstFrameReturnsExactCurrentMessageAndStoresIt)
{
  PointcloudDensifier densifier;
  const auto current = make_cloud();

  const auto result = densifier.process(current, std::nullopt);

  EXPECT_EQ(result.status, DensifierStatus::kFirstFrame);
  EXPECT_EQ(result.cloud, current);
  EXPECT_TRUE(densifier.has_history());
}

TEST(PointcloudDensifier, AppliesPreviousToCurrentOdomTransformWithCorrectSign)
{
  PointcloudDensifier densifier;
  densifier.process(
    make_cloud({{30.0F, 0.0F, 0.0F, 10U}}, 10, 0U), std::nullopt);
  auto transform = identity_transform();
  transform.translation[0] = 1.0;
  const auto current = make_cloud({{40.0F, 0.0F, 0.0F, 20U}}, 10, 100000000U);

  const auto result = densifier.process(current, transform);

  ASSERT_EQ(result.status, DensifierStatus::kFused);
  ASSERT_EQ(result.historical_points_added, 1U);
  ASSERT_EQ(result.cloud.width, 2U);
  EXPECT_FLOAT_EQ(point_at(result.cloud, 0U).x, 40.0F);
  EXPECT_FLOAT_EQ(point_at(result.cloud, 1U).x, 31.0F);
}

TEST(PointcloudDensifier, StoresRawCurrentInputAndNeverRecursivelyAccumulates)
{
  PointcloudDensifier densifier;
  densifier.process(make_cloud({{30.0F, 0.0F, 0.0F, 1U}}, 10, 0U), std::nullopt);
  densifier.process(
    make_cloud({{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U),
    identity_transform());

  const auto result = densifier.process(
    make_cloud({{50.0F, 0.0F, 0.0F, 3U}}, 10, 200000000U),
    identity_transform());

  ASSERT_EQ(result.cloud.width, 2U);
  EXPECT_EQ(point_at(result.cloud, 0U).label, 3U);
  EXPECT_EQ(point_at(result.cloud, 1U).label, 2U);
}

TEST(PointcloudDensifier, AppliesRoiAfterTransformingHistoryIntoCurrentFrame)
{
  PointcloudDensifier densifier;
  densifier.process(
    make_cloud({{19.5F, 0.0F, 0.0F, 1U}}, 10, 0U), std::nullopt);
  auto transform = identity_transform();
  transform.translation[0] = 1.0;

  const auto result = densifier.process(
    make_cloud({{60.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U), transform);

  ASSERT_EQ(result.cloud.width, 2U);
  EXPECT_FLOAT_EQ(point_at(result.cloud, 1U).x, 20.5F);
}

TEST(PointcloudDensifier, CurrentRecordsKeepOrderAndBytesAndClaimVoxelsFirst)
{
  PointcloudDensifier densifier;
  densifier.process(
    make_cloud(
      {{25.05F, 0.0F, 0.0F, 11U}, {26.0F, 0.0F, 0.0F, 12U}}, 10, 0U,
      "lidar_link", 1U, 0U),
    std::nullopt);
  const auto current = make_cloud(
    {{25.01F, 0.0F, 0.0F, 21U}, {25.02F, 0.0F, 0.0F, 22U}}, 10,
    100000000U, "lidar_link", 1U, 5U);

  const auto result = densifier.process(current, identity_transform());

  ASSERT_EQ(result.cloud.width, 3U);
  EXPECT_EQ(point_at(result.cloud, 0U).label, 21U);
  EXPECT_EQ(point_at(result.cloud, 1U).label, 22U);
  EXPECT_EQ(point_at(result.cloud, 2U).label, 12U);
  EXPECT_EQ(
    std::vector<std::uint8_t>(
      result.cloud.data.begin(), result.cloud.data.begin() + 32),
    std::vector<std::uint8_t>(current.data.begin(), current.data.begin() + 32));
}

TEST(PointcloudDensifier, HistoricalNonXyzBytesRemainExact)
{
  PointcloudDensifier densifier;
  const auto previous = make_cloud(
    {{30.0F, 1.0F, 2.0F, 0xDEADBEEFU}}, 10, 0U, "lidar_link", 1U, 0U);
  densifier.process(previous, std::nullopt);
  auto transform = identity_transform();
  transform.translation = {1.0, 2.0, 3.0};

  const auto result = densifier.process(
    make_cloud({{60.0F, 0.0F, 0.0F, 7U}}, 10, 100000000U), transform);

  ASSERT_EQ(result.cloud.width, 2U);
  const auto historical = point_at(result.cloud, 1U);
  EXPECT_FLOAT_EQ(historical.x, 31.0F);
  EXPECT_FLOAT_EQ(historical.y, 3.0F);
  EXPECT_FLOAT_EQ(historical.z, 5.0F);
  EXPECT_EQ(historical.label, 0xDEADBEEFU);
}

TEST(PointcloudDensifier, FusionCompactsOrganizedInputWithoutChangingRecords)
{
  PointcloudDensifier densifier;
  densifier.process(
    make_cloud({{30.0F, 0.0F, 0.0F, 9U}}, 10, 0U, "lidar_link", 1U, 7U),
    std::nullopt);
  const auto current = make_cloud(
    {{40.0F, 0.0F, 0.0F, 1U}, {41.0F, 0.0F, 0.0F, 2U}}, 10,
    100000000U, "lidar_link", 2U, 5U);

  const auto result = densifier.process(current, identity_transform());

  ASSERT_EQ(result.status, DensifierStatus::kFused);
  EXPECT_EQ(result.cloud.height, 1U);
  EXPECT_EQ(result.cloud.width, 3U);
  EXPECT_EQ(result.cloud.row_step, 3U * current.point_step);
  EXPECT_EQ(result.cloud.data.size(), result.cloud.row_step);
  EXPECT_EQ(point_at(result.cloud, 0U).label, 1U);
  EXPECT_EQ(point_at(result.cloud, 1U).label, 2U);
  EXPECT_EQ(point_at(result.cloud, 2U).label, 9U);
}

TEST(PointcloudDensifier, ValidCycleWithNoEligibleHistoryPreservesOrganizedCurrentExactly)
{
  PointcloudDensifier densifier;
  densifier.process(
    make_cloud({{10.0F, 0.0F, 0.0F, 9U}}, 10, 0U), std::nullopt);
  const auto current = make_cloud(
    {{40.0F, 0.0F, 0.0F, 1U}, {41.0F, 0.0F, 0.0F, 2U}}, 10,
    100000000U, "lidar_link", 2U, 5U);

  const auto result = densifier.process(current, identity_transform());

  EXPECT_EQ(result.status, DensifierStatus::kNoEligibleHistory);
  EXPECT_EQ(result.cloud, current);
}

TEST(PointcloudDensifier, RotationTransformsHistoryIntoCurrentFrame)
{
  PointcloudDensifier densifier;
  densifier.process(
    make_cloud({{30.0F, 1.0F, 0.0F, 9U}}, 10, 0U), std::nullopt);
  auto transform = identity_transform();
  constexpr double kHalfYaw = 0.05;
  transform.quaternion_xyzw = {0.0, 0.0, std::sin(kHalfYaw), std::cos(kHalfYaw)};

  const auto result = densifier.process(
    make_cloud({{60.0F, 0.0F, 0.0F, 1U}}, 10, 100000000U), transform);

  ASSERT_EQ(result.status, DensifierStatus::kFused);
  EXPECT_NEAR(point_at(result.cloud, 1U).x, 29.75029F, 1.0e-4F);
  EXPECT_NEAR(point_at(result.cloud, 1U).y, 3.99001F, 1.0e-4F);
}

TEST(PointcloudDensifier, ReportsTypedFallbacksAndReplacesHistory)
{
  const std::vector<std::tuple<DensifierStatus, sensor_msgs::msg::PointCloud2,
    std::optional<DensifierTransform>>> cases = [&]() {
      std::vector<std::tuple<DensifierStatus, sensor_msgs::msg::PointCloud2,
        std::optional<DensifierTransform>>> values;
      values.emplace_back(
        DensifierStatus::kNonIncreasingStamp,
        make_cloud({{40.0F, 0.0F, 0.0F, 2U}}, 10, 0U), identity_transform());
      values.emplace_back(
        DensifierStatus::kStaleHistory,
        make_cloud({{40.0F, 0.0F, 0.0F, 2U}}, 11, 0U), identity_transform());
      auto schema = make_cloud({{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U);
      schema.fields[3].name = "class_id";
      values.emplace_back(
        DensifierStatus::kSchemaMismatch, schema, identity_transform());
      values.emplace_back(
        DensifierStatus::kFrameMismatch,
        make_cloud({{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U, "other"),
        identity_transform());
      values.emplace_back(
        DensifierStatus::kTransformUnavailable,
        make_cloud({{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U), std::nullopt);
      auto translation_jump = identity_transform();
      translation_jump.translation[0] = 5.01;
      values.emplace_back(
        DensifierStatus::kTransformTranslationJump,
        make_cloud({{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U),
        translation_jump);
      auto rotation_jump = identity_transform();
      rotation_jump.quaternion_xyzw = {0.0, 0.0, std::sin(0.18), std::cos(0.18)};
      values.emplace_back(
        DensifierStatus::kTransformRotationJump,
        make_cloud({{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U),
        rotation_jump);
      return values;
    }();

  for (const auto & [status, current, transform] : cases) {
    PointcloudDensifier densifier;
    densifier.process(
      make_cloud({{30.0F, 0.0F, 0.0F, 1U}}, 10, 0U), std::nullopt);
    expect_current_only(densifier, current, transform, status);

    auto next = current;
    next.header.stamp.nanosec += 100000000U;
    const float next_x = 50.0F;
    const std::uint32_t next_label = 3U;
    std::memcpy(next.data.data(), &next_x, sizeof(next_x));
    std::memcpy(next.data.data() + 12U, &next_label, sizeof(next_label));
    const auto reset_result = densifier.process(next, identity_transform());
    ASSERT_EQ(reset_result.status, DensifierStatus::kFused);
    ASSERT_EQ(reset_result.cloud.width, 2U);
    EXPECT_EQ(point_at(reset_result.cloud, 0U).label, 3U);
    EXPECT_EQ(point_at(reset_result.cloud, 1U).label, 2U);
  }
}

TEST(PointcloudDensifier, MalformedCurrentClearsHistoryAndNextValidFrameRestarts)
{
  PointcloudDensifier densifier;
  densifier.process(make_cloud({}, 10, 0U), std::nullopt);
  auto malformed = make_cloud({{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U);
  malformed.row_step -= 1U;

  expect_current_only(
    densifier, malformed, identity_transform(),
    DensifierStatus::kMalformedCurrent);
  EXPECT_FALSE(densifier.has_history());

  const auto valid = make_cloud({{50.0F, 0.0F, 0.0F, 3U}}, 10, 200000000U);
  const auto restarted = densifier.process(valid, identity_transform());
  EXPECT_EQ(restarted.status, DensifierStatus::kFirstFrame);
  EXPECT_EQ(restarted.cloud, valid);
}

TEST(PointcloudDensifier, NonfiniteCurrentIsMalformedEvenOnFirstFrameAndIsNotStored)
{
  PointcloudDensifier densifier;
  auto malformed = make_cloud();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::memcpy(malformed.data.data(), &nan, sizeof(nan));

  expect_current_only(
    densifier, malformed, std::nullopt, DensifierStatus::kMalformedCurrent);
  EXPECT_FALSE(densifier.has_history());

  const auto valid = make_cloud({{40.0F, 0.0F, 0.0F, 3U}}, 10, 100000000U);
  EXPECT_EQ(
    densifier.process(valid, identity_transform()).status,
    DensifierStatus::kFirstFrame);
}

TEST(PointcloudDensifier, InvalidTransformAndUnsafeVoxelIndexAreNumericalFallbacks)
{
  PointcloudDensifier densifier;
  densifier.process(
    make_cloud({{30.0F, 0.0F, 0.0F, 1U}}, 10, 0U), std::nullopt);
  auto invalid = identity_transform();
  invalid.quaternion_xyzw[0] = std::numeric_limits<double>::quiet_NaN();
  expect_current_only(
    densifier,
    make_cloud({{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U), invalid,
    DensifierStatus::kNumericalFailure);

  PointcloudDensifier overflow_densifier;
  overflow_densifier.process(
    make_cloud({{30.0F, 0.0F, 0.0F, 1U}}, 10, 0U), std::nullopt);
  expect_current_only(
    overflow_densifier,
    make_cloud(
      {{std::numeric_limits<float>::max(), 0.0F, 0.0F, 2U}}, 10,
      100000000U),
    identity_transform(), DensifierStatus::kNumericalFailure);
}

TEST(PointcloudDensifier, RejectsDuplicateOrNonFloatXyzAndBigEndianClouds)
{
  PointcloudDensifier densifier;
  auto duplicate = make_cloud();
  duplicate.fields.push_back(duplicate.fields[0]);
  expect_current_only(
    densifier, duplicate, std::nullopt, DensifierStatus::kMalformedCurrent);

  auto wrong_type = make_cloud({}, 11, 0U);
  wrong_type.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT64;
  expect_current_only(
    densifier, wrong_type, identity_transform(),
    DensifierStatus::kMalformedCurrent);

  auto big_endian = make_cloud({}, 12, 0U);
  big_endian.is_bigendian = true;
  expect_current_only(
    densifier, big_endian, identity_transform(),
    DensifierStatus::kMalformedCurrent);
}

TEST(PointcloudDensifier, RejectsOverlappingXyzByteRangesWithoutPoisoningHistory)
{
  auto malformed = make_cloud(
    {{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U);
  malformed.fields[1].offset = 2U;

  expect_malformed_schema_clears_history(malformed);
}

TEST(PointcloudDensifier, RejectsXyzAndNonXyzByteOverlapWithoutChangingCurrent)
{
  auto malformed = make_cloud(
    {{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U);
  malformed.fields[3].offset = 10U;

  expect_malformed_schema_clears_history(malformed);
}

TEST(PointcloudDensifier, AllowsOverlapBetweenDeclaredNonXyzAliasFields)
{
  PointcloudDensifier densifier;
  auto previous = make_cloud(
    {{30.0F, 0.0F, 0.0F, 0x12345678U}}, 10, 0U);
  previous.fields.push_back(
    sensor_msgs::msg::PointField().set__name("flags").set__offset(14U)
    .set__datatype(sensor_msgs::msg::PointField::UINT16).set__count(1U));
  auto current = make_cloud(
    {{40.0F, 0.0F, 0.0F, 0x90ABCDEFU}}, 10, 100000000U);
  current.fields.push_back(previous.fields.back());

  ASSERT_EQ(
    densifier.process(previous, std::nullopt).status,
    DensifierStatus::kFirstFrame);
  const auto result = densifier.process(current, identity_transform());

  ASSERT_EQ(result.status, DensifierStatus::kFused);
  ASSERT_EQ(result.historical_points_added, 1U);
  ASSERT_EQ(result.cloud.width, 2U);
  EXPECT_EQ(point_at(result.cloud, 0U).label, 0x90ABCDEFU);
  EXPECT_EQ(point_at(result.cloud, 1U).label, 0x12345678U);
}

TEST(PointcloudDensifier, RejectsInvalidDeclaredFieldDatatypeCountAndRange)
{
  std::vector<sensor_msgs::msg::PointCloud2> malformed_clouds;
  auto unknown_datatype = make_cloud(
    {{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U);
  unknown_datatype.fields[3].datatype = 0U;
  malformed_clouds.push_back(unknown_datatype);

  auto zero_count = make_cloud(
    {{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U);
  zero_count.fields[3].count = 0U;
  malformed_clouds.push_back(zero_count);

  auto past_point_step = make_cloud(
    {{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U);
  past_point_step.fields[3].offset = 15U;
  malformed_clouds.push_back(past_point_step);

  auto huge_array = make_cloud(
    {{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U);
  huge_array.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT64;
  huge_array.fields[3].count = std::numeric_limits<std::uint32_t>::max();
  huge_array.fields[3].offset = std::numeric_limits<std::uint32_t>::max();
  malformed_clouds.push_back(huge_array);

  auto duplicate_name = make_cloud(
    {{40.0F, 0.0F, 0.0F, 2U}}, 10, 100000000U);
  duplicate_name.point_step = 20U;
  duplicate_name.row_step = 23U;
  duplicate_name.data.resize(duplicate_name.row_step, 0xA5U);
  duplicate_name.fields.push_back(
    sensor_msgs::msg::PointField().set__name("label").set__offset(16U)
    .set__datatype(sensor_msgs::msg::PointField::UINT32).set__count(1U));
  malformed_clouds.push_back(duplicate_name);

  for (std::size_t index = 0U; index < malformed_clouds.size(); ++index) {
    SCOPED_TRACE(index);
    expect_malformed_schema_clears_history(malformed_clouds[index]);
  }
}

}  // namespace
