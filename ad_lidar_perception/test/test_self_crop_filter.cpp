#include "ad_lidar_perception/preprocessing/self_crop_filter.hpp"

#include <gtest/gtest.h>

#include <sensor_msgs/msg/point_field.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using ad_lidar_perception::preprocessing::RigidTransform3;
using ad_lidar_perception::preprocessing::SelfCropBounds;
using ad_lidar_perception::preprocessing::crop_self_points;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

constexpr std::uint32_t kPointStep = 22U;

struct InputPoint
{
  float x;
  float y;
  float z;
  float intensity;
  std::uint16_t ring;
  float time;
};

PointField field(
  std::string name, const std::uint32_t offset, const std::uint8_t datatype)
{
  PointField result;
  result.name = std::move(name);
  result.offset = offset;
  result.datatype = datatype;
  result.count = 1U;
  return result;
}

std::vector<PointField> strict_fields()
{
  return {
    field("x", 0U, PointField::FLOAT32),
    field("y", 4U, PointField::FLOAT32),
    field("z", 8U, PointField::FLOAT32),
    field("intensity", 12U, PointField::FLOAT32),
    field("ring", 16U, PointField::UINT16),
    field("time", 18U, PointField::FLOAT32),
  };
}

template<typename T>
void write_value(std::vector<std::uint8_t> & data, const std::size_t offset, const T value)
{
  ASSERT_LE(offset + sizeof(T), data.size());
  std::memcpy(data.data() + offset, &value, sizeof(T));
}

template<typename T>
T read_value(const std::vector<std::uint8_t> & data, const std::size_t offset)
{
  T value{};
  EXPECT_LE(offset + sizeof(T), data.size());
  std::memcpy(&value, data.data() + offset, sizeof(T));
  return value;
}

PointCloud2 make_cloud(
  const std::uint32_t height, const std::uint32_t width,
  const std::uint32_t row_padding, const std::vector<InputPoint> & points)
{
  EXPECT_EQ(points.size(), static_cast<std::size_t>(height) * width);
  PointCloud2 cloud;
  cloud.header.stamp.sec = 42;
  cloud.header.stamp.nanosec = 123456789U;
  cloud.header.frame_id = "lidar_link";
  cloud.height = height;
  cloud.width = width;
  cloud.fields = strict_fields();
  cloud.is_bigendian = false;
  cloud.point_step = kPointStep;
  cloud.row_step = width * kPointStep + row_padding;
  cloud.data.assign(static_cast<std::size_t>(cloud.row_step) * height, 0xE7U);
  cloud.is_dense = false;
  for (std::uint32_t row = 0U; row < height; ++row) {
    for (std::uint32_t column = 0U; column < width; ++column) {
      const auto index = static_cast<std::size_t>(row) * width + column;
      const auto offset = static_cast<std::size_t>(row) * cloud.row_step +
        static_cast<std::size_t>(column) * cloud.point_step;
      const auto & point = points[index];
      write_value(cloud.data, offset + 0U, point.x);
      write_value(cloud.data, offset + 4U, point.y);
      write_value(cloud.data, offset + 8U, point.z);
      write_value(cloud.data, offset + 12U, point.intensity);
      write_value(cloud.data, offset + 16U, point.ring);
      write_value(cloud.data, offset + 18U, point.time);
    }
  }
  return cloud;
}

PointCloud2 make_cloud(const std::vector<InputPoint> & points)
{
  return make_cloud(1U, static_cast<std::uint32_t>(points.size()), 0U, points);
}

void expect_invalid(
  const PointCloud2 & cloud, const SelfCropBounds & bounds,
  const std::optional<RigidTransform3> & transform, const std::string & message)
{
  try {
    static_cast<void>(crop_self_points(cloud, bounds, transform));
    FAIL() << "expected std::invalid_argument containing " << message;
  } catch (const std::invalid_argument & error) {
    EXPECT_NE(std::string(error.what()).find(message), std::string::npos)
      << "actual error: " << error.what();
  } catch (const std::exception & error) {
    FAIL() << "wrong exception class: " << error.what();
  }
}

TEST(SelfCropFilter, RemovesEveryInclusiveBoundFaceAndDefinesEmptyOutput)
{
  const SelfCropBounds bounds{-1.0, 4.0, -1.0, 1.0, -0.25, 2.0};
  const std::vector<InputPoint> points{
    {-1.0F, 0.0F, 0.0F, 1.0F, 1U, 0.01F},
    {4.0F, 0.0F, 0.0F, 2.0F, 2U, 0.02F},
    {0.0F, -1.0F, 0.0F, 3.0F, 3U, 0.03F},
    {0.0F, 1.0F, 0.0F, 4.0F, 4U, 0.04F},
    {0.0F, 0.0F, -0.25F, 5.0F, 5U, 0.05F},
    {0.0F, 0.0F, 2.0F, 6.0F, 6U, 0.06F},
  };

  const auto result = crop_self_points(
    make_cloud(points), bounds, RigidTransform3{});

  EXPECT_EQ(result.input_points, 6U);
  EXPECT_EQ(result.removed_points, 6U);
  EXPECT_EQ(result.nonfinite_points, 0U);
  EXPECT_EQ(result.cloud.header.frame_id, "lidar_link");
  EXPECT_EQ(result.cloud.height, 1U);
  EXPECT_EQ(result.cloud.width, 0U);
  EXPECT_EQ(result.cloud.point_step, kPointStep);
  EXPECT_EQ(result.cloud.row_step, 0U);
  EXPECT_TRUE(result.cloud.data.empty());
  EXPECT_TRUE(result.cloud.is_dense);
}

TEST(SelfCropFilter, UsesTranslatedAndRotatedBaseTransformOnlyForClassification)
{
  const std::vector<InputPoint> points{
    {1.0F, 0.0F, 0.0F, 10.0F, 7U, 0.11F},
    {0.0F, 2.0F, 0.0F, 20.0F, 8U, 0.22F},
  };
  SelfCropBounds bounds{1.9, 2.1, 0.9, 1.1, -0.1, 0.1};
  RigidTransform3 transform;
  transform.translation_x_m = 2.0;
  transform.quaternion_z = std::sqrt(2.0);
  transform.quaternion_w = std::sqrt(2.0);

  const auto input = make_cloud(points);
  const auto result = crop_self_points(input, bounds, transform);

  ASSERT_EQ(result.cloud.width, 1U);
  EXPECT_EQ(result.removed_points, 1U);
  EXPECT_FLOAT_EQ(read_value<float>(result.cloud.data, 0U), 0.0F);
  EXPECT_FLOAT_EQ(read_value<float>(result.cloud.data, 4U), 2.0F);
  EXPECT_FLOAT_EQ(read_value<float>(result.cloud.data, 8U), 0.0F);
  EXPECT_EQ(result.cloud.header.frame_id, "lidar_link");
  EXPECT_EQ(result.cloud.header.stamp, input.header.stamp);
}

TEST(SelfCropFilter, CopiesEverySurvivingRecordByteIncludingNonXyzFields)
{
  const std::vector<InputPoint> points{
    {0.0F, 0.0F, 0.0F, 1.25F, 11U, 0.125F},
    {9.0F, -8.0F, 7.0F, -123.5F, 65530U, 0.875F},
  };
  const auto input = make_cloud(points);
  const auto result = crop_self_points(input, SelfCropBounds{}, RigidTransform3{});

  ASSERT_EQ(result.cloud.width, 1U);
  ASSERT_EQ(result.cloud.data.size(), kPointStep);
  EXPECT_EQ(
    std::memcmp(result.cloud.data.data(), input.data.data() + kPointStep, kPointStep),
    0);
  EXPECT_EQ(result.cloud.fields, input.fields);
  EXPECT_EQ(result.cloud.is_bigendian, input.is_bigendian);
  EXPECT_TRUE(result.cloud.is_dense);
  EXPECT_EQ(result.cloud.header, input.header);
}

TEST(SelfCropFilter, CompactsOrganizedRowsWithoutCopyingPaddingAndKeepsOrder)
{
  const std::vector<InputPoint> points{
    {10.0F, 0.0F, 0.0F, 1.0F, 10U, 0.01F},
    {0.0F, 0.0F, 0.0F, 2.0F, 11U, 0.02F},
    {20.0F, 0.0F, 0.0F, 3.0F, 12U, 0.03F},
    {30.0F, 0.0F, 0.0F, 4.0F, 13U, 0.04F},
  };
  const auto input = make_cloud(2U, 2U, 9U, points);

  const auto result = crop_self_points(input, SelfCropBounds{}, RigidTransform3{});

  EXPECT_EQ(result.cloud.height, 1U);
  EXPECT_EQ(result.cloud.width, 3U);
  EXPECT_EQ(result.cloud.row_step, 3U * kPointStep);
  ASSERT_EQ(result.cloud.data.size(), 3U * kPointStep);
  const std::array<std::size_t, 3> source_offsets{
    0U, static_cast<std::size_t>(input.row_step),
    static_cast<std::size_t>(input.row_step) + kPointStep};
  for (std::size_t index = 0U; index < source_offsets.size(); ++index) {
    EXPECT_EQ(
      std::memcmp(
        result.cloud.data.data() + index * kPointStep,
        input.data.data() + source_offsets[index], kPointStep),
      0);
  }
}

TEST(SelfCropFilter, DiscardsOrganizedInvalidReturnsAndKeepsFiniteSurvivorsExact)
{
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  const auto infinity = std::numeric_limits<float>::infinity();
  const std::vector<InputPoint> points{
    {10.0F, 2.0F, 3.0F, 1.0F, 10U, 0.01F},
    {nan, nan, nan, 2.0F, 11U, 0.02F},
    {0.0F, 0.0F, 0.0F, 3.0F, 12U, 0.03F},
    {infinity, 4.0F, 5.0F, 4.0F, 13U, 0.04F},
    {20.0F, -2.0F, 6.0F, 5.0F, 14U, 0.05F},
    {1.0F, 0.0F, 0.0F, 6.0F, 15U, 0.06F},
  };
  const auto input = make_cloud(2U, 3U, 7U, points);

  const auto result = crop_self_points(input, SelfCropBounds{}, RigidTransform3{});

  EXPECT_EQ(result.input_points, 6U);
  EXPECT_EQ(result.removed_points, 2U);
  EXPECT_EQ(result.nonfinite_points, 2U);
  EXPECT_EQ(result.cloud.header, input.header);
  EXPECT_EQ(result.cloud.fields, input.fields);
  EXPECT_EQ(result.cloud.is_bigendian, input.is_bigendian);
  EXPECT_EQ(result.cloud.height, 1U);
  EXPECT_EQ(result.cloud.width, 2U);
  EXPECT_EQ(result.cloud.point_step, kPointStep);
  EXPECT_EQ(result.cloud.row_step, 2U * kPointStep);
  EXPECT_TRUE(result.cloud.is_dense);
  ASSERT_EQ(result.cloud.data.size(), 2U * kPointStep);
  const std::array<std::size_t, 2> survivor_offsets{
    0U,
    static_cast<std::size_t>(input.row_step) + kPointStep,
  };
  for (std::size_t index = 0U; index < survivor_offsets.size(); ++index) {
    EXPECT_EQ(
      std::memcmp(
        result.cloud.data.data() + index * kPointStep,
        input.data.data() + survivor_offsets[index], kPointStep),
      0);
  }
}

TEST(SelfCropFilter, FailsClosedForMissingTransformAndMalformedLayout)
{
  const auto valid = make_cloud(
    std::vector<InputPoint>{{10.0F, 0.0F, 0.0F, 1.0F, 1U, 0.0F}});
  expect_invalid(valid, SelfCropBounds{}, std::nullopt, "transform");

  auto malformed = valid;
  malformed.point_step = 21U;
  expect_invalid(malformed, SelfCropBounds{}, RigidTransform3{}, "point_step");
}

TEST(SelfCropFilter, RejectsNonfiniteOrReversedBoundsAndInvalidTransform)
{
  const auto cloud = make_cloud(
    std::vector<InputPoint>{{10.0F, 0.0F, 0.0F, 1.0F, 1U, 0.0F}});
  auto bounds = SelfCropBounds{};
  bounds.min_x_m = 2.0;
  bounds.max_x_m = 1.0;
  expect_invalid(cloud, bounds, RigidTransform3{}, "x bounds");

  bounds = SelfCropBounds{};
  bounds.max_y_m = std::numeric_limits<double>::infinity();
  expect_invalid(cloud, bounds, RigidTransform3{}, "bounds must be finite");

  auto transform = RigidTransform3{};
  transform.translation_z_m = std::numeric_limits<double>::quiet_NaN();
  expect_invalid(cloud, SelfCropBounds{}, transform, "translation");

  transform = RigidTransform3{};
  transform.quaternion_w = 0.0;
  expect_invalid(cloud, SelfCropBounds{}, transform, "quaternion");
}

}  // namespace
