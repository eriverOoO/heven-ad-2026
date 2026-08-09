#include "ad_lidar_perception/preprocessing/point_layout_converter.hpp"

#include <gtest/gtest.h>

#include <sensor_msgs/msg/point_field.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using ad_lidar_perception::preprocessing::ConverterConfig;
using ad_lidar_perception::preprocessing::convert_morai_xyzirt_to_point_xyzirc;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

constexpr std::uint32_t kInputPointStep = 22U;
constexpr std::uint32_t kOutputPointStep = 16U;

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
  cloud.header.stamp.sec = 123;
  cloud.header.stamp.nanosec = 456789U;
  cloud.header.frame_id = "lidar_link";
  cloud.height = height;
  cloud.width = width;
  cloud.fields = strict_fields();
  cloud.is_bigendian = false;
  cloud.point_step = kInputPointStep;
  cloud.row_step = width * kInputPointStep + row_padding;
  cloud.data.assign(static_cast<std::size_t>(cloud.row_step) * height, 0xA5U);
  cloud.is_dense = true;
  for (std::uint32_t row = 0U; row < height; ++row) {
    for (std::uint32_t column = 0U; column < width; ++column) {
      const auto point_index = static_cast<std::size_t>(row) * width + column;
      const auto offset = static_cast<std::size_t>(row) * cloud.row_step +
        static_cast<std::size_t>(column) * cloud.point_step;
      const auto & point = points[point_index];
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
  const PointCloud2 & cloud, const ConverterConfig & config,
  const std::string & message)
{
  try {
    static_cast<void>(convert_morai_xyzirt_to_point_xyzirc(cloud, config));
    FAIL() << "expected std::invalid_argument containing " << message;
  } catch (const std::invalid_argument & error) {
    EXPECT_NE(std::string(error.what()).find(message), std::string::npos)
      << "actual error: " << error.what();
  } catch (const std::exception & error) {
    FAIL() << "wrong exception class: " << error.what();
  }
}

TEST(PointLayoutConverter, ConvertsEveryPointWithoutEgoCropOrTransform)
{
  const std::vector<InputPoint> points{
    {0.0F, 0.0F, 0.0F, 12.5F, 4U, 0.01F},
    {8.0F, -2.0F, 3.0F, 300.0F, 65535U, 0.02F},
  };
  ConverterConfig config;
  config.return_type = 9U;

  const auto result = convert_morai_xyzirt_to_point_xyzirc(
    make_cloud(points), config);

  EXPECT_EQ(result.stats.input_points, 2U);
  EXPECT_EQ(result.stats.output_points, 2U);
  EXPECT_EQ(result.cloud.header.frame_id, "lidar_link");
  EXPECT_EQ(result.cloud.width, 2U);
  EXPECT_EQ(result.cloud.height, 1U);
  EXPECT_EQ(result.cloud.point_step, kOutputPointStep);
  EXPECT_EQ(result.cloud.row_step, 2U * kOutputPointStep);
  ASSERT_EQ(result.cloud.data.size(), 2U * kOutputPointStep);
  EXPECT_FLOAT_EQ(read_value<float>(result.cloud.data, 0U), 0.0F);
  EXPECT_FLOAT_EQ(read_value<float>(result.cloud.data, 16U), 8.0F);
  EXPECT_EQ(result.cloud.data[12U], 13U);
  EXPECT_EQ(result.cloud.data[13U], 9U);
  EXPECT_EQ(read_value<std::uint16_t>(result.cloud.data, 14U), 4U);
  EXPECT_EQ(result.cloud.data[28U], 255U);
  EXPECT_EQ(read_value<std::uint16_t>(result.cloud.data, 30U), 65535U);
}

TEST(PointLayoutConverter, AppliesIntensityPolicyWithoutChangingXyz)
{
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  const std::vector<InputPoint> points{
    {-1.0F, 2.0F, 3.0F, -10.0F, 1U, 0.0F},
    {4.0F, 5.0F, 6.0F, nan, 2U, 0.0F},
  };
  ConverterConfig config;
  config.intensity_scale = 2.0;
  config.intensity_offset = 5.0;
  config.nonfinite_intensity = 77U;

  const auto result = convert_morai_xyzirt_to_point_xyzirc(
    make_cloud(points), config);

  EXPECT_FLOAT_EQ(read_value<float>(result.cloud.data, 0U), -1.0F);
  EXPECT_FLOAT_EQ(read_value<float>(result.cloud.data, 4U), 2.0F);
  EXPECT_FLOAT_EQ(read_value<float>(result.cloud.data, 8U), 3.0F);
  EXPECT_EQ(result.cloud.data[12U], 0U);
  EXPECT_EQ(result.cloud.data[28U], 77U);
  EXPECT_EQ(result.stats.clamped_low, 1U);
  EXPECT_EQ(result.stats.nonfinite_intensity, 1U);
}

TEST(PointLayoutConverter, CompactsOrganizedInputInRowMajorOrder)
{
  const std::vector<InputPoint> points{
    {1.0F, 0.0F, 0.0F, 1.0F, 10U, 0.0F},
    {2.0F, 0.0F, 0.0F, 2.0F, 11U, 0.0F},
    {3.0F, 0.0F, 0.0F, 3.0F, 12U, 0.0F},
    {4.0F, 0.0F, 0.0F, 4.0F, 13U, 0.0F},
  };

  const auto result = convert_morai_xyzirt_to_point_xyzirc(
    make_cloud(2U, 2U, 7U, points), ConverterConfig{});

  ASSERT_EQ(result.cloud.width, 4U);
  for (std::size_t index = 0U; index < points.size(); ++index) {
    EXPECT_FLOAT_EQ(
      read_value<float>(result.cloud.data, index * kOutputPointStep),
      points[index].x);
    EXPECT_EQ(
      read_value<std::uint16_t>(
        result.cloud.data, index * kOutputPointStep + 14U),
      points[index].ring);
  }
}

TEST(PointLayoutConverter, AcceptsWellFormedEmptyInput)
{
  const auto result = convert_morai_xyzirt_to_point_xyzirc(
    make_cloud(1U, 0U, 0U, {}), ConverterConfig{});
  EXPECT_EQ(result.cloud.header.frame_id, "lidar_link");
  EXPECT_EQ(result.cloud.height, 1U);
  EXPECT_EQ(result.cloud.width, 0U);
  EXPECT_EQ(result.cloud.row_step, 0U);
  EXPECT_TRUE(result.cloud.data.empty());
}

TEST(PointLayoutConverter, RejectsMalformedStrictLayoutAndNonfiniteCoordinates)
{
  const auto valid = make_cloud(
    std::vector<InputPoint>{{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.0F}});
  auto malformed = valid;
  malformed.fields.pop_back();
  expect_invalid(malformed, ConverterConfig{}, "exactly six fields");

  malformed = valid;
  malformed.point_step = 21U;
  expect_invalid(malformed, ConverterConfig{}, "point_step");

  malformed = valid;
  malformed.data.pop_back();
  expect_invalid(malformed, ConverterConfig{}, "data size");

  malformed = valid;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  write_value(malformed.data, 4U, nan);
  expect_invalid(malformed, ConverterConfig{}, "nonfinite coordinate");
}

TEST(PointLayoutConverter, RejectsNonfiniteIntensityConfiguration)
{
  const auto cloud = make_cloud(
    std::vector<InputPoint>{{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.0F}});
  auto config = ConverterConfig{};
  config.intensity_scale = std::numeric_limits<double>::infinity();
  expect_invalid(cloud, config, "intensity scale");
  config.intensity_scale = 1.0;
  config.intensity_offset = std::numeric_limits<double>::quiet_NaN();
  expect_invalid(cloud, config, "intensity offset");
}

TEST(PointLayoutConverter, EmitsExactXyzircLayoutBytesAndFullIntensityStatistics)
{
  const auto nan = std::numeric_limits<float>::quiet_NaN();
  const auto inf = std::numeric_limits<float>::infinity();
  const std::vector<InputPoint> points{
    {1.0F, 2.0F, 3.0F, -1.0F, 0U, 0.01F},
    {-4.0F, 5.0F, 6.0F, 0.0F, 15U, 0.02F},
    {7.0F, -8.0F, 9.0F, 12.4F, 65535U, 0.03F},
    {10.0F, 11.0F, -12.0F, 12.5F, 2U, 0.04F},
    {13.0F, 14.0F, 15.0F, 255.0F, 3U, 0.05F},
    {16.0F, 17.0F, 18.0F, 300.0F, 4U, 0.06F},
    {19.0F, 20.0F, 21.0F, nan, 5U, 0.07F},
    {22.0F, 23.0F, 24.0F, inf, 6U, 0.08F},
    {25.0F, 26.0F, 27.0F, -inf, 7U, 0.09F},
  };
  ConverterConfig config;
  config.nonfinite_intensity = 42U;
  config.return_type = 9U;

  const auto result = convert_morai_xyzirt_to_point_xyzirc(
    make_cloud(points), config);

  ASSERT_EQ(result.cloud.fields.size(), 6U);
  const std::array<std::string, 6> expected_names{
    "x", "y", "z", "intensity", "return_type", "channel"};
  const std::array<std::uint32_t, 6> expected_offsets{
    0U, 4U, 8U, 12U, 13U, 14U};
  const std::array<std::uint8_t, 6> expected_datatypes{
    PointField::FLOAT32, PointField::FLOAT32, PointField::FLOAT32,
    PointField::UINT8, PointField::UINT8, PointField::UINT16};
  for (std::size_t index = 0U; index < expected_names.size(); ++index) {
    EXPECT_EQ(result.cloud.fields[index].name, expected_names[index]);
    EXPECT_EQ(result.cloud.fields[index].offset, expected_offsets[index]);
    EXPECT_EQ(result.cloud.fields[index].datatype, expected_datatypes[index]);
    EXPECT_EQ(result.cloud.fields[index].count, 1U);
  }
  EXPECT_EQ(result.cloud.header.stamp.sec, 123);
  EXPECT_EQ(result.cloud.header.stamp.nanosec, 456789U);
  EXPECT_EQ(result.cloud.header.frame_id, "lidar_link");
  EXPECT_EQ(result.cloud.height, 1U);
  EXPECT_EQ(result.cloud.width, points.size());
  EXPECT_FALSE(result.cloud.is_bigendian);
  EXPECT_TRUE(result.cloud.is_dense);
  EXPECT_EQ(result.cloud.point_step, kOutputPointStep);
  EXPECT_EQ(result.cloud.row_step, points.size() * kOutputPointStep);
  ASSERT_EQ(result.cloud.data.size(), points.size() * kOutputPointStep);

  const std::array<std::uint8_t, 9> expected_intensity{
    0U, 0U, 12U, 13U, 255U, 255U, 42U, 42U, 42U};
  for (std::size_t index = 0U; index < points.size(); ++index) {
    const auto output_offset = index * kOutputPointStep;
    EXPECT_EQ(
      std::memcmp(
        result.cloud.data.data() + output_offset,
        &points[index].x, sizeof(float)),
      0);
    EXPECT_EQ(
      std::memcmp(
        result.cloud.data.data() + output_offset + 4U,
        &points[index].y, sizeof(float)),
      0);
    EXPECT_EQ(
      std::memcmp(
        result.cloud.data.data() + output_offset + 8U,
        &points[index].z, sizeof(float)),
      0);
    EXPECT_EQ(result.cloud.data[output_offset + 12U], expected_intensity[index]);
    EXPECT_EQ(result.cloud.data[output_offset + 13U], 9U);
    EXPECT_EQ(
      read_value<std::uint16_t>(result.cloud.data, output_offset + 14U),
      points[index].ring);
  }
  EXPECT_EQ(result.stats.input_points, points.size());
  EXPECT_EQ(result.stats.output_points, points.size());
  EXPECT_EQ(result.stats.clamped_low, 1U);
  EXPECT_EQ(result.stats.clamped_high, 1U);
  EXPECT_EQ(result.stats.nonfinite_intensity, 3U);
}

TEST(PointLayoutConverter, AppliesScaleOffsetRoundingAndBothClampStatistics)
{
  const std::vector<InputPoint> points{
    {1.0F, 0.0F, 0.0F, -100.0F, 0U, 0.0F},
    {2.0F, 0.0F, 0.0F, -2.0F, 1U, 0.0F},
    {3.0F, 0.0F, 0.0F, 100.0F, 2U, 0.0F},
  };
  ConverterConfig config;
  config.intensity_scale = 2.5;
  config.intensity_offset = 10.0;

  const auto result = convert_morai_xyzirt_to_point_xyzirc(
    make_cloud(points), config);

  ASSERT_EQ(result.cloud.width, 3U);
  EXPECT_EQ(result.cloud.data[12U], 0U);
  EXPECT_EQ(result.cloud.data[28U], 5U);
  EXPECT_EQ(result.cloud.data[44U], 255U);
  EXPECT_EQ(result.stats.clamped_low, 1U);
  EXPECT_EQ(result.stats.clamped_high, 1U);
}

TEST(PointLayoutConverter, PreservesGenericLidarFrameWithoutANameContract)
{
  auto input = make_cloud(
    std::vector<InputPoint>{{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.0F}});
  input.header.frame_id = "lidar_link";

  const auto result = convert_morai_xyzirt_to_point_xyzirc(
    input, ConverterConfig{});

  EXPECT_EQ(result.cloud.header.frame_id, "lidar_link");
}

TEST(PointLayoutConverter, RejectsFieldSetOrderAndDuplicates)
{
  const auto valid = make_cloud(
    std::vector<InputPoint>{{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.0F}});
  auto missing = valid;
  missing.fields.pop_back();
  expect_invalid(missing, ConverterConfig{}, "exactly six fields");

  auto extra = valid;
  extra.fields.push_back(field("extra", 22U, PointField::UINT8));
  expect_invalid(extra, ConverterConfig{}, "exactly six fields");

  auto duplicate = valid;
  duplicate.fields[5] = duplicate.fields[4];
  expect_invalid(duplicate, ConverterConfig{}, "field 5");

  auto reordered = valid;
  std::swap(reordered.fields[0], reordered.fields[1]);
  expect_invalid(reordered, ConverterConfig{}, "field 0");
}

TEST(PointLayoutConverter, RejectsWrongDatatypeOffsetAndCountForEveryField)
{
  const auto valid = make_cloud(
    std::vector<InputPoint>{{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.0F}});
  for (std::size_t index = 0U; index < valid.fields.size(); ++index) {
    auto datatype = valid;
    datatype.fields[index].datatype = PointField::UINT8;
    expect_invalid(
      datatype, ConverterConfig{}, "field " + std::to_string(index));

    auto offset = valid;
    ++offset.fields[index].offset;
    expect_invalid(offset, ConverterConfig{}, "field " + std::to_string(index));

    auto count = valid;
    count.fields[index].count = 2U;
    expect_invalid(count, ConverterConfig{}, "field " + std::to_string(index));
  }
}

TEST(PointLayoutConverter, RejectsEndianStrideRowsDataSizeAndOutputOverflow)
{
  const auto valid = make_cloud(
    std::vector<InputPoint>{{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.0F}});
  auto big_endian = valid;
  big_endian.is_bigendian = true;
  expect_invalid(big_endian, ConverterConfig{}, "little-endian");

  for (const auto point_step : {21U, 23U}) {
    auto wrong_step = valid;
    wrong_step.point_step = point_step;
    expect_invalid(wrong_step, ConverterConfig{}, "point_step");
  }

  auto short_row = valid;
  short_row.row_step = 21U;
  short_row.data.resize(21U);
  expect_invalid(short_row, ConverterConfig{}, "row_step");

  auto truncated = valid;
  truncated.data.pop_back();
  expect_invalid(truncated, ConverterConfig{}, "data size");

  auto trailing = valid;
  trailing.data.push_back(0U);
  expect_invalid(trailing, ConverterConfig{}, "data size");

  auto zero_height = valid;
  zero_height.height = 0U;
  zero_height.row_step = 0U;
  zero_height.data.clear();
  expect_invalid(zero_height, ConverterConfig{}, "height");

  auto excessive_count = valid;
  excessive_count.width = 2U;
  excessive_count.height = std::numeric_limits<std::uint32_t>::max();
  excessive_count.row_step = 44U;
  excessive_count.data.clear();
  expect_invalid(excessive_count, ConverterConfig{}, "point count");

  auto excessive_output = valid;
  excessive_output.height =
    std::numeric_limits<std::uint32_t>::max() / kOutputPointStep + 1U;
  excessive_output.row_step = kInputPointStep;
  excessive_output.data.clear();
  expect_invalid(excessive_output, ConverterConfig{}, "compact output");
}

TEST(PointLayoutConverter, RejectsWholeCloudOnAnyNonfiniteCoordinate)
{
  const std::array<float, 3> nonfinite_values{
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::infinity()};
  for (const auto value : nonfinite_values) {
    for (const std::size_t coordinate_index : {0U, 1U, 2U}) {
      InputPoint point{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.0F};
      if (coordinate_index == 0U) {
        point.x = value;
      } else if (coordinate_index == 1U) {
        point.y = value;
      } else {
        point.z = value;
      }
      expect_invalid(
        make_cloud(std::vector<InputPoint>{point}), ConverterConfig{},
        "nonfinite coordinate");
    }
  }
}

}  // namespace
