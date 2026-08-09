#include "ad_lidar_perception/preprocessing/gravity_leveler.hpp"
#include "ad_lidar_perception/preprocessing/xyzirt_layout.hpp"

#include <gtest/gtest.h>

#include <sensor_msgs/msg/point_field.hpp>

#include <array>
#include <cmath>
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

using ad_lidar_perception::preprocessing::GravityLevelingTransform;
using ad_lidar_perception::preprocessing::XyzirtCloudView;
using ad_lidar_perception::preprocessing::derive_leveled_frame;
using ad_lidar_perception::preprocessing::level_xyzirt_cloud;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

constexpr std::uint32_t kPointStep = 22U;
constexpr double kPi = 3.14159265358979323846;

struct Point
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
void write_value(std::vector<std::uint8_t> & bytes, const std::size_t offset, const T value)
{
  ASSERT_LE(offset + sizeof(T), bytes.size());
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

PointCloud2 make_cloud(
  const std::vector<Point> & points, const std::uint32_t height = 1U,
  const std::uint32_t row_padding = 0U, const std::string & frame = "roof_lidar_link")
{
  EXPECT_GT(height, 0U);
  EXPECT_EQ(points.size() % height, 0U);
  PointCloud2 cloud;
  cloud.header.stamp.sec = 42;
  cloud.header.stamp.nanosec = 123456789U;
  cloud.header.frame_id = frame;
  cloud.height = height;
  cloud.width = static_cast<std::uint32_t>(points.size() / height);
  cloud.fields = strict_fields();
  cloud.is_bigendian = false;
  cloud.point_step = kPointStep;
  cloud.row_step = cloud.width * cloud.point_step + row_padding;
  cloud.data.assign(static_cast<std::size_t>(cloud.row_step) * cloud.height, 0xD3U);
  cloud.is_dense = false;
  for (std::uint32_t row = 0U; row < cloud.height; ++row) {
    for (std::uint32_t column = 0U; column < cloud.width; ++column) {
      const auto index = static_cast<std::size_t>(row) * cloud.width + column;
      const auto offset = static_cast<std::size_t>(row) * cloud.row_step +
        static_cast<std::size_t>(column) * cloud.point_step;
      write_value(cloud.data, offset + 0U, points[index].x);
      write_value(cloud.data, offset + 4U, points[index].y);
      write_value(cloud.data, offset + 8U, points[index].z);
      write_value(cloud.data, offset + 12U, points[index].intensity);
      write_value(cloud.data, offset + 16U, points[index].ring);
      write_value(cloud.data, offset + 18U, points[index].time);
    }
  }
  return cloud;
}

GravityLevelingTransform transform_from_rpy(
  const std::array<double, 3> & translation, const double roll,
  const double pitch, const double yaw, const double quaternion_scale = 1.0)
{
  const auto cr = std::cos(roll * 0.5);
  const auto sr = std::sin(roll * 0.5);
  const auto cp = std::cos(pitch * 0.5);
  const auto sp = std::sin(pitch * 0.5);
  const auto cy = std::cos(yaw * 0.5);
  const auto sy = std::sin(yaw * 0.5);
  return GravityLevelingTransform{
    translation,
    {
      quaternion_scale * (sr * cp * cy - cr * sp * sy),
      quaternion_scale * (cr * sp * cy + sr * cp * sy),
      quaternion_scale * (cr * cp * sy - sr * sp * cy),
      quaternion_scale * (cr * cp * cy + sr * sp * sy),
    },
  };
}

std::array<double, 9> rotation_matrix(const GravityLevelingTransform & transform)
{
  const auto & q = transform.quaternion_xyzw;
  const auto norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const auto x = q[0] / norm;
  const auto y = q[1] / norm;
  const auto z = q[2] / norm;
  const auto w = q[3] / norm;
  return {
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
    2.0 * (x * z + y * w), 2.0 * (x * y + z * w),
    1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
    2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
    1.0 - 2.0 * (x * x + y * y),
  };
}

std::array<double, 9> multiply(
  const std::array<double, 9> & left, const std::array<double, 9> & right)
{
  std::array<double, 9> result{};
  for (std::size_t row = 0U; row < 3U; ++row) {
    for (std::size_t column = 0U; column < 3U; ++column) {
      for (std::size_t inner = 0U; inner < 3U; ++inner) {
        result[row * 3U + column] +=
          left[row * 3U + inner] * right[inner * 3U + column];
      }
    }
  }
  return result;
}

std::array<double, 3> point_xyz(const PointCloud2 & cloud, const std::size_t index)
{
  const auto point = XyzirtCloudView(cloud).point(index);
  return {point.x, point.y, point.z};
}

void expect_invalid(
  const PointCloud2 & input, const GravityLevelingTransform & odom_from_base,
  const GravityLevelingTransform & base_from_lidar, const std::string & message)
{
  try {
    static_cast<void>(level_xyzirt_cloud(
        input, odom_from_base, base_from_lidar, "roof_lidar_leveled_frame"));
    FAIL() << "expected std::invalid_argument containing " << message;
  } catch (const std::invalid_argument & error) {
    EXPECT_NE(std::string(error.what()).find(message), std::string::npos)
      << "actual error: " << error.what();
  } catch (const std::exception & error) {
    FAIL() << "wrong exception class: " << error.what();
  }
}

TEST(GravityLeveler, AppliesSignedWorldRollAndPitchCorrection)
{
  const double angle = 0.25;
  const auto roll_cloud = make_cloud({{0.0F, 1.0F, 0.0F, 1.0F, 1U, 0.0F}});
  const auto pitch_cloud = make_cloud({{0.0F, 0.0F, 1.0F, 1.0F, 1U, 0.0F}});
  for (const double sign : {-1.0, 1.0}) {
    const auto roll = level_xyzirt_cloud(
      roll_cloud, transform_from_rpy({}, sign * angle, 0.0, 0.0),
      GravityLevelingTransform{}, "roof_lidar_leveled_frame");
    const auto pitch = level_xyzirt_cloud(
      pitch_cloud, transform_from_rpy({}, 0.0, sign * angle, 0.0),
      GravityLevelingTransform{}, "roof_lidar_leveled_frame");
    const auto roll_xyz = point_xyz(roll.cloud, 0U);
    const auto pitch_xyz = point_xyz(pitch.cloud, 0U);
    EXPECT_NEAR(roll_xyz[1], std::cos(angle), 1.0e-6);
    EXPECT_NEAR(roll_xyz[2], sign * std::sin(angle), 1.0e-6);
    EXPECT_NEAR(pitch_xyz[0], sign * std::sin(angle), 1.0e-6);
    EXPECT_NEAR(pitch_xyz[2], std::cos(angle), 1.0e-6);
  }
}

TEST(GravityLeveler, PureCombinedYawIsExactPointDataIdentity)
{
  const auto input = make_cloud(
    {
      {1.25F, -2.5F, 3.75F, 9.5F, 42U, 0.01F},
      {-4.5F, 5.25F, -6.0F, -7.5F, 65530U, 0.09F},
    },
    2U, 11U);
  const auto result = level_xyzirt_cloud(
    input, transform_from_rpy({10.0, -2.0, 0.5}, 0.0, 0.0, 0.4),
    transform_from_rpy({1.2, -0.4, 1.4}, 0.0, 0.0, -0.7, 17.0),
    "roof_lidar_leveled_frame");

  EXPECT_EQ(result.cloud.data, input.data);
  EXPECT_EQ(result.cloud.header.stamp, input.header.stamp);
  EXPECT_EQ(result.cloud.header.frame_id, "roof_lidar_leveled_frame");
}

TEST(GravityLeveler, HandlesCombinedBasePoseAndNontrivialLidarExtrinsic)
{
  const auto input = make_cloud({{3.0F, -2.0F, 0.5F, 8.0F, 7U, 0.04F}});
  const auto result = level_xyzirt_cloud(
    input, transform_from_rpy({10.0, -3.0, 2.0}, 0.2, -0.3, 0.4),
    transform_from_rpy({1.2, -0.4, 1.4}, 0.1, 0.2, -0.5),
    "roof_lidar_leveled_frame");

  const auto xyz = point_xyz(result.cloud, 0U);
  EXPECT_NEAR(xyz[0], 2.99207331, 1.0e-6);
  EXPECT_NEAR(xyz[1], -2.03237769, 1.0e-6);
  EXPECT_NEAR(xyz[2], -0.40858075, 1.0e-6);
  EXPECT_NEAR(result.base_from_level.translation[0], 1.2, 1.0e-12);
  EXPECT_NEAR(result.base_from_level.translation[1], -0.4, 1.0e-12);
  EXPECT_NEAR(result.base_from_level.translation[2], 1.4, 1.0e-12);
}

TEST(GravityLeveler, KeepsThePhysicalLidarOriginInvariant)
{
  const auto input = make_cloud({{0.0F, 0.0F, 0.0F, 5.0F, 3U, 0.0F}});
  const auto result = level_xyzirt_cloud(
    input, transform_from_rpy({4.0, 5.0, 6.0}, -0.4, 0.3, 1.1),
    transform_from_rpy({1.1, -0.2, 1.7}, 0.2, -0.1, 0.6),
    "roof_lidar_leveled_frame");

  const auto xyz = point_xyz(result.cloud, 0U);
  EXPECT_DOUBLE_EQ(xyz[0], 0.0);
  EXPECT_DOUBLE_EQ(xyz[1], 0.0);
  EXPECT_DOUBLE_EQ(xyz[2], 0.0);
  EXPECT_NEAR(result.base_from_level.translation[0], 1.1, 1.0e-12);
  EXPECT_NEAR(result.base_from_level.translation[1], -0.2, 1.0e-12);
  EXPECT_NEAR(result.base_from_level.translation[2], 1.7, 1.0e-12);
}

TEST(GravityLeveler, RetainsActualLidarYawAndProducesNormalizedTransformQuaternion)
{
  const auto odom_from_base =
    transform_from_rpy({2.0, 3.0, 4.0}, -0.3, 0.2, 0.7, 9.0);
  const auto base_from_lidar =
    transform_from_rpy({1.2, 0.1, 1.4}, 0.1, -0.25, -0.4, 0.2);
  const auto result = level_xyzirt_cloud(
    make_cloud({{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.0F}}),
    odom_from_base, base_from_lidar, "roof_lidar_leveled_frame");

  const auto world_from_lidar = multiply(
    rotation_matrix(odom_from_base), rotation_matrix(base_from_lidar));
  const auto world_from_level = multiply(
    rotation_matrix(odom_from_base), rotation_matrix(result.base_from_level));
  const auto lidar_yaw = std::atan2(world_from_lidar[3], world_from_lidar[0]);
  const auto level_yaw = std::atan2(world_from_level[3], world_from_level[0]);
  EXPECT_NEAR(level_yaw, lidar_yaw, 1.0e-12);
  EXPECT_NEAR(std::atan2(world_from_level[7], world_from_level[8]), 0.0, 1.0e-12);
  EXPECT_NEAR(-std::asin(world_from_level[6]), 0.0, 1.0e-12);
  const auto & q = result.base_from_level.quaternion_xyzw;
  EXPECT_NEAR(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 1.0, 1.0e-12);
}

TEST(GravityLeveler, PreservesShapeMetadataOrderNonXyzBytesAndRowPadding)
{
  const auto input = make_cloud(
    {
      {1.0F, 2.0F, 3.0F, 11.25F, 8U, 0.01F},
      {4.0F, 5.0F, 6.0F, -22.5F, 9U, 0.02F},
      {7.0F, 8.0F, 9.0F, 33.75F, 10U, 0.03F},
      {10.0F, 11.0F, 12.0F, 44.0F, 11U, 0.04F},
    },
    2U, 13U);
  const auto result = level_xyzirt_cloud(
    input, transform_from_rpy({}, 0.2, -0.1, 0.5),
    transform_from_rpy({}, -0.3, 0.25, -0.2), "roof_lidar_leveled_frame");

  EXPECT_EQ(result.cloud.height, input.height);
  EXPECT_EQ(result.cloud.width, input.width);
  EXPECT_EQ(result.cloud.fields, input.fields);
  EXPECT_EQ(result.cloud.is_bigendian, input.is_bigendian);
  EXPECT_EQ(result.cloud.point_step, input.point_step);
  EXPECT_EQ(result.cloud.row_step, input.row_step);
  EXPECT_EQ(result.cloud.is_dense, input.is_dense);
  for (std::uint32_t row = 0U; row < input.height; ++row) {
    for (std::uint32_t column = 0U; column < input.width; ++column) {
      const auto offset = static_cast<std::size_t>(row) * input.row_step +
        static_cast<std::size_t>(column) * input.point_step;
      EXPECT_EQ(
        std::memcmp(result.cloud.data.data() + offset + 12U,
          input.data.data() + offset + 12U, 10U),
        0);
    }
    const auto padding_offset = static_cast<std::size_t>(row) * input.row_step +
      static_cast<std::size_t>(input.width) * input.point_step;
    EXPECT_EQ(
      std::memcmp(result.cloud.data.data() + padding_offset,
        input.data.data() + padding_offset, 13U),
      0);
  }
}

TEST(GravityLeveler, RejectsMalformedNonfiniteAndFloatOverflowTransactionally)
{
  const auto valid = make_cloud({{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.0F}});
  auto malformed = valid;
  malformed.point_step = 21U;
  expect_invalid(
    malformed, GravityLevelingTransform{}, GravityLevelingTransform{}, "point_step");

  auto nonfinite = valid;
  write_value(nonfinite.data, 0U, std::numeric_limits<float>::quiet_NaN());
  const auto nonfinite_before = nonfinite;
  expect_invalid(
    nonfinite, GravityLevelingTransform{}, GravityLevelingTransform{}, "nonfinite");
  EXPECT_EQ(nonfinite, nonfinite_before);

  const auto maximum = std::numeric_limits<float>::max();
  const auto overflow = make_cloud({{maximum, 0.0F, -maximum, 4.0F, 5U, 0.0F}});
  const auto overflow_before = overflow;
  expect_invalid(
    overflow, transform_from_rpy({}, 0.0, kPi / 4.0, 0.0),
    GravityLevelingTransform{}, "float32");
  EXPECT_EQ(overflow, overflow_before);
}

TEST(GravityLeveler, NormalizesFiniteQuaternionsAndRejectsZeroOrNonfiniteOnes)
{
  const auto input = make_cloud({{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.0F}});
  const auto normalized = level_xyzirt_cloud(
    input, transform_from_rpy({}, 0.1, -0.2, 0.3, 1.0e200),
    transform_from_rpy({}, -0.2, 0.1, -0.4, 1.0e-200),
    "roof_lidar_leveled_frame");
  EXPECT_EQ(normalized.cloud.header.frame_id, "roof_lidar_leveled_frame");

  auto invalid = GravityLevelingTransform{};
  invalid.quaternion_xyzw = {0.0, 0.0, 0.0, 0.0};
  expect_invalid(input, invalid, GravityLevelingTransform{}, "nonzero");
  invalid = GravityLevelingTransform{};
  invalid.quaternion_xyzw[2] = std::numeric_limits<double>::infinity();
  expect_invalid(input, GravityLevelingTransform{}, invalid, "finite");
}

TEST(GravityLeveler, DerivesGenericRelativeFramesAndRejectsInvalidFrames)
{
  EXPECT_EQ(derive_leveled_frame("lidar_link"), "lidar_leveled_frame");
  EXPECT_EQ(derive_leveled_frame("roof_lidar_link"), "roof_lidar_leveled_frame");
  EXPECT_EQ(derive_leveled_frame("generic_sensor"), "generic_sensor_leveled_frame");
  EXPECT_THROW(static_cast<void>(derive_leveled_frame("")), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(derive_leveled_frame("/lidar_link")), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(derive_leveled_frame("roof lidar")), std::invalid_argument);

  const auto input = make_cloud({{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.0F}});
  EXPECT_THROW(
    static_cast<void>(level_xyzirt_cloud(
        input, GravityLevelingTransform{}, GravityLevelingTransform{},
        "some_other_frame")),
    std::invalid_argument);
}

}  // namespace
