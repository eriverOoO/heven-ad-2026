#include "ad_lidar_perception/preprocessing/finite_point_filter.hpp"

#include <gtest/gtest.h>

#include <sensor_msgs/msg/point_field.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{

using ad_lidar_perception::preprocessing::filter_finite_xyz;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

PointField field(
  std::string name, const std::uint32_t offset,
  const std::uint8_t datatype = PointField::FLOAT32)
{
  PointField result;
  result.name = std::move(name);
  result.offset = offset;
  result.datatype = datatype;
  result.count = 1U;
  return result;
}

template<typename T>
void write_value(
  std::vector<std::uint8_t> & data, const std::size_t offset,
  const T value)
{
  ASSERT_LE(offset + sizeof(T), data.size());
  std::memcpy(data.data() + offset, &value, sizeof(T));
}

PointCloud2 cloud()
{
  PointCloud2 result;
  result.header.stamp.sec = 123;
  result.header.frame_id = "lidar_link";
  result.height = 2U;
  result.width = 2U;
  result.fields = {
    field("x", 0U), field("y", 4U), field("z", 8U),
    field("intensity", 12U)};
  result.is_bigendian = false;
  result.point_step = 16U;
  result.row_step = 36U;
  result.data.assign(72U, 0xA5U);
  result.is_dense = true;

  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const float points[4][4] = {
    {1.0F, 2.0F, 3.0F, 10.0F},
    {nan, 5.0F, 6.0F, 20.0F},
    {7.0F, inf, 9.0F, 30.0F},
    {10.0F, 11.0F, 12.0F, 40.0F},
  };
  for (std::size_t index = 0U; index < 4U; ++index) {
    const std::size_t row = index / 2U;
    const std::size_t column = index % 2U;
    const std::size_t offset = row * result.row_step + column * result.point_step;
    for (std::size_t field_index = 0U; field_index < 4U; ++field_index) {
      write_value(result.data, offset + field_index * sizeof(float),
        points[index][field_index]);
    }
  }
  return result;
}

TEST(FinitePointFilter, CompactsOrganizedRowsAndPreservesEveryFiniteRecord)
{
  const auto input = cloud();
  const auto result = filter_finite_xyz(input);

  EXPECT_EQ(result.stats.input_points, 4U);
  EXPECT_EQ(result.stats.output_points, 2U);
  EXPECT_EQ(result.stats.removed_nonfinite, 2U);
  EXPECT_EQ(result.cloud.header, input.header);
  EXPECT_EQ(result.cloud.fields, input.fields);
  EXPECT_EQ(result.cloud.height, 1U);
  EXPECT_EQ(result.cloud.width, 2U);
  EXPECT_EQ(result.cloud.point_step, 16U);
  EXPECT_EQ(result.cloud.row_step, 32U);
  EXPECT_EQ(result.cloud.data.size(), 32U);
  EXPECT_TRUE(result.cloud.is_dense);
  EXPECT_EQ(
    std::memcmp(result.cloud.data.data(), input.data.data(), 16U), 0);
  EXPECT_EQ(
    std::memcmp(
      result.cloud.data.data() + 16U,
      input.data.data() + input.row_step + input.point_step,
      16U),
    0);
}

TEST(FinitePointFilter, RejectsMalformedFieldsRowsAndEndian)
{
  auto input = cloud();
  input.fields.push_back(field("x", 0U));
  EXPECT_THROW(filter_finite_xyz(input), std::invalid_argument);

  input = cloud();
  input.row_step = input.width * input.point_step - 1U;
  EXPECT_THROW(filter_finite_xyz(input), std::invalid_argument);

  input = cloud();
  input.is_bigendian = true;
  EXPECT_THROW(filter_finite_xyz(input), std::invalid_argument);
}

}  // namespace
