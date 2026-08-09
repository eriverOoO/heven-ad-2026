#include "ad_lidar_perception/preprocessing/motion_deskewer.hpp"
#include "ad_lidar_perception/preprocessing/motion_history.hpp"
#include "ad_lidar_perception/preprocessing/xyzirt_layout.hpp"

#include <gtest/gtest.h>

#include <sensor_msgs/msg/point_field.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{

using ad_lidar_perception::preprocessing::DeskewMode;
using ad_lidar_perception::preprocessing::MotionDeskewOptions;
using ad_lidar_perception::preprocessing::MotionDeskewRetryability;
using ad_lidar_perception::preprocessing::MotionHistory;
using ad_lidar_perception::preprocessing::MotionCoverageStatus;
using ad_lidar_perception::preprocessing::MotionHistoryUpdate;
using ad_lidar_perception::preprocessing::PendingDeskewAction;
using ad_lidar_perception::preprocessing::RigidTransform3d;
using ad_lidar_perception::preprocessing::XyzirtCloudView;
using ad_lidar_perception::preprocessing::deskew_xyzirt_cloud;
using ad_lidar_perception::preprocessing::rigid_transform_from_quaternion;
using ad_lidar_perception::preprocessing::ros_stamp_seconds;
using ad_lidar_perception::preprocessing::pending_deskew_action;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

constexpr std::uint32_t kPointStep = 22U;

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

template<typename T>
T read_value(const std::vector<std::uint8_t> & bytes, const std::size_t offset)
{
  T value{};
  EXPECT_LE(offset + sizeof(T), bytes.size());
  std::memcpy(&value, bytes.data() + offset, sizeof(T));
  return value;
}

PointCloud2 make_cloud(
  const std::vector<Point> & points, const std::uint32_t height = 1U,
  const std::uint32_t row_padding = 0U, const std::string & frame = "lidar_link")
{
  EXPECT_GT(height, 0U);
  EXPECT_EQ(points.size() % height, 0U);
  PointCloud2 cloud;
  cloud.header.stamp.sec = 100;
  cloud.header.stamp.nanosec = 0U;
  cloud.header.frame_id = frame;
  cloud.height = height;
  cloud.width = static_cast<std::uint32_t>(points.size() / height);
  cloud.fields = strict_fields();
  cloud.is_bigendian = false;
  cloud.point_step = kPointStep;
  cloud.row_step = cloud.width * cloud.point_step + row_padding;
  cloud.data.assign(static_cast<std::size_t>(cloud.row_step) * cloud.height, 0xA5U);
  cloud.is_dense = false;
  for (std::uint32_t row = 0U; row < cloud.height; ++row) {
    for (std::uint32_t column = 0U; column < cloud.width; ++column) {
      const auto point_index = static_cast<std::size_t>(row) * cloud.width + column;
      const auto offset =
        static_cast<std::size_t>(row) * cloud.row_step +
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

double point_time(const PointCloud2 & cloud, const std::size_t index)
{
  return static_cast<double>(XyzirtCloudView(cloud).point(index).time);
}

std::array<double, 3> point_xyz(const PointCloud2 & cloud, const std::size_t index)
{
  const auto point = XyzirtCloudView(cloud).point(index);
  return {point.x, point.y, point.z};
}

MotionHistory constant_history(
  const double end_offset, const double velocity,
  const std::array<double, 3> & angular_rate = {0.0, 0.0, 0.0})
{
  MotionHistory history(1.0, 4096U);
  EXPECT_EQ(
    history.add_wheel_sample(100.0, velocity),
    MotionHistoryUpdate::kAccepted);
  EXPECT_EQ(
    history.add_wheel_sample(100.0 + end_offset, velocity),
    MotionHistoryUpdate::kAccepted);
  EXPECT_EQ(
    history.add_imu_sample(100.0, angular_rate),
    MotionHistoryUpdate::kAccepted);
  if (end_offset > 0.10) {
    EXPECT_EQ(
      history.add_imu_sample(100.0 + 0.5 * end_offset, angular_rate),
      MotionHistoryUpdate::kAccepted);
  }
  EXPECT_EQ(
    history.add_imu_sample(100.0 + end_offset, angular_rate),
    MotionHistoryUpdate::kAccepted);
  return history;
}

MotionDeskewOptions options()
{
  MotionDeskewOptions result;
  result.maximum_scan_duration_sec = 0.20;
  result.maximum_point_count = 300000U;
  result.maximum_imu_gap_sec = 0.12;
  result.maximum_wheel_gap_sec = 0.20;
  result.maximum_integration_step_sec = 0.005;
  return result;
}

void expect_success(const ad_lidar_perception::preprocessing::MotionDeskewResult & result)
{
  ASSERT_TRUE(result.cloud.has_value()) << result.error;
  EXPECT_TRUE(result.error.empty());
}

TEST(XyzirtLayout, CompiledViewAcceptsOnlyStrictLittleEndianTwentyTwoByteRecords)
{
  const auto cloud = make_cloud({{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.125F}});
  const XyzirtCloudView view(cloud);
  ASSERT_EQ(view.size(), 1U);
  EXPECT_EQ(view.point_offset(0U), 0U);
  const auto point = view.point(0U);
  EXPECT_FLOAT_EQ(point.x, 1.0F);
  EXPECT_FLOAT_EQ(point.y, 2.0F);
  EXPECT_FLOAT_EQ(point.z, 3.0F);
  EXPECT_FLOAT_EQ(point.intensity, 4.0F);
  EXPECT_EQ(point.ring, 5U);
  EXPECT_FLOAT_EQ(point.time, 0.125F);

  auto wrong_step = cloud;
  wrong_step.point_step = 24U;
  EXPECT_THROW(static_cast<void>(XyzirtCloudView{wrong_step}), std::invalid_argument);
  auto reordered = cloud;
  std::swap(reordered.fields[0], reordered.fields[1]);
  EXPECT_THROW(static_cast<void>(XyzirtCloudView{reordered}), std::invalid_argument);
  auto big_endian = cloud;
  big_endian.is_bigendian = true;
  EXPECT_THROW(static_cast<void>(XyzirtCloudView{big_endian}), std::invalid_argument);
  auto truncated = cloud;
  truncated.data.pop_back();
  EXPECT_THROW(static_cast<void>(XyzirtCloudView{truncated}), std::invalid_argument);
}

TEST(MotionDeskewer, DefaultLimitsMatchTheInitialSafetyContract)
{
  const MotionDeskewOptions defaults;
  EXPECT_DOUBLE_EQ(defaults.maximum_scan_duration_sec, 0.20);
  EXPECT_EQ(defaults.maximum_point_count, 300000U);
  EXPECT_DOUBLE_EQ(defaults.maximum_imu_gap_sec, 0.12);
  EXPECT_DOUBLE_EQ(defaults.maximum_wheel_gap_sec, 0.20);
  EXPECT_DOUBLE_EQ(defaults.maximum_integration_step_sec, 0.005);
}

TEST(MotionDeskewer, DefaultImuGapAcceptsObservedOneHundredSixPointFiveMillisecondBracket)
{
  const auto input = make_cloud({{1.0F, 0.0F, 0.0F, 1.0F, 1U, 0.05F}});
  MotionHistory history(1.0, 4096U);
  history.add_wheel_sample(100.0, 0.0);
  history.add_wheel_sample(100.06, 0.0);
  history.add_imu_sample(99.99, {0.0, 0.0, 0.0});
  history.add_imu_sample(100.0965, {0.0, 0.0, 0.0});

  expect_success(deskew_xyzirt_cloud(
    input, history, RigidTransform3d{}, MotionDeskewOptions{}));
}

TEST(MotionDeskewer, DefaultImuGapRejectsBracketOverOneHundredTwentyMilliseconds)
{
  const auto input = make_cloud({{1.0F, 0.0F, 0.0F, 1.0F, 1U, 0.05F}});
  MotionHistory history(1.0, 4096U);
  history.add_wheel_sample(100.0, 0.0);
  history.add_wheel_sample(100.06, 0.0);
  history.add_imu_sample(99.99, {0.0, 0.0, 0.0});
  history.add_imu_sample(100.111, {0.0, 0.0, 0.0});

  const auto result = deskew_xyzirt_cloud(
    input, history, RigidTransform3d{}, MotionDeskewOptions{});

  EXPECT_FALSE(result.cloud.has_value());
  EXPECT_EQ(result.retryability, MotionDeskewRetryability::kPermanent);
  EXPECT_NE(result.error.find("IMU interpolation gap"), std::string::npos);
}

TEST(MotionDeskewer, UsesOneRosStampConversionForNonzeroNanosecondCoverage)
{
  auto input = make_cloud({{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.1F}});
  input.header.stamp.sec = 1;
  input.header.stamp.nanosec = 99991U;
  const auto start = ros_stamp_seconds(input.header.stamp);
  const auto prior_node_value =
    static_cast<double>(1000000000LL + input.header.stamp.nanosec) * 1.0e-9;
  EXPECT_NE(start, prior_node_value);
  const auto duration = point_time(input, 0U);
  MotionHistory history(1.0, 4096U);
  history.add_wheel_sample(start, 0.0);
  history.add_wheel_sample(start + duration, 0.0);
  history.add_imu_sample(start, {0.0, 0.0, 0.0});
  history.add_imu_sample(start + duration, {0.0, 0.0, 0.0});

  expect_success(deskew_xyzirt_cloud(
    input, history, RigidTransform3d{}, options()));
}

TEST(MotionDeskewer, ZeroMotionIsAnExactByteForByteIdentity)
{
  const auto input = make_cloud(
    {
      {1.0F, 2.0F, 3.0F, 17.25F, 7U, 0.0F},
      {-4.0F, 5.0F, 6.0F, 99.5F, 8U, 0.05F},
      {7.0F, -8.0F, 9.0F, -3.0F, 9U, 0.05F},
      {10.0F, 11.0F, -12.0F, 42.0F, 10U, 0.1F},
    },
    2U, 9U);
  const auto end = point_time(input, 3U);
  const auto result = deskew_xyzirt_cloud(
    input, constant_history(end, 0.0), RigidTransform3d{}, options());

  expect_success(result);
  EXPECT_EQ(result.cloud.value(), input);
}

TEST(MotionDeskewer, AppliesSignedForwardAndReverseLongitudinalTranslation)
{
  const auto input = make_cloud({{10.0F, 2.0F, -1.0F, 4.0F, 5U, 0.1F}});
  const auto duration = point_time(input, 0U);
  for (const double velocity : {2.0, -2.0}) {
    const auto result = deskew_xyzirt_cloud(
      input, constant_history(duration, velocity), RigidTransform3d{}, options());
    expect_success(result);
    const auto xyz = point_xyz(result.cloud.value(), 0U);
    EXPECT_NEAR(xyz[0], 10.0 + velocity * duration, 1.0e-5);
    EXPECT_NEAR(xyz[1], 2.0, 1.0e-6);
    EXPECT_NEAR(xyz[2], -1.0, 1.0e-6);
  }
}

TEST(MotionDeskewer, PreservesTheSignOfYawRollAndPitchRates)
{
  const double angle = 0.1;
  const auto yaw_input = make_cloud({{1.0F, 0.0F, 0.0F, 1.0F, 1U, 0.1F}});
  const auto roll_input = make_cloud({{0.0F, 1.0F, 0.0F, 1.0F, 1U, 0.1F}});
  const auto pitch_input = make_cloud({{0.0F, 0.0F, 1.0F, 1.0F, 1U, 0.1F}});
  const auto duration = point_time(yaw_input, 0U);
  for (const double sign : {-1.0, 1.0}) {
    const auto yaw = deskew_xyzirt_cloud(
      yaw_input, constant_history(duration, 0.0, {0.0, 0.0, sign}),
      RigidTransform3d{}, options());
    const auto roll = deskew_xyzirt_cloud(
      roll_input, constant_history(duration, 0.0, {sign, 0.0, 0.0}),
      RigidTransform3d{}, options());
    const auto pitch = deskew_xyzirt_cloud(
      pitch_input, constant_history(duration, 0.0, {0.0, sign, 0.0}),
      RigidTransform3d{}, options());
    expect_success(yaw);
    expect_success(roll);
    expect_success(pitch);
    const auto yaw_xyz = point_xyz(yaw.cloud.value(), 0U);
    const auto roll_xyz = point_xyz(roll.cloud.value(), 0U);
    const auto pitch_xyz = point_xyz(pitch.cloud.value(), 0U);
    EXPECT_NEAR(yaw_xyz[0], std::cos(angle), 1.0e-5);
    EXPECT_NEAR(yaw_xyz[1], sign * std::sin(angle), 1.0e-5);
    EXPECT_NEAR(roll_xyz[1], std::cos(angle), 1.0e-5);
    EXPECT_NEAR(roll_xyz[2], sign * std::sin(angle), 1.0e-5);
    EXPECT_NEAR(pitch_xyz[0], sign * std::sin(angle), 1.0e-5);
    EXPECT_NEAR(pitch_xyz[2], std::cos(angle), 1.0e-5);
  }
}

TEST(MotionDeskewer, MatchesTheAnalyticForwardAndYawCircularArc)
{
  const auto input = make_cloud({{0.0F, 0.0F, 0.0F, 1.0F, 2U, 0.2F}});
  const auto duration = point_time(input, 0U);
  const auto result = deskew_xyzirt_cloud(
    input, constant_history(duration, 2.0, {0.0, 0.0, 1.0}),
    RigidTransform3d{}, options());

  expect_success(result);
  const auto xyz = point_xyz(result.cloud.value(), 0U);
  EXPECT_NEAR(xyz[0], 2.0 * std::sin(duration), 1.0e-5);
  EXPECT_NEAR(xyz[1], 2.0 * (1.0 - std::cos(duration)), 1.0e-5);
  EXPECT_NEAR(xyz[2], 0.0, 1.0e-6);
}

TEST(MotionDeskewer, LinearlyInterpolatesControlsInsteadOfUsingNearestSamples)
{
  const auto input = make_cloud({{0.0F, 0.0F, 0.0F, 1.0F, 1U, 0.1F}});
  const auto duration = point_time(input, 0U);
  MotionHistory history(1.0, 4096U);
  history.add_wheel_sample(100.0, 0.0);
  history.add_wheel_sample(100.0 + duration, 2.0);
  history.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  history.add_imu_sample(100.0 + duration, {0.0, 0.0, 0.0});

  const auto result =
    deskew_xyzirt_cloud(input, history, RigidTransform3d{}, options());

  expect_success(result);
  EXPECT_NEAR(point_xyz(result.cloud.value(), 0U)[0], duration, 1.0e-5);
}

TEST(MotionDeskewer, AppliesTheFullLidarLeverArm)
{
  const auto input = make_cloud({{0.0F, 0.0F, 0.0F, 1.0F, 1U, 0.1F}});
  const auto duration = point_time(input, 0U);
  const auto base_from_lidar = rigid_transform_from_quaternion(
    {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0});
  const auto result = deskew_xyzirt_cloud(
    input, constant_history(duration, 0.0, {0.0, 0.0, 1.0}),
    base_from_lidar, options());

  expect_success(result);
  const auto xyz = point_xyz(result.cloud.value(), 0U);
  EXPECT_NEAR(xyz[0], -std::sin(duration), 1.0e-5);
  EXPECT_NEAR(xyz[1], std::cos(duration) - 1.0, 1.0e-5);
  EXPECT_NEAR(xyz[2], 0.0, 1.0e-6);
}

TEST(MotionDeskewer, TwoDimensionalModeZerosOnlyRollAndPitchRates)
{
  const auto input = make_cloud({{1.0F, 1.0F, 0.0F, 1.0F, 1U, 0.1F}});
  const auto duration = point_time(input, 0U);
  const auto history = constant_history(duration, 1.0, {1.0, 0.0, 1.0});
  auto three_dimensional = options();
  auto two_dimensional = options();
  two_dimensional.mode = DeskewMode::kTwoDimensional;

  const auto result_3d = deskew_xyzirt_cloud(
    input, history, RigidTransform3d{}, three_dimensional);
  const auto result_2d = deskew_xyzirt_cloud(
    input, history, RigidTransform3d{}, two_dimensional);

  expect_success(result_3d);
  expect_success(result_2d);
  const auto xyz_3d = point_xyz(result_3d.cloud.value(), 0U);
  const auto xyz_2d = point_xyz(result_2d.cloud.value(), 0U);
  EXPECT_GT(std::abs(xyz_3d[2]), 1.0e-4);
  EXPECT_NEAR(xyz_2d[2], 0.0, 1.0e-6);
  EXPECT_GT(xyz_2d[0], 0.0);
  EXPECT_GT(xyz_2d[1], 0.0);
}

TEST(MotionDeskewer, AcceptsInterleavedFiringTimesAndPreservesInvalidReturns)
{
  const auto quiet_nan = std::numeric_limits<float>::quiet_NaN();
  const auto infinity = std::numeric_limits<float>::infinity();
  const auto input = make_cloud(
    {
      {10.0F, 1.0F, 2.0F, 10.0F, 0U, 0.08F},
      {20.0F, 3.0F, 4.0F, 20.0F, 8U, 0.01F},
      {quiet_nan, quiet_nan, quiet_nan, 30.0F, 1U, 0.10F},
      {infinity, 5.0F, 6.0F, 40.0F, 9U, 0.06F},
      {30.0F, 7.0F, 8.0F, 50.0F, 2U, 0.08F},
      {40.0F, 9.0F, 10.0F, 60.0F, 10U, 0.04F},
    },
    2U, 11U, "lidar_link");
  const auto maximum_offset = point_time(input, 2U);

  // Coverage must include the maximum firing offset, even when that record is
  // an invalid return and the final record has an earlier time.
  const auto insufficient = deskew_xyzirt_cloud(
    input, constant_history(0.08, 2.0), RigidTransform3d{}, options());
  EXPECT_FALSE(insufficient.cloud.has_value());
  EXPECT_EQ(insufficient.retryability, MotionDeskewRetryability::kRetryable);

  const auto result = deskew_xyzirt_cloud(
    input, constant_history(maximum_offset, 2.0), RigidTransform3d{}, options());

  expect_success(result);
  const auto & output = result.cloud.value();
  EXPECT_EQ(output.header, input.header);
  EXPECT_EQ(output.height, input.height);
  EXPECT_EQ(output.width, input.width);
  EXPECT_EQ(output.fields, input.fields);
  EXPECT_EQ(output.is_bigendian, input.is_bigendian);
  EXPECT_EQ(output.point_step, input.point_step);
  EXPECT_EQ(output.row_step, input.row_step);
  EXPECT_EQ(output.is_dense, input.is_dense);
  ASSERT_EQ(output.data.size(), input.data.size());

  EXPECT_NEAR(point_xyz(output, 0U)[0], 10.0 + 2.0 * 0.08, 1.0e-5);
  EXPECT_NEAR(point_xyz(output, 1U)[0], 20.0 + 2.0 * 0.01, 1.0e-5);
  EXPECT_NEAR(point_xyz(output, 4U)[0], 30.0 + 2.0 * 0.08, 1.0e-5);
  EXPECT_NEAR(point_xyz(output, 5U)[0], 40.0 + 2.0 * 0.04, 1.0e-5);

  std::vector<bool> xyz_byte(input.data.size(), false);
  const XyzirtCloudView view(input);
  for (std::size_t index = 0U; index < view.size(); ++index) {
    const auto offset = view.point_offset(index);
    const auto valid_xyz =
      std::isfinite(view.point(index).x) &&
      std::isfinite(view.point(index).y) &&
      std::isfinite(view.point(index).z);
    for (std::size_t byte = 0U; byte < 12U; ++byte) {
      if (valid_xyz) {
        xyz_byte[offset + byte] = true;
      } else {
        EXPECT_EQ(output.data[offset + byte], input.data[offset + byte])
          << "invalid return byte " << byte << " at point " << index;
      }
    }
  }
  for (std::size_t byte = 0U; byte < input.data.size(); ++byte) {
    if (!xyz_byte[byte]) {
      EXPECT_EQ(output.data[byte], input.data[byte]) << "byte " << byte;
    }
  }
}

TEST(MotionDeskewer, RejectsInvalidPointTimesDurationAndPointCount)
{
  const std::array<float, 3> invalid_times{
    -0.01F,
    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
  };
  for (const auto time : invalid_times) {
    const auto input = make_cloud({{1.0F, 2.0F, 3.0F, 4.0F, 5U, time}});
    const auto result = deskew_xyzirt_cloud(
      input, constant_history(0.1, 0.0), RigidTransform3d{}, options());
    EXPECT_FALSE(result.cloud.has_value());
  }

  const auto too_long = make_cloud({{1.0F, 2.0F, 3.0F, 4.0F, 5U, 0.201F}});
  EXPECT_FALSE(deskew_xyzirt_cloud(
    too_long, constant_history(0.3, 0.0), RigidTransform3d{}, options()).cloud);

  auto one_point_only = options();
  one_point_only.maximum_point_count = 1U;
  const auto two_points = make_cloud(
    {
      {1.0F, 0.0F, 0.0F, 1.0F, 1U, 0.0F},
      {2.0F, 0.0F, 0.0F, 1.0F, 2U, 0.01F},
    });
  EXPECT_FALSE(deskew_xyzirt_cloud(
    two_points, constant_history(0.01, 0.0), RigidTransform3d{}, one_point_only).cloud);
}

TEST(MotionDeskewer, RequiresBothHistoriesToBracketExactEndpointsWithoutLargeGaps)
{
  const auto input = make_cloud({{1.0F, 0.0F, 0.0F, 1.0F, 1U, 0.1F}});
  const auto duration = point_time(input, 0U);
  const auto exact = constant_history(duration, 0.0);
  expect_success(deskew_xyzirt_cloud(
    input, exact, RigidTransform3d{}, options()));

  MotionHistory missing_wheel_start(1.0, 4096U);
  missing_wheel_start.add_wheel_sample(100.001, 0.0);
  missing_wheel_start.add_wheel_sample(100.0 + duration, 0.0);
  missing_wheel_start.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  missing_wheel_start.add_imu_sample(100.0 + duration, {0.0, 0.0, 0.0});
  EXPECT_FALSE(deskew_xyzirt_cloud(
    input, missing_wheel_start, RigidTransform3d{}, options()).cloud);

  MotionHistory missing_imu_end(1.0, 4096U);
  missing_imu_end.add_wheel_sample(100.0, 0.0);
  missing_imu_end.add_wheel_sample(100.0 + duration, 0.0);
  missing_imu_end.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  missing_imu_end.add_imu_sample(100.0 + duration - 0.001, {0.0, 0.0, 0.0});
  EXPECT_FALSE(deskew_xyzirt_cloud(
    input, missing_imu_end, RigidTransform3d{}, options()).cloud);

  MotionHistory excessive_gap(1.0, 4096U);
  excessive_gap.add_wheel_sample(99.95, 0.0);
  excessive_gap.add_wheel_sample(100.25, 0.0);
  excessive_gap.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  excessive_gap.add_imu_sample(100.0 + duration, {0.0, 0.0, 0.0});
  EXPECT_FALSE(deskew_xyzirt_cloud(
    input, excessive_gap, RigidTransform3d{}, options()).cloud);
}

TEST(MotionDeskewer, IgnoresLargeHistoryGapEndingExactlyAtScanStart)
{
  const auto input = make_cloud({{1.0F, 0.0F, 0.0F, 1.0F, 1U, 0.05F}});
  const auto duration = point_time(input, 0U);
  MotionHistory history(2.0, 4096U);
  history.add_wheel_sample(99.0, 0.0);
  history.add_wheel_sample(100.0, 0.0);
  history.add_wheel_sample(100.0 + duration, 0.0);
  history.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  history.add_imu_sample(100.0 + duration, {0.0, 0.0, 0.0});
  auto strict_gap = options();
  strict_gap.maximum_wheel_gap_sec = 0.10;

  expect_success(deskew_xyzirt_cloud(
    input, history, RigidTransform3d{}, strict_gap));
}

TEST(MotionDeskewer, IgnoresLargeHistoryGapStartingExactlyAtScanEnd)
{
  const auto input = make_cloud({{1.0F, 0.0F, 0.0F, 1.0F, 1U, 0.05F}});
  const auto duration = point_time(input, 0U);
  MotionHistory history(2.0, 4096U);
  history.add_wheel_sample(100.0, 0.0);
  history.add_wheel_sample(100.0 + duration, 0.0);
  history.add_wheel_sample(101.0, 0.0);
  history.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  history.add_imu_sample(100.0 + duration, {0.0, 0.0, 0.0});
  auto strict_gap = options();
  strict_gap.maximum_wheel_gap_sec = 0.10;

  expect_success(deskew_xyzirt_cloud(
    input, history, RigidTransform3d{}, strict_gap));
}

TEST(MotionDeskewer, RejectsLargeHistoryGapCrossingPositiveScanInterior)
{
  const auto input = make_cloud({{1.0F, 0.0F, 0.0F, 1.0F, 1U, 0.05F}});
  const auto duration = point_time(input, 0U);
  MotionHistory history(2.0, 4096U);
  history.add_wheel_sample(99.99, 0.0);
  history.add_wheel_sample(100.0 + duration + 0.01, 0.0);
  history.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  history.add_imu_sample(100.0 + duration, {0.0, 0.0, 0.0});
  auto strict_gap = options();
  strict_gap.maximum_wheel_gap_sec = 0.02;

  const auto result = deskew_xyzirt_cloud(
    input, history, RigidTransform3d{}, strict_gap);
  EXPECT_FALSE(result.cloud.has_value());
  EXPECT_NE(result.error.find("gap"), std::string::npos);
}

TEST(MotionHistory, BackwardEpochClearsBothBoundedSeriesBeforeAcceptingNewData)
{
  MotionHistory history(1.0, 3U);
  history.add_wheel_sample(10.0, 1.0);
  history.add_wheel_sample(10.1, 2.0);
  history.add_imu_sample(10.0, {1.0, 2.0, 3.0});
  history.add_imu_sample(10.1, {4.0, 5.0, 6.0});

  EXPECT_EQ(
    history.add_wheel_sample(9.0, -1.0),
    MotionHistoryUpdate::kEpochReset);
  EXPECT_EQ(history.wheel_sample_count(), 1U);
  EXPECT_EQ(history.imu_sample_count(), 0U);

  history.add_wheel_sample(9.1, -2.0);
  history.add_wheel_sample(9.2, -3.0);
  history.add_wheel_sample(9.3, -4.0);
  EXPECT_EQ(history.wheel_sample_count(), 3U);
  EXPECT_THROW(
    history.add_imu_sample(
      std::numeric_limits<double>::quiet_NaN(), {0.0, 0.0, 0.0}),
    std::invalid_argument);
}

TEST(MotionHistory, CategorizesRetryableAndPermanentCoverageFailures)
{
  MotionHistory empty(1.0, 4096U);
  EXPECT_EQ(
    empty.coverage(100.0, 100.1, 0.2, 0.1).status,
    MotionCoverageStatus::kAwaitingFuture);

  MotionHistory awaiting_end(1.0, 4096U);
  awaiting_end.add_wheel_sample(100.0, 0.0);
  awaiting_end.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  EXPECT_EQ(
    awaiting_end.coverage(100.0, 100.1, 0.2, 0.1).status,
    MotionCoverageStatus::kAwaitingFuture);

  MotionHistory missing_past(1.0, 4096U);
  missing_past.add_wheel_sample(100.01, 0.0);
  missing_past.add_wheel_sample(100.1, 0.0);
  missing_past.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  missing_past.add_imu_sample(100.1, {0.0, 0.0, 0.0});
  EXPECT_EQ(
    missing_past.coverage(100.0, 100.1, 0.2, 0.1).status,
    MotionCoverageStatus::kMissingPast);

  MotionHistory excessive_gap(1.0, 4096U);
  excessive_gap.add_wheel_sample(100.0, 0.0);
  excessive_gap.add_wheel_sample(100.1, 0.0);
  excessive_gap.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  excessive_gap.add_imu_sample(100.1, {0.0, 0.0, 0.0});
  EXPECT_EQ(
    excessive_gap.coverage(100.0, 100.1, 0.05, 0.05).status,
    MotionCoverageStatus::kExcessiveGap);
}

TEST(MotionHistory, KnownInteriorGapIsPermanentWhileScanEndIsStillPending)
{
  MotionHistory history(1.0, 4096U);
  history.add_wheel_sample(100.0, 0.0);
  history.add_wheel_sample(100.05, 0.0);
  history.add_imu_sample(100.0, {0.0, 0.0, 0.0});

  EXPECT_EQ(
    history.coverage(100.0, 100.1, 0.01, 0.1).status,
    MotionCoverageStatus::kExcessiveGap);
}

TEST(MotionDeskewer, ExposesTypedRetryabilityForPendingPolicy)
{
  const auto input = make_cloud({{1.0F, 0.0F, 0.0F, 1.0F, 1U, 0.1F}});
  const auto duration = point_time(input, 0U);

  MotionHistory awaiting(1.0, 4096U);
  awaiting.add_wheel_sample(100.0, 0.0);
  awaiting.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  const auto retryable = deskew_xyzirt_cloud(
    input, awaiting, RigidTransform3d{}, options());
  EXPECT_EQ(retryable.retryability, MotionDeskewRetryability::kRetryable);
  EXPECT_EQ(pending_deskew_action(retryable), PendingDeskewAction::kRetry);

  const auto malformed = make_cloud(
    {{1.0F, 0.0F, 0.0F, 1.0F, 1U, -0.01F}});
  const auto permanent = deskew_xyzirt_cloud(
    malformed, constant_history(duration, 0.0), RigidTransform3d{}, options());
  EXPECT_EQ(permanent.retryability, MotionDeskewRetryability::kPermanent);
  EXPECT_EQ(pending_deskew_action(permanent), PendingDeskewAction::kDrop);

  const auto success = deskew_xyzirt_cloud(
    input, constant_history(duration, 0.0), RigidTransform3d{}, options());
  EXPECT_EQ(pending_deskew_action(success), PendingDeskewAction::kPublish);
}

TEST(MotionDeskewer, MissingPastAndHistoricalGapArePermanentFailures)
{
  const auto input = make_cloud({{1.0F, 0.0F, 0.0F, 1.0F, 1U, 0.1F}});
  const auto duration = point_time(input, 0U);
  MotionHistory missing_past(1.0, 4096U);
  missing_past.add_wheel_sample(100.01, 0.0);
  missing_past.add_wheel_sample(100.0 + duration, 0.0);
  missing_past.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  missing_past.add_imu_sample(100.0 + duration, {0.0, 0.0, 0.0});
  EXPECT_EQ(
    deskew_xyzirt_cloud(
      input, missing_past, RigidTransform3d{}, options()).retryability,
    MotionDeskewRetryability::kPermanent);

  MotionHistory gap(1.0, 4096U);
  gap.add_wheel_sample(100.0, 0.0);
  gap.add_wheel_sample(100.0 + duration, 0.0);
  gap.add_imu_sample(100.0, {0.0, 0.0, 0.0});
  gap.add_imu_sample(100.0 + duration, {0.0, 0.0, 0.0});
  auto strict = options();
  strict.maximum_wheel_gap_sec = 0.01;
  strict.maximum_imu_gap_sec = 0.01;
  EXPECT_EQ(
    deskew_xyzirt_cloud(
      input, gap, RigidTransform3d{}, strict).retryability,
    MotionDeskewRetryability::kPermanent);
}

TEST(MotionDeskewer, PreservesShapePaddingFrameAndEveryNonXyzByteExactly)
{
  const auto input = make_cloud(
    {
      {1.0F, 2.0F, 3.0F, 10.0F, 11U, 0.0F},
      {4.0F, 5.0F, 6.0F, 20.0F, 12U, 0.03F},
      {7.0F, 8.0F, 9.0F, 30.0F, 13U, 0.06F},
      {10.0F, 11.0F, 12.0F, 40.0F, 14U, 0.09F},
    },
    2U, 13U, "lidar_link");
  const auto duration = point_time(input, 3U);
  const auto result = deskew_xyzirt_cloud(
    input, constant_history(duration, 1.0), RigidTransform3d{}, options());

  expect_success(result);
  const auto & output = result.cloud.value();
  EXPECT_EQ(output.header, input.header);
  EXPECT_EQ(output.header.frame_id, "lidar_link");
  EXPECT_EQ(output.height, input.height);
  EXPECT_EQ(output.width, input.width);
  EXPECT_EQ(output.fields, input.fields);
  EXPECT_EQ(output.is_bigendian, input.is_bigendian);
  EXPECT_EQ(output.point_step, input.point_step);
  EXPECT_EQ(output.row_step, input.row_step);
  EXPECT_EQ(output.is_dense, input.is_dense);
  ASSERT_EQ(output.data.size(), input.data.size());

  std::vector<bool> xyz_byte(input.data.size(), false);
  const XyzirtCloudView view(input);
  for (std::size_t index = 0U; index < view.size(); ++index) {
    const auto offset = view.point_offset(index);
    for (std::size_t byte = 0U; byte < 12U; ++byte) {
      xyz_byte[offset + byte] = true;
    }
  }
  for (std::size_t byte = 0U; byte < input.data.size(); ++byte) {
    if (!xyz_byte[byte]) {
      EXPECT_EQ(output.data[byte], input.data[byte]) << "byte " << byte;
    }
  }
}

TEST(MotionDeskewer, DenseCloudWithInvalidCoordinateReturnsNoPartiallyMutatedCloud)
{
  auto input = make_cloud(
    {
      {1.0F, 2.0F, 3.0F, 10.0F, 11U, 0.0F},
      {4.0F, 5.0F, 6.0F, 20.0F, 12U, 0.1F},
    });
  const auto original = input;
  const auto second_offset = XyzirtCloudView(input).point_offset(1U);
  write_value(
    input.data, second_offset + 8U,
    std::numeric_limits<float>::quiet_NaN());
  input.is_dense = true;
  const auto invalid_input = input;
  const auto result = deskew_xyzirt_cloud(
    input, constant_history(0.1, 2.0), RigidTransform3d{}, options());

  EXPECT_FALSE(result.cloud.has_value());
  EXPECT_NE(result.error.find("finite XYZ"), std::string::npos);
  EXPECT_EQ(input, invalid_input);
  EXPECT_NE(input, original);
}

TEST(MotionDeskewer, RejectsFiniteDoubleResultOutsideFloatCoordinateRange)
{
  const auto input = make_cloud(
    {{std::numeric_limits<float>::max(), 0.0F, 0.0F, 1.0F, 1U, 0.1F}});
  const auto duration = point_time(input, 0U);
  const auto result = deskew_xyzirt_cloud(
    input, constant_history(duration, 4.0e39), RigidTransform3d{}, options());

  EXPECT_FALSE(result.cloud.has_value());
  EXPECT_NE(result.error.find("float"), std::string::npos);
}

}  // namespace
