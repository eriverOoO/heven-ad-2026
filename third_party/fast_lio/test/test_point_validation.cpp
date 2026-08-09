#include <gtest/gtest.h>

#include <limits>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <fast_lio/point_validation.hpp>

TEST(PointValidation, AcceptsOnlyFiniteCartesianCoordinates) {
  EXPECT_TRUE(fast_lio::finite_xyz(1.0F, -2.0F, 3.0F));
  EXPECT_FALSE(fast_lio::finite_xyz(
      std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F));
  EXPECT_FALSE(fast_lio::finite_xyz(
      0.0F, std::numeric_limits<float>::infinity(), 0.0F));
  EXPECT_FALSE(fast_lio::finite_xyz(
      0.0F, 0.0F, -std::numeric_limits<float>::infinity()));
}

namespace {

sensor_msgs::msg::PointCloud2 valid_morai_cloud() {
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.height = 1;
  cloud.width = 10;
  cloud.point_step = 24;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.resize(cloud.row_step);
  cloud.fields = {
    sensor_msgs::msg::PointField().set__name("x").set__offset(0).set__datatype(
      sensor_msgs::msg::PointField::FLOAT32).set__count(1),
    sensor_msgs::msg::PointField().set__name("y").set__offset(4).set__datatype(
      sensor_msgs::msg::PointField::FLOAT32).set__count(1),
    sensor_msgs::msg::PointField().set__name("z").set__offset(8).set__datatype(
      sensor_msgs::msg::PointField::FLOAT32).set__count(1),
    sensor_msgs::msg::PointField().set__name("intensity").set__offset(12).set__datatype(
      sensor_msgs::msg::PointField::FLOAT32).set__count(1),
    sensor_msgs::msg::PointField().set__name("time").set__offset(16).set__datatype(
      sensor_msgs::msg::PointField::FLOAT32).set__count(1),
    sensor_msgs::msg::PointField().set__name("ring").set__offset(20).set__datatype(
      sensor_msgs::msg::PointField::UINT16).set__count(1),
  };
  return cloud;
}

}  // namespace

TEST(PointValidation, RequiresExactMoraiFieldTypesAndCounts) {
  auto cloud = valid_morai_cloud();
  EXPECT_TRUE(fast_lio::valid_morai_pointcloud_layout(cloud));

  cloud.fields.back().datatype = sensor_msgs::msg::PointField::UINT8;
  EXPECT_FALSE(fast_lio::valid_morai_pointcloud_layout(cloud));

  cloud = valid_morai_cloud();
  cloud.fields.front().count = 3;
  EXPECT_FALSE(fast_lio::valid_morai_pointcloud_layout(cloud));
}

TEST(PointValidation, RejectsMalformedOrUnboundedCloudShape) {
  auto cloud = valid_morai_cloud();
  EXPECT_TRUE(fast_lio::valid_pointcloud_shape(cloud, 100));

  cloud.data.pop_back();
  EXPECT_FALSE(fast_lio::valid_pointcloud_shape(cloud, 100));

  cloud = valid_morai_cloud();
  cloud.width = 101;
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.data.resize(cloud.row_step);
  EXPECT_FALSE(fast_lio::valid_pointcloud_shape(cloud, 100));
}

TEST(PointValidation, AcceptsOnlyFiniteBoundedRelativePointTime) {
  EXPECT_TRUE(fast_lio::valid_relative_point_time(0.0F, 0.2));
  EXPECT_TRUE(fast_lio::valid_relative_point_time(0.1F, 0.2));
  EXPECT_FALSE(fast_lio::valid_relative_point_time(-0.001F, 0.2));
  EXPECT_FALSE(fast_lio::valid_relative_point_time(0.201F, 0.2));
  EXPECT_FALSE(fast_lio::valid_relative_point_time(
      std::numeric_limits<float>::quiet_NaN(), 0.2));
}

TEST(PointValidation, RequiresAValidRelativeConfiguredLidarFrame) {
  EXPECT_TRUE(fast_lio::valid_relative_frame("lidar_link"));
  EXPECT_FALSE(fast_lio::valid_relative_frame(""));
  EXPECT_FALSE(fast_lio::valid_relative_frame("/lidar_link"));
  EXPECT_FALSE(fast_lio::valid_relative_frame("lidar//link"));
  EXPECT_FALSE(fast_lio::valid_relative_frame("lidar link"));
}

TEST(PointValidation, AcceptsOnlyTheConfiguredLidarFrame) {
  EXPECT_TRUE(fast_lio::matches_lidar_frame("lidar_link", "lidar_link"));
  EXPECT_FALSE(fast_lio::matches_lidar_frame("", "lidar_link"));
  EXPECT_FALSE(fast_lio::matches_lidar_frame("unexpected_sensor_frame", "lidar_link"));
  EXPECT_FALSE(fast_lio::matches_lidar_frame("/unexpected_sensor_frame", "lidar_link"));
}
