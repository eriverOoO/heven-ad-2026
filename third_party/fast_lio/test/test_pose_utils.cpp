#include <gtest/gtest.h>

#include <fast_lio/pose_utils.hpp>

#include <limits>

TEST(PoseUtils, ConvertsBaseInitialPoseToImuAndBackUsingTheSensorMount) {
  geometry_msgs::msg::Pose base_pose;
  base_pose.position.x = 10.0;
  base_pose.position.y = -2.0;
  base_pose.position.z = 0.3;
  base_pose.orientation.w = 1.0;
  const auto base_T_imu = fast_lio::rigid_transform({0.0, 0.0, 1.2, 0.0, 0.0, 0.0});

  const auto odom_T_imu = fast_lio::to_isometry(base_pose) * base_T_imu;
  EXPECT_DOUBLE_EQ(odom_T_imu.translation().x(), 10.0);
  EXPECT_DOUBLE_EQ(odom_T_imu.translation().y(), -2.0);
  EXPECT_DOUBLE_EQ(odom_T_imu.translation().z(), 1.5);

  const auto recovered_base = fast_lio::to_pose(odom_T_imu * base_T_imu.inverse());
  EXPECT_DOUBLE_EQ(recovered_base.position.x, base_pose.position.x);
  EXPECT_DOUBLE_EQ(recovered_base.position.y, base_pose.position.y);
  EXPECT_DOUBLE_EQ(recovered_base.position.z, base_pose.position.z);
  EXPECT_DOUBLE_EQ(recovered_base.orientation.w, 1.0);
}

TEST(PoseUtils, ExpressesWorldVelocityInThePublishedBaseFrame) {
  const Eigen::Matrix3d world_R_base =
      Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const Eigen::Vector3d world_velocity(0.0, 4.0, 0.5);

  const auto base_velocity =
      fast_lio::world_velocity_to_body(world_R_base, world_velocity);

  EXPECT_NEAR(base_velocity.x(), 4.0, 1.0e-12);
  EXPECT_NEAR(base_velocity.y(), 0.0, 1.0e-12);
  EXPECT_NEAR(base_velocity.z(), 0.5, 1.0e-12);
}

TEST(PoseUtils, ValidatesFiniteNonzeroPoseQuaternion) {
  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 1.0;
  EXPECT_TRUE(fast_lio::valid_pose(pose));

  pose.position.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(fast_lio::valid_pose(pose));

  pose = geometry_msgs::msg::Pose();
  pose.orientation.w = 0.0;
  EXPECT_FALSE(fast_lio::valid_pose(pose));
}

TEST(PoseUtils, ValidatesProperFiniteRotationMatrix) {
  EXPECT_TRUE(fast_lio::valid_rotation_matrix(
      {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
  EXPECT_FALSE(fast_lio::valid_rotation_matrix(
      {2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
  EXPECT_FALSE(fast_lio::valid_rotation_matrix(
      {-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
  EXPECT_FALSE(fast_lio::valid_rotation_matrix(
      {1.0, 0.0, 0.0}));
}
