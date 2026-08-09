#ifndef AD_LIDAR_PERCEPTION__PREPROCESSING__POINTCLOUD_DENSIFIER_HPP_
#define AD_LIDAR_PERCEPTION__PREPROCESSING__POINTCLOUD_DENSIFIER_HPP_

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <array>
#include <cstddef>
#include <optional>

namespace ad_lidar_perception::preprocessing
{

struct DensifierConfig
{
  double voxel_size_m{0.30};
  double roi_min_x_m{20.0};
  double roi_max_x_m{100.0};
  double roi_min_y_m{-12.0};
  double roi_max_y_m{12.0};
  double maximum_history_age_sec{0.25};
  double maximum_translation_jump_m{5.0};
  double maximum_rotation_jump_rad{0.35};
};

struct DensifierTransform
{
  std::array<double, 3> translation{};
  std::array<double, 4> quaternion_xyzw{{0.0, 0.0, 0.0, 1.0}};
};

enum class DensifierStatus
{
  kFirstFrame,
  kFused,
  kNoEligibleHistory,
  kNonIncreasingStamp,
  kStaleHistory,
  kSchemaMismatch,
  kFrameMismatch,
  kTransformUnavailable,
  kTransformTranslationJump,
  kTransformRotationJump,
  kMalformedCurrent,
  kMalformedHistory,
  kNumericalFailure,
};

struct DensifierResult
{
  sensor_msgs::msg::PointCloud2 cloud;
  DensifierStatus status;
  std::size_t historical_points_added{0U};
};

class PointcloudDensifier
{
public:
  explicit PointcloudDensifier(DensifierConfig config = {});

  [[nodiscard]] bool has_history() const noexcept;
  DensifierResult process(
    const sensor_msgs::msg::PointCloud2 & current,
    const std::optional<DensifierTransform> & current_from_previous);

private:
  DensifierConfig config_;
  std::optional<sensor_msgs::msg::PointCloud2> history_;
};

}  // namespace ad_lidar_perception::preprocessing

#endif  // AD_LIDAR_PERCEPTION__PREPROCESSING__POINTCLOUD_DENSIFIER_HPP_
