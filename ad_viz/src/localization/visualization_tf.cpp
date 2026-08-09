#include "ad_viz/localization/visualization_tf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>

namespace ad_viz::localization
{

tf2_msgs::msg::TFMessage without_transform_edge(
  const tf2_msgs::msg::TFMessage & input,
  const std::string & parent_frame,
  const std::string & child_frame)
{
  tf2_msgs::msg::TFMessage output;
  output.transforms.reserve(input.transforms.size());
  std::copy_if(
    input.transforms.begin(), input.transforms.end(),
    std::back_inserter(output.transforms),
    [&parent_frame, &child_frame](
      const geometry_msgs::msg::TransformStamped & transform) {
      return transform.header.frame_id != parent_frame ||
             transform.child_frame_id != child_frame;
    });
  return output;
}

std::optional<geometry_msgs::msg::TransformStamped>
visualization_transform_from_odometry(
  const nav_msgs::msg::Odometry & odometry,
  const std::string & expected_parent_frame,
  const std::string & expected_child_frame)
{
  if (odometry.header.frame_id != expected_parent_frame ||
    odometry.child_frame_id != expected_child_frame)
  {
    return std::nullopt;
  }
  const auto & position = odometry.pose.pose.position;
  const auto & orientation = odometry.pose.pose.orientation;
  const std::array<double, 7> values{
    position.x, position.y, position.z,
    orientation.x, orientation.y, orientation.z, orientation.w};
  if (!std::all_of(
      values.begin(), values.end(),
      [](const double value) {return std::isfinite(value);}))
  {
    return std::nullopt;
  }
  const double orientation_norm_squared =
    orientation.x * orientation.x + orientation.y * orientation.y +
    orientation.z * orientation.z + orientation.w * orientation.w;
  if (std::abs(orientation_norm_squared - 1.0) > 1.0e-3) {
    return std::nullopt;
  }

  geometry_msgs::msg::TransformStamped output;
  output.header = odometry.header;
  output.child_frame_id = odometry.child_frame_id;
  output.transform.translation.x = position.x;
  output.transform.translation.y = position.y;
  output.transform.translation.z = position.z;
  output.transform.rotation = orientation;
  return output;
}

}  // namespace ad_viz::localization
