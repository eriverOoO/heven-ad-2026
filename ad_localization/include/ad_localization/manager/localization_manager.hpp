#ifndef AD_LOCALIZATION__MANAGER__LOCALIZATION_MANAGER_HPP_
#define AD_LOCALIZATION__MANAGER__LOCALIZATION_MANAGER_HPP_

#include <cstdint>
#include <optional>
#include <string>

#include "builtin_interfaces/msg/time.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"

namespace ad_localization
{

struct LocalizationManagerConfig
{
  std::string map_frame{"map"};
  std::string odom_frame{"odom"};
  std::string base_frame{"base_link"};
};

class LocalizationManager
{
public:
  explicit LocalizationManager(LocalizationManagerConfig config);

  std::optional<nav_msgs::msg::Odometry> accept(
    const nav_msgs::msg::Odometry & candidate);
  void reset() noexcept;

private:
  LocalizationManagerConfig config_;
  std::optional<std::int64_t> last_stamp_ns_;
};

geometry_msgs::msg::TransformStamped odometry_transform(
  const nav_msgs::msg::Odometry & odometry);

geometry_msgs::msg::TransformStamped map_to_odom_transform(
  const LocalizationManagerConfig & config,
  const builtin_interfaces::msg::Time & stamp);

}  // namespace ad_localization

#endif  // AD_LOCALIZATION__MANAGER__LOCALIZATION_MANAGER_HPP_
