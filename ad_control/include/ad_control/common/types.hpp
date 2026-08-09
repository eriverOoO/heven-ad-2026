#ifndef AD_CONTROL__COMMON__TYPES_HPP_
#define AD_CONTROL__COMMON__TYPES_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ad_control {

struct Point3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct Pose2 {
  double x{0.0};
  double y{0.0};
  double yaw_rad{0.0};
};

struct PhysicalCommand {
  double accel{0.0};
  double brake{1.0};
  double steering_rad{0.0};
};

struct Trajectory {
  std::vector<Pose2> poses;
};

struct ControllerResult {
  bool valid{false};
  PhysicalCommand command{};
  std::string reason{"invalid"};
  std::optional<Point3> target;
  std::optional<Trajectory> local_trajectory;
  std::optional<double> target_speed_mps;
};

struct OccupancyGrid {
  Pose2 origin;
  double resolution{0.0};
  std::size_t width{0};
  std::size_t height{0};
  std::vector<std::int8_t> cells;
  bool valid{false};
  bool fresh{false};
};

inline bool operator==(const PhysicalCommand &lhs, const PhysicalCommand &rhs) {
  return lhs.accel == rhs.accel && lhs.brake == rhs.brake &&
         lhs.steering_rad == rhs.steering_rad;
}

inline bool operator==(const Point3 &lhs, const Point3 &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

inline bool operator==(const Pose2 &lhs, const Pose2 &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.yaw_rad == rhs.yaw_rad;
}

inline bool operator==(const Trajectory &lhs, const Trajectory &rhs) {
  return lhs.poses == rhs.poses;
}

inline bool operator==(const ControllerResult &lhs,
                       const ControllerResult &rhs) {
  return lhs.valid == rhs.valid && lhs.command == rhs.command &&
         lhs.reason == rhs.reason && lhs.target == rhs.target &&
         lhs.local_trajectory == rhs.local_trajectory &&
         lhs.target_speed_mps == rhs.target_speed_mps;
}

struct Route {
  std::vector<Point3> points;
  bool closed{false};
};

struct RouteProgressState {
  bool initialized{false};
  std::size_t target_index{0};
  std::size_t lap_count{0};
  bool terminal_full_brake{false};
};

} // namespace ad_control

#endif // AD_CONTROL__COMMON__TYPES_HPP_
