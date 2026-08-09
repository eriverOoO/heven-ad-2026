"""Low-speed GPS/IMU pure-pursuit route follower."""

import json
import math
import os
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import String

from .route_tracking import (
    cumulative_distances,
    latlon_to_utm52,
    lookahead_index,
    nearest_index,
    normalized_pure_pursuit_steer,
    quaternion_yaw,
)


class RouteFollower(Node):
    """Publish bounded commands; the control bridge remains the safety owner."""

    def __init__(self) -> None:
        super().__init__("route_follower")
        self.declare_parameter("route_file", os.getenv("HEVEN_ROUTE_FILE", ""))
        self.declare_parameter("origin_easting", 302595.0)
        self.declare_parameter("origin_northing", 4124145.0)
        self.declare_parameter("lookahead_m", 7.0)
        self.declare_parameter("wheelbase_m", 3.0)
        self.declare_parameter("max_wheel_angle_deg", 40.0)
        self.declare_parameter("speed_mps", 1.0)
        self.declare_parameter("sensor_timeout_sec", 0.5)
        self.declare_parameter("link_timeout_sec", 0.5)
        self.declare_parameter("nearest_search_ahead_points", 80)
        self.declare_parameter("stop_after_progress_m", 0.0)
        route_file = str(self.get_parameter("route_file").value)
        if not route_file:
            raise RuntimeError("route_file or HEVEN_ROUTE_FILE must be set")
        with open(os.path.expanduser(route_file), encoding="utf-8") as stream:
            document = json.load(stream)
        self._route = document["points"]
        self._link_ranges = document.get("link_ranges", [])
        if len(self._route) < 2:
            raise RuntimeError("route must contain at least two points")
        self._origin_easting = float(self.get_parameter("origin_easting").value)
        self._origin_northing = float(self.get_parameter("origin_northing").value)
        self._lookahead = float(self.get_parameter("lookahead_m").value)
        self._wheelbase = float(self.get_parameter("wheelbase_m").value)
        self._max_wheel_angle = math.radians(
            float(self.get_parameter("max_wheel_angle_deg").value)
        )
        self._speed = float(self.get_parameter("speed_mps").value)
        self._timeout = float(self.get_parameter("sensor_timeout_sec").value)
        self._link_timeout = float(self.get_parameter("link_timeout_sec").value)
        self._nearest_search_ahead = int(
            self.get_parameter("nearest_search_ahead_points").value
        )
        self._stop_after_progress = float(
            self.get_parameter("stop_after_progress_m").value
        )
        if self._stop_after_progress < 0.0:
            raise RuntimeError("stop_after_progress_m cannot be negative")
        self._route_distances = cumulative_distances(self._route)
        self._position = None
        self._yaw = None
        self._gps_time = float("-inf")
        self._imu_time = float("-inf")
        self._link_id = None
        self._link_time = float("-inf")
        self._progress = None
        self._start_progress = None
        self._completed = False
        self._command_pub = self.create_publisher(Twist, "/vehicle/command", 10)
        self._gps_sub = self.create_subscription(NavSatFix, "/gps/fix", self._on_gps, 10)
        self._imu_sub = self.create_subscription(Imu, "/imu/data", self._on_imu, 10)
        self._link_sub = self.create_subscription(
            String, "/vehicle/status/link_id", self._on_link, 10
        )
        self._timer = self.create_timer(0.05, self._on_timer)
        self.get_logger().info(f"Loaded {len(self._route)} route points from {route_file}")

    def _on_gps(self, message: NavSatFix) -> None:
        if not math.isfinite(message.latitude) or not math.isfinite(message.longitude):
            return
        easting, northing = latlon_to_utm52(message.latitude, message.longitude)
        self._position = (
            easting - self._origin_easting,
            northing - self._origin_northing,
        )
        self._gps_time = time.monotonic()

    def _on_imu(self, message: Imu) -> None:
        q = message.orientation
        self._yaw = quaternion_yaw(q.w, q.x, q.y, q.z)
        self._imu_time = time.monotonic()

    def _on_link(self, message: String) -> None:
        self._link_id = message.data
        self._link_time = time.monotonic()

    def _initial_progress(self, x: float, y: float) -> int | None:
        if not self._link_ranges:
            return nearest_index(self._route, x, y)
        candidates = []
        for item in self._link_ranges:
            if item["link_id"] != self._link_id:
                continue
            candidates.extend(range(int(item["start_index"]), int(item["end_index"]) + 1))
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda index: (
                (self._route[index][0] - x) ** 2
                + (self._route[index][1] - y) ** 2
            ),
        )

    def _on_timer(self) -> None:
        command = Twist()
        now = time.monotonic()
        if (
            self._position is None
            or self._yaw is None
            or now - self._gps_time > self._timeout
            or now - self._imu_time > self._timeout
            or (
                self._link_ranges
                and (
                    self._link_id is None
                    or now - self._link_time > self._link_timeout
                )
            )
        ):
            self._command_pub.publish(command)
            return
        x, y = self._position
        if self._progress is None:
            self._progress = self._initial_progress(x, y)
            if self._progress is None:
                self.get_logger().error(
                    f"Current MORAI link {self._link_id!r} is not in this route; "
                    "refusing to move",
                    throttle_duration_sec=2.0,
                )
                self._command_pub.publish(command)
                return
            self._start_progress = self._progress
            self.get_logger().info(
                f"Initialized route progress at point {self._progress} "
                f"on link {self._link_id}"
            )
        else:
            self._progress = nearest_index(
                self._route,
                x,
                y,
                self._progress,
                max_search_ahead=self._nearest_search_ahead,
            )
        if self._completed:
            self._command_pub.publish(command)
            return
        if self._stop_after_progress > 0.0:
            travelled = (
                self._route_distances[self._progress]
                - self._route_distances[self._start_progress]
            )
            if travelled >= self._stop_after_progress:
                self._completed = True
                self.get_logger().warning(
                    f"Reached test distance {travelled:.1f} m; commanding stop"
                )
                self._command_pub.publish(command)
                return
        target_index = lookahead_index(self._route, self._progress, self._lookahead)
        target_x, target_y = self._route[target_index][:2]
        distance_to_end = math.hypot(
            self._route[-1][0] - x,
            self._route[-1][1] - y,
        )
        if target_index == len(self._route) - 1 and distance_to_end < 2.0:
            self._command_pub.publish(command)
            return
        command.linear.x = self._speed
        command.angular.z = normalized_pure_pursuit_steer(
            x=x,
            y=y,
            yaw=self._yaw,
            target_x=target_x,
            target_y=target_y,
            wheelbase=self._wheelbase,
            lookahead=self._lookahead,
            max_wheel_angle_rad=self._max_wheel_angle,
        )
        self._command_pub.publish(command)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RouteFollower()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
