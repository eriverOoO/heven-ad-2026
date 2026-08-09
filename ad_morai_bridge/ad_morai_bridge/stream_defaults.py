from dataclasses import dataclass


@dataclass(frozen=True)
class StreamDefaults:
    enabled: bool
    port: int
    topic: str
    frame_id: str
    mode: str
    receive_buffer_bytes: int = 4 * 1024 * 1024


STREAMS = {
    "competition_status": StreamDefaults(
        True, 1909, "/ad/vehicle/status", "map", "ego"
    ),
    "collisions": StreamDefaults(
        True, 9092, "/ad/safety/collisions", "map", "collisions"
    ),
    "camera_front": StreamDefaults(
        True, 9291, "/ad/sensors/camera/front/compressed",
        "camera_front_optical_frame", "camera", 16 * 1024 * 1024,
    ),
    "camera_left": StreamDefaults(
        True, 9293, "/ad/sensors/camera/left/compressed",
        "camera_left_optical_frame", "camera", 16 * 1024 * 1024,
    ),
    "camera_right": StreamDefaults(
        True, 9295, "/ad/sensors/camera/right/compressed",
        "camera_right_optical_frame", "camera", 16 * 1024 * 1024,
    ),
    "camera_traffic_light": StreamDefaults(
        True, 9307, "/ad/sensors/camera/traffic_light/compressed",
        "camera_traffic_light_optical_frame", "camera", 16 * 1024 * 1024,
    ),
    "gps": StreamDefaults(
        True, 9297, "/ad/sensors/gps/fix", "gps_link", "gps"
    ),
    "imu": StreamDefaults(
        True, 9299, "/ad/sensors/imu/data", "imu_link", "imu"
    ),
    "velodyne": StreamDefaults(
        True, 2368, "/ad/sensors/lidar/raw", "lidar_link", "velodyne",
        16 * 1024 * 1024,
    ),
}
