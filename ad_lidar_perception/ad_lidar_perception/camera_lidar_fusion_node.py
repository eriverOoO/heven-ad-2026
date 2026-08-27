"""POST-FREEZE EXTENSION 2, Stage 1: opt-in, EXPERIMENTAL, DISABLED-BY-DEFAULT
ROS2 node for camera+LiDAR late fusion.

    /ad/perception/objects/detected      (LiDAR 3D, unchanged, read-only)
  + /vision/dynamic_obstacle/detections  (camera 2D semantic, ad_camera_perception)
  ->  camera_lidar_fusion_core.fuse()
  ->  /experiment/perception/objects/camera_lidar_fused   (experimental)

This node does NOT modify the production graph, any detector, the tracker, or
`/ad/perception/objects/detected`. It is not launched by any production launch
file. `enabled` defaults to False.

Stage 4 hardening makes the disabled node dependency/graph-neutral, requires
an explicit audited calibration file when enabled, and applies the Stage-2
conservative geometry rule before fusion. A real bag smoke test still depends
on the host providing rosbag2_storage_mcap and vision_msgs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from ad_lidar_perception.camera_lidar_fusion_core import (
    CameraIntrinsics,
    CameraModel,
    FusionConfig,
    fuse,
    rigid_transform_from_tf_chain,
)
from ad_lidar_perception.camera_lidar_fusion_ros import (
    detected_objects_to_lidar_detections,
    detection2d_array_to_camera_detections,
    fused_result_to_detected_objects,
)
from ad_lidar_perception.camera_lidar_fusion_stage2_core import (
    GeometryFilterRule,
    passes_geometry_filter,
)


class CameraLidarFusionNode(Node):
    def __init__(self, **node_kwargs) -> None:
        super().__init__("camera_lidar_fusion", **node_kwargs)

        self.declare_parameter("enabled", False)
        self.declare_parameter("lidar_detections_topic", "/ad/perception/objects/detected")
        self.declare_parameter("camera_detections_topic", "/vision/dynamic_obstacle/detections")
        self.declare_parameter("fused_topic", "/experiment/perception/objects/camera_lidar_fused")
        self.declare_parameter("association_measure", "iou")
        self.declare_parameter("iou_gate", 0.10)
        self.declare_parameter("center_distance_gate", 0.10)
        self.declare_parameter("camera_min_confidence", 0.25)
        self.declare_parameter("max_sync_dt_ms", 50.0)
        self.declare_parameter("include_lidar_only", True)
        self.declare_parameter("conservative_geometry_filter_enabled", True)
        self.declare_parameter("geometry_min_dimension_m", 0.15)
        # Camera model: audited tf_static JSON + the three-hop frame chain.
        # For morai_cam4 the intrinsics remain explicitly reconstructed.
        self.declare_parameter("tf_static_json", "")
        self.declare_parameter("image_width", 1280)
        self.declare_parameter("image_height", 720)
        self.declare_parameter("horizontal_fov_deg", 90.0)
        self.declare_parameter("calibration_quality", "B_reconstructed")

        self.enabled = bool(self.get_parameter("enabled").value)
        self._pub = None
        self._camera_subscription = None
        self._lidar_subscription = None
        self.frames_processed = 0
        self.output_messages = 0
        self.frames_skipped_no_camera = 0
        self.frames_skipped_sync = 0
        self.last_fusion_latency_ms = 0.0

        # A disabled research feature must also be dependency- and graph-neutral.
        # In particular, do not import optional vision_msgs, read calibration,
        # or create publishers/subscribers in Mode 0.
        if not self.enabled:
            self.get_logger().warn(
                "camera_lidar_fusion is disabled (enabled:=false). Set enabled:=true to run."
            )
            return

        self._config = FusionConfig(
            association_measure=str(self.get_parameter("association_measure").value),
            iou_gate=float(self.get_parameter("iou_gate").value),
            center_distance_gate=float(self.get_parameter("center_distance_gate").value),
            camera_min_confidence=float(self.get_parameter("camera_min_confidence").value),
        )
        self._max_sync_ns = float(self.get_parameter("max_sync_dt_ms").value) * 1e6
        self._include_lidar_only = bool(self.get_parameter("include_lidar_only").value)
        self._geometry_filter_enabled = bool(
            self.get_parameter("conservative_geometry_filter_enabled").value
        )
        self._geometry_rule = GeometryFilterRule(
            "stage2_conservative_runtime",
            min_dimension_m=float(self.get_parameter("geometry_min_dimension_m").value),
        )
        self._camera_model = self._build_camera_model()

        self._latest_camera = None  # (stamp_ns, Detection2DArray)

        try:
            from autoware_perception_msgs.msg import (
                DetectedObject, DetectedObjects, ObjectClassification, Shape,
            )
            from vision_msgs.msg import Detection2DArray
        except ImportError as exc:  # pragma: no cover - environment dependent
            self.get_logger().error(
                f"camera_lidar_fusion needs autoware_perception_msgs + vision_msgs: {exc}"
            )
            raise

        self._msg = dict(
            DetectedObjects=DetectedObjects, DetectedObject=DetectedObject,
            ObjectClassification=ObjectClassification, Shape=Shape,
        )

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self._pub = self.create_publisher(
            DetectedObjects, str(self.get_parameter("fused_topic").value), qos
        )
        self._camera_subscription = self.create_subscription(
            Detection2DArray, str(self.get_parameter("camera_detections_topic").value),
            self._on_camera, qos,
        )
        self._lidar_subscription = self.create_subscription(
            DetectedObjects, str(self.get_parameter("lidar_detections_topic").value),
            self._on_lidar, qos,
        )
        self.get_logger().info(
            f"camera_lidar_fusion up: measure={self._config.association_measure} "
            f"intrinsics={self._camera_model.intrinsics.calibration_quality}"
        )

    def _build_camera_model(self) -> CameraModel:
        width = int(self.get_parameter("image_width").value)
        height = int(self.get_parameter("image_height").value)
        fov = float(self.get_parameter("horizontal_fov_deg").value)
        quality = str(self.get_parameter("calibration_quality").value)
        intr = CameraIntrinsics.from_fov(width, height, fov, calibration_quality=quality)

        tf_json = str(self.get_parameter("tf_static_json").value)
        calibration_path = Path(tf_json).expanduser() if tf_json else None
        if calibration_path is None or not calibration_path.is_file():
            raise RuntimeError(
                "enabled camera_lidar_fusion requires tf_static_json pointing to "
                "the audited calibration dump; identity fallback is prohibited"
            )
        try:
            with calibration_path.open(encoding="utf-8") as stream:
                tf = {item["child_frame"]: item for item in json.load(stream)}
            matrix = rigid_transform_from_tf_chain([
                {**tf["lidar_link"], "invert": False},
                {**tf["camera_front_link"], "invert": True},
                {**tf["camera_front_optical_frame"], "invert": True},
            ])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid tf_static_json calibration: {calibration_path}") from exc
        return CameraModel(intr, matrix)

    def _stamp_ns(self, header) -> int:
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    def _on_camera(self, message) -> None:
        self._latest_camera = (self._stamp_ns(message.header), message)

    def _on_lidar(self, message) -> None:
        if self._latest_camera is None:
            self.frames_skipped_no_camera += 1
            return
        lidar_stamp = self._stamp_ns(message.header)
        cam_stamp, cam_msg = self._latest_camera
        if abs(lidar_stamp - cam_stamp) > self._max_sync_ns:
            self.frames_skipped_sync += 1
            self.get_logger().warn("skipping frame: camera/LiDAR sync gap too large", throttle_duration_sec=5.0)
            return

        lidar_dets = detected_objects_to_lidar_detections(message)
        if self._geometry_filter_enabled:
            lidar_dets = [
                detection
                for detection in lidar_dets
                if passes_geometry_filter(detection, self._geometry_rule)
            ]
        camera_dets = detection2d_array_to_camera_detections(cam_msg)
        started = time.perf_counter()
        result = fuse(lidar_dets, camera_dets, self._camera_model, self._config)
        self.last_fusion_latency_ms = (time.perf_counter() - started) * 1000.0
        self.frames_processed += 1
        out = fused_result_to_detected_objects(
            result, message.header,
            detected_objects_cls=self._msg["DetectedObjects"],
            detected_object_cls=self._msg["DetectedObject"],
            object_classification_cls=self._msg["ObjectClassification"],
            shape_cls=self._msg["Shape"],
            include_lidar_only=self._include_lidar_only,
        )
        self._pub.publish(out)
        self.output_messages += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = CameraLidarFusionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
