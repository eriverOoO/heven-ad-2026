#!/usr/bin/env python3
"""Optional OpenPCDet CenterPoint ROS detector; Euclidean remains the default."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from autoware_perception_msgs.msg import (
    DetectedObject, DetectedObjectKinematics, DetectedObjects, ObjectClassification, Shape,
)
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

from ad_lidar_perception.centerpoint_ros import (
    CLASS_NAMES,
    detections_to_message,
    filter_detections,
    pointcloud2_to_xyzi,
    validate_backend,
)


MESSAGE_TYPES = {
    "DetectedObject": DetectedObject,
    "DetectedObjectKinematics": DetectedObjectKinematics,
    "DetectedObjects": DetectedObjects,
    "ObjectClassification": ObjectClassification,
    "Shape": Shape,
}


class CenterPointDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("ad_centerpoint_detector")
        self.declare_parameter("detector_backend", "euclidean")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("score_threshold", 0.1)
        self.declare_parameter("max_detections", 500)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("point_cloud_range", [-4.0, -25.0, -3.0, 100.0, 25.0, 5.0])
        self.declare_parameter("enabled", False)
        self.declare_parameter("mock_mode", False)
        self.declare_parameter("openpcdet_root", "")
        self.declare_parameter("input_topic", "/ad/perception/lidar/cropped")
        self.declare_parameter("output_topic", "/ad/perception/objects/detected")
        self.backend = validate_backend(str(self.get_parameter("detector_backend").value))
        self.enabled = bool(self.get_parameter("enabled").value)
        self.mock_mode = bool(self.get_parameter("mock_mode").value)
        if self.enabled and self.backend != "centerpoint":
            raise ValueError("this node may only be enabled for the centerpoint backend")
        self.model = None
        self.dataset = None
        self.torch = None
        if self.enabled and not self.mock_mode:
            self._load_model()
        self.publisher = self.create_publisher(
            DetectedObjects, str(self.get_parameter("output_topic").value), 10
        )
        self.subscription = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("input_topic").value),
            self._on_cloud,
            qos_profile_sensor_data,
        )

    def _load_model(self) -> None:
        import torch
        from easydict import EasyDict

        share = Path(get_package_share_directory("ad_lidar_perception"))
        tools = share / "centerpoint_offline"
        sys.path.insert(0, str(tools))
        from openpcdet_runtime import import_smoke_components
        from train_morai_centerpoint import DEFAULT_DATA_CONFIG, DEFAULT_MODEL_CONFIG, load_yaml_config

        root = Path(str(self.get_parameter("openpcdet_root").value))
        checkpoint_path = Path(str(self.get_parameter("checkpoint_path").value))
        if not root.is_dir() or not checkpoint_path.is_file():
            raise FileNotFoundError("openpcdet_root and checkpoint_path must exist")
        if not str(self.get_parameter("device").value).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("the pinned OpenPCDet CenterPoint runtime requires CUDA")
        data_cfg = EasyDict(load_yaml_config(DEFAULT_DATA_CONFIG))
        configured_range = list(self.get_parameter("point_cloud_range").value)
        if configured_range != list(data_cfg.POINT_CLOUD_RANGE):
            raise ValueError("point_cloud_range must match the pinned MORAI data config")
        model_cfg = EasyDict(load_yaml_config(DEFAULT_MODEL_CONFIG))
        dataset_template, centerpoint_class = import_smoke_components(root)

        class RosPointDataset(dataset_template):
            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError(index)

        self.dataset = RosPointDataset(data_cfg, list(CLASS_NAMES), training=False, root_path=None, logger=None)
        self.model = centerpoint_class(model_cfg.MODEL, len(CLASS_NAMES), self.dataset).cuda().eval()
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        torch.cuda.reset_peak_memory_stats()
        self.torch = torch

    def _on_cloud(self, message: PointCloud2) -> None:
        if not self.enabled:
            return
        callback_started = time.perf_counter()
        preprocessing_started = time.perf_counter()
        if message.header.frame_id != "lidar_link":
            self.get_logger().error("CenterPoint requires frame_id lidar_link")
            return
        points = pointcloud2_to_xyzi(message)
        preprocessing_ms = (time.perf_counter() - preprocessing_started) * 1000.0
        model_ms = 0.0
        raw = []
        if self.enabled and not self.mock_mode:
            sample = self.dataset.prepare_data({"frame_id": "ros", "points": points})
            batch = self.dataset.collate_batch([sample])
            from openpcdet_runtime import load_batch_to_cuda
            load_batch_to_cuda(batch)
            self.torch.cuda.synchronize()
            model_started = time.perf_counter()
            with self.torch.no_grad():
                predictions, _ = self.model(batch)
            self.torch.cuda.synchronize()
            model_ms = (time.perf_counter() - model_started) * 1000.0
            prediction = predictions[0]
            boxes = prediction["pred_boxes"].detach().cpu().numpy()
            scores = prediction["pred_scores"].detach().cpu().numpy()
            labels = prediction["pred_labels"].detach().cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                if not 1 <= int(label) <= len(CLASS_NAMES):
                    raise ValueError(f"unknown CenterPoint class ID {label}")
                raw.append({"class_name": CLASS_NAMES[int(label) - 1], "score": float(score), "box_lidar": box[:7]})
        post_started = time.perf_counter()
        filtered = filter_detections(
            raw,
            float(self.get_parameter("score_threshold").value),
            int(self.get_parameter("max_detections").value),
        )
        output = detections_to_message(header=message.header, detections=filtered, message_types=MESSAGE_TYPES)
        postprocessing_ms = (time.perf_counter() - post_started) * 1000.0
        self.publisher.publish(output)
        total_ms = (time.perf_counter() - callback_started) * 1000.0
        gpu_text = ""
        if self.torch is not None:
            gpu_text = (
                f" gpu_allocated_mib={self.torch.cuda.memory_allocated() / (1024 ** 2):.1f}"
                f" gpu_peak_mib={self.torch.cuda.max_memory_allocated() / (1024 ** 2):.1f}"
            )
        self.get_logger().info(
            f"centerpoint_latency_ms preprocessing={preprocessing_ms:.3f} "
            f"model_forward={model_ms:.3f} postprocessing={postprocessing_ms:.3f} "
            f"total={total_ms:.3f}{gpu_text}"
        )


def main() -> None:
    rclpy.init()
    node = CenterPointDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
