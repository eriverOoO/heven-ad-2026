"""Runtime-level checks for the disabled-by-default fusion node."""

import pytest

rclpy = pytest.importorskip("rclpy")

from ad_lidar_perception.camera_lidar_fusion_node import CameraLidarFusionNode


def test_disabled_node_is_optional_dependency_and_graph_neutral():
    rclpy.init()
    node = None
    try:
        node = CameraLidarFusionNode()
        assert node.enabled is False
        assert node._pub is None
        assert node._camera_subscription is None
        assert node._lidar_subscription is None
        assert node.frames_processed == 0
        assert node.output_messages == 0
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
