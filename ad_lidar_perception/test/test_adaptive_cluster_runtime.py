import struct
import threading
import time
import unittest
from pathlib import Path

from autoware_perception_msgs.msg import DetectedObjects
from launch import LaunchDescription
import launch_testing
import launch_testing.actions
import launch_testing.asserts
from launch_ros.actions import Node
import pytest
from rcl_interfaces.srv import GetParameters
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node as RclpyNode
from rclpy.parameter import parameter_value_to_python
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
import yaml


INPUT_TOPIC = "/ad/perception/lidar/nonground_finite"
OBJECTS_TOPIC = "/ad/perception/objects/detected"
CLUSTERS_TOPIC = "/ad/perception/lidar/clusters"


@pytest.mark.launch_test
def generate_test_description():
    cluster_process = Node(
        package="ad_lidar_perception",
        executable="ad_adaptive_euclidean_cluster_node",
        name="ad_adaptive_cluster_runtime_fixture",
        output="screen",
    )
    return (
        LaunchDescription(
            [cluster_process, launch_testing.actions.ReadyToTest()]
        ),
        {"cluster_process": cluster_process},
    )


def _cloud(stamp, points):
    message = PointCloud2()
    message.header.frame_id = "lidar_link"
    message.header.stamp = stamp
    message.height = 1
    message.width = len(points)
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    message.point_step = 12
    message.row_step = message.point_step * message.width
    message.is_dense = True
    message.data = b"".join(struct.pack("<fff", *point) for point in points)
    return message


def _line_component(start_x, end_x, *, y=0.0, spacing=0.4):
    step_count = int(round((end_x - start_x) / spacing))
    assert step_count > 0
    return [
        (start_x + spacing * index, y, 0.2)
        for index in range(step_count + 1)
    ]


def _stamp_key(message_or_stamp):
    stamp = getattr(message_or_stamp, "header", None)
    stamp = stamp.stamp if stamp is not None else message_or_stamp
    return stamp.sec, stamp.nanosec


def _intensity_ids(message):
    intensity_field = next(
        field for field in message.fields if field.name == "intensity"
    )
    byte_order = ">" if message.is_bigendian else "<"
    return {
        struct.unpack_from(
            f"{byte_order}f",
            message.data,
            index * message.point_step + intensity_field.offset,
        )[0]
        for index in range(message.width * message.height)
    }


class ClusterRuntimeDriver(RclpyNode):
    def __init__(self, context):
        super().__init__("adaptive_cluster_runtime_driver", context=context)
        self.objects = []
        self.clusters = []
        self.lock = threading.Lock()
        self.publisher = self.create_publisher(
            PointCloud2, INPUT_TOPIC, qos_profile_sensor_data
        )
        self.create_subscription(
            DetectedObjects, OBJECTS_TOPIC, self._on_objects, 10
        )
        self.create_subscription(
            PointCloud2, CLUSTERS_TOPIC, self._on_clusters, 10
        )
        self.parameter_client = self.create_client(
            GetParameters,
            "/ad_adaptive_cluster_runtime_fixture/get_parameters",
        )

    def _on_objects(self, message):
        with self.lock:
            self.objects.append(message)

    def _on_clusters(self, message):
        with self.lock:
            self.clusters.append(message)

    def snapshot(self):
        with self.lock:
            return list(self.objects), list(self.clusters)

    def get_remote_parameters(self, names):
        if not self.parameter_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("adaptive cluster parameter service unavailable")
        request = GetParameters.Request()
        request.names = names
        future = self.parameter_client.call_async(request)
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            raise RuntimeError("adaptive cluster parameter request timed out")
        response = future.result()
        return {
            name: parameter_value_to_python(value)
            for name, value in zip(names, response.values)
        }


class TestAdaptiveClusterRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = Context()
        rclpy.init(context=cls.context)
        cls.driver = ClusterRuntimeDriver(cls.context)
        cls.executor = SingleThreadedExecutor(context=cls.context)
        cls.executor.add_node(cls.driver)
        cls.spin_thread = threading.Thread(
            target=cls.executor.spin, daemon=True
        )
        cls.spin_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.executor.shutdown(timeout_sec=3.0)
        cls.spin_thread.join(timeout=3.0)
        cls.executor.remove_node(cls.driver)
        cls.driver.destroy_node()
        rclpy.shutdown(context=cls.context)

    def _publish_and_wait(self, points):
        stamp = self.driver.get_clock().now().to_msg()
        expected_key = _stamp_key(stamp)
        message = _cloud(stamp, points)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self.driver.publisher.publish(message)
            objects, clusters = self.driver.snapshot()
            matching_objects = [
                output
                for output in objects
                if _stamp_key(output) == expected_key
            ]
            matching_clusters = [
                output
                for output in clusters
                if _stamp_key(output) == expected_key
            ]
            if matching_objects and matching_clusters:
                return matching_objects[-1], matching_clusters[-1]
            time.sleep(0.05)
        self.fail("matching detected and debug outputs were not published")

    def test_checked_in_yaml_matches_measured_d0_production_profile(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "clustering"
            / "adaptive_euclidean_cluster.yaml"
        )
        parameters = yaml.safe_load(config_path.read_text())["/**"][
            "ros__parameters"
        ]

        self.assertEqual(
            parameters,
            {
                "input_topic": INPUT_TOPIC,
                "objects_topic": OBJECTS_TOPIC,
                "clusters_topic": CLUSTERS_TOPIC,
                "near_tolerance_m": 0.45,
                "far_tolerance_m": 1.60,
                "far_range_m": 45.0,
                "minimum_points_far_range_m": 45.0,
                "near_min_cluster_size": 5,
                "far_min_cluster_size": 2,
                "max_cluster_size": 20000,
                "maximum_dynamic_object_diagonal_m": 12.0,
                "use_height": False,
                "crop": {
                    "min_x_m": -4.0,
                    "max_x_m": 100.0,
                    "min_y_m": -25.0,
                    "max_y_m": 25.0,
                    "min_z_m": -1.0,
                    "max_z_m": 3.0,
                },
            },
        )

    def test_direct_node_defaults_match_measured_d0_production_profile(self):
        names = [
            "input_topic",
            "objects_topic",
            "clusters_topic",
            "near_tolerance_m",
            "far_tolerance_m",
            "far_range_m",
            "minimum_points_far_range_m",
            "near_min_cluster_size",
            "far_min_cluster_size",
            "max_cluster_size",
            "maximum_dynamic_object_diagonal_m",
            "use_height",
            "crop.min_x_m",
            "crop.max_x_m",
            "crop.min_y_m",
            "crop.max_y_m",
            "crop.min_z_m",
            "crop.max_z_m",
        ]

        self.assertEqual(
            self.driver.get_remote_parameters(names),
            {
                "input_topic": INPUT_TOPIC,
                "objects_topic": OBJECTS_TOPIC,
                "clusters_topic": CLUSTERS_TOPIC,
                "near_tolerance_m": 0.45,
                "far_tolerance_m": 1.60,
                "far_range_m": 45.0,
                "minimum_points_far_range_m": 45.0,
                "near_min_cluster_size": 5,
                "far_min_cluster_size": 2,
                "max_cluster_size": 20000,
                "maximum_dynamic_object_diagonal_m": 12.0,
                "use_height": False,
                "crop.min_x_m": -4.0,
                "crop.max_x_m": 100.0,
                "crop.min_y_m": -25.0,
                "crop.max_y_m": 25.0,
                "crop.min_z_m": -1.0,
                "crop.max_z_m": 3.0,
            },
        )

    def test_far_sparse_returns_reach_both_outputs(self):
        points = [
            (60.0, 0.0, 0.2),
            (60.0, 0.65, 0.2),
            (60.0, 1.30, 0.2),
        ]
        objects, clusters = self._publish_and_wait(points)

        self.assertEqual(len(objects.objects), 1)
        self.assertEqual(clusters.width, len(points))
        self.assertIn("intensity", [field.name for field in clusters.fields])

    def test_long_structure_stays_one_debug_component_but_not_detection(self):
        structure = _line_component(30.0, 50.0)

        objects, clusters = self._publish_and_wait(structure)

        self.assertEqual(objects.header, clusters.header)
        self.assertEqual(objects.objects, [])
        self.assertEqual(clusters.width, len(structure))
        self.assertEqual(_intensity_ids(clusters), {1.0})

    def test_vehicle_sized_component_is_detected_beside_debug_only_structure(self):
        vehicle = _line_component(10.0, 19.6)
        structure = _line_component(30.0, 50.0)

        objects, clusters = self._publish_and_wait(vehicle + structure)

        self.assertEqual(len(objects.objects), 1)
        self.assertAlmostEqual(objects.objects[0].shape.dimensions.x, 9.6, places=4)
        self.assertEqual(clusters.width, len(vehicle) + len(structure))
        self.assertEqual(_intensity_ids(clusters), {1.0, 2.0})

    def test_published_debug_box_dimensions_control_dynamic_threshold(self):
        raw_extent_at_threshold = _line_component(10.0, 22.0)

        objects, clusters = self._publish_and_wait(raw_extent_at_threshold)

        self.assertEqual(objects.objects, [])
        self.assertEqual(_intensity_ids(clusters), {1.0})

    def test_connected_vehicle_and_structure_is_not_heuristically_split(self):
        absorbed_vehicle = _line_component(10.0, 19.6)
        connected_structure = _line_component(20.0, 30.0)

        objects, clusters = self._publish_and_wait(
            absorbed_vehicle + connected_structure
        )

        self.assertEqual(objects.objects, [])
        self.assertEqual(_intensity_ids(clusters), {1.0})


@launch_testing.post_shutdown_test()
class TestAdaptiveClusterShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info, cluster_process):
        launch_testing.asserts.assertExitCodes(
            proc_info, process=cluster_process
        )
