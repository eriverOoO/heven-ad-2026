import copy
from importlib.util import module_from_spec, spec_from_file_location
import math
from pathlib import Path
import struct
import subprocess
import threading
import time
import unittest
import xml.etree.ElementTree as ET

from ad_interfaces.msg import (
    PredictedObject,
    PredictedObjectArray,
    PredictedState,
)
from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from geometry_msgs.msg import TransformStamped
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing
import launch_testing.actions
import launch_testing.asserts
from launch_ros.actions import Node as LaunchNode
from nav_msgs.msg import OccupancyGrid
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.serialization import serialize_message
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE / "config"
LAUNCH = PACKAGE / "launch"
CASE_NAMES = ("static_combined", "dynamic")
CASE_DOMAINS = {"static_combined": 93, "dynamic": 94}
TEST_PREDICTION_TIMEOUT_SEC = 2.0


def _installed_launch(name):
    return PythonLaunchDescriptionSource(
        str(
            Path(get_package_share_directory("ad_lidar_perception"))
            / "launch"
            / name
        )
    )


@pytest.mark.launch_test
@launch_testing.parametrize("case_name", CASE_NAMES)
def generate_test_description(case_name):
    if case_name == "static_combined":
        actions = [
            IncludeLaunchDescription(_installed_launch("occupancy_grid.launch.py")),
            IncludeLaunchDescription(
                _installed_launch("combined_occupancy_grid.launch.py")
            ),
        ]
    else:
        actions = [
            LaunchNode(
                package="ad_lidar_perception",
                executable="ad_dynamic_occupancy_grid_node",
                name="ad_dynamic_occupancy_grid",
                output="screen",
                parameters=[
                    str(
                        Path(get_package_share_directory("ad_lidar_perception"))
                        / "config"
                        / "occupancy_grid"
                        / "dynamic.yaml"
                    ),
                    {
                        "prediction_timeout_sec": (
                            TEST_PREDICTION_TIMEOUT_SEC
                        )
                    },
                ],
            )
        ]
    return (
        LaunchDescription(
            [
                SetEnvironmentVariable(
                    "ROS_DOMAIN_ID", str(CASE_DOMAINS[case_name])
                ),
                *actions,
                launch_testing.actions.ReadyToTest(),
            ]
        ),
        {"case_name": case_name},
    )


class RecordingNode:
    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(self)


def _load_launch(name):
    path = LAUNCH / name
    spec = spec_from_file_location(name.replace(".", "_"), path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _record_leaf(name):
    module = _load_launch(name)
    RecordingNode.calls.clear()
    module.Node = RecordingNode
    description = module.generate_launch_description()
    arguments = {
        entity.name
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert "namespace" not in arguments
    assert len(RecordingNode.calls) == 1
    node = RecordingNode.calls[0].kwargs
    assert "namespace" not in node
    return node


def _declared_launch_arguments(name):
    return {
        entity.name
        for entity in _load_launch(name).generate_launch_description().entities
        if isinstance(entity, DeclareLaunchArgument)
    }


def _parameters(path, node_name):
    return yaml.safe_load(path.read_text(encoding="utf-8"))[node_name][
        "ros__parameters"
    ]


def _assert_static_configuration_preserves_geometric_contract():
    static = _parameters(
        CONFIG / "occupancy_grid" / "static.yaml", "ad_lidar_perception"
    )
    assert static == {
        "target_frame": "base_link",
        "x_min": -4.0,
        "x_max": 100.0,
        "y_min": -10.0,
        "y_max": 10.0,
        "z_min": 0.1,
        "z_max": 2.0,
        "resolution": 0.1,
        "inflation_radius_m": 1.8,
        "inflation_cost_scaling_factor": 2.0,
        "ego_clear_x_min": -1.0,
        "ego_clear_x_max": 4.05,
        "ego_clear_y_min": -1.15,
        "ego_clear_y_max": 1.15,
        "transform_timeout_sec": 0.05,
        "persistence.duration_sec": 0.5,
        "persistence.fixed_frame": "odom",
        "persistence.maximum_clouds": 8,
        "road_gate.enabled": True,
        "road_gate.maximum_pending_messages": 8,
        "topics.points": "/ad/sensors/lidar/points",
        "topics.drivable_mask": "/ad/planning/drivable_mask",
        "topics.occupancy_grid": "/ad/perception/occupancy/static",
        "visualization.topics.static_ungated": (
            "/ad/viz/perception/occupancy/static_ungated"
        ),
    }


def _assert_leaf_launch_contracts():
    static = _record_leaf("occupancy_grid.launch.py")
    assert static["package"] == "ad_lidar_perception"
    assert static["executable"] == "ad_lidar_perception_node"
    assert static["name"] == "ad_lidar_perception"
    assert _declared_launch_arguments("occupancy_grid.launch.py") == {
        "points_topic",
        "drivable_mask_topic",
        "static_ungated_topic",
    }
    assert str(static["parameters"][0]).endswith(
        "config/occupancy_grid/static.yaml"
    )
    assert static["parameters"][1].keys() == {
        "topics.points",
        "topics.drivable_mask",
        "visualization.topics.static_ungated",
    }

    dynamic = _record_leaf("dynamic_occupancy_grid.launch.py")
    assert dynamic["package"] == "ad_lidar_perception"
    assert dynamic["executable"] == "ad_dynamic_occupancy_grid_node"
    assert dynamic["name"] == "ad_dynamic_occupancy_grid"
    assert len(dynamic["parameters"]) == 2
    assert str(dynamic["parameters"][0]).endswith(
        "config/occupancy_grid/dynamic.yaml"
    )
    assert dynamic["parameters"][1].keys() == {"topics.drivable_mask"}

    combined = _record_leaf("combined_occupancy_grid.launch.py")
    assert combined["package"] == "ad_lidar_perception"
    assert combined["executable"] == "ad_combined_occupancy_grid_node"
    assert combined["name"] == "ad_combined_occupancy_grid"
    assert len(combined["parameters"]) == 1
    assert str(combined["parameters"][0]).endswith(
        "config/occupancy_grid/combined.yaml"
    )

    assert _parameters(
        CONFIG / "occupancy_grid" / "dynamic.yaml", "ad_dynamic_occupancy_grid"
    ) == {
        "target_frame": "base_link",
        "source_frame": "odom",
        "transform_timeout_sec": 0.05,
        "prediction_timeout_sec": 0.50,
        "stale_check_period_sec": 0.10,
        "x_min": -4.0,
        "x_max": 100.0,
        "y_min": -10.0,
        "y_max": 10.0,
        "resolution": 0.1,
        "covariance_sigma": 2.0,
        "minimum_inflation_m": 0.20,
        "occupied_cost": 100,
        "maximum_cells_per_object": 20000,
        "road_gate.enabled": True,
        "road_gate.maximum_pending_messages": 8,
        "topics.predicted_objects": "/ad/perception/objects/predicted",
        "topics.drivable_mask": "/ad/planning/drivable_mask",
        "topics.dynamic_grid": "/ad/perception/occupancy/dynamic",
    }
    assert _parameters(
        CONFIG / "occupancy_grid" / "combined.yaml", "ad_combined_occupancy_grid"
    ) == {
        "maximum_pending_messages": 8,
        "topics.static_grid": "/ad/perception/occupancy/static",
        "topics.dynamic_grid": "/ad/perception/occupancy/dynamic",
        "topics.combined_grid": "/ad/perception/occupancy/combined",
        "topics.compatibility_grid": "/ad/perception/occupancy_grid",
    }


def _assert_manifest_dependencies():
    package = ET.parse(PACKAGE / "package.xml").getroot()
    dependencies = {
        element.text
        for element in package
        if element.tag in {"depend", "build_depend", "exec_depend"}
    }
    assert {
        "ad_interfaces",
        "geometry_msgs",
        "nav_msgs",
        "rclcpp",
        "sensor_msgs",
        "tf2",
        "tf2_geometry_msgs",
        "tf2_ros",
        "tf2_sensor_msgs",
    } <= dependencies


def _stamp_ns(message):
    return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec


def _defined_cdr_bytes(message):
    serialized = bytearray(serialize_message(message))
    # Fast-CDR leaves alignment padding unspecified. Canonicalize only the
    # padding following Header.frame_id so every defined serialized byte is
    # still compared.
    frame_length = struct.unpack_from("<I", serialized, 12)[0]
    frame_end = 16 + frame_length
    metadata_start = (frame_end + 3) & ~3
    serialized[frame_end:metadata_start] = b"\x00" * (
        metadata_start - frame_end
    )
    origin_unaligned = metadata_start + 20
    origin_start = 4 + (((origin_unaligned - 4) + 7) & ~7)
    serialized[origin_unaligned:origin_start] = b"\x00" * (
        origin_start - origin_unaligned
    )
    return bytes(serialized)


def _time_message(nanoseconds):
    return Time(nanoseconds=nanoseconds).to_msg()


def _cloud(stamp_ns, frame_id="base_link", malformed=False):
    message = PointCloud2()
    message.header.stamp = _time_message(stamp_ns)
    message.header.frame_id = frame_id
    message.height = 1
    message.width = 1
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = 12
    message.data = list(struct.pack("<fff", 5.0, 0.0, 2.1))
    message.is_dense = True
    if malformed:
        message.row_step = 11
    return message


def _empty_cloud(stamp_ns):
    message = _cloud(stamp_ns)
    message.width = 0
    message.row_step = 0
    message.data = []
    return message


def _occupied_cloud(stamp_ns):
    message = _cloud(stamp_ns)
    message.data = list(struct.pack("<fff", 20.0, 0.0, 1.0))
    return message


def _padded_cloud(stamp_ns):
    message = _cloud(stamp_ns)
    message.height = 2
    message.row_step = 16
    point = struct.pack("<fff", 5.0, 0.0, 2.0)
    message.data = list(point + b"\x00" * 4 + point + b"\x00" * 4)
    return message


def _dynamic_grid(stamp_ns, occupied_index=123):
    message = OccupancyGrid()
    message.header.stamp = _time_message(stamp_ns)
    message.header.frame_id = "base_link"
    message.info.resolution = 0.1
    message.info.width = 1040
    message.info.height = 200
    message.info.origin.position.x = -4.0
    message.info.origin.position.y = -10.0
    message.info.origin.orientation.w = 1.0
    message.data = [0] * (message.info.width * message.info.height)
    if occupied_index is not None:
        message.data[occupied_index] = 100
    return message


def _drivable_mask(stamp_ns):
    message = _dynamic_grid(stamp_ns, occupied_index=None)
    message.data = [0] * len(message.data)
    return message


def _predicted_array(stamp_ns):
    message = PredictedObjectArray()
    message.header.stamp = _time_message(stamp_ns)
    message.header.frame_id = "odom"

    obj = PredictedObject()
    obj.existence_probability = 0.0
    obj.classification = PredictedObject.UNKNOWN
    obj.classification_probability = 0.0
    obj.dimensions.x = 1.0
    obj.dimensions.y = 1.0
    obj.dimensions.z = 1.0
    obj.initial_pose.pose.position.x = 2.0
    obj.initial_pose.pose.orientation.w = 1.0
    for seconds, x in ((0.5, 3.0), (1.0, 4.0)):
        state = PredictedState()
        state.time_from_start.sec = int(seconds)
        state.time_from_start.nanosec = int(
            round((seconds - int(seconds)) * 1_000_000_000)
        )
        state.pose.pose.position.x = x
        state.pose.pose.orientation.w = 1.0
        obj.states.append(state)
    message.objects.append(obj)
    return message


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class OccupancyDriver(Node):
    def __init__(self, case_name, context):
        super().__init__(
            f"occupancy_layer_contract_{case_name}", context=context
        )
        self.case_name = case_name
        self.lock = threading.Lock()
        self.static = {}
        self.ungated = {}
        self.combined = {}
        self.alias = {}
        self.dynamic = []

        self.points_publisher = self.create_publisher(
            PointCloud2,
            "/ad/sensors/lidar/points",
            qos_profile_sensor_data,
        )
        self.synthetic_dynamic_publisher = self.create_publisher(
            OccupancyGrid,
            "/ad/perception/occupancy/dynamic",
            qos_profile_sensor_data,
        )
        self.drivable_mask_publisher = self.create_publisher(
            OccupancyGrid,
            "/ad/planning/drivable_mask",
            qos_profile_sensor_data,
        )
        prediction_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.prediction_publisher = self.create_publisher(
            PredictedObjectArray,
            "/ad/perception/objects/predicted",
            prediction_qos,
        )
        self.create_subscription(
            OccupancyGrid,
            "/ad/perception/occupancy/static",
            lambda message: self._store(self.static, message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            OccupancyGrid,
            "/ad/viz/perception/occupancy/static_ungated",
            lambda message: self._store(self.ungated, message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            OccupancyGrid,
            "/ad/perception/occupancy/combined",
            lambda message: self._store(self.combined, message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            OccupancyGrid,
            "/ad/perception/occupancy_grid",
            lambda message: self._store(self.alias, message),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            OccupancyGrid,
            "/ad/perception/occupancy/dynamic",
            self._store_dynamic,
            qos_profile_sensor_data,
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = None
        if case_name == "static_combined":
            self.static_tf_broadcaster = StaticTransformBroadcaster(self)
            transform = TransformStamped()
            transform.header.frame_id = "odom"
            transform.child_frame_id = "base_link"
            transform.transform.rotation.w = 1.0
            self.static_tf_broadcaster.sendTransform(transform)

    def _store(self, destination, message):
        with self.lock:
            destination[_stamp_ns(message)] = message

    def _store_dynamic(self, message):
        with self.lock:
            self.dynamic.append(message)

    def wait_for_graph(self):
        if self.case_name == "static_combined":
            return _wait_until(
                lambda: (
                    self.points_publisher.get_subscription_count() == 1
                    and self.drivable_mask_publisher.get_subscription_count()
                    == 1
                    and self.synthetic_dynamic_publisher.get_subscription_count()
                    == 2
                    and self.count_publishers(
                        "/ad/perception/occupancy/static"
                    )
                    == 1
                    and self.count_publishers(
                        "/ad/viz/perception/occupancy/static_ungated"
                    )
                    == 1
                    and self.count_publishers(
                        "/ad/perception/occupancy/combined"
                    )
                    == 1
                    and self.count_publishers("/ad/perception/occupancy_grid")
                    == 1
                ),
                timeout=5.0,
            )
        return _wait_until(
            lambda: (
                self.prediction_publisher.get_subscription_count() == 1
                and self.drivable_mask_publisher.get_subscription_count() == 1
                and self.count_publishers(
                    "/ad/perception/occupancy/dynamic"
                )
                >= 1
            ),
            timeout=5.0,
        )

    def graph_snapshot(self):
        return {
            "points_subscribers": self.points_publisher.get_subscription_count(),
            "drivable_mask_subscribers": (
                self.drivable_mask_publisher.get_subscription_count()
            ),
            "dynamic_subscribers": (
                self.synthetic_dynamic_publisher.get_subscription_count()
            ),
            "prediction_subscribers": (
                self.prediction_publisher.get_subscription_count()
            ),
            "static_publishers": self.count_publishers(
                "/ad/perception/occupancy/static"
            ),
            "static_ungated_publishers": self.count_publishers(
                "/ad/viz/perception/occupancy/static_ungated"
            ),
            "dynamic_publishers": self.count_publishers(
                "/ad/perception/occupancy/dynamic"
            ),
            "combined_publishers": self.count_publishers(
                "/ad/perception/occupancy/combined"
            ),
            "alias_publishers": self.count_publishers(
                "/ad/perception/occupancy_grid"
            ),
        }

    def wait_for_layers(self, stamp_ns, timeout=3.0):
        return _wait_until(
            lambda: all(
                stamp_ns in messages
                for messages in (
                    self.static,
                    self.ungated,
                    self.combined,
                    self.alias,
                )
            ),
            timeout=timeout,
        )

    def wait_for_static_layers(self, stamp_ns, timeout=3.0):
        return _wait_until(
            lambda: all(
                stamp_ns in messages
                for messages in (self.static, self.ungated)
            ),
            timeout=timeout,
        )

    def wait_for_combined_layers(self, stamp_ns, timeout=3.0):
        return _wait_until(
            lambda: all(
                stamp_ns in messages
                for messages in (self.combined, self.alias)
            ),
            timeout=timeout,
        )

    def publish_dynamic(self, dynamic):
        for _ in range(3):
            self.synthetic_dynamic_publisher.publish(dynamic)
            time.sleep(0.02)

    def publish_dynamic_then_cloud(self, dynamic, cloud):
        self.publish_dynamic(dynamic)
        self.publish_cloud(cloud)

    def publish_clear_then_cloud(self, cloud, mask=None):
        self.publish_dynamic(
            _dynamic_grid(_stamp_ns(cloud), occupied_index=None)
        )
        self.publish_cloud(cloud, mask=mask)

    def publish_cloud(self, cloud, mask=None):
        if mask is None:
            mask = _drivable_mask(_stamp_ns(cloud))
        self.drivable_mask_publisher.publish(mask)
        self.points_publisher.publish(cloud)

    def publish_prediction(self, prediction, mask=None):
        if mask is None:
            mask = _drivable_mask(_stamp_ns(prediction))
        self.drivable_mask_publisher.publish(mask)
        self.prediction_publisher.publish(prediction)

    def publish_transform(self, stamp_ns):
        transform = TransformStamped()
        transform.header.stamp = _time_message(stamp_ns)
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.rotation.w = 1.0
        for _ in range(3):
            self.tf_broadcaster.sendTransform(transform)
            time.sleep(0.03)


class TestOccupancyLayers(unittest.TestCase):
    def test_contract(self, case_name):
        _assert_static_configuration_preserves_geometric_contract()
        _assert_leaf_launch_contracts()
        _assert_manifest_dependencies()

        context = Context()
        rclpy.init(
            context=context, domain_id=CASE_DOMAINS[case_name]
        )
        driver = OccupancyDriver(case_name, context)
        executor = MultiThreadedExecutor(num_threads=2, context=context)
        executor.add_node(driver)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        shutdown_complete = False
        spin_thread_stopped = False
        try:
            self.assertTrue(
                driver.wait_for_graph(),
                f"occupancy graph did not match: {driver.graph_snapshot()}",
            )
            if case_name == "static_combined":
                self._test_static_combined(driver)
            else:
                self._test_dynamic(driver)
        finally:
            try:
                shutdown_complete = executor.shutdown(timeout_sec=2.0)
            finally:
                try:
                    spin_thread.join(timeout=2.0)
                    spin_thread_stopped = not spin_thread.is_alive()
                finally:
                    try:
                        driver.destroy_node()
                    finally:
                        if context.ok():
                            context.shutdown()
        self.assertTrue(
            shutdown_complete, "occupancy test executor did not shut down"
        )
        self.assertTrue(
            spin_thread_stopped, "occupancy executor thread leaked"
        )

    def _test_static_combined(self, driver):
        first_stamp = driver.get_clock().now().nanoseconds
        driver.points_publisher.publish(_cloud(first_stamp))
        time.sleep(0.25)
        self.assertNotIn(
            first_stamp,
            driver.static,
            "road-gated static OGM published without an exact-stamp mask",
        )
        self.assertNotIn(
            first_stamp,
            driver.ungated,
            "ungated debug OGM published without its planning pair",
        )
        first_static_ready = False
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            driver.publish_cloud(_cloud(first_stamp))
            if driver.wait_for_static_layers(first_stamp, timeout=0.1):
                first_static_ready = True
                break
        self.assertTrue(first_static_ready)
        time.sleep(0.25)
        self.assertNotIn(
            first_stamp,
            driver.combined,
            "combined OGM published without an exact-stamp dynamic layer",
        )
        self.assertNotIn(first_stamp, driver.alias)

        # A matching dynamic layer arriving after the static layer must
        # complete the pair immediately; otherwise late tracker output leaves
        # the planner blocked until another LiDAR scan.
        driver.publish_dynamic(
            _dynamic_grid(first_stamp, occupied_index=None)
        )
        self.assertTrue(driver.wait_for_combined_layers(first_stamp))
        static = driver.static[first_stamp]
        combined = driver.combined[first_stamp]
        alias = driver.alias[first_stamp]
        self.assertEqual(_defined_cdr_bytes(static), _defined_cdr_bytes(combined))
        self.assertEqual(_defined_cdr_bytes(combined), _defined_cdr_bytes(alias))
        self.assertTrue(all(cell == 0 for cell in static.data))

        compatible_stamp = driver.get_clock().now().nanoseconds
        dynamic = _dynamic_grid(compatible_stamp)
        driver.publish_dynamic_then_cloud(
            dynamic, _cloud(compatible_stamp)
        )
        self.assertTrue(driver.wait_for_layers(compatible_stamp))
        compatible_static = driver.static[compatible_stamp]
        compatible_combined = driver.combined[compatible_stamp]
        compatible_alias = driver.alias[compatible_stamp]
        self.assertEqual(compatible_static.data[123], 0)
        self.assertEqual(compatible_combined.data[123], 100)
        differing_cells = [
            (index, combined_cell, alias_cell)
            for index, (combined_cell, alias_cell) in enumerate(
                zip(compatible_combined.data, compatible_alias.data)
            )
            if combined_cell != alias_cell
        ][:10]
        self.assertEqual(
            _defined_cdr_bytes(compatible_combined),
            _defined_cdr_bytes(compatible_alias),
            (
                f"headers={compatible_combined.header!r},"
                f"{compatible_alias.header!r}; "
                f"info={compatible_combined.info!r},"
                f"{compatible_alias.info!r}; "
                f"differing_cells={differing_cells}"
            ),
        )
        self.assertEqual(
            compatible_combined.header, compatible_static.header
        )
        self.assertEqual(compatible_combined.info, compatible_static.info)

        retry_stamp = driver.get_clock().now().nanoseconds
        clear = _dynamic_grid(retry_stamp, occupied_index=None)
        driver.publish_dynamic(clear)
        corrected = _dynamic_grid(retry_stamp, occupied_index=124)
        driver.publish_dynamic_then_cloud(corrected, _cloud(retry_stamp))
        self.assertTrue(driver.wait_for_layers(retry_stamp))
        self.assertEqual(
            driver.combined[retry_stamp].data[124],
            100,
            "an equal-stamp explicit clear blocked the corrected retry",
        )

        watermark_stamp = driver.get_clock().now().nanoseconds
        watermark_clear = _dynamic_grid(
            watermark_stamp, occupied_index=None
        )
        driver.publish_dynamic(watermark_clear)
        delayed_occupied = _dynamic_grid(
            watermark_stamp - 1, occupied_index=125
        )
        driver.publish_dynamic_then_cloud(
            delayed_occupied, _cloud(watermark_stamp)
        )
        self.assertTrue(driver.wait_for_layers(watermark_stamp))
        self.assertEqual(
            driver.combined[watermark_stamp].data[125],
            0,
            "an occupied sample older than the clear watermark was replayed",
        )

        rollback_stamp = driver.get_clock().now().nanoseconds
        rollback_occupied = _dynamic_grid(
            rollback_stamp, occupied_index=126
        )
        driver.publish_dynamic(rollback_occupied)
        rollback_clear = _dynamic_grid(
            rollback_stamp - 1, occupied_index=None
        )
        driver.publish_dynamic_then_cloud(
            rollback_clear, _cloud(rollback_stamp)
        )
        self.assertTrue(driver.wait_for_layers(rollback_stamp))
        self.assertEqual(
            driver.combined[rollback_stamp].data[126],
            100,
            "a stale clear invalidated a newer exact-stamp occupied layer",
        )

        malformed_stamp = driver.get_clock().now().nanoseconds
        malformed = _dynamic_grid(malformed_stamp)
        malformed.data = [100]
        driver.publish_dynamic_then_cloud(malformed, _cloud(malformed_stamp))
        self.assertTrue(driver.wait_for_static_layers(malformed_stamp))
        time.sleep(0.25)
        self.assertNotIn(
            malformed_stamp,
            driver.combined,
            "malformed dynamic input must fail closed",
        )
        driver.publish_dynamic(
            _dynamic_grid(malformed_stamp, occupied_index=None)
        )
        self.assertTrue(driver.wait_for_combined_layers(malformed_stamp))

        geometry_stamp = driver.get_clock().now().nanoseconds
        wrong_geometry = _dynamic_grid(geometry_stamp)
        wrong_geometry.info.resolution = 0.2
        driver.publish_dynamic_then_cloud(
            wrong_geometry, _cloud(geometry_stamp)
        )
        self.assertTrue(driver.wait_for_static_layers(geometry_stamp))
        time.sleep(0.25)
        self.assertNotIn(
            geometry_stamp,
            driver.combined,
            "incompatible rolling-grid geometry must fail closed",
        )

        frame_stamp = driver.get_clock().now().nanoseconds
        wrong_frame = _dynamic_grid(frame_stamp)
        wrong_frame.header.frame_id = "odom"
        driver.publish_dynamic_then_cloud(wrong_frame, _cloud(frame_stamp))
        self.assertTrue(driver.wait_for_static_layers(frame_stamp))
        time.sleep(0.25)
        self.assertNotIn(
            frame_stamp,
            driver.combined,
            "incompatible rolling-grid frame must fail closed",
        )

        skew_stamp = driver.get_clock().now().nanoseconds
        wrong_stamp = _dynamic_grid(skew_stamp - 1)
        driver.publish_dynamic_then_cloud(wrong_stamp, _cloud(skew_stamp))
        self.assertTrue(driver.wait_for_static_layers(skew_stamp))
        time.sleep(0.25)
        self.assertNotIn(
            skew_stamp,
            driver.combined,
            "one-nanosecond-skewed rolling grids were merged",
        )

        no_publish_stamp = driver.get_clock().now().nanoseconds
        driver.publish_cloud(
            _cloud(no_publish_stamp, malformed=True)
        )
        time.sleep(0.25)
        self.assertNotIn(no_publish_stamp, driver.static)
        self.assertNotIn(no_publish_stamp, driver.combined)
        self.assertNotIn(no_publish_stamp, driver.alias)

        empty_stamp = driver.get_clock().now().nanoseconds
        driver.publish_cloud(_empty_cloud(empty_stamp))
        time.sleep(0.25)
        self.assertFalse(
            empty_stamp in driver.static,
            "an empty cloud must fail closed before iterator construction",
        )
        self.assertFalse(empty_stamp in driver.combined)
        self.assertFalse(empty_stamp in driver.alias)

        padded_stamp = driver.get_clock().now().nanoseconds
        driver.publish_cloud(_padded_cloud(padded_stamp))
        time.sleep(0.25)
        self.assertFalse(
            padded_stamp in driver.static,
            "organized row padding is unsupported by the ROS iterator path",
        )
        self.assertFalse(padded_stamp in driver.combined)
        self.assertFalse(padded_stamp in driver.alias)

        bigendian_stamp = driver.get_clock().now().nanoseconds
        bigendian = _cloud(bigendian_stamp)
        bigendian.is_bigendian = True
        driver.publish_cloud(bigendian)
        time.sleep(0.25)
        self.assertFalse(
            bigendian_stamp in driver.static,
            "big-endian XYZ must fail closed before native float iteration",
        )
        self.assertFalse(bigendian_stamp in driver.combined)
        self.assertFalse(bigendian_stamp in driver.alias)

        missing_tf_stamp = driver.get_clock().now().nanoseconds
        driver.publish_cloud(
            _cloud(missing_tf_stamp, frame_id="lidar_without_tf")
        )
        time.sleep(0.25)
        self.assertNotIn(missing_tf_stamp, driver.static)
        self.assertNotIn(missing_tf_stamp, driver.combined)
        self.assertNotIn(missing_tf_stamp, driver.alias)

        zero_stamp = 0
        driver.publish_cloud(_cloud(zero_stamp))
        time.sleep(0.25)
        self.assertNotIn(zero_stamp, driver.static)
        self.assertNotIn(zero_stamp, driver.combined)
        self.assertNotIn(zero_stamp, driver.alias)

        observed_stamp = driver.get_clock().now().nanoseconds
        driver.publish_clear_then_cloud(_occupied_cloud(observed_stamp))
        self.assertTrue(driver.wait_for_layers(observed_stamp))
        self.assertEqual(max(driver.static[observed_stamp].data), 100)

        persisted_stamp = observed_stamp + 100_000_000
        driver.publish_clear_then_cloud(_cloud(persisted_stamp))
        self.assertTrue(driver.wait_for_layers(persisted_stamp))
        self.assertEqual(
            max(driver.static[persisted_stamp].data),
            100,
            "a sparse static return disappeared before persistence timeout",
        )

        expired_stamp = observed_stamp + 600_000_000
        driver.publish_clear_then_cloud(_cloud(expired_stamp))
        self.assertTrue(driver.wait_for_layers(expired_stamp))
        self.assertTrue(
            all(cell == 0 for cell in driver.static[expired_stamp].data),
            "a static return remained after persistence timeout",
        )

        offroad_stamp = observed_stamp + 700_000_000
        offroad_mask = _drivable_mask(offroad_stamp)
        offroad_mask.data = [100] * len(offroad_mask.data)
        driver.publish_clear_then_cloud(
            _occupied_cloud(offroad_stamp), mask=offroad_mask
        )
        self.assertTrue(driver.wait_for_layers(offroad_stamp))
        self.assertTrue(
            all(cell == 0 for cell in driver.static[offroad_stamp].data),
            "the planning static OGM retained a road-mask-rejected return",
        )
        self.assertEqual(
            max(driver.ungated[offroad_stamp].data),
            100,
            "the same-stamp visualization OGM lost the off-road return",
        )
        self.assertEqual(
            driver.ungated[offroad_stamp].header.stamp,
            driver.static[offroad_stamp].header.stamp,
        )

    def _test_dynamic(self, driver):
        time.sleep(0.25)
        self.assertEqual(driver.dynamic, [])

        executable = (
            Path(get_package_prefix("ad_lidar_perception"))
            / "lib"
            / "ad_lidar_perception"
            / "ad_dynamic_occupancy_grid_node"
        )
        try:
            nonintegral = subprocess.run(
                [
                    str(executable),
                    "--ros-args",
                    "-p",
                    "x_max:=24.05",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except subprocess.TimeoutExpired:
            self.fail("dynamic node accepted a non-integral grid extent")
        self.assertNotEqual(nonintegral.returncode, 0)
        self.assertIn(
            "integer number of cells",
            nonintegral.stdout + nonintegral.stderr,
        )

        valid_stamp = driver.get_clock().now().nanoseconds
        driver.publish_transform(valid_stamp)
        driver.prediction_publisher.publish(_predicted_array(valid_stamp))
        time.sleep(0.25)
        self.assertFalse(
            any(_stamp_ns(message) == valid_stamp for message in driver.dynamic),
            "road-gated dynamic OGM published without an exact-stamp mask",
        )
        driver.drivable_mask_publisher.publish(_drivable_mask(valid_stamp))
        self.assertTrue(
            _wait_until(
                lambda: any(
                    _stamp_ns(message) == valid_stamp
                    and any(cell == 100 for cell in message.data)
                    for message in driver.dynamic
                )
            ),
            "valid current footprint did not produce occupancy",
        )
        occupied = next(
            message
            for message in driver.dynamic
            if _stamp_ns(message) == valid_stamp
            and any(cell == 100 for cell in message.data)
        )
        for x, expected_cost in ((2.0, 100), (3.0, 0), (4.0, 0)):
            grid_x = int(math.floor((x + 4.0) / 0.1))
            grid_y = int(math.floor(10.0 / 0.1))
            self.assertEqual(
                occupied.data[grid_y * occupied.info.width + grid_x],
                expected_cost,
                "planning OGM must not collapse future predictions into "
                "timeless occupied cells",
            )

        newer_stamp = driver.get_clock().now().nanoseconds
        driver.publish_transform(newer_stamp)
        before = len(driver.dynamic)
        driver.publish_prediction(_predicted_array(newer_stamp))
        self.assertTrue(
            _wait_until(
                lambda: len(driver.dynamic) > before
                and _stamp_ns(driver.dynamic[-1]) == newer_stamp
                and any(cell == 100 for cell in driver.dynamic[-1].data)
            )
        )

        before = len(driver.dynamic)
        driver.publish_prediction(_predicted_array(valid_stamp))
        self.assertTrue(_wait_until(lambda: len(driver.dynamic) > before))
        time.sleep(0.15)
        out_of_order_outputs = driver.dynamic[before:]
        self.assertTrue(out_of_order_outputs)
        self.assertTrue(
            all(
                all(cell == 0 for cell in message.data)
                for message in out_of_order_outputs
            ),
            "out-of-order prediction transiently published occupied cells",
        )

        before = len(driver.dynamic)
        driver.publish_prediction(_predicted_array(valid_stamp))
        self.assertTrue(_wait_until(lambda: len(driver.dynamic) > before))
        time.sleep(0.15)
        replay_outputs = driver.dynamic[before:]
        self.assertTrue(
            all(
                all(cell == 0 for cell in message.data)
                for message in replay_outputs
            ),
            "clearing the active cache forgot the latest valid stamp watermark",
        )

        while (
            driver.get_clock().now().nanoseconds - valid_stamp
            <= int((TEST_PREDICTION_TIMEOUT_SEC + 0.05) * 1_000_000_000)
        ):
            time.sleep(0.01)
        before = len(driver.dynamic)
        driver.publish_prediction(_predicted_array(valid_stamp))
        self.assertTrue(_wait_until(lambda: len(driver.dynamic) > before))
        time.sleep(0.15)
        stale_input_outputs = driver.dynamic[before:]
        self.assertTrue(stale_input_outputs)
        self.assertTrue(
            all(
                all(cell == 0 for cell in message.data)
                for message in stale_input_outputs
            ),
            "already-stale prediction transiently published occupied cells",
        )

        invalid_stamp = driver.get_clock().now().nanoseconds
        invalid = _predicted_array(invalid_stamp)
        invalid.objects[0].existence_probability = math.nan
        driver.publish_transform(invalid_stamp)
        before = len(driver.dynamic)
        driver.publish_prediction(invalid)
        self.assertTrue(
            _wait_until(
                lambda: len(driver.dynamic) > before
                and _stamp_ns(driver.dynamic[-1]) == invalid_stamp
                and all(cell == 0 for cell in driver.dynamic[-1].data)
            )
        )

        delayed_stamp = invalid_stamp - 1
        before = len(driver.dynamic)
        driver.publish_prediction(_predicted_array(delayed_stamp))
        self.assertTrue(
            _wait_until(
                lambda: len(driver.dynamic) > before
                and _stamp_ns(driver.dynamic[-1]) == invalid_stamp
                and all(cell == 0 for cell in driver.dynamic[-1].data)
            )
        )
        time.sleep(0.15)
        delayed_outputs = driver.dynamic[before:]
        self.assertTrue(
            all(
                _stamp_ns(message) == invalid_stamp
                and all(cell == 0 for cell in message.data)
                for message in delayed_outputs
            ),
            "out-of-order input advanced the clear beyond its watermark",
        )

        before = len(driver.dynamic)
        driver.publish_prediction(_predicted_array(invalid_stamp))
        self.assertTrue(
            _wait_until(
                lambda: len(driver.dynamic) > before
                and _stamp_ns(driver.dynamic[-1]) == invalid_stamp
                and any(cell == 100 for cell in driver.dynamic[-1].data)
            ),
            "equal-stamp corrected prediction was not accepted after clear",
        )

        extra_stamp = driver.get_clock().now().nanoseconds
        extra = _predicted_array(extra_stamp)
        extra_state = copy.deepcopy(extra.objects[0].states[-1])
        extra_state.time_from_start.sec = 2
        extra_state.time_from_start.nanosec = 0
        extra.objects[0].states.append(extra_state)
        driver.publish_transform(extra_stamp)
        before = len(driver.dynamic)
        driver.publish_prediction(extra)
        self.assertTrue(
            _wait_until(
                lambda: len(driver.dynamic) > before
                and any(cell == 100 for cell in driver.dynamic[-1].data)
            )
        )

        stale_stamp = driver.get_clock().now().nanoseconds
        driver.publish_transform(stale_stamp)
        before = len(driver.dynamic)
        driver.publish_prediction(_predicted_array(stale_stamp))
        self.assertTrue(
            _wait_until(
                lambda: len(driver.dynamic) > before
                and _stamp_ns(driver.dynamic[-1]) == stale_stamp
                and any(cell == 100 for cell in driver.dynamic[-1].data)
            )
        )
        occupied_count = len(driver.dynamic)
        self.assertTrue(
            _wait_until(
                lambda: len(driver.dynamic) > occupied_count
                and _stamp_ns(driver.dynamic[-1]) > stale_stamp
                and all(cell == 0 for cell in driver.dynamic[-1].data),
                timeout=TEST_PREDICTION_TIMEOUT_SEC + 1.0,
            ),
            "stale prediction did not clear the dynamic layer",
        )
        clear_count = len(driver.dynamic)
        time.sleep(0.25)
        self.assertEqual(len(driver.dynamic), clear_count)


@launch_testing.post_shutdown_test()
class TestOccupancyLayerShutdown(unittest.TestCase):
    def test_clean_shutdown(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info)
