"""Opt-in isolated ROS integration for the parallel ESKF candidate graph."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import pytest


RUN_INTEGRATION = os.environ.get("AD_RUN_ROS_INTEGRATION") == "1"
CANDIDATES = (
    "baseline",
    "bias_covariance",
    "observable_bias",
    "combined_bias",
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ESKF_PATH = PACKAGE_ROOT.parent / "ad_localization" / "config" / "eskf.yaml"


def _stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)


@pytest.mark.skipif(not RUN_INTEGRATION, reason="set AD_RUN_ROS_INTEGRATION=1")
def test_parallel_candidates_share_inputs_initialize_and_publish_without_tf():
    import rclpy
    from geometry_msgs.msg import PoseStamped, TwistWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float64MultiArray
    from tf2_msgs.msg import TFMessage

    domain_id = str(80 + os.getpid() % 100)
    environment = {**os.environ, "ROS_DOMAIN_ID": domain_id}
    os.environ["ROS_DOMAIN_ID"] = domain_id
    log = tempfile.TemporaryFile(mode="w+")
    process = subprocess.Popen(
        [
            "ros2",
            "launch",
            "ad_morai_bridge_dev",
            "eskf_parallel_ab.launch.py",
            "run_id:=integration",
            "harness_enabled:=false",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=environment,
    )

    rclpy.init()
    node = rclpy.create_node(f"eskf_parallel_probe_{domain_id}")
    reliable = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    imu_pub = node.create_publisher(
        Imu, "/ad/localization/input/eskf_imu", qos_profile_sensor_data
    )
    gnss_pub = node.create_publisher(
        PoseStamped, "/ad/localization/input/gnss_pose", reliable
    )
    wheel_pub = node.create_publisher(
        TwistWithCovarianceStamped,
        "/ad/localization/input/wheel_speed",
        qos_profile_sensor_data,
    )
    initial_pubs = {
        name: node.create_publisher(
            PoseStamped,
            f"/ad/experiment/eskf/integration/{name}/initial_pose",
            reliable,
        )
        for name in CANDIDATES
    }
    odometry = {name: [] for name in CANDIDATES}
    initialized = {name: False for name in CANDIDATES}
    subscriptions = []
    for name in CANDIDATES:
        subscriptions.append(
            node.create_subscription(
                Odometry,
                f"/ad/experiment/eskf/integration/{name}/odometry",
                lambda message, candidate=name: odometry[candidate].append(message),
                reliable,
            )
        )

        def on_initialization(message, candidate=name):
            if message.data and int(message.data[0]) == 1:
                initialized[candidate] = True

        subscriptions.append(
            node.create_subscription(
                Float64MultiArray,
                f"/ad/experiment/eskf/integration/eskf_{name}/debug/initialization",
                on_initialization,
                reliable,
            )
        )
    tf_messages = []
    subscriptions.append(
        node.create_subscription(
            TFMessage, "/tf", tf_messages.append, qos_profile_sensor_data
        )
    )

    try:
        readiness_deadline = time.monotonic() + 10.0
        while time.monotonic() < readiness_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if (
                process.poll() is None
                and imu_pub.get_subscription_count() == len(CANDIDATES)
                and gnss_pub.get_subscription_count() == len(CANDIDATES)
                and wheel_pub.get_subscription_count() == len(CANDIDATES)
                and all(pub.get_subscription_count() == 1 for pub in initial_pubs.values())
            ):
                break
        else:
            raise AssertionError("parallel candidate subscriptions did not become ready")

        started = time.monotonic()
        tick = 0
        while time.monotonic() - started < 5.0 and not all(initialized.values()):
            stamp = node.get_clock().now().to_msg()
            initial = PoseStamped()
            initial.header.stamp = stamp
            initial.header.frame_id = "odom"
            initial.pose.position.x = 1.0
            initial.pose.position.y = 2.0
            initial.pose.position.z = 3.0
            initial.pose.orientation.w = 1.0
            if tick < 10:
                for publisher in initial_pubs.values():
                    publisher.publish(initial)

            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = "base_link"
            imu.orientation.w = 1.0
            imu.orientation_covariance[0] = 0.01
            imu.orientation_covariance[4] = 0.01
            imu.orientation_covariance[8] = 0.01
            imu.linear_acceleration.z = 9.80665
            imu_pub.publish(imu)

            gnss = PoseStamped()
            gnss.header = initial.header
            gnss.pose = initial.pose
            gnss_pub.publish(gnss)

            wheel = TwistWithCovarianceStamped()
            wheel.header.stamp = stamp
            wheel.header.frame_id = "base_link"
            wheel_pub.publish(wheel)
            # Keep the synthetic source near the live IMU rate and drain all
            # currently-ready callbacks.  A single spin_once at a much higher
            # publish rate can starve the one-shot initialization diagnostic
            # behind four continuously arriving odometry streams.
            for _ in range(12):
                rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.02)
            tick += 1

        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.0)

        assert all(initialized.values())
        assert all(odometry[name] for name in CANDIDATES)
        assert tf_messages == []
        assert process.poll() is None
    except Exception:
        _stop_process(process)
        log.seek(0)
        pytest.fail("parallel ESKF integration failed:\n" + log.read())
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
        _stop_process(process)
        log.close()

    assert process.returncode == 0


@pytest.mark.skipif(not RUN_INTEGRATION, reason="set AD_RUN_ROS_INTEGRATION=1")
def test_harness_node_can_destroy_its_ros_entities_once(tmp_path):
    import rclpy

    from ad_morai_bridge_dev.eskf_experiment.node import (
        _RosEskfExperimentNode,
    )

    domain_id = str(180 + os.getpid() % 40)
    os.environ["ROS_DOMAIN_ID"] = domain_id
    sensor_file = tmp_path / "sensor.json"
    sensor_file.write_text("{}\n", encoding="utf-8")
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            "run_id:=destroy_probe",
            "-p",
            f"experiment_config:={PACKAGE_ROOT / 'config' / 'eskf_parallel_ab.yaml'}",
            "-p",
            f"eskf_config:={ESKF_PATH}",
            "-p",
            f"data_root:={tmp_path}",
            "-p",
            f"repository_root:={PACKAGE_ROOT.parent}",
            "-p",
            f"active_sensor_file:={sensor_file}",
        ]
    )
    node = None
    try:
        node = _RosEskfExperimentNode()
        node.destroy_node()
        node = None
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except ValueError:
                pass
        if rclpy.ok():
            rclpy.shutdown()
