"""Opt-in end-to-end test for the adapter and pinned upstream ESKF."""

from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import pytest


RUN_INTEGRATION = os.environ.get("AD_RUN_ROS_INTEGRATION") == "1"
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_ROOT))


def _spin_until(node, predicate, timeout_sec):
    import rclpy

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
        if predicate():
            return True
    return False


def _spin_for(node, duration_sec):
    _spin_until(node, lambda: False, duration_sec)


def _stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)


@pytest.mark.skipif(not RUN_INTEGRATION, reason="set AD_RUN_ROS_INTEGRATION=1")
@pytest.mark.parametrize(
    ("domain_offset", "orientation_source", "yaw_offset_rad", "expected_success"),
    [
        pytest.param(0, "imu", 0.0, True, id="corrected-imu-only"),
        pytest.param(
            1,
            "vehicle_status",
            0.0,
            False,
            id="reject-uncorrected-vehicle-status",
        ),
        pytest.param(
            2,
            "imu",
            0.1,
            False,
            id="reject-second-initial-yaw-offset",
        ),
    ],
)
def test_eskf_configuration_requires_the_single_corrected_imu_initialization_path(
    domain_offset,
    orientation_source,
    yaw_offset_rad,
    expected_success,
):
    import rclpy
    from lifecycle_msgs.msg import Transition
    from lifecycle_msgs.srv import ChangeState

    domain_id = 40 + os.getpid() % 150 + domain_offset
    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    environment = os.environ.copy()
    log = tempfile.TemporaryFile(mode="w+")
    process = subprocess.Popen(
        [
            "ros2",
            "run",
            "ad_localization",
            "ad_localization_node",
            "--ros-args",
            "-p",
            "localization_backend:=eskf",
            "-p",
            f"initial_orientation_source:={orientation_source}",
            "-p",
            f"initial_orientation_yaw_offset_rad:={yaw_offset_rad}",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=environment,
    )

    rclpy.init()
    probe = rclpy.create_node(f"eskf_configuration_probe_{domain_id}")
    client = probe.create_client(ChangeState, "/ad_localization/change_state")
    try:
        assert client.wait_for_service(timeout_sec=8.0)
        request = ChangeState.Request()
        request.transition.id = Transition.TRANSITION_CONFIGURE
        future = client.call_async(request)
        assert _spin_until(probe, future.done, 5.0)
        assert future.result() is not None
        assert future.result().success is expected_success
    except Exception:
        _stop_process(process)
        log.seek(0)
        pytest.fail("ESKF configuration contract failed:\n" + log.read())
    finally:
        probe.destroy_node()
        rclpy.shutdown()
        _stop_process(process)
        log.close()


@pytest.mark.skipif(not RUN_INTEGRATION, reason="set AD_RUN_ROS_INTEGRATION=1")
def test_composed_eskf_initializes_propagates_and_recovers_gnss():
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
    from fixtures.eskf_synthetic_input import SyntheticLocalizationInput
    from geometry_msgs.msg import Pose2D, PoseStamped, TwistWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import Imu
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    domain_id = str(120 + os.getpid() % 80)
    os.environ["ROS_DOMAIN_ID"] = domain_id
    environment = os.environ.copy()
    logs = [tempfile.TemporaryFile(mode="w+") for _ in range(2)]
    processes = [
        subprocess.Popen(
            ["ros2", "launch", "ad_description", "description.launch.py"],
            stdout=logs[0],
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=environment,
        ),
        subprocess.Popen(
            [
                "ros2",
                "launch",
                "ad_localization",
                "localization.launch.py",
                "localization_backend:=eskf",
            ],
            stdout=logs[1],
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=environment,
        ),
    ]

    rclpy.init()
    fixture = SyntheticLocalizationInput()
    reliable = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE)
    odometry: list[Odometry] = []
    poses: list[Pose2D] = []
    initial_poses: list[PoseStamped] = []
    wheels: list[TwistWithCovarianceStamped] = []
    diagnostics: list[DiagnosticArray] = []
    corrected_imus: list[Imu] = []
    subscriptions = [
        fixture.create_subscription(
            Odometry, "/ad/localization/odometry", odometry.append, reliable
        ),
        fixture.create_subscription(
            Pose2D, "/ad/localization/pose2d", poses.append, reliable
        ),
        fixture.create_subscription(
            PoseStamped,
            "/ad/localization/input/initial_pose",
            initial_poses.append,
            reliable,
        ),
        fixture.create_subscription(
            TwistWithCovarianceStamped,
            "/ad/localization/input/wheel_speed",
            wheels.append,
            reliable,
        ),
        fixture.create_subscription(
            DiagnosticArray, "/diagnostics", diagnostics.append, reliable
        ),
        fixture.create_subscription(
            Imu,
            "/ad/localization/input/eskf_imu",
            corrected_imus.append,
            qos_profile_sensor_data,
        ),
    ]
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, fixture, spin_thread=False)

    try:
        assert _spin_until(
            fixture,
            lambda: (
                fixture.gps_publisher.get_subscription_count() >= 1
                and fixture.imu_publisher.get_subscription_count() == 1
                and fixture.count_subscribers(
                    "/ad/localization/input/eskf_imu"
                )
                >= 2
                and fixture.status_publisher.get_subscription_count() >= 1
            ),
            8.0,
        )
        assert all(process.poll() is None for process in processes)
        assert _spin_until(
            fixture, lambda: bool(odometry and poses and corrected_imus), 12.0
        )
        corrected_imu = corrected_imus[-1]
        assert corrected_imu.header.frame_id == "imu_link"
        assert corrected_imu.orientation.z == pytest.approx(
            math.sin(-0.02350724531030645 / 2.0), abs=1e-12
        )
        assert corrected_imu.orientation.w == pytest.approx(
            math.cos(-0.02350724531030645 / 2.0), abs=1e-12
        )
        assert corrected_imu.angular_velocity == Imu().angular_velocity
        assert corrected_imu.linear_acceleration.x == 0.0
        assert corrected_imu.linear_acceleration.z == pytest.approx(9.80665)
        assert list(corrected_imu.orientation_covariance) == [
            0.01,
            0.0,
            0.0,
            0.0,
            0.01,
            0.0,
            0.0,
            0.0,
            0.01,
        ]
        assert all(
            math.isfinite(value)
            for value in (
                odometry[-1].pose.pose.position.x,
                odometry[-1].pose.pose.position.y,
                odometry[-1].pose.pose.position.z,
                poses[-1].x,
                poses[-1].y,
                poses[-1].theta,
            )
        )
        orientation = odometry[-1].pose.pose.orientation
        assert all(
            math.isfinite(value)
            for value in (orientation.x, orientation.y, orientation.z, orientation.w)
        )
        assert math.sqrt(
            orientation.x**2
            + orientation.y**2
            + orientation.z**2
            + orientation.w**2
        ) == pytest.approx(1.0, abs=1e-6)
        assert _spin_until(
            fixture,
            lambda: tf_buffer.can_transform("map", "gps_link", Time()),
            3.0,
        )
        assert tf_buffer.can_transform("map", "imu_link", Time())
        assert tf_buffer.can_transform("map", "base_link", Time())

        fixture.speed_mps = 2.0
        fixture.gear = 4
        _spin_for(fixture, 0.4)
        fixture.gear = 2
        _spin_for(fixture, 0.4)
        assert any(message.twist.twist.linear.x > 0.0 for message in wheels)
        assert any(message.twist.twist.linear.x < 0.0 for message in wheels)

        fixture.speed_mps = 0.0
        fixture.gear = 4
        _spin_for(fixture, 2.0)
        stable_initial_pose_count = len(initial_poses)
        before_blackout_count = len(odometry)
        before_blackout_x = odometry[-1].pose.pose.position.x
        before_blackout_y = odometry[-1].pose.pose.position.y
        fixture.status_enabled = False
        fixture.acceleration_x_mps2 = 1.0
        fixture.gps_enabled = False
        _spin_for(fixture, 2.0)
        assert len(odometry) > before_blackout_count
        propagated_distance = math.hypot(
            odometry[-1].pose.pose.position.x - before_blackout_x,
            odometry[-1].pose.pose.position.y - before_blackout_y,
        )
        assert propagated_distance > 0.1
        assert len(initial_poses) == stable_initial_pose_count
        assert any(
            item.name == "ad_localization/input_adapter"
            and item.level == DiagnosticStatus.WARN
            and "GNSS unavailable" in item.message
            for message in diagnostics
            for item in message.status
        )

        fixture.gps_enabled = True
        fixture.status_enabled = True
        fixture.acceleration_x_mps2 = 0.0
        assert _spin_until(
            fixture,
            lambda: any(
                item.name == "ad_localization/input_adapter"
                and item.level == DiagnosticStatus.OK
                for message in diagnostics
                for item in message.status
            ),
            3.0,
        )
        assert len(initial_poses) == stable_initial_pose_count
        recovered_position = odometry[-1].pose.pose.position
        jump = math.hypot(
            recovered_position.x - before_blackout_x,
            recovered_position.y - before_blackout_y,
        )
        assert jump < 5.0
        assert all(process.poll() is None for process in processes)
    except Exception:
        for process in processes:
            _stop_process(process)
        captured = []
        for log in logs:
            log.seek(0)
            captured.append(log.read())
        pytest.fail("ESKF integration failed:\n" + "\n".join(captured))
    finally:
        tf_listener.unregister()
        for subscription in subscriptions:
            fixture.destroy_subscription(subscription)
        fixture.destroy_node()
        rclpy.shutdown()
        for process in processes:
            _stop_process(process)
        for log in logs:
            log.close()

    assert all(process.returncode == 0 for process in processes)
