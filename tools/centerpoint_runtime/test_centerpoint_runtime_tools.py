import importlib.util
from pathlib import Path
import sys


MODULE = Path(__file__).with_name("check_heven_centerpoint_runtime.py")
SPEC = importlib.util.spec_from_file_location("centerpoint_runtime_health", MODULE)
health = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = health
SPEC.loader.exec_module(health)


def test_parse_ros_topic_info_counts_and_type():
    assert health.parse_topic_info(
        "Type: sensor_msgs/msg/PointCloud2\nPublisher count: 1\nSubscription count: 2\n"
    ) == ("sensor_msgs/msg/PointCloud2", 1, 2)


def test_missing_input_is_waiting_not_detector_failure():
    result = health.classify_topic(
        "points_xyzirc", "/points", "sensor_msgs/msg/PointCloud2", None, None, None
    )
    assert result.status == "WAITING_FOR_SENSOR"


def test_missing_downstream_is_blocked_and_wrong_type_fails():
    blocked = health.classify_topic(
        "tracked", "/tracked", "pkg/msg/Tracked", None, None, None
    )
    wrong = health.classify_topic(
        "detected", "/detected", "pkg/msg/Detected", "pkg/msg/Wrong", 1, 1
    )
    assert blocked.status == "BLOCKED"
    assert wrong.status == "FAIL"


def test_runtime_summary_uses_stamps_and_latency_percentiles():
    summary_spec = importlib.util.spec_from_file_location(
        "centerpoint_runtime_summary", MODULE.with_name("summarize_detection_runtime.py")
    )
    summary_module = importlib.util.module_from_spec(summary_spec)
    assert summary_spec.loader is not None
    sys.modules[summary_spec.name] = summary_module
    summary_spec.loader.exec_module(summary_module)
    result = summary_module.summarize([
        {"backend": "centerpoint", "header_stamp_ns": 1_000_000_000, "latency_ms": 11.0},
        {"backend": "centerpoint", "header_stamp_ns": 1_100_000_000, "latency_ms": 17.0},
        {"backend": "centerpoint", "header_stamp_ns": 1_200_000_000, "latency_ms": 13.0},
    ])
    assert result["output_hz"] == 10.0
    assert result["latency_p95_ms"] == 17.0
