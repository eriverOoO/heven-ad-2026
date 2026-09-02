"""Unit tests for the sensor-bag readiness preflight (metadata-only core).

These tests exercise only ``classify_bag_metadata`` - pure ``metadata.yaml``
parsing - so they run without a sourced ROS environment, RViz, CUDA, torch,
CenterPoint, or KalmanNet. They never touch a real rosbag and never write
inside the repository's own tracked tree (everything is built under
``tmp_path``).
"""

from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_sensor_bag_ready as m  # noqa: E402


def _topic_entry(name, type_name, count):
    return {
        "topic_metadata": {
            "name": name,
            "type": type_name,
            "serialization_format": "cdr",
        },
        "message_count": count,
    }


def _write_bag(
    tmp_path,
    *,
    topics=None,
    duration_ns=10_000_000_000,
    write_storage_file=True,
    storage_relative_path="bag_0.mcap",
    mutate_metadata=None,
):
    bag = tmp_path / "sample_bag"
    bag.mkdir()
    metadata = {
        "rosbag2_bagfile_information": {
            "storage_identifier": "mcap",
            "relative_file_paths": [storage_relative_path],
            "duration": {"nanoseconds": duration_ns},
            "topics_with_message_count": topics or [],
        }
    }
    if mutate_metadata is not None:
        mutate_metadata(metadata)
    (bag / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    if write_storage_file:
        (bag / storage_relative_path).write_bytes(b"\x89MCAP0\r\n")
    return bag


def _full_replay_topics(camera=True, tf_dynamic_count=120):
    topics = [
        _topic_entry(m.LIDAR_TOPIC, m.LIDAR_TYPE, 300),
        _topic_entry(m.ODOMETRY_TOPIC, m.ODOMETRY_TYPE, 300),
        _topic_entry(m.TF_TOPIC, m.TF_TYPE, tf_dynamic_count),
        _topic_entry(m.TF_STATIC_TOPIC, m.TF_TYPE, 1),
    ]
    if camera:
        topics.append(_topic_entry(m.CAMERA_TOPIC, m.CAMERA_TYPE, 700))
    return topics


def _recompute_topics(camera=False):
    topics = [
        _topic_entry(m.LIDAR_TOPIC, m.LIDAR_TYPE, 300),
        _topic_entry(m.GPS_TOPIC, m.GPS_TYPE, 700),
        _topic_entry(m.IMU_TOPIC, m.IMU_TYPE, 1400),
        _topic_entry(m.VEHICLE_STATUS_TOPIC, m.VEHICLE_STATUS_TYPE, 1400),
        _topic_entry(m.TF_STATIC_TOPIC, m.TF_TYPE, 1),
    ]
    if camera:
        topics.append(_topic_entry(m.CAMERA_TOPIC, m.CAMERA_TYPE, 700))
    return topics


# 1. full replay bag manifest -> READY_FULL_REPLAY
def test_full_replay_bag_is_ready_full_replay(tmp_path):
    bag = _write_bag(tmp_path, topics=_full_replay_topics())
    result = m.classify_bag_metadata(bag)
    assert result.status == m.STATUS_READY_FULL_REPLAY
    assert result.topics[m.CAMERA_TOPIC].present is True


# 6. camera optional - full-replay bag without camera is still READY_FULL_REPLAY
def test_camera_is_optional_for_full_replay(tmp_path):
    bag = _write_bag(tmp_path, topics=_full_replay_topics(camera=False))
    result = m.classify_bag_metadata(bag)
    assert result.status == m.STATUS_READY_FULL_REPLAY
    assert result.topics[m.CAMERA_TOPIC].present is False
    assert any("camera" in reason.lower() for reason in result.reasons)


# 2. raw ego but no odometry -> READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE
def test_raw_ego_sensors_without_odometry_is_recompute_ready(tmp_path):
    bag = _write_bag(tmp_path, topics=_recompute_topics())
    result = m.classify_bag_metadata(bag)
    assert result.status == m.STATUS_READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE


def test_matches_morai_cam4_shape_exactly(tmp_path):
    # Mirrors the real morai_cam4_20260813_163222 bag's topic shape (Phase 17):
    # LiDAR + camera + raw ego sensors, zero recorded /tf, no odometry.
    topics = _recompute_topics(camera=True) + [
        _topic_entry(m.TF_TOPIC, m.TF_TYPE, 0),
    ]
    bag = _write_bag(tmp_path, topics=topics)
    result = m.classify_bag_metadata(bag)
    assert result.status == m.STATUS_READY_REPLAY_WITH_LOCALIZATION_RECOMPUTE
    assert result.status != m.STATUS_READY_FULL_REPLAY


# 3. no LiDAR -> MISSING_LIDAR
def test_no_lidar_topic_is_missing_lidar(tmp_path):
    topics = [t for t in _full_replay_topics() if t["topic_metadata"]["name"] != m.LIDAR_TOPIC]
    bag = _write_bag(tmp_path, topics=topics)
    result = m.classify_bag_metadata(bag)
    assert result.status == m.STATUS_MISSING_LIDAR


def test_lidar_topic_present_with_zero_messages_is_missing_lidar(tmp_path):
    topics = _full_replay_topics()
    for entry in topics:
        if entry["topic_metadata"]["name"] == m.LIDAR_TOPIC:
            entry["message_count"] = 0
    bag = _write_bag(tmp_path, topics=topics)
    result = m.classify_bag_metadata(bag)
    assert result.status == m.STATUS_MISSING_LIDAR


# 4. no TF -> MISSING_TF (odometry recorded but dynamic TF absent/empty)
def test_odometry_without_dynamic_tf_is_missing_tf(tmp_path):
    topics = _full_replay_topics(tf_dynamic_count=0)
    bag = _write_bag(tmp_path, topics=topics)
    result = m.classify_bag_metadata(bag)
    assert result.status == m.STATUS_MISSING_TF


def test_lidar_only_no_localization_signal_at_all(tmp_path):
    topics = [
        _topic_entry(m.LIDAR_TOPIC, m.LIDAR_TYPE, 300),
        _topic_entry(m.TF_STATIC_TOPIC, m.TF_TYPE, 1),
    ]
    bag = _write_bag(tmp_path, topics=topics)
    result = m.classify_bag_metadata(bag)
    assert result.status == m.STATUS_MISSING_LOCALIZATION


def test_incomplete_raw_ego_sensor_set_is_partial_sensor_only(tmp_path):
    topics = [
        _topic_entry(m.LIDAR_TOPIC, m.LIDAR_TYPE, 300),
        _topic_entry(m.GPS_TOPIC, m.GPS_TYPE, 700),
        # imu and vehicle_status both missing.
    ]
    bag = _write_bag(tmp_path, topics=topics)
    result = m.classify_bag_metadata(bag)
    assert result.status == m.STATUS_PARTIAL_SENSOR_ONLY


# 5. metadata without storage file -> BROKEN
def test_metadata_without_storage_file_is_broken(tmp_path):
    bag = _write_bag(
        tmp_path, topics=_full_replay_topics(), write_storage_file=False
    )
    result = m.classify_bag_metadata(bag)
    assert result.status == m.STATUS_BROKEN
    assert "does not exist on disk" in result.reasons[0]


def test_missing_metadata_yaml_is_broken(tmp_path):
    bag = tmp_path / "empty_dir"
    bag.mkdir()
    result = m.classify_bag_metadata(bag)
    assert result.status == m.STATUS_BROKEN


def test_nonexistent_bag_path_is_broken(tmp_path):
    result = m.classify_bag_metadata(tmp_path / "does_not_exist")
    assert result.status == m.STATUS_BROKEN


def test_invalid_metadata_yaml_is_broken(tmp_path):
    bag = tmp_path / "bad_bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("not: [a, valid, rosbag2, schema", encoding="utf-8")
    result = m.classify_bag_metadata(bag)
    assert result.status == m.STATUS_BROKEN


# type mismatch is reported (not silently ignored)
def test_wrong_message_type_is_flagged_in_reasons(tmp_path):
    topics = _full_replay_topics()
    for entry in topics:
        if entry["topic_metadata"]["name"] == m.LIDAR_TOPIC:
            entry["topic_metadata"]["type"] = "sensor_msgs/msg/LaserScan"
    bag = _write_bag(tmp_path, topics=topics)
    result = m.classify_bag_metadata(bag)
    assert result.topics[m.LIDAR_TOPIC].type_matches_expected is False
    assert any("does not match expected" in reason for reason in result.reasons)


# 9. deterministic report output
def test_classification_is_deterministic(tmp_path):
    bag = _write_bag(tmp_path, topics=_full_replay_topics())
    first = m.classify_bag_metadata(bag).to_dict()
    second = m.classify_bag_metadata(bag).to_dict()
    assert first == second


# 10. no bag modifications
def test_classification_never_writes_to_the_bag(tmp_path):
    bag = _write_bag(tmp_path, topics=_full_replay_topics())
    before = {
        path: path.stat().st_mtime_ns for path in bag.rglob("*") if path.is_file()
    }
    m.classify_bag_metadata(bag)
    after = {
        path: path.stat().st_mtime_ns for path in bag.rglob("*") if path.is_file()
    }
    assert before == after
    assert set(before) == set(after)


def test_to_dict_is_json_serializable(tmp_path):
    import json

    bag = _write_bag(tmp_path, topics=_full_replay_topics())
    result = m.classify_bag_metadata(bag)
    json.dumps(result.to_dict())  # must not raise
