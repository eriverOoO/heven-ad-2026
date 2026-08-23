from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_camera_lidar_tracking.sh"


def test_runner_is_portable_and_launches_the_one_command_entrypoint():
    source = RUNNER.read_text(encoding="utf-8")

    assert RUNNER.stat().st_mode & 0o111
    assert "/home/" not in source
    assert "HEVEN_AD_WS_PATH" in source
    assert "camera_lidar_tracking_replay.launch.py" in source
    assert "rosbag2_storage_mcap" in source
    assert "compressed_image_transport" in source
    assert "metadata.yaml" in source


def test_runner_defaults_to_untracked_camera_bag_and_half_speed():
    source = RUNNER.read_text(encoding="utf-8")

    assert "morai_cam4_20260813_163222" in source
    assert 'rate="${2:-0.5}"' in source
