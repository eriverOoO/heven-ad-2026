from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_camera_lidar_tracking.sh"


def run_runner(*arguments):
    return subprocess.run(
        [str(RUNNER), *arguments],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def test_runner_is_portable_and_launches_the_preset_entrypoint():
    source = RUNNER.read_text(encoding="utf-8")

    assert RUNNER.stat().st_mode & 0o111
    assert "/home/" not in source
    assert "HEVEN_AD_WS_PATH" in source
    assert "HEVEN_CENTERPOINT_CHECKPOINT" in source
    assert "HEVEN_KALMANNET_CHECKPOINT" in source
    assert "camera_lidar_tracking_replay.launch.py" in source
    assert "rosbag2_storage_mcap" in source
    assert "compressed_image_transport" in source
    assert "metadata.yaml" in source
    assert 'for required_topic in "${required_topics[@]}"' in source


def test_runner_lists_all_ten_modes_without_requiring_ros_runtime():
    result = run_runner("--list-modes")

    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 10
    assert lines[0].lstrip().startswith("1: Euclidean / GIoU / Greedy")
    assert lines[6].lstrip().endswith("KalmanNet")
    assert lines[9].lstrip().startswith(
        "10: CenterPoint / Mahalanobis Hybrid 10m"
    )
    assert "(default)" in lines[2]


def test_runner_rejects_unknown_mode_before_runtime_checks():
    result = run_runner("--mode", "11")

    assert result.returncode != 0
    assert "1 through 10" in result.stderr


def test_runner_keeps_positional_bag_and_rate_compatibility():
    source = RUNNER.read_text(encoding="utf-8")

    assert "morai_cam4_20260813_163222" in source
    assert 'rate="0.5"' in source
    assert 'bag_path="${positional[0]}"' in source
    assert 'rate="${positional[1]}"' in source
    assert "references/openpcdet" in source
    assert "models/experimental/dense_kalmannet_v2.pt" in source
    assert "models/experimental/centerpoint_t14_reproduction.pth" in source


def test_runner_exposes_all_model_artifact_and_device_arguments():
    help_result = run_runner("--help")

    assert help_result.returncode == 0
    for option in (
        "--mode",
        "--centerpoint-checkpoint",
        "--openpcdet-root",
        "--centerpoint-device",
        "--kalmannet-checkpoint",
        "--kalmannet-device",
    ):
        assert option in help_result.stderr


def test_runner_pins_the_two_frozen_checkpoint_hashes():
    source = RUNNER.read_text(encoding="utf-8")

    assert "466c8181a377682e032bb32579c8ddb65807b5feebd8625a750b1d5538ddbc95" in source
    assert "956604975e5204b2c584c3e8fe16a7ba9346097980566ecbbb22c2001fbf7d48" in source
    assert 'verify_sha256 "CenterPoint"' in source
    assert 'verify_sha256 "KalmanNet"' in source
