import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _stamp(nanoseconds):
    return SimpleNamespace(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


def _header(nanoseconds):
    return SimpleNamespace(stamp=_stamp(nanoseconds))


def _imu(nanoseconds):
    return SimpleNamespace(header=_header(nanoseconds))


def _timing(nanoseconds, *, stream="imu", published=True):
    return SimpleNamespace(
        header=_header(nanoseconds),
        stream=stream,
        source_valid=True,
        source_selected=False,
        source_rejected=False,
        duplicate=False,
        stamp_regression=False,
        normalized_published=published,
    )


def _statistics(count, *, stream="imu"):
    return SimpleNamespace(
        stream=stream,
        packets=count,
        bytes=count * 10,
        malformed=0,
        dropped=0,
        bind_errors=0,
        source_selected=0,
        arrival_fallback=count,
        source_rejected=0,
        duplicates=0,
        stamp_regressions=0,
    )


def test_parser_rejects_unsupported_rate_and_nonpositive_or_nonfinite_intervals():
    from ad_morai_bridge_dev.imu_timing.eval import parse_arguments

    base = [
        "--expected-rate-hz",
        "20",
        "--duration-sec",
        "1",
        "--warmup-sec",
        "1",
        "--startup-timeout-sec",
        "1",
        "--drain-sec",
        "1",
        "--run-id",
        "run-1",
    ]
    for arguments in (
        [*base, "--expected-rate-hz", "40"],
        [*base, "--duration-sec", "0"],
        [*base, "--warmup-sec", "nan"],
        [*base, "--startup-timeout-sec", "inf"],
        [*base, "--drain-sec", "-1"],
    ):
        with pytest.raises(SystemExit):
            parse_arguments(arguments)


def test_report_directory_fails_closed_for_bad_data_root_run_id_reuse_and_escape(
    tmp_path, monkeypatch
):
    from ad_morai_bridge_dev.imu_timing.eval import create_report_artifact

    monkeypatch.delenv("AD_DATA_DIR", raising=False)
    with pytest.raises(ValueError, match="AD_DATA_DIR"):
        create_report_artifact("safe-run")

    monkeypatch.setenv("AD_DATA_DIR", "relative-data")
    with pytest.raises(ValueError, match="absolute"):
        create_report_artifact("safe-run")

    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))
    for run_id in ("../escape", ".", "a/b"):
        with pytest.raises(ValueError, match="run_id"):
            create_report_artifact(run_id)

    artifact = create_report_artifact("safe-run")
    assert artifact.report == tmp_path / "experiments/morai_imu_timing/safe-run/report.json"
    with pytest.raises(FileExistsError, match="already exists"):
        create_report_artifact("safe-run")

    escaped = tmp_path / "outside"
    escaped.mkdir()
    base = tmp_path / "experiments/morai_imu_timing"
    base.unlink() if base.is_symlink() else None
    # A symlinked base is a path escape even though the lexical run ID is safe.
    if base.exists():
        # The prior artifact made a real base, so test escaping under a fresh root.
        other_root = tmp_path / "other-root"
        other_root.mkdir()
        monkeypatch.setenv("AD_DATA_DIR", str(other_root))
        base = other_root / "experiments/morai_imu_timing"
    base.parent.mkdir(parents=True, exist_ok=True)
    base.symlink_to(escaped, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe artifact directory"):
        create_report_artifact("escaped")


def test_optional_sensor_profile_requires_valid_imu_periods(tmp_path):
    from ad_morai_bridge_dev.imu_timing.eval import sensor_profile_provenance

    with pytest.raises(ValueError, match="does not exist"):
        sensor_profile_provenance(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"IMUList": [{"ic": {"sensorPeriod": 0}}]}', encoding="utf-8")
    with pytest.raises(ValueError, match="sensorPeriod"):
        sensor_profile_provenance(invalid)

    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps({"IMUList": [{"m_SensorUniqueID": 7, "ic": {"sensorPeriod": 0.02}}]}),
        encoding="utf-8",
    )
    provenance = sensor_profile_provenance(profile)
    assert provenance["sha256"]
    assert provenance["imu_periods"] == [
        {"index": 0, "unique_id": 7, "sensor_period_sec": 0.02}
    ]

    dual_key = tmp_path / "dual-key.json"
    dual_key.write_text(
        json.dumps(
            {
                "IMUList": [
                    {
                        "m_SensorUniqueID": 7,
                        "UNIQUEID": 7,
                        "ic": {"sensorPeriod": 0.02},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert sensor_profile_provenance(dual_key)["imu_periods"][0]["unique_id"] == 7

    disagreeing = tmp_path / "disagreeing.json"
    disagreeing.write_text(
        json.dumps(
            {
                "IMUList": [
                    {
                        "m_SensorUniqueID": 7,
                        "UNIQUEID": 8,
                        "ic": {"sensorPeriod": 0.02},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identifier"):
        sensor_profile_provenance(disagreeing)


def test_sensor_profile_rejects_swappable_symlink_and_oversized_integer_period(
    tmp_path,
):
    from ad_morai_bridge_dev.imu_timing.eval import sensor_profile_provenance

    first = tmp_path / "first.json"
    first.write_text(
        json.dumps({"IMUList": [{"m_SensorUniqueID": 7, "ic": {"sensorPeriod": 0.02}}]}),
        encoding="utf-8",
    )
    second = tmp_path / "second.json"
    second.write_text(
        json.dumps({"IMUList": [{"m_SensorUniqueID": 7, "ic": {"sensorPeriod": 0.05}}]}),
        encoding="utf-8",
    )
    swappable = tmp_path / "profile-link.json"
    swappable.symlink_to(first)
    # A link can be changed after validation; reject it before any bytes,
    # metadata, or canonical path are claimed as one provenance record.
    with pytest.raises(ValueError, match="symlink"):
        sensor_profile_provenance(swappable)
    swappable.unlink()
    swappable.symlink_to(second)
    with pytest.raises(ValueError, match="symlink"):
        sensor_profile_provenance(swappable)

    oversized = tmp_path / "oversized-period.json"
    oversized.write_text(
        '{"IMUList":[{"m_SensorUniqueID":7,"ic":{"sensorPeriod":' + "1" + "0" * 400 + "}}]}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sensorPeriod"):
        sensor_profile_provenance(oversized)


def test_report_writer_rejects_post_validation_run_directory_symlink_swap(
    tmp_path, monkeypatch
):
    import ad_morai_bridge_dev.imu_timing.eval as evaluation

    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))
    artifact = evaluation.create_report_artifact("swap-run")
    outside = tmp_path / "outside"
    outside.mkdir()
    original_validate = evaluation._validate_report_target

    def validate_then_swap(path):
        result = original_validate(path)
        artifact.run_directory.rmdir()
        artifact.run_directory.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(evaluation, "_validate_report_target", validate_then_swap)
    with pytest.raises(ValueError, match="unsafe artifact directory"):
        evaluation.write_report(artifact.report, {"safe": True})
    assert not (outside / "report.json").exists()


def test_main_shuts_down_when_ros_node_construction_fails(tmp_path, monkeypatch):
    import ad_morai_bridge_dev.imu_timing.eval as evaluation

    class FakeRclpy:
        def __init__(self):
            self.initialized = False
            self.try_shutdown_calls = 0

        def init(self, *, args):
            self.initialized = True

        def try_shutdown(self):
            self.try_shutdown_calls += 1

    fake_rclpy = FakeRclpy()
    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        evaluation,
        "_ros_dependencies",
        lambda: (fake_rclpy, object, object, object, object, (object,) * 4),
    )

    def fail_constructor(_collector):
        raise RuntimeError("subscription setup failed")

    monkeypatch.setattr(evaluation, "ImuTimingEvaluationNode", fail_constructor)
    with pytest.raises(RuntimeError, match="subscription setup failed"):
        evaluation.main(
            [
                "--expected-rate-hz", "20", "--duration-sec", "1",
                "--warmup-sec", "1", "--startup-timeout-sec", "1",
                "--drain-sec", "1", "--run-id", "partial-init",
            ]
        )
    assert fake_rclpy.initialized is True
    assert fake_rclpy.try_shutdown_calls == 1


def test_collector_requires_four_actual_surfaces_and_excludes_warmup_and_edge_callbacks():
    from ad_morai_bridge_dev.imu_timing.eval import TimingCollector

    collector = TimingCollector(warmup_ns=5, duration_ns=10, drain_ns=3)
    collector.on_normalized(_imu(100), receipt_ns=1)
    collector.on_full(_imu(100), receipt_ns=1)
    collector.on_timing(_timing(100), receipt_ns=1)
    assert collector.phase == "waiting"
    collector.on_statistics(_statistics(10), receipt_ns=1)
    assert collector.phase == "warmup"

    collector.tick(6)
    assert collector.phase == "measuring"
    collector.on_normalized(_imu(200), receipt_ns=5)  # warmup, excluded
    for receipt_ns, stamp_ns in ((6, 600), (15, 1500), (16, 1600)):
        collector.on_normalized(_imu(stamp_ns), receipt_ns=receipt_ns)
        collector.on_full(_imu(stamp_ns), receipt_ns=receipt_ns)
        collector.on_timing(_timing(stamp_ns), receipt_ns=receipt_ns)
    collector.tick(16)
    assert collector.phase == "draining"
    collector.on_statistics(_statistics(12), receipt_ns=17)
    collector.tick(19)

    window = collector.frozen_window()
    assert window.normalized_headers == (600, 1500)
    assert window.full_headers == (600, 1500)
    assert window.timing_headers == (600, 1500)
    assert window.measurement_duration_ns == 10
    assert window.statistics_discovered is True
    assert window.start_counters.packets == 10
    assert window.end_counters.packets == 12


def test_collector_tick_stays_in_warmup_until_warmup_deadline():
    from ad_morai_bridge_dev.imu_timing.eval import TimingCollector

    collector = TimingCollector(warmup_ns=10, duration_ns=20, drain_ns=3)
    for callback, message in (
        (collector.on_normalized, _imu(1)),
        (collector.on_full, _imu(1)),
        (collector.on_timing, _timing(1)),
        (collector.on_statistics, _statistics(1)),
    ):
        callback(message, receipt_ns=100)

    for receipt_ns in (101, 109):
        collector.tick(receipt_ns)
        assert collector.phase == "warmup"
        lifecycle = collector.lifecycle()
        assert lifecycle["measurement_start_monotonic_ns"] is None
        assert lifecycle["measurement_end_monotonic_ns"] is None


def test_frozen_header_range_accepts_drain_late_counterpart_but_rejects_outside_header():
    from ad_morai_bridge_dev.imu_timing.eval import TimingCollector

    collector = TimingCollector(warmup_ns=1, duration_ns=10, drain_ns=3)
    for callback, message in (
        (collector.on_normalized, _imu(1)),
        (collector.on_full, _imu(1)),
        (collector.on_timing, _timing(1)),
        (collector.on_statistics, _statistics(4)),
    ):
        callback(message, receipt_ns=0)
    collector.tick(1)
    # Timing/full establish H inside the measurement.  Their normalized mate
    # reaches this callback only after the measurement endpoint, during drain.
    collector.on_full(_imu(700), receipt_ns=5)
    collector.on_timing(_timing(700), receipt_ns=5)
    collector.tick(11)
    collector.on_normalized(_imu(700), receipt_ns=12)
    collector.on_normalized(_imu(999), receipt_ns=12)
    collector.on_statistics(_statistics(5), receipt_ns=12)
    collector.on_statistics(_statistics(6), receipt_ns=13)
    collector.tick(14)

    window = collector.frozen_window()
    assert window.normalized_headers == (700,)
    assert window.full_headers == (700,)
    assert window.timing_headers == (700,)
    # The end snapshot is the latest IMU-only post-window sample in drain.
    assert window.end_counters.packets == 6


def test_collector_filters_shared_topics_to_imu_and_adapts_exact_contract_fields():
    from ad_morai_bridge_dev.imu_timing.eval import TimingCollector

    collector = TimingCollector(warmup_ns=1, duration_ns=2, drain_ns=1)
    collector.on_normalized(_imu(1), receipt_ns=0)
    collector.on_full(_imu(1), receipt_ns=0)
    collector.on_timing(_timing(1, stream="gps"), receipt_ns=0)
    collector.on_statistics(_statistics(4, stream="gps"), receipt_ns=0)
    assert collector.phase == "waiting"
    collector.on_timing(_timing(2), receipt_ns=0)
    collector.on_statistics(_statistics(4), receipt_ns=0)
    collector.tick(1)
    collector.on_normalized(_imu(10), receipt_ns=1)
    collector.on_full(_imu(10), receipt_ns=1)
    collector.on_timing(_timing(10, published=False), receipt_ns=1)
    collector.tick(3)
    collector.on_statistics(_statistics(5), receipt_ns=3)
    collector.tick(4)

    window = collector.frozen_window()
    assert len(window.audits) == 1
    audit = window.audits[0]
    assert audit.header_stamp_ns == 10
    assert audit.stream == "imu"
    assert audit.normalized_published is False
    assert window.start_counters.packets == 4
    assert window.end_counters.packets == 5


def test_frozen_collector_and_report_use_pure_evaluator_and_strict_counter_encoding(
    tmp_path, monkeypatch
):
    import ad_morai_bridge_dev.imu_timing.eval as evaluation

    collector = evaluation.TimingCollector(warmup_ns=1, duration_ns=2, drain_ns=1)
    for callback, message in (
        (collector.on_normalized, _imu(1)),
        (collector.on_full, _imu(1)),
        (collector.on_timing, _timing(1)),
        (collector.on_statistics, _statistics(1)),
    ):
        callback(message, receipt_ns=0)
    collector.tick(1)
    for callback, message in (
        (collector.on_normalized, _imu(100)),
        (collector.on_full, _imu(100)),
        (collector.on_timing, _timing(100)),
    ):
        callback(message, receipt_ns=1)
    collector.tick(3)
    collector.on_statistics(_statistics(2), receipt_ns=3)
    collector.tick(4)
    frozen = collector.frozen_window()
    collector.on_normalized(_imu(999), receipt_ns=5)
    assert collector.frozen_window() is frozen

    monkeypatch.setattr(evaluation, "evaluate_transport_contract", lambda summary: (2, ("from_pure",)))
    payload, exit_code = evaluation.evaluate_frozen_window(frozen, expected_rate_hz=20)
    assert exit_code == 2
    assert payload["reasons"] == ["from_pure"]

    monkeypatch.setenv("AD_DATA_DIR", str(tmp_path))
    artifact = evaluation.create_report_artifact("strict-json")
    evaluation.write_report(artifact.report, payload)
    with pytest.raises(FileExistsError, match="report already exists"):
        evaluation.write_report(artifact.report, {"replacement": True})
    loaded = json.loads(artifact.report.read_text(encoding="utf-8"))
    assert loaded["summary"]["header_multisets"]["normalized"] == [
        {"count": 1, "stamp_ns": 100}
    ]
