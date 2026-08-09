import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import derive_global_path as path_tool
from derive_global_path import DerivedPath, derive_path


SCRIPT = SCRIPTS / "derive_global_path.py"


def run_tool(input_path, output_path, *arguments):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=2.0,
    )


def write_path(path, text):
    path.write_bytes(text.encode("utf-8"))


def assert_points_close(actual, expected):
    assert len(actual) == len(expected)
    for actual_point, expected_point in zip(actual, expected):
        assert actual_point == pytest.approx(expected_point)


def test_derives_a_typed_open_path_with_consecutive_duplicates_removed():
    derived = derive_path(
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (2.0, 0.0, 3.0)),
        spacing_m=1.0,
        smooth_passes=0,
    )

    assert isinstance(derived, DerivedPath)
    assert derived.closed is False
    assert derived.points == ((0.0, 0.0, 1.0), (1.0, 0.0, 2.0), (2.0, 0.0, 3.0))


def test_resamples_an_open_three_meter_segment_at_uniform_one_meter_spacing():
    derived = derive_path(
        ((0.0, 0.0, 0.0), (3.0, 0.0, 3.0)),
        spacing_m=1.0,
        smooth_passes=0,
    )

    assert derived.points == (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 2.0),
        (3.0, 0.0, 3.0),
    )


def test_non_divisible_open_path_keeps_requested_samples_and_exact_endpoint():
    derived = derive_path(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0)),
        spacing_m=0.3,
        smooth_passes=0,
    )

    assert_points_close(
        derived.points,
        (
            (0.0, 0.0, 0.0),
            (0.3, 0.0, 0.3),
            (0.6, 0.0, 0.6),
            (0.9, 0.0, 0.9),
            (1.0, 0.0, 1.0),
        ),
    )
    assert derived.points[-1] == (1.0, 0.0, 1.0)


def test_divisible_decimal_open_path_has_no_near_duplicate_endpoint():
    derived = derive_path(
        ((0.0, 0.0, 0.0), (0.8, 0.0, 0.0)),
        spacing_m=0.1,
        smooth_passes=0,
    )

    assert len(derived.points) == 9
    assert derived.points[-1] == (0.8, 0.0, 0.0)
    assert derived.points[-2][0] == pytest.approx(0.7)


def test_tiny_open_path_keeps_both_start_and_endpoint():
    derived = derive_path(
        ((0.0, 0.0, 0.0), (1.0e-15, 0.0, 0.0)),
        spacing_m=1.0,
        smooth_passes=0,
        duplicate_tolerance_m=1.0e-16,
    )

    assert derived.points == (
        (0.0, 0.0, 0.0),
        (1.0e-15, 0.0, 0.0),
    )


def test_endpoint_merge_tolerance_does_not_swallow_a_valid_spacing_sample():
    endpoint = 1.0 + 5.0e-13
    derived = derive_path(
        ((0.0, 0.0, 0.0), (endpoint, 0.0, 0.0)),
        spacing_m=1.0,
        smooth_passes=0,
        duplicate_tolerance_m=1.0e-14,
    )

    assert derived.points == (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (endpoint, 0.0, 0.0),
    )


def test_derive_path_revalidates_the_resampled_minimum_point_count(monkeypatch):
    monkeypatch.setattr(
        path_tool,
        "_resample",
        lambda *args, **kwargs: [(0.0, 0.0, 0.0)],
    )

    with pytest.raises(ValueError, match="derived path requires at least two points"):
        derive_path(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            spacing_m=1.0,
            smooth_passes=0,
        )


def test_smoothing_preserves_open_endpoints():
    derived = derive_path(
        ((0.0, 0.0, 0.0), (1.0, 2.0, 0.0), (2.0, 0.0, 0.0)),
        spacing_m=math.sqrt(5.0),
        smooth_passes=1,
    )

    assert derived.closed is False
    assert derived.points[0] == (0.0, 0.0, 0.0)
    assert derived.points[-1] == (2.0, 0.0, 0.0)
    assert derived.points[1] == pytest.approx((1.0, 1.0, 0.0))


def test_closed_path_wraps_smoothing_neighbors_and_repeats_first_output_point():
    derived = derive_path(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        spacing_m=2.0,
        smooth_passes=1,
    )

    assert derived.closed is True
    assert derived.points[0] == derived.points[-1]
    assert derived.points[0] == (0.5, 0.5, 0.0)
    assert len(derived.points) == 5


def test_non_divisible_closed_path_distributes_the_remainder_around_the_loop():
    derived = derive_path(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        spacing_m=3.0,
        smooth_passes=0,
    )

    assert_points_close(
        derived.points,
        (
            (0.0, 0.0, 0.0),
            (2.0, 2.0 / 3.0, 0.0),
            (2.0 / 3.0, 2.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
    )


def test_closed_path_with_spacing_larger_than_loop_keeps_two_intervals():
    derived = derive_path(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        spacing_m=10.0,
        smooth_passes=0,
    )

    assert derived.points == (
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
    )


def test_duplicate_tolerance_removes_points_inside_and_on_the_boundary():
    derived = derive_path(
        (
            (0.0, 0.0, 0.0),
            (0.0005, 0.0, 10.0),
            (0.001, 0.0, 20.0),
            (1.0, 0.0, 1.0),
        ),
        spacing_m=1.0,
        smooth_passes=0,
        duplicate_tolerance_m=0.001,
    )

    assert derived.points == ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0))


def test_duplicate_tolerance_rejects_a_path_that_collapses_to_one_point():
    with pytest.raises(ValueError):
        derive_path(
            ((0.0, 0.0, 0.0), (0.0005, 0.0, 1.0), (0.001, 0.0, 2.0)),
            spacing_m=1.0,
            smooth_passes=0,
            duplicate_tolerance_m=0.001,
        )


@pytest.mark.parametrize(
    ("spacing_m", "smooth_passes", "duplicate_tolerance_m"),
    [
        (0.0, 0, 1.0e-6),
        (-1.0, 0, 1.0e-6),
        (1.0, -1, 1.0e-6),
        (1.0, 0, 0.0),
        (1.0, 0, -1.0),
        (math.nan, 0, 1.0e-6),
        (math.inf, 0, 1.0e-6),
        (1.0, 0, math.nan),
        (1.0, 0, math.inf),
    ],
)
def test_rejects_invalid_derivation_parameters(
    spacing_m, smooth_passes, duplicate_tolerance_m
):
    with pytest.raises(ValueError):
        derive_path(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            spacing_m=spacing_m,
            smooth_passes=smooth_passes,
            duplicate_tolerance_m=duplicate_tolerance_m,
        )


def test_extreme_finite_xy_geometry_is_rejected_without_an_unbounded_loop(tmp_path):
    input_path = tmp_path / "source.txt"
    output_path = tmp_path / "derived.txt"
    write_path(input_path, "-1e308 0\n1e308 0\n")

    completed = run_tool(input_path, output_path, "--spacing-m", "0.5")

    assert completed.returncode != 0
    assert "finite" in completed.stderr
    assert not output_path.exists()
    assert not Path(f"{output_path}.json").exists()


def test_stable_interpolation_keeps_extreme_finite_z_values_finite():
    derived = derive_path(
        ((0.0, 0.0, -1.0e308), (1.0, 0.0, 1.0e308)),
        spacing_m=0.5,
        smooth_passes=0,
    )

    assert derived.points[1] == (0.5, 0.0, 0.0)
    assert all(math.isfinite(value) for point in derived.points for value in point)


def test_repeated_smoothing_keeps_large_finite_results_finite():
    derived = derive_path(
        (
            (0.0, 0.0, 1.0e308),
            (1.0, 1.0, 1.0e308),
            (2.0, 0.0, 1.0e308),
        ),
        spacing_m=math.sqrt(2.0),
        smooth_passes=3,
    )

    assert all(math.isfinite(value) for point in derived.points for value in point)
    assert all(point[2] == pytest.approx(1.0e308) for point in derived.points)


def test_resampling_scans_source_segments_monotonically(monkeypatch):
    calls = 0
    real_distance = path_tool._xy_distance

    def counting_distance(first, second):
        nonlocal calls
        calls += 1
        return real_distance(first, second)

    monkeypatch.setattr(path_tool, "_xy_distance", counting_distance)
    derived = derive_path(
        tuple((float(index), 0.0, 0.0) for index in range(101)),
        spacing_m=0.25,
        smooth_passes=0,
    )

    assert len(derived.points) == 401
    assert calls < 1000


@pytest.mark.parametrize(
    "text",
    [
        "0,0\n1 1\n",
        "0 0\n1 1 0\n",
        "0,0\nnan,1\n",
        "0 0\n1 nope\n",
        "0 0 0 0\n1 0 0\n",
        "0 0\n",
    ],
)
def test_cli_rejects_mixed_or_invalid_input_without_changing_source(tmp_path, text):
    input_path = tmp_path / "source.txt"
    output_path = tmp_path / "derived.txt"
    write_path(input_path, text)
    before = input_path.read_bytes()

    completed = run_tool(input_path, output_path)

    assert completed.returncode != 0
    assert input_path.read_bytes() == before
    assert not output_path.exists()
    assert not Path(f"{output_path}.json").exists()


def test_cli_rejects_identical_input_and_output_without_changing_source(tmp_path):
    input_path = tmp_path / "source.txt"
    write_path(input_path, "0 0\n1 0\n")
    before = input_path.read_bytes()

    completed = run_tool(input_path, input_path)

    assert completed.returncode != 0
    assert input_path.read_bytes() == before


@pytest.mark.parametrize("existing_name", ("derived.txt", "derived.txt.json"))
def test_cli_refuses_existing_output_or_sidecar_without_changing_source(
    tmp_path, existing_name
):
    input_path = tmp_path / "source.txt"
    output_path = tmp_path / "derived.txt"
    write_path(input_path, "0 0\n1 0\n")
    existing_path = tmp_path / existing_name
    existing_bytes = b"existing\x00bytes\n"
    existing_path.write_bytes(existing_bytes)
    before = input_path.read_bytes()

    completed = run_tool(input_path, output_path)

    assert completed.returncode != 0
    assert input_path.read_bytes() == before
    assert existing_path.read_bytes() == existing_bytes
    if existing_path == output_path:
        assert not Path(f"{output_path}.json").exists()
    else:
        assert not output_path.exists()


def test_derive_file_reads_one_immutable_input_snapshot(tmp_path, monkeypatch):
    input_path = tmp_path / "source.txt"
    output_path = tmp_path / "derived.txt"
    source_bytes = b"0 0\n1 0\n"
    input_path.write_bytes(source_bytes)
    real_read_bytes = Path.read_bytes
    input_reads = 0

    def counted_read_bytes(path):
        nonlocal input_reads
        if path == input_path:
            input_reads += 1
        return real_read_bytes(path)

    def forbidden_read_text(path, *args, **kwargs):
        raise AssertionError(f"second text read attempted for {path}")

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    monkeypatch.setattr(Path, "read_text", forbidden_read_text)

    path_tool.derive_file(
        input_path,
        output_path,
        spacing_m=0.5,
        smooth_passes=0,
        duplicate_tolerance_m=1.0e-6,
    )

    assert input_reads == 1
    with Path(f"{output_path}.json").open(encoding="utf-8") as stream:
        sidecar = json.load(stream)
    assert sidecar["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert sidecar["source_point_count"] == 2


@pytest.mark.parametrize("failed_write", (1, 2))
def test_partial_temp_write_failure_leaves_no_output_sidecar_or_temp(
    tmp_path, monkeypatch, failed_write
):
    output_path = tmp_path / "derived.txt"
    calls = 0

    def injected_write(file_descriptor, payload):
        nonlocal calls
        calls += 1
        os.write(file_descriptor, payload[:1])
        if calls == failed_write:
            raise OSError("injected partial write")
        os.write(file_descriptor, payload[1:])
        os.fsync(file_descriptor)

    monkeypatch.setattr(path_tool, "_write_fully", injected_write, raising=False)

    with pytest.raises(OSError, match="injected partial write"):
        path_tool._write_new_files(output_path, b"output", b"sidecar")

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "failed_publish",
    (1, 2),
    ids=("output", "sidecar"),
)
def test_link_success_then_immediate_failure_leaves_no_published_or_temp_files(
    tmp_path, monkeypatch, failed_publish
):
    output_path = tmp_path / "derived.txt"
    real_publish = path_tool._publish_no_clobber
    calls = 0

    def injected_publish(temporary, destination, expected_identity):
        nonlocal calls
        calls += 1
        if calls == failed_publish:
            os.link(temporary, destination, follow_symlinks=False)
            raise OSError("injected failure immediately after link")
        real_publish(temporary, destination, expected_identity)

    monkeypatch.setattr(path_tool, "_publish_no_clobber", injected_publish)

    with pytest.raises(OSError, match="immediately after link"):
        path_tool._write_new_files(output_path, b"output", b"sidecar")

    assert list(tmp_path.iterdir()) == []


def test_cli_writes_linked_sidecar_and_preserves_source_bytes(tmp_path):
    input_path = tmp_path / "source.txt"
    output_path = tmp_path / "derived.txt"
    write_path(input_path, "0,0,0\n1,0,1\n1,0,1\n3,0,3\n")
    before = input_path.read_bytes()

    completed = run_tool(
        input_path,
        output_path,
        "--spacing-m",
        "1.0",
        "--smooth-passes",
        "0",
        "--duplicate-tolerance-m",
        "0.001",
    )

    assert completed.returncode == 0, completed.stderr
    assert input_path.read_bytes() == before
    sidecar = json.loads(Path(f"{output_path}.json").read_text(encoding="utf-8"))
    result_bytes = output_path.read_bytes()
    assert sidecar == {
        "closed": False,
        "derived_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "derived_point_count": 4,
        "duplicate_tolerance_m": 0.001,
        "smooth_passes": 0,
        "source_point_count": 4,
        "source_sha256": hashlib.sha256(before).hexdigest(),
        "spacing_m": 1.0,
    }
