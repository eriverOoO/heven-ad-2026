from scripts import prepare_checkpoint_segment as segment


def test_extract_segment_selects_ordered_endpoints_and_removes_duplicates():
    points = [
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    ]
    result, start_index, end_index = segment.extract_segment(
        points,
        {3: (0.0, 0.0), 6: (2.0, 0.0)},
        3,
        6,
    )

    assert start_index == 2
    assert end_index == 4
    assert result == points[2:5]
    assert segment.segment_length(result) == 2.0


def test_write_atomic_is_idempotent(tmp_path):
    output = tmp_path / "derived" / "cp3_to_cp6.txt"
    points = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]

    segment.write_atomic(output, points)
    first = output.stat()
    segment.write_atomic(output, points)

    assert output.read_text(encoding="utf-8") == (
        "0.0 0.0 1.0\n1.0 0.0 1.0\n"
    )
    assert output.stat().st_mtime_ns == first.st_mtime_ns


def test_corridor_matches_segment_digest_and_has_strict_stations():
    points = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (2.0, 1.0, 1.0),
    ]
    digest = "a" * 64

    corridor = segment.build_route_corridor(
        points,
        global_path_sha256=digest,
        lane_sequence_id="checkpoint:3-6",
    )
    lane = corridor["lanes"][0]

    assert corridor["source_sha256"] == {"global_path": digest}
    assert corridor["primary_lane_sequence_id"] == "checkpoint:3-6"
    assert lane["adjacent_lane_sequence_ids"] == {
        "left": [],
        "right": [],
    }
    assert lane["source_link_ids"] == [
        "global-path-segment:checkpoint:3-6"
    ]
    stations = [point["route_s_m"] for point in lane["points"]]
    assert all(second > first for first, second in zip(stations, stations[1:]))
