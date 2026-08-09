import sqlite3
import struct
from pathlib import Path

from heven_slam.rtabmap_db_quality import analyze_database, format_report


def _transform(x: float, y: float = 0.0, z: float = 0.0) -> bytes:
    return struct.pack(
        "<12f",
        1.0,
        0.0,
        0.0,
        x,
        0.0,
        1.0,
        0.0,
        y,
        0.0,
        0.0,
        1.0,
        z,
    )


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE Node (
            id INTEGER PRIMARY KEY,
            map_id INTEGER NOT NULL,
            stamp FLOAT,
            pose BLOB,
            gps BLOB
        );
        CREATE TABLE Data (id INTEGER PRIMARY KEY, scan BLOB);
        CREATE TABLE Link (
            from_id INTEGER NOT NULL,
            to_id INTEGER NOT NULL,
            type INTEGER NOT NULL,
            transform BLOB
        );
        """
    )
    return connection


def test_rejects_disconnected_non_returning_segments(tmp_path: Path):
    path = tmp_path / "segments.db"
    connection = _database(path)
    connection.executemany(
        "INSERT INTO Node VALUES (?, ?, ?, ?, ?)",
        [
            (1, 0, 0.0, _transform(0.0), b"gps"),
            (2, 0, 1.0, _transform(10.0), b"gps"),
            (3, 1, 2.0, _transform(0.0), b"gps"),
            (4, 1, 3.0, _transform(10.0), b"gps"),
        ],
    )
    connection.executemany(
        "INSERT INTO Data VALUES (?, ?)", [(node_id, b"scan") for node_id in range(1, 5)]
    )
    connection.executemany(
        "INSERT INTO Link VALUES (?, ?, 0, ?)",
        [(1, 2, _transform(1.0)), (3, 4, _transform(1.0))],
    )
    connection.commit()
    connection.close()

    report = analyze_database(path)

    assert report.graph_components == 2
    assert report.closure_links == 0
    assert not report.ready_for_loop_validation
    assert "NOT READY" in format_report(report)


def test_accepts_connected_closed_route_with_loop_closure(tmp_path: Path):
    path = tmp_path / "closed.db"
    connection = _database(path)
    poses = [0.0, 30.0, 0.0]
    connection.executemany(
        "INSERT INTO Node VALUES (?, 0, ?, ?, ?)",
        [
            (node_id, float(node_id), _transform(x), b"gps")
            for node_id, x in enumerate(poses, start=1)
        ],
    )
    connection.executemany(
        "INSERT INTO Data VALUES (?, ?)", [(node_id, b"scan") for node_id in range(1, 4)]
    )
    connection.executemany(
        "INSERT INTO Link VALUES (?, ?, ?, ?)",
        [
            (1, 2, 0, _transform(1.0)),
            (2, 3, 0, _transform(1.0)),
            (3, 1, 1, _transform(0.1)),
        ],
    )
    connection.commit()
    connection.close()

    report = analyze_database(path)

    assert report.graph_components == 1
    assert report.closure_links == 1
    assert report.sessions[0].closed_loop_candidate
    assert report.ready_for_loop_validation
