import math
import sys
from pathlib import Path

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_morai_dataset import (
    Actor,
    NearestIndex,
    TimedSample,
    _num_points_inside_box,
    _preflight_bags,
    _same_static_transform_contract,
    _transform_box,
)
from geometry import RigidTransform, quaternion_from_rpy


def test_nearest_index_rejects_large_source_gap():
    index = NearestIndex([TimedSample(100, 1000, "value")])
    assert index.nearest(109, 10).value == "value"
    assert index.nearest(111, 10) is None


def test_transform_composition_and_inverse_round_trip():
    first = RigidTransform((1.0, 2.0, 3.0), quaternion_from_rpy(0.0, 0.0, 0.5))
    second = RigidTransform((-2.0, 0.0, 1.0), quaternion_from_rpy(0.0, 0.0, -0.2))
    point = (4.0, 5.0, 6.0)
    composed = first.compose(second)
    assert composed.apply(point) == pytest.approx(first.apply(second.apply(point)))
    assert composed.inverse().apply(composed.apply(point)) == pytest.approx(point)


def test_vehicle_center_uses_rear_axle_geometry_without_dimension_swap():
    actor = Actor(
        1,
        1,
        (0.0, 0.0, 0.0),
        0.0,
        (4.0, 2.0, 1.5),
        0.8,
        2.8,
        0.4,
    )
    box = _transform_box(
        actor,
        {
            "name": "vehicle",
            "center_policy": "rear_axle_ground_to_box_center",
            "length_tolerance_m": 0.01,
        },
        RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )
    assert (box["x"], box["y"], box["z"]) == pytest.approx((1.6, 0.0, 0.75))
    assert (box["length"], box["width"], box["height"]) == (4.0, 2.0, 1.5)
    assert box["yaw"] == pytest.approx(0.0)


def test_yaw_is_expressed_in_target_frame_and_normalized():
    actor = Actor(3, 0, (0.0, 0.0, 0.0), 0.0, (0.3, 0.6, 1.8), 0.0, 0.0, 0.0)
    box = _transform_box(
        actor,
        {"name": "pedestrian", "center_policy": "ground_center_to_box_center"},
        RigidTransform((0.0, 0.0, 0.0), quaternion_from_rpy(0.0, 0.0, math.pi / 2.0)),
    )
    assert box["yaw"] == pytest.approx(math.pi / 2.0)


def test_vehicle_length_mismatch_is_rejected():
    actor = Actor(1, 1, (0.0, 0.0, 0.0), 0.0, (5.0, 2.0, 1.5), 0.8, 2.8, 0.4)
    with pytest.raises(ValueError, match="vehicle_length_geometry_mismatch"):
        _transform_box(
            actor,
            {
                "name": "vehicle",
                "center_policy": "rear_axle_ground_to_box_center",
                "length_tolerance_m": 0.01,
            },
            RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        )


def test_num_points_inside_box_uses_orientation_height_and_inclusive_edges():
    points = np.array(
        [
            [0.0, 2.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 2.1, 0.0],
            [0.0, -2.1, 1.0, 0.0],
        ]
    )
    box = {
        "x": 0.0,
        "y": 0.0,
        "z": 1.0,
        "length": 4.0,
        "width": 2.0,
        "height": 2.0,
        "yaw": math.pi / 2.0,
    }
    assert _num_points_inside_box(points, box) == 2


def test_preflight_requires_explicit_whole_scene_split_and_scenario(tmp_path):
    bag = tmp_path / "scene_a"
    bag.mkdir()
    scenario = tmp_path / "scenarios" / "a.json"
    scenario.parent.mkdir()
    scenario.write_text("{}\n", encoding="utf-8")
    config = {
        "dataset": {"version": "dataset_v1"},
        "scenes": {
            "scene_a": {"split": "val", "scenario": "scenarios/a.json"}
        },
    }
    assert _preflight_bags([bag], config, tmp_path) == [
        (
            bag.resolve(),
            "scene_a",
            {"split": "val", "scenario": "scenarios/a.json"},
        )
    ]

    config["scenes"]["scene_a"]["split"] = "unassigned"
    with pytest.raises(RuntimeError, match="explicit train/val/test"):
        _preflight_bags([bag], config, tmp_path)

    config["scenes"]["scene_a"]["split"] = "val"
    scenario.unlink()
    with pytest.raises(RuntimeError, match="scenario evidence does not exist"):
        _preflight_bags([bag], config, tmp_path)


def test_preflight_rejects_duplicate_scene_basenames(tmp_path):
    first = tmp_path / "first" / "scene_a"
    second = tmp_path / "second" / "scene_a"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    config = {
        "dataset": {"version": "dataset_v1"},
        "scenes": {
            "scene_a": {"split": "train", "scenario": "scenarios/a.json"}
        },
    }
    with pytest.raises(RuntimeError, match="duplicate scene name"):
        _preflight_bags([first, second], config)


def test_static_transform_contract_accepts_quaternion_sign_only():
    first = {
        "base_link->lidar_link": {
            "translation": [1.0, 2.0, 3.0],
            "quaternion_xyzw": [0.0, 0.0, 0.5, 0.5],
        }
    }
    equivalent = {
        "base_link->lidar_link": {
            "translation": [1.0, 2.0, 3.0],
            "quaternion_xyzw": [0.0, 0.0, -0.5, -0.5],
        }
    }
    moved = {
        "base_link->lidar_link": {
            "translation": [1.01, 2.0, 3.0],
            "quaternion_xyzw": [0.0, 0.0, 0.5, 0.5],
        }
    }
    assert _same_static_transform_contract(first, equivalent)
    assert not _same_static_transform_contract(first, moved)
