"""Tests for versioned ROS parameter files."""

from pathlib import Path

import yaml

from ad_camera_perception.utils.parameters import (
    validate_normalized_crop,
    validate_normalized_polygon,
)


CONFIG_DIRECTORY = Path(__file__).parents[1] / "config"


def _parameters(file_name, node_name):
    """Load one node's ROS parameter mapping."""
    document = yaml.safe_load((CONFIG_DIRECTORY / file_name).read_text())
    return document[node_name]["ros__parameters"]


def test_dynamic_obstacle_configuration_regions_are_valid():
    """Dynamic detection crop and danger polygon use normalized coordinates."""
    detector = _parameters("dynamic_obstacle.yaml", "dynamic_obstacle_detector")
    evaluator = _parameters("dynamic_obstacle.yaml", "dynamic_obstacle_evaluator")

    validate_normalized_crop(detector["crop_normalized"])
    validate_normalized_polygon(evaluator["danger_roi_normalized"])
    assert detector["excluded_class_ids"] == [9, 10, 11, 12]
    assert detector["image_transport"] == "compressed"
    assert detector["image_topic"] == "/ad/sensors/camera/front/compressed"
    visualizer = _parameters(
        "dynamic_obstacle.yaml", "dynamic_obstacle_visualizer"
    )
    assert visualizer["show_window"] is False


def test_traffic_light_configuration_regions_are_valid():
    """Traffic-light detection and target regions use normalized coordinates."""
    detector = _parameters("traffic_light.yaml", "traffic_light_detector")
    evaluator = _parameters("traffic_light.yaml", "traffic_light_evaluator")
    visualizer = _parameters("traffic_light.yaml", "traffic_light_visualizer")

    assert validate_normalized_crop(detector["target_roi_normalized"]) == (
        0.142857,
        0.0,
        0.857143,
        1.0,
    )
    assert evaluator["target_roi_normalized"] == detector["target_roi_normalized"]
    assert visualizer["target_roi_normalized"] == detector["target_roi_normalized"]
    assert detector["image_topic"] == (
        "/ad/sensors/camera/traffic_light/compressed"
    )
    assert detector["model_path"] == "models/yolov7_best.pt"
    assert detector["image_size"] == 640
    assert evaluator["voting_window_frames"] == 5
    assert evaluator["minimum_vote_frames"] == 3
    assert visualizer["show_window"] is True
    assert visualizer["window_name"] == "mm2025_traffic_light"
