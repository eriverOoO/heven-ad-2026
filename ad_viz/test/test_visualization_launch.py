from pathlib import Path

import yaml


PACKAGE = Path(__file__).resolve().parents[1]


def load_rviz_config(name="ad.rviz"):
    return yaml.safe_load(
        (PACKAGE / "rviz" / name).read_text(encoding="utf-8")
    )


def iter_displays(name="ad.rviz"):
    """Every leaf display, flattened out of the rviz_common/Group tree."""
    pending = list(load_rviz_config(name)["Visualization Manager"]["Displays"])
    leaves = []
    while pending:
        display = pending.pop(0)
        if not isinstance(display, dict):
            continue
        if display.get("Class") == "rviz_common/Group":
            pending = list(display.get("Displays", [])) + pending
        else:
            leaves.append(display)
    return leaves


def display_groups(name="ad.rviz"):
    return {
        display["Name"]: [child["Name"] for child in display["Displays"]]
        for display in load_rviz_config(name)["Visualization Manager"]["Displays"]
        if display.get("Class") == "rviz_common/Group"
    }


def displays_by_name(name="ad.rviz"):
    return {
        display["Name"]: display
        for display in iter_displays(name)
        if "Name" in display
    }


def test_rviz_uses_the_canonical_frames_and_topics():
    config = load_rviz_config()
    manager = config["Visualization Manager"]

    assert manager["Global Options"]["Fixed Frame"] == "map"
    assert manager["Views"]["Current"]["Class"] == (
        "rviz_default_plugins/TopDownOrtho"
    )
    assert manager["Views"]["Current"]["Target Frame"] == "base_link"


def test_rviz_has_exact_occupancy_layers_with_live_qos():
    displays = displays_by_name()
    expected = {
        "Combined Occupancy": (
            "/ad/perception/occupancy/combined",
            True,
        ),
        "Static Occupancy": (
            "/ad/perception/occupancy/static",
            False,
        ),
        "Dynamic Occupancy": (
            "/ad/perception/occupancy/dynamic",
            False,
        ),
    }
    assert set(expected).issubset(displays)
    for name, (topic, enabled) in expected.items():
        display = displays[name]
        assert display["Class"] == "rviz_default_plugins/Map"
        assert display["Enabled"] is enabled
        assert display["Topic"]["Value"] == topic
        assert display["Topic"]["Durability Policy"] == "Volatile"
        assert display["Update Topic"]["Value"]
        assert display["Update Topic"]["Durability Policy"] == "Volatile"


def test_rviz_has_each_lidar_stage_with_sensor_qos():
    displays = displays_by_name()
    expected = {
        "Raw LiDAR": (
            "/ad/sensors/lidar/points",
            False,
            "FlatColor",
            3,
        ),
        "Deskewed LiDAR": (
            "/ad/perception/lidar/deskewed",
            False,
            "FlatColor",
            3,
        ),
        "Cropped LiDAR": (
            "/ad/perception/lidar/cropped",
            False,
            "FlatColor",
            3,
        ),
        "Leveled LiDAR": (
            "/ad/perception/lidar/leveled",
            False,
            "FlatColor",
            3,
        ),
        "Ground": (
            "/ad/perception/lidar/ground",
            True,
            "FlatColor",
            3,
        ),
        "Nonground": (
            "/ad/perception/lidar/nonground",
            True,
            "FlatColor",
            4,
        ),
        "PointXYZIRC": (
            "/ad/perception/lidar/points_xyzirc",
            False,
            "FlatColor",
            3,
        ),
        "Euclidean Clusters": (
            "/ad/perception/lidar/clusters",
            True,
            "Intensity",
            7,
        ),
    }
    for name, (topic, enabled, transformer, size_pixels) in expected.items():
        display = displays[name]
        assert display["Class"] == "rviz_default_plugins/PointCloud2"
        assert display["Enabled"] is enabled
        assert display["Value"] is enabled
        assert display["Topic"]["Value"] == topic
        assert display["Color Transformer"] == transformer
        assert display["Size (Pixels)"] == size_pixels
        assert display["Topic"]["Reliability Policy"] == "Best Effort"
        assert display["Topic"]["Durability Policy"] == "Volatile"
    assert displays["Raw LiDAR"]["Color"] == "255; 64; 64"
    assert displays["Raw LiDAR"]["Alpha"] == 0.55
    assert displays["Deskewed LiDAR"]["Color"] == "64; 255; 64"
    assert displays["Cropped LiDAR"]["Color"] == "64; 128; 255"
    assert displays["Leveled LiDAR"]["Color"] == "255; 255; 64"
    assert displays["Ground"]["Color"] == "64; 255; 255"
    assert displays["Nonground"]["Color"] == "255; 160; 64"
    clusters = displays["Euclidean Clusters"]
    assert clusters["Channel Name"] == "intensity"
    assert clusters["Use rainbow"] is True


def test_rviz_has_every_bridge_camera_with_optional_views_disabled():
    displays = displays_by_name()
    expected = {
        "Front Camera": (
            "/ad/sensors/camera/front/compressed",
            True,
        ),
        "Left Camera": (
            "/ad/sensors/camera/left/compressed",
            False,
        ),
        "Right Camera": (
            "/ad/sensors/camera/right/compressed",
            False,
        ),
        "Traffic Light Camera": (
            "/ad/sensors/camera/traffic_light/compressed",
            False,
        ),
    }

    for name, (topic, enabled) in expected.items():
        display = displays[name]
        assert display["Class"] == "rviz_default_plugins/Image"
        assert display["Enabled"] is enabled
        assert display["Value"] is enabled
        assert "Queue Size" not in display
        assert display["Topic"]["Value"] == topic
        assert display["Topic"]["Depth"] == 2
        assert display["Topic"]["History Policy"] == "Keep Last"
        assert display["Topic"]["Reliability Policy"] == "Best Effort"
        assert display["Topic"]["Durability Policy"] == "Volatile"


def test_rviz_camera_docks_match_the_default_subscription_state():
    config = load_rviz_config()
    main_window_state = str(
        config["Window Geometry"]["QMainWindow State"]
    ).lower()
    expected_visibility = {
        "Front Camera": "01",
        "Left Camera": "00",
        "Right Camera": "00",
        "Traffic Light Camera": "00",
    }

    # QMainWindow serializes each dock name as UTF-16BE followed by its
    # visibility byte. Keeping the optional docks hidden matters because RViz
    # enables an Image display (and its DDS subscription) when its dock opens.
    for name, visibility in expected_visibility.items():
        encoded_name = name.encode("utf-16-be").hex()
        name_offset = main_window_state.index(encoded_name)
        visibility_offset = name_offset + len(encoded_name)
        assert (
            main_window_state[
                visibility_offset : visibility_offset + 2
            ]
            == visibility
        )


def test_rviz_has_predictions_and_truthful_planner_diagnostics():
    displays = displays_by_name()
    expected = {
        "Predicted Objects": (
            "rviz_default_plugins/MarkerArray",
            "/ad/viz/perception/objects",
            True,
        ),
        "Predicted Corridor Relevance": (
            "rviz_default_plugins/MarkerArray",
            "/ad/viz/planner/relevant_objects",
            True,
        ),
        "Occupancy Corridor Relevance": (
            "rviz_default_plugins/MarkerArray",
            "/ad/viz/planner/occupancy_relevance",
            True,
        ),
        "Local Command Rollout": (
            "rviz_default_plugins/Path",
            "/ad/viz/planner/local_path",
            True,
        ),
        "Local Motion Candidates": (
            "rviz_default_plugins/MarkerArray",
            "/ad/viz/planner/candidate_paths",
            True,
        ),
        "MPPI Transformed Reference": (
            "rviz_default_plugins/Path",
            "/ad/viz/planner/mppi/transformed_reference",
            False,
        ),
        "MPPI Candidate Trajectories": (
            "rviz_default_plugins/MarkerArray",
            "/ad/viz/planner/mppi/trajectories",
            False,
        ),
    }
    for name, (class_name, topic, enabled) in expected.items():
        display = displays[name]
        assert display["Class"] == class_name
        assert display["Enabled"] is enabled
        assert display["Topic"]["Value"] == topic
        assert display["Topic"]["Reliability Policy"] == "Reliable"
        assert display["Topic"]["Durability Policy"] == "Volatile"

    planner_diagnostic_names = {
        name
        for name in displays
        if name.startswith("MPPI ")
        or name in {"Local Command Rollout", "Local Motion Candidates"}
    }
    assert planner_diagnostic_names == {
        "Local Command Rollout",
        "Local Motion Candidates",
        "MPPI Transformed Reference",
        "MPPI Candidate Trajectories",
    }
    candidate_name = "MPPI Candidate Trajectories".lower()
    for dishonest in ("optimal", "selected", "command"):
        assert dishonest not in candidate_name


def test_rviz_has_exactly_one_path_tracking_display_with_latched_qos():
    matches = [
        display
        for display in iter_displays()
        if display.get("Name") == "Path Tracking"
    ]
    assert len(matches) == 1
    display = matches[0]
    assert display["Class"] == "rviz_default_plugins/MarkerArray"
    assert display["Enabled"] is True
    assert display["Value"] is True
    assert display["Topic"]["Value"] == "/ad/viz/planner/path_tracking"
    assert display["Topic"]["Reliability Policy"] == "Reliable"
    assert display["Topic"]["Durability Policy"] == "Transient Local"


def test_rviz_planner_target_is_visualization_only():
    display = displays_by_name()["Planner Target"]
    assert display["Class"] == "rviz_default_plugins/Marker"
    assert display["Topic"]["Value"] == "/ad/viz/planner/target"


def test_main_rviz_visualizes_fixed_map_matching_and_handoff():
    config = load_rviz_config()
    manager = config["Visualization Manager"]
    displays = iter_displays()
    topics = {
        display["Name"]: display["Topic"]["Value"]
        for display in displays
        if "Topic" in display
    }

    assert manager["Global Options"]["Fixed Frame"] == "map"
    assert manager["Views"]["Current"]["Target Frame"] == "base_link"
    assert topics["Fixed PCD Map"] == "/ad/localization/fastlio/map"
    assert (
        topics["Registered LiDAR"]
        == "/ad/localization/fastlio/registered_points"
    )
    assert (
        topics["Selected Localization Route Elevation"]
        == "/ad/viz/localization/odometry_route_elevation"
    )
    assert (
        topics["Selected Localization Ground Projection"]
        == "/ad/viz/localization/odometry_ground"
    )
    assert (
        topics["FastLIO Candidate"]
        == "/ad/localization/backends/fastlio/odometry"
    )
    assert (
        topics["GNSS IMU Candidate"]
        == "/ad/localization/backends/gnss_imu/odometry"
    )

    fixed_map = next(
        display for display in displays if display["Name"] == "Fixed PCD Map"
    )
    assert fixed_map["Topic"]["Reliability Policy"] == "Reliable"
    assert fixed_map["Topic"]["Durability Policy"] == "Transient Local"


def test_localization_odometry_displays_coexist_with_live_qos():
    displays = displays_by_name()
    expected = {
        "Selected Localization Route Elevation": (
            "/ad/viz/localization/odometry_route_elevation",
            True,
        ),
        "Selected Localization Ground Projection": (
            "/ad/viz/localization/odometry_ground",
            False,
        ),
    }

    for name, (topic, enabled) in expected.items():
        display = displays[name]
        assert display["Class"] == "rviz_default_plugins/Odometry"
        assert display["Enabled"] is enabled
        assert display["Topic"]["Value"] == topic
        assert display["Topic"]["Reliability Policy"] == "Reliable"
        assert display["Topic"]["Durability Policy"] == "Volatile"


def test_rviz_groups_every_display_by_pipeline_stage():
    groups = display_groups()

    assert list(groups) == [
        "Scene",
        "Cameras",
        "Localization",
        "LiDAR Preprocessing",
        "Occupancy Grid",
        "Objects and Prediction",
        "Planning and Control",
        "MPPI Debug",
    ]
    assert groups["Scene"] == ["Grid", "TF", "Robot Model"]
    assert groups["Cameras"] == [
        "Front Camera",
        "Left Camera",
        "Right Camera",
        "Traffic Light Camera",
    ]
    assert groups["LiDAR Preprocessing"] == [
        "Raw LiDAR",
        "Deskewed LiDAR",
        "Cropped LiDAR",
        "Leveled LiDAR",
        "Ground",
        "Nonground",
        "PointXYZIRC",
        "Euclidean Clusters",
    ]

    # No display may sit outside a group, and none may be listed twice.
    grouped = [name for names in groups.values() for name in names]
    assert sorted(grouped) == sorted(displays_by_name())
    assert len(grouped) == len(set(grouped))


def test_rviz_shows_every_localization_backend_candidate():
    displays = displays_by_name()
    expected = {
        "GNSS IMU Candidate": "gnss_imu",
        "Quaternion Wheel GNSS EKF Candidate": "quaternion_wheel_gnss_ekf",
        "IMU Quaternion Encoder Candidate": "imu_quaternion_encoder",
        "ESKF Candidate": "eskf",
        "FastLIO Candidate": "fastlio",
    }

    for name, backend in expected.items():
        display = displays[name]
        assert display["Class"] == "rviz_default_plugins/Odometry"
        assert display["Topic"]["Value"] == (
            f"/ad/localization/backends/{backend}/odometry"
        )

    # Every candidate is distinguishable from the others on screen.
    colors = {displays[name]["Shape"]["Color"] for name in expected}
    assert len(colors) == len(expected)


def test_rviz_grid_spans_the_whole_course():
    grid = displays_by_name()["Grid"]

    assert grid["Cell Size"] == 1
    assert grid["Plane Cell Count"] == 200


def test_visualization_launch_starts_ground_odometry_node():
    source = (PACKAGE / "launch" / "visualization.launch.py").read_text(
        encoding="utf-8"
    )

    assert source.count('executable="ad_localization_ground_odometry"') == 1
    assert source.count('name="ad_localization_ground_odometry"') == 1


def test_visualization_launch_exposes_marker_clock_rollback_threshold():
    source = (PACKAGE / "launch" / "visualization.launch.py").read_text(
        encoding="utf-8"
    )

    assert source.count('"viz_clock_rollback_reset_sec"') == 2
    assert source.count('"clock_rollback_reset_sec"') == 1


def test_visualization_launch_starts_route_elevation_odometry_node():
    source = (PACKAGE / "launch" / "visualization.launch.py").read_text(
        encoding="utf-8"
    )

    assert (
        source.count(
            'executable="ad_localization_route_elevation_odometry"'
        )
        == 1
    )
    assert (
        source.count('name="ad_localization_route_elevation_odometry"')
        == 1
    )


def test_visualization_launch_isolates_rviz_on_visualization_tf_topics():
    source = (PACKAGE / "launch" / "visualization.launch.py").read_text(
        encoding="utf-8"
    )

    assert source.count('executable="ad_visualization_tf_relay"') == 1
    assert source.count('name="ad_visualization_tf_relay"') == 1
    assert source.count('(\"/tf\", \"/ad/viz/tf\")') == 1
    assert source.count('(\"/tf_static\", \"/ad/viz/tf_static\")') == 1
