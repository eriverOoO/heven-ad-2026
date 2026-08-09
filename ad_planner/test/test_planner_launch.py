from pathlib import Path
import hashlib
import importlib.util

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
import pytest
import yaml


PACKAGE = Path(__file__).resolve().parents[1]


def _load_planner_launch_module():
    path = PACKAGE / "launch" / "planner.launch.py"
    spec = importlib.util.spec_from_file_location("ad_planner_launch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _planner_parameters(path=PACKAGE / "config" / "planner.yaml"):
    return yaml.safe_load(path.read_text(encoding="utf-8"))["ad_planner"][
        "ros__parameters"
    ]


def _controller_parameters(parameters, prefix):
    prefix_with_dot = f"{prefix}."
    return {
        name.removeprefix(prefix_with_dot): value
        for name, value in parameters.items()
        if name.startswith(prefix_with_dot)
    }


def _launch_context(**values):
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "data_dir": str(PACKAGE / "test" / "fixtures"),
            "config_file": str(PACKAGE / "config" / "planner.yaml"),
            "path_file": "path_space.txt",
            "path_tracking_backend": "",
            "target_speed_mps": "",
            "perception_enabled": "",
            **values,
        }
    )
    return context


def _normalized_parameter_mapping(parameter, context):
    return {
        perform_substitutions(context, names): (
            yaml.safe_load(perform_substitutions(context, value))
            if isinstance(value, tuple)
            else value
        )
        for names, value in parameter.items()
    }


def _parameter_overrides(node, context):
    return _normalized_parameter_mapping(node._Node__parameters[-1], context)


def _normalized_parameter_sequence(node, context):
    parameters = node._Node__parameters
    assert len(parameters) == 4
    assert isinstance(parameters[0], ParameterFile)
    assert isinstance(parameters[1], ParameterFile)
    assert isinstance(parameters[2], dict)
    assert isinstance(parameters[3], dict)
    return (
        perform_substitutions(context, parameters[0].param_file),
        perform_substitutions(context, parameters[1].param_file),
        parameters[2],
        _parameter_overrides(node, context),
    )


def test_planner_yaml_uses_ad_topics_and_safe_defaults():
    parameters = _planner_parameters()
    topics = [value for key, value in parameters.items() if key.startswith("topics.")]
    assert topics
    assert all(topic.startswith("/ad/") for topic in topics)
    assert parameters["perception.enabled"] is True
    assert parameters["local_motion.backend"] == "dwa"
    assert "local_planner.backend" not in parameters
    assert parameters["path_tracking.backend"] == "stanley"
    assert parameters["tuning.lease_required"] is False
    assert parameters["tuning.lease_timeout_sec"] == 1.0
    assert parameters["topics.tuning_lease"] == "/ad/tuning/lease"


def test_path_tracking_backends_have_complete_independent_effective_configs(
    monkeypatch,
):
    parameters = _planner_parameters()
    expected = {
        "stanley": {
            "target_speed_mps": 16.25,
            "cross_track_gain": 0.91,
            "speed_softening_mps": 2.57,
            "lookahead_time_s": 0.16,
            "lookahead_min_m": 1.5,
            "lookahead_max_m": 5.0,
            "curvature_lookahead_m": 1.0,
            "curvature_window_radius": 5,
            "heading_error_gain": 1.0,
            "lateral_acceleration_mps2": 6.0,
            "minimum_speed_mps": 1.3888888889,
            "maximum_speed_mps": 16.6666666667,
            "acceleration_mps2": 2.0,
            "deceleration_mps2": 2.0,
            "speed_zones.count": 1,
            "speed_zones.0.start_xy_m": [
                38.868875371112615, -480.68740975673563,
            ],
            "speed_zones.0.end_xy_m": [
                -81.83284234744308, -547.3316347631321,
            ],
            "speed_zones.0.maximum_speed_mps": 8.3333333333,
            "launch_speed_mps": 1.3888888889,
            "launch_ramp_s": 4.0,
            "forward_window": 200,
            "maximum_laps": 1,
            "speed_pid.kp": 1.08,
            "speed_pid.ki": 0.0,
            "speed_pid.kd": 0.036,
            "speed_pid.integral_limit": 100.0,
            "speed_pid.derivative_limit": 10.0,
            "speed_pid.derivative_filter_time_constant_s": 0.1,
            "brake_pid.kp": 0.2,
            "brake_pid.ki": 0.0,
            "brake_pid.kd": 0.01,
        },
        "profile_stanley": {
            "target_speed_mps": 16.25,
            "cross_track_gain": 1.2590347192291622,
            "speed_softening_mps": 1.8151037682456395,
            "lookahead_time_s": 0.0829957038411917,
            "lookahead_min_m": 1.5,
            "lookahead_max_m": 5.0,
            "curvature_lookahead_m": 2.818227339182336,
            "curvature_window_radius": 5,
            "heading_error_gain": 0.8204460682192964,
            "lateral_acceleration_mps2": 6.0,
            "forward_window": 200,
            "maximum_laps": 1,
            "speed_pid.kp": 0.35523299734554153,
            "speed_pid.ki": 0.0,
            "speed_pid.kd": 0.007949498723771662,
            "speed_pid.integral_limit": 100.0,
            "speed_pid.derivative_limit": 10.0,
            "speed_pid.derivative_filter_time_constant_s": 0.1,
            "brake_pid.kp": 0.1869594609763207,
            "brake_pid.ki": 0.0,
            "brake_pid.kd": 0.014300904664999885,
            "minimum_speed_mps": 1.3888888889,
            "maximum_speed_mps": 16.6666666667,
            "acceleration_mps2": 5.0,
            "deceleration_mps2": 2.0,
            "speed_zones.count": 1,
            "speed_zones.0.start_xy_m": [
                38.868875371112615, -480.68740975673563,
            ],
            "speed_zones.0.end_xy_m": [
                -81.83284234744308, -547.3316347631321,
            ],
            "speed_zones.0.maximum_speed_mps": 8.3333333333,
            "launch_speed_mps": 1.3888888889,
            "launch_ramp_s": 3.0,
            "longitudinal_profile.speed_mps": [
                1.388888889, 2.777777778, 4.166666667, 5.555555556,
                6.944444444, 8.333333333, 9.722222222, 11.111111111,
                12.5, 13.888888889, 15.277777778,
            ],
            "longitudinal_profile.acceleration_mps2": [
                4.422133433, 4.003169592, 3.688007844, 3.594037245,
                3.564077446, 3.558017657, 3.532980823, 3.5313351,
                3.503538525, 3.505896377, 3.487892617,
            ],
            "longitudinal_profile.deceleration_mps2": [
                3.670262249, 3.617239635, 3.586904013, 3.464571008,
                2.118385009, 1.832400025, 1.829604241, 1.837794547,
                1.835002459, 1.845794044, 1.825491215,
            ],
            "longitudinal_profile.braking_delay_s": 0.11589156,
        },
    }
    for backend, backend_expected in expected.items():
        assert _controller_parameters(parameters, backend) == backend_expected

    module = _load_planner_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    vehicle_parameters = module._load_vehicle_parameters()
    assert vehicle_parameters["stanley.control_point_x_m"] == 3.0
    assert vehicle_parameters["profile_stanley.control_point_x_m"] == 3.0
    assert "stanley.control_point_x_m" not in parameters
    assert "profile_stanley.control_point_x_m" not in parameters
    assert parameters["profile_stanley.maximum_speed_mps"] <= 16.6667


def test_visualization_config_is_canonical_and_preserves_operational_topics():
    parameters = _planner_parameters()
    assert parameters["visualization.profile_sample_stride"] > 0
    assert {
        name: value
        for name, value in parameters.items()
        if name.startswith("visualization.topics.")
    } == {
        "visualization.topics.static_ungated": (
            "/ad/viz/perception/occupancy/static_ungated"
        ),
        "visualization.topics.local_path": "/ad/viz/planner/local_path",
        "visualization.topics.target": "/ad/viz/planner/target",
        "visualization.topics.candidate_paths": (
            "/ad/viz/planner/candidate_paths"
        ),
        "visualization.topics.path_tracking": (
            "/ad/viz/planner/path_tracking"
        ),
        "visualization.topics.occupancy_relevance": (
            "/ad/viz/planner/occupancy_relevance"
        ),
        "visualization.topics.relevant_objects": (
            "/ad/viz/planner/relevant_objects"
        ),
    }
    assert parameters["topics.path"] == "/ad/planner/path"
    assert parameters["topics.target_speed"] == "/ad/planner/target_speed"
    assert parameters["topics.command"] == "/ad/control/command"


def test_local_motion_backend_files_match_the_package_contract():
    local_planning = PACKAGE / "config" / "local_planning"
    expected = {"dwa", "frenet_lattice", "mppi_nav2"}
    assert {path.stem for path in local_planning.glob("*.yaml")} == expected

    backend_parameters = {}
    for backend in expected:
        document = yaml.safe_load(
            (local_planning / f"{backend}.yaml").read_text(encoding="utf-8")
        )
        if backend == "mppi_nav2":
            assert set(document) == {
                "ad_planner",
                "controller_server",
                "local_costmap",
            }
        else:
            assert set(document) == {"ad_planner"}
        backend_parameters[backend] = document["ad_planner"]["ros__parameters"]

    dwa = backend_parameters["dwa"]
    assert dwa["dwa.minimum_speed_mps"] == 0.0
    assert dwa["dwa.maximum_speed_mps"] == 16.25
    assert dwa["dwa.speed_step_mps"] == 1.0
    assert dwa["dwa.minimum_steering_rad"] == -0.52
    assert dwa["dwa.maximum_steering_rad"] == 0.52
    assert dwa["dwa.steering_step_rad"] == 0.04
    assert dwa["dwa.simulation_dt"] == 0.2
    assert dwa["dwa.horizon_sec"] == 1.5
    assert dwa["dwa.dynamic_window_time_sec"] == 0.5
    assert dwa["dwa.maximum_acceleration_mps2"] == 5.0
    assert dwa["dwa.maximum_deceleration_mps2"] == 1.8
    assert dwa["dwa.emergency_deceleration_mps2"] == 6.0
    assert dwa["dwa.maximum_lateral_acceleration_mps2"] == 6.0
    assert dwa["local_motion.maximum_speed_mps"] == 16.25
    assert dwa["local_motion.maximum_acceleration_mps2"] == 5.0
    assert dwa["local_motion.maximum_deceleration_mps2"] == 6.0
    assert dwa["local_motion.maximum_lateral_acceleration_mps2"] == 6.0
    assert dwa["local_motion.footprint_front_m"] == 3.845
    assert dwa["local_motion.footprint_rear_m"] == 0.79
    assert dwa["local_motion.footprint_half_width_m"] == 0.945
    assert dwa["dwa.maximum_path_distance_m"] == 4.5
    assert dwa["dwa.prediction.covariance_sigma"] == 2.0
    assert dwa["dwa.prediction.minimum_margin_m"] == 0.2
    assert dwa["dwa.footprint.occupied_threshold"] == 20
    assert dwa["dwa.footprint.maximum_cells_to_check"] == 8192
    assert dwa["dwa.progress_weight"] == 1.0
    assert {
        "dwa.goal_weight",
        "dwa.heading_weight",
        "dwa.clearance_weight",
        "dwa.smoothness_weight",
        "dwa.path_distance_weight",
        "dwa.speed_weight",
        "dwa.speed_pid.kp",
        "dwa.speed_pid.kd",
    }.issubset(dwa)
    assert backend_parameters["frenet_lattice"] == {
        "frenet_lattice.lateral_targets_m": [-1.0, 0.0, 1.0],
        "frenet_lattice.target_speeds_mps": [2.0, 4.0, 6.0],
        "frenet_lattice.durations_s": [2.0, 3.0, 4.0],
        "frenet_lattice.sample_dt_s": 0.1,
        "frenet_lattice.maximum_curvature_inv_m": 0.2,
        "frenet_lattice.maximum_acceleration_mps2": 3.0,
        "frenet_lattice.maximum_lateral_acceleration_mps2": 3.0,
        "frenet_lattice.maximum_jerk_mps3": 5.0,
        "frenet_lattice.maximum_lateral_transition_m": 5.0,
        "frenet_lattice.footprint_clearance_m": 0.2,
        "frenet_lattice.occupied_threshold": 100,
        "frenet_lattice.maximum_cells_to_check": 4096,
        "frenet_lattice.progress_weight": 1.0,
        "frenet_lattice.clearance_weight": 1.0,
        "frenet_lattice.jerk_weight": 0.1,
        "frenet_lattice.lateral_offset_weight": 0.2,
        "frenet_lattice.lane_change_weight": 1.0,
        "frenet_lattice.continuity_weight": 1.0,
        "frenet_lattice.maximum_candidates": 512,
        "local_motion.maximum_speed_mps": 16.25,
        "local_motion.maximum_acceleration_mps2": 5.0,
        "local_motion.maximum_deceleration_mps2": 6.0,
        "local_motion.maximum_lateral_acceleration_mps2": 6.0,
        "local_motion.maximum_jerk_mps3": 5.0,
        "local_motion.footprint_front_m": 3.845,
        "local_motion.footprint_rear_m": 0.79,
        "local_motion.footprint_half_width_m": 0.945,
    }
    assert backend_parameters["mppi_nav2"] == {
        "mppi_nav2.cmd_vel_topic": "/ad/planner/mppi/cmd_vel",
        "mppi_nav2.command_timeout_s": 0.20,
        "mppi_nav2.diagnostic_rollout_dt_s": 0.10,
        "mppi_nav2.diagnostic_rollout_horizon_s": 3.0,
        "mppi_nav2.wheelbase_m": 3.0,
        "mppi_nav2.maximum_road_wheel_angle_rad": 0.588,
        "mppi_nav2.steering_rate_limit_rad_s": 0.35,
        "mppi_nav2.near_zero_speed_mps": 0.05,
        "mppi_nav2.path_refresh_period_s": 0.25,
    }


def test_perception_trigger_corridor_is_configurable_from_yaml():
    parameters = yaml.safe_load(
        (PACKAGE / "config" / "planner.yaml").read_text(encoding="utf-8")
    )["ad_planner"]["ros__parameters"]
    expected = {
        "perception.route_aligned_activation": True,
        "perception.clear_release_duration_sec": 2.0,
        "perception.speed_aware_lookahead": True,
        "perception.near_x_m": 1.0,
        "perception.minimum_lookahead_m": 20.0,
        "perception.maximum_lookahead_m": 99.0,
        "perception.front_bumper_x_m": 3.845,
        "perception.reaction_time_sec": 1.0,
        "perception.braking_deceleration_mps2": 1.8,
        "perception.stopping_margin_m": 5.0,
        "perception.forward_check_pose.x": 5.5,
        "perception.forward_check_pose.y": 0.0,
        "perception.forward_check_pose.yaw_rad": 0.0,
        "perception.forward_check_footprint.half_length_m": 4.2,
        "perception.forward_check_footprint.half_width_m": 1.2,
        "perception.forward_check_footprint.clearance_m": 0.1,
        "perception.forward_check_footprint.occupied_threshold": 50,
        "perception.forward_check_footprint.maximum_cells_to_check": 32768,
    }
    assert {name: parameters.get(name) for name in expected} == expected


def test_planner_launch_exposes_only_generic_arguments(monkeypatch):
    module = _load_planner_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    description = module.generate_launch_description()
    arguments = {
        action.name: action
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert set(arguments) == {
        "data_dir",
        "config_file",
        "path_file",
        "route_corridor_file",
        "path_tracking_backend",
        "target_speed_mps",
        "local_motion_prediction_mode",
        "perception_enabled",
        "tuning_lease_required",
    }
    assert perform_substitutions(LaunchContext(), arguments["config_file"].default_value) == str(
        PACKAGE / "config" / "planner.yaml"
    )
    assert perform_substitutions(
        LaunchContext(), arguments["path_file"].default_value
    ) == ""
    assert perform_substitutions(
        LaunchContext(), arguments["path_tracking_backend"].default_value
    ) == ""
    assert perform_substitutions(
        LaunchContext(), arguments["target_speed_mps"].default_value
    ) == ""
    assert perform_substitutions(
        LaunchContext(), arguments["perception_enabled"].default_value
    ) == ""
    assert perform_substitutions(
        LaunchContext(),
        arguments["local_motion_prediction_mode"].default_value,
    ) == ""
    assert perform_substitutions(
        LaunchContext(),
        arguments["tuning_lease_required"].default_value,
    ) == ""
    opaque_actions = [
        action for action in description.entities if isinstance(action, OpaqueFunction)
    ]
    assert len(opaque_actions) == 1
    context = _launch_context()
    actions = opaque_actions[0].execute(context)
    assert actions
    assert all(isinstance(action, Node) for action in actions)
    assert all(action._Node__node_namespace is None for action in actions)


def test_dwa_selection_hashes_the_reference_corridor_source_path(
    monkeypatch, tmp_path
):
    module = _load_planner_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    data_dir = str(tmp_path)
    for path_override, expected_path in (
        ("", "path/2026_molit_comp_global_path.txt"),
        ("path/derived/example.txt", "path/derived/example.txt"),
    ):
        path = tmp_path / expected_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected_path.encode())
        context = _launch_context(
            data_dir=data_dir,
            path_file=path_override,
        )
        node = module._create_planner_node(context)
        common_file, backend_file, vehicle_parameters, overrides = (
            _normalized_parameter_sequence(node, context)
        )
        assert common_file == str(PACKAGE / "config" / "planner.yaml")
        assert backend_file == str(
            PACKAGE / "config" / "local_planning" / "dwa.yaml"
        )
        assert vehicle_parameters
        assert overrides == {
            "data_dir": data_dir,
            "path_file": expected_path,
            "route_corridor.expected_global_path_sha256": (
                hashlib.sha256(path.read_bytes()).hexdigest()
            ),
        }


def test_path_tracking_backend_override_is_validated_and_forwarded(monkeypatch):
    module = _load_planner_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    context = _launch_context(path_tracking_backend="profile_stanley")
    node = module._create_planner_node(context)
    assert _parameter_overrides(node, context)["path_tracking.backend"] == (
        "profile_stanley"
    )

    with pytest.raises(RuntimeError, match="unsupported path_tracking.backend"):
        module._create_planner_node(
            _launch_context(path_tracking_backend="unknown")
        )


def test_speed_overrides_follow_the_selected_path_tracking_backend(monkeypatch):
    module = _load_planner_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    context = _launch_context(
        path_tracking_backend="profile_stanley",
        target_speed_mps="4.0",
    )

    node = module._create_planner_node(context)

    assert _parameter_overrides(node, context) == {
        "data_dir": str(PACKAGE / "test" / "fixtures"),
        "path_file": "path_space.txt",
        "path_tracking.backend": "profile_stanley",
        "profile_stanley.target_speed_mps": 4.0,
        "route_corridor.expected_global_path_sha256": hashlib.sha256(
            (PACKAGE / "test" / "fixtures" / "path_space.txt").read_bytes()
        ).hexdigest(),
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("target_speed_mps", "not-a-number"),
    ],
)
def test_speed_overrides_reject_non_positive_or_non_numeric_values(
    monkeypatch, name, value
):
    module = _load_planner_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )

    with pytest.raises(RuntimeError, match=name):
        module._create_planner_node(
            _launch_context(
                path_tracking_backend="profile_stanley",
                **{name: value},
            )
        )


def test_speed_overrides_require_a_supported_configured_backend(
    monkeypatch, tmp_path
):
    module = _load_planner_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    config = tmp_path / "planner.yaml"
    config.write_text(
        "ad_planner:\n  ros__parameters:\n"
        "    local_motion.backend: dwa\n"
        "    path_tracking.backend: []\n"
        "    path_file: path_space.txt\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="supported path_tracking.backend"):
        module._create_planner_node(
            _launch_context(
                config_file=str(config),
                target_speed_mps="4.0",
            )
        )


def test_perception_override_is_validated_and_forwarded(monkeypatch):
    module = _load_planner_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    for launch_value, expected in (("true", True), ("false", False)):
        context = _launch_context(perception_enabled=launch_value)
        node = module._create_planner_node(context)
        assert _parameter_overrides(node, context)[
            "perception.enabled"
        ] is expected

    with pytest.raises(RuntimeError, match="perception_enabled"):
        module._create_planner_node(
            _launch_context(perception_enabled="sometimes")
        )


def test_frenet_selection_loads_only_its_file_and_hashes_active_path(
    monkeypatch, tmp_path
):
    module = _load_planner_launch_module()
    common = tmp_path / "planner.yaml"
    common.write_text(
        "ad_planner:\n  ros__parameters:\n"
        "    local_motion.backend: frenet_lattice\n"
        "    path_file: path/active.txt\n",
        encoding="utf-8",
    )
    active_path = tmp_path / "path" / "active.txt"
    active_path.parent.mkdir()
    active_path.write_bytes(b"known global path\n")
    loaded = []
    original_loader = module._load_parameter_file

    def record_load(path):
        loaded.append(Path(path).name)
        return original_loader(path)

    monkeypatch.setattr(module, "_load_parameter_file", record_load)
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    context = _launch_context(
        config_file=str(common), data_dir=str(tmp_path), path_file=""
    )
    node = module._create_planner_node(context)

    common_file, backend_file, vehicle_parameters, overrides = (
        _normalized_parameter_sequence(node, context)
    )
    assert loaded == ["planner.yaml", "frenet_lattice.yaml"]
    assert common_file == str(common)
    assert backend_file == str(
        PACKAGE / "config" / "local_planning" / "frenet_lattice.yaml"
    )
    assert vehicle_parameters
    assert overrides["path_file"] == "path/active.txt"
    assert overrides[
        "route_corridor.expected_global_path_sha256"
    ] == hashlib.sha256(active_path.read_bytes()).hexdigest()


def test_frenet_explicit_path_override_is_forwarded_and_hashed(
    monkeypatch, tmp_path
):
    module = _load_planner_launch_module()
    common = tmp_path / "planner.yaml"
    common.write_text(
        "ad_planner:\n  ros__parameters:\n"
        "    local_motion.backend: frenet_lattice\n"
        "    path_file: path/yaml-default.txt\n",
        encoding="utf-8",
    )
    yaml_path = tmp_path / "path" / "yaml-default.txt"
    derived_path = tmp_path / "path" / "derived" / "example.txt"
    derived_path.parent.mkdir(parents=True)
    yaml_path.write_bytes(b"yaml path must not be selected\n")
    derived_path.write_bytes(b"explicit derived path\n")
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )

    context = _launch_context(
        config_file=str(common),
        data_dir=str(tmp_path),
        path_file="path/derived/example.txt",
    )
    node = module._create_planner_node(context)
    _, _, _, overrides = _normalized_parameter_sequence(node, context)

    assert overrides["path_file"] == "path/derived/example.txt"
    assert overrides[
        "route_corridor.expected_global_path_sha256"
    ] == hashlib.sha256(derived_path.read_bytes()).hexdigest()
    assert overrides[
        "route_corridor.expected_global_path_sha256"
    ] != hashlib.sha256(yaml_path.read_bytes()).hexdigest()


def test_unsupported_backend_and_missing_local_motion_path_fail_explicitly(
    monkeypatch, tmp_path
):
    module = _load_planner_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    unsupported = tmp_path / "unsupported.yaml"
    unsupported.write_text(
        "ad_planner:\n  ros__parameters:\n"
        "    local_motion.backend: unsupported\n",
        encoding="utf-8",
    )
    missing_path = tmp_path / "missing-path.yaml"
    missing_path.write_text(
        "ad_planner:\n  ros__parameters:\n"
        "    local_motion.backend: frenet_lattice\n"
        "    path_file: path/missing.txt\n",
        encoding="utf-8",
    )
    invalid_path = tmp_path / "invalid-path.yaml"
    invalid_path.write_text(
        "ad_planner:\n  ros__parameters:\n"
        "    local_motion.backend: frenet_lattice\n"
        "    path_file: []\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsupported local_motion.backend"):
        module._create_planner_node(_launch_context(config_file=str(unsupported)))
    with pytest.raises(RuntimeError, match="global path"):
        module._create_planner_node(
            _launch_context(
                config_file=str(missing_path),
                data_dir=str(tmp_path),
                path_file="",
            )
        )
    with pytest.raises(RuntimeError, match="configured global path"):
        module._create_planner_node(
            _launch_context(
                config_file=str(invalid_path),
                data_dir=str(tmp_path),
                path_file="",
            )
        )


@pytest.mark.parametrize(
    "backend",
    ["[dwa]", "{name: dwa}"],
)
def test_malformed_backend_value_fails_as_an_unsupported_backend(
    monkeypatch, tmp_path, backend
):
    module = _load_planner_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )
    common = tmp_path / "planner.yaml"
    common.write_text(
        "ad_planner:\n  ros__parameters:\n"
        f"    local_motion.backend: {backend}\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsupported local_motion.backend"):
        module._create_planner_node(_launch_context(config_file=str(common)))


@pytest.mark.parametrize(
    "backend_contents",
    [None, "ad_planner:\n  ros__parameters: []\n"],
)
def test_missing_or_malformed_selected_backend_file_fails_explicitly(
    monkeypatch, tmp_path, backend_contents
):
    module = _load_planner_launch_module()
    planner_share = tmp_path / "ad_planner"
    backend_dir = planner_share / "config" / "local_planning"
    backend_dir.mkdir(parents=True)
    if backend_contents is not None:
        (backend_dir / "frenet_lattice.yaml").write_text(
            backend_contents, encoding="utf-8"
        )
    common = tmp_path / "planner.yaml"
    common.write_text(
        "ad_planner:\n  ros__parameters:\n"
        "    local_motion.backend: frenet_lattice\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(
            planner_share if package == "ad_planner" else PACKAGE.parent / package
        ),
    )

    with pytest.raises(RuntimeError, match="invalid planner parameter file"):
        module._create_planner_node(_launch_context(config_file=str(common)))


def test_planner_launch_loads_dwa_geometry_from_ad_description(monkeypatch):
    vehicle_path = PACKAGE.parent / "ad_description" / "config" / "vehicle_parameters.yaml"
    vehicle = yaml.safe_load(vehicle_path.read_text(encoding="utf-8"))["vehicle"]
    expected = {
        "stanley.control_point_x_m": vehicle["control"][
            "lateral_control_point_x_m"
        ],
        "profile_stanley.control_point_x_m": vehicle["control"][
            "lateral_control_point_x_m"
        ],
        "dwa.wheelbase_m": vehicle["geometry"]["wheelbase_m"],
        **{
            f"dwa.footprint.{name}": value
            for name, value in vehicle["collision_footprint"].items()
            if name
            in {
                "center_offset_x_m",
                "half_length_m",
                "half_width_m",
                "clearance_m",
            }
        },
        "perception.front_bumper_x_m": vehicle["geometry"][
            "front_bumper_x_m"
        ],
        "traffic.front_bumper_x_m": vehicle["geometry"][
            "front_bumper_x_m"
        ],
        "local_motion.wheelbase_m": vehicle["geometry"]["wheelbase_m"],
        "local_motion.footprint_front_m": vehicle["geometry"][
            "front_bumper_x_m"
        ],
        "local_motion.footprint_rear_m": abs(
            vehicle["geometry"]["rear_bumper_x_m"]
        ),
        "local_motion.footprint_half_width_m": vehicle["geometry"]["width_m"]
        / 2,
        "local_motion.maximum_steering_rad": vehicle["steering"][
            "initial_bicycle_max_road_wheel_angle_rad"
        ],
    }

    module = _load_planner_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(PACKAGE.parent / package),
    )

    vehicle_parameters = module._load_vehicle_parameters()
    assert vehicle_parameters == expected
    assert "dwa.footprint.occupied_threshold" not in vehicle_parameters
    assert "dwa.footprint.maximum_cells_to_check" not in vehicle_parameters

    planner_parameters = yaml.safe_load(
        (PACKAGE / "config" / "planner.yaml").read_text(encoding="utf-8")
    )["ad_planner"]["ros__parameters"]
    assert not any(
        name == "dwa.wheelbase_m" or name.startswith("dwa.footprint.")
        for name in planner_parameters
    )


def test_planner_launch_loads_stanley_control_point_from_vehicle_yaml(
    monkeypatch, tmp_path
):
    description_share = tmp_path / "ad_description"
    vehicle_config = description_share / "config" / "vehicle_parameters.yaml"
    vehicle_config.parent.mkdir(parents=True)
    vehicle_config.write_text(
        "vehicle:\n"
        "  control:\n"
        "    lateral_control_point_x_m: 7.25\n"
        "  geometry:\n"
        "    wheelbase_m: 3.0\n"
        "    front_bumper_x_m: 3.8\n"
        "    rear_bumper_x_m: -0.8\n"
        "    width_m: 1.8\n"
        "  steering:\n"
        "    initial_bicycle_max_road_wheel_angle_rad: 0.6\n"
        "  collision_footprint:\n"
        "    center_offset_x_m: 1.5\n",
        encoding="utf-8",
    )

    module = _load_planner_launch_module()
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package: str(description_share),
    )

    vehicle_parameters = module._load_vehicle_parameters()
    assert vehicle_parameters["stanley.control_point_x_m"] == 7.25
    assert vehicle_parameters["profile_stanley.control_point_x_m"] == 7.25
    assert vehicle_parameters["local_motion.footprint_front_m"] == 3.8
    assert vehicle_parameters["traffic.front_bumper_x_m"] == 3.8
    assert vehicle_parameters["local_motion.footprint_rear_m"] == 0.8

    vehicle = yaml.safe_load(
        (
            PACKAGE.parent
            / "ad_description"
            / "config"
            / "vehicle_parameters.yaml"
        ).read_text(encoding="utf-8")
    )["vehicle"]
    assert vehicle["control"]["lateral_control_point_x_m"] == 3.0

    planner_parameters = yaml.safe_load(
        (PACKAGE / "config" / "planner.yaml").read_text(encoding="utf-8")
    )["ad_planner"]["ros__parameters"]
    assert "stanley.control_point_x_m" not in planner_parameters
    assert "profile_stanley.control_point_x_m" not in planner_parameters


def test_planner_uses_current_tuned_profile_stanley_candidate():
    parameters = yaml.safe_load((PACKAGE / "config" / "planner.yaml").read_text())[
        "ad_planner"
    ]["ros__parameters"]

    assert parameters["path_tracking.backend"] == "stanley"
    assert parameters["profile_stanley.target_speed_mps"] == 16.25
    assert parameters["profile_stanley.lookahead_time_s"] == 0.0829957038411917
    assert parameters["profile_stanley.lookahead_min_m"] == 1.5
    assert parameters["profile_stanley.lookahead_max_m"] == 5.0
    assert parameters["profile_stanley.curvature_lookahead_m"] == 2.818227339182336
    assert parameters["profile_stanley.cross_track_gain"] == 1.2590347192291622
    assert parameters["profile_stanley.speed_softening_mps"] == 1.8151037682456395
    assert parameters["profile_stanley.lateral_acceleration_mps2"] == 6.0
    assert parameters["profile_stanley.acceleration_mps2"] == 5.0
    assert parameters["profile_stanley.launch_ramp_s"] == 3.0
    assert parameters["profile_stanley.forward_window"] == 200
    assert parameters["profile_stanley.speed_pid.kp"] == 0.35523299734554153
    assert parameters["profile_stanley.speed_pid.ki"] == 0.0
    assert parameters["profile_stanley.speed_pid.kd"] == 0.007949498723771662
    assert parameters["profile_stanley.brake_pid.kp"] == 0.1869594609763207
    assert parameters["profile_stanley.brake_pid.kd"] == 0.014300904664999885
    assert parameters["stanley.lateral_acceleration_mps2"] == 6.0
    speeds = parameters["profile_stanley.longitudinal_profile.speed_mps"]
    acceleration = parameters[
        "profile_stanley.longitudinal_profile.acceleration_mps2"
    ]
    deceleration = parameters[
        "profile_stanley.longitudinal_profile.deceleration_mps2"
    ]
    assert len(speeds) == len(acceleration) == len(deceleration) == 11
    assert speeds[0] == pytest.approx(5.0 / 3.6)
    assert speeds[-1] == pytest.approx(55.0 / 3.6)
    assert acceleration[0] == pytest.approx(4.422133433)
    assert acceleration[-1] == pytest.approx(3.487892617)
    assert parameters[
        "profile_stanley.longitudinal_profile.braking_delay_s"
    ] == pytest.approx(0.11589156)

    assert parameters["speed_pid.kp"] == 0.3
    assert parameters["speed_pid.ki"] == 0.0
    assert parameters["speed_pid.kd"] == 0.01
