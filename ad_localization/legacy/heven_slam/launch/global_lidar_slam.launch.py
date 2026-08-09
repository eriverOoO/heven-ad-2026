"""Run RTAB-Map 3D LiDAR mapping or localization on MORAI sensor topics."""

import os
from pathlib import Path

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _bool_argument(context: LaunchContext, name: str) -> bool:
    return LaunchConfiguration(name).perform(context).lower() in ("1", "true", "yes")


def _launch_setup(context: LaunchContext, *args, **kwargs):
    mode = LaunchConfiguration("mode").perform(context).lower()
    if mode not in ("mapping", "localization"):
        raise RuntimeError("mode must be either 'mapping' or 'localization'")

    database_path = LaunchConfiguration("database_path").perform(context)
    if not database_path:
        map_directory = Path(
            os.environ.get("HEVEN_MAP_DIR", "~/.ros/heven_maps")
        ).expanduser()
        database_path = str(map_directory / "global_map.db")
    database_path = str(Path(database_path).expanduser())
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)

    reset_database = _bool_argument(context, "reset_database")
    if mode == "localization" and reset_database:
        raise RuntimeError("reset_database cannot be true in localization mode")
    if mode == "localization" and not Path(database_path).is_file():
        raise RuntimeError(f"localization database does not exist: {database_path}")

    voxel_size = float(LaunchConfiguration("voxel_size").perform(context))
    icp_strategy = int(LaunchConfiguration("icp_strategy").perform(context))
    point_to_plane = _bool_argument(context, "point_to_plane")
    max_translation = float(
        LaunchConfiguration("max_icp_translation").perform(context)
    )
    use_gps_priors = _bool_argument(context, "use_gps_priors")
    use_imu = _bool_argument(context, "use_imu")
    enable_deskewing = _bool_argument(context, "enable_deskewing")
    use_sim_time = _bool_argument(context, "use_sim_time")
    if enable_deskewing and not use_imu:
        raise RuntimeError("enable_deskewing requires use_imu:=true")
    fixed_frame = "base_link_stabilized" if use_imu else ""
    deskewed_topic = "/slam/deskewed_points"
    lidar_topic = deskewed_topic if enable_deskewing else "/velodyne_points"

    shared_parameters = {
        "use_sim_time": use_sim_time,
        "frame_id": "base_link",
        "qos": 1,
        "wait_for_transform": 0.2,
        "Icp/PointToPlane": "true" if point_to_plane else "false",
        "Icp/Iterations": "10",
        "Icp/VoxelSize": str(voxel_size),
        "Icp/Epsilon": "0.001",
        "Icp/PointToPlaneK": "20",
        "Icp/PointToPlaneRadius": "0",
        "Icp/PointToPlaneGroundNormalsUp": "1.0",
        "Icp/PointToPlaneMinComplexity": "0.02",
        "Icp/PointToPlaneLowComplexityStrategy": "1",
        "Icp/Force4DoF": "true",
        "Icp/MaxTranslation": str(max_translation),
        "Icp/MaxCorrespondenceDistance": str(voxel_size * 10.0),
        "Icp/Strategy": str(icp_strategy),
        "Icp/OutlierRatio": "0.7",
        "Reg/Force3DoF": "true",
    }
    odometry_parameters = {
        "expected_update_rate": 15.0,
        "odom_frame_id": "slam_odom",
        # Match adjacent completed VLP-16 scans. RTAB-Map still owns the global
        # pose graph and loop closures; using a growing F2M odometry submap made
        # MORAI's first moving scan look like a 0.5 m instantaneous correction.
        "Odom/Strategy": "1",
        # Average the velocity guess across multiple scans so variable MORAI
        # packet delivery delay cannot amplify one estimate into a huge guess.
        "Odom/GuessMotion": "true",
        "Odom/GuessSmoothingDelay": "1.0",
        "Odom/ScanKeyFrameThr": "0.4",
        "OdomF2M/ScanSubtractRadius": str(voxel_size),
        "OdomF2M/ScanMaxSize": "15000",
        "OdomF2M/BundleAdjustment": "false",
        "Icp/CorrespondenceRatio": "0.05",
    }
    if use_imu:
        odometry_parameters["guess_frame_id"] = fixed_frame
        odometry_parameters["wait_imu_to_init"] = True
    slam_parameters = {
        "database_path": database_path,
        "subscribe_depth": False,
        "subscribe_rgb": False,
        "subscribe_odom_info": False,
        "subscribe_scan_cloud": True,
        "map_frame_id": "map",
        "odom_sensor_sync": True,
        "map_always_update": True,
        "RGBD/ProximityMaxGraphDepth": "0",
        "RGBD/ProximityPathMaxNeighbors": "1",
        "RGBD/AngularUpdate": "0.05",
        "RGBD/LinearUpdate": "0.05",
        "RGBD/CreateOccupancyGrid": "false",
        "Mem/NotLinkedNodesKept": "false",
        "Mem/STMSize": "30",
        "Reg/Strategy": "1",
        "Icp/CorrespondenceRatio": "0.2",
        "Optimizer/PriorsIgnored": "false" if use_gps_priors else "true",
        "Mem/IncrementalMemory": "false" if mode == "localization" else "true",
        "Mem/InitWMWithAllNodes": "true" if mode == "localization" else "false",
    }
    common_remappings = [
        ("odom", "/slam/odom"),
        ("scan_cloud", lidar_topic),
        ("imu", "/imu/data" if use_imu else "/imu/not_used"),
    ]

    nodes = [
        Node(
            package="rtabmap_odom",
            executable="icp_odometry",
            name="icp_odometry",
            output="screen",
            parameters=[shared_parameters, odometry_parameters],
            remappings=common_remappings,
        ),
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[shared_parameters, slam_parameters],
            remappings=common_remappings + [
                ("gps/fix", "/gps/fix"),
                ("cloud_map", "/slam/cloud_map"),
            ],
            arguments=["-d"] if reset_database else [],
        ),
    ]
    if use_imu:
        nodes.insert(
            0,
            Node(
                package="rtabmap_util",
                executable="imu_to_tf",
                name="slam_imu_to_tf",
                output="screen",
                parameters=[{
                    "fixed_frame_id": fixed_frame,
                    "base_frame_id": "base_link",
                    "wait_for_transform_duration": 0.001,
                }],
                remappings=[("imu/data", "/imu/data")],
            ),
        )
    if enable_deskewing:
        nodes.insert(
            1,
            Node(
                package="rtabmap_util",
                executable="lidar_deskewing",
                name="slam_lidar_deskewing",
                output="screen",
                parameters=[{
                    "fixed_frame_id": fixed_frame,
                    "wait_for_transform": 0.2,
                    "slerp": True,
                }],
                remappings=[
                    ("input_cloud", "/velodyne_points"),
                    ("output_cloud", deskewed_topic),
                ],
            ),
        )
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "mode",
            default_value="mapping",
            description="mapping or localization",
        ),
        DeclareLaunchArgument(
            "database_path",
            default_value="",
            description="RTAB-Map database; defaults under HEVEN_MAP_DIR",
        ),
        DeclareLaunchArgument(
            "reset_database",
            default_value="false",
            description="delete the selected mapping database before startup",
        ),
        DeclareLaunchArgument(
            "voxel_size",
            default_value="0.5",
            description="outdoor LiDAR downsampling size in metres",
        ),
        DeclareLaunchArgument(
            "icp_strategy",
            default_value="1",
            description="RTAB-Map ICP backend; 0=PCL, 1=libpointmatcher",
        ),
        DeclareLaunchArgument(
            "point_to_plane",
            default_value="true",
            description="use point-to-plane ICP with low-complexity constraints",
        ),
        DeclareLaunchArgument(
            "max_icp_translation",
            default_value="1.0",
            description="reject implausible single ICP translations in metres",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="consume /clock instead of the WSL wall clock",
        ),
        DeclareLaunchArgument(
            "use_gps_priors",
            default_value="false",
            description="use GPS constraints during graph optimization",
        ),
        DeclareLaunchArgument(
            "enable_deskewing",
            default_value="false",
            description="deskew LiDAR using the IMU-stabilized frame",
        ),
        DeclareLaunchArgument(
            "use_imu",
            default_value="false",
            description="use IMU orientation to initialize and guide ICP odometry",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
