# heven_slam

Experimental RTAB-Map 3D LiDAR SLAM retained from the pre-migration stack. Its
launch file still consumes the legacy `/gps/fix`, `/imu/data`, and
`/velodyne_points` topic contract, so it is not wired to `ad_morai_bridge` in
the current migration stage.

## Mapping

Do not start this package as part of the current `/ad/...` stack. The command
below is retained only to document the legacy experiment after its database
and sensor inputs have been supplied explicitly:

```bash
ros2 launch heven_slam global_lidar_slam.launch.py reset_database:=true
```

Further runs omit `reset_database:=true` to continue the same database. The
default database is `${HEVEN_MAP_DIR:-~/.ros/heven_maps}/global_map.db`.

## Export a global PCD

Keep mapping running and request the current assembled cloud:

```bash
ros2 run heven_slam save_cloud_pcd
```

The default output is
`${HEVEN_MAP_DIR:-~/.ros/heven_maps}/global_map.pcd`. Map DB and PCD artifacts
must not be committed.

## Validate a mapping run

Check that the database contains a connected returning trajectory and at least
one loop closure before treating an exported point cloud as a global map:

```bash
ros2 run heven_slam analyze_rtabmap_db \
  ~/.ros/heven_maps/global_map.db
```

The command exits with status 2 when the run is not yet suitable for evaluating
loop-closure drift. A point cloud can still be exported in that state, but it
must not be interpreted as a loop-corrected full-course map.

## Localization against the saved map

```bash
ros2 launch heven_slam global_lidar_slam.launch.py mode:=localization
```

GPS measurements are recorded and used to select loop candidates. Graph
optimization ignores GPS priors by default until the competition noise model
is known. Use `use_gps_priors:=true` only for a controlled comparison.

LiDAR deskewing is off by default because MORAI emits a completed simulated
rotation and the current prototype prioritizes repeatable timestamps. It can be
evaluated separately with `use_imu:=true enable_deskewing:=true`. LiDAR-only ICP
is the default baseline; `use_imu:=true` enables the inertial comparison.
