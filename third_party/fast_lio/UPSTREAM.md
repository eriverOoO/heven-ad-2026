# Upstream attribution and local changes

This directory is a deliberately small ROS 2 derivative of
[Ericsii/FAST_LIO_ROS2](https://github.com/Ericsii/FAST_LIO_ROS2.git), pinned
to commit `2fffc570a25d0df172720bac034fbdb6a13d2162` (short
`2fffc570`).  It retains the upstream GPL-2.0 `LICENSE`, the upstream IKFoM
toolkit, and ikd-Tree (upstream ikd-Tree revision
`e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4`).

HEVEN changes on 2026-07-23:

- replace the Livox/ROS1-compatibility entry point with a MORAI PointCloud2
  path only; no Livox message, `pcl_ros`, or Python plotting dependency is
  retained;
- split the ROS executable entry point from the node and expose explicit,
  testable mapping/localization policy;
- add initial-pose gating, fixed-map loading, immutable localization, atomic
  ikd-tree snapshot saving, diagnostics, and REP-103 base/IMU/LiDAR pose
  conversion;
- use only the repository canonical ROS graph documented in `config/fastlio.yaml`.

The derivative remains GPL-2.0-or-later.  Upstream copyright and license
notices in vendored files must remain intact.
