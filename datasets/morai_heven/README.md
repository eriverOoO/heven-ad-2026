# MORAI HEVEN 3D detection dataset

This dataset was exported from recorded HEVEN topics. It contains no model,
training output, or CenterPoint integration.

## Contents

- `points/*.bin`: little-endian float32 XYZI, one finite point per row.
- `labels/*.json`: source timestamps, point counts, transform provenance, and
  lidar-frame 3D boxes.
- `splits/*.txt`: sample IDs.
- `skipped_frames.csv`: every rejected source frame and reason.
- `metadata.json`: source/config/environment and aggregate statistics.

## Export summary

- Source frames: 1778
- Exported frames: 1764
- Skipped frames: 14
- GT objects in configured lidar ROI: 2951
- Class distribution: `{"obstacle": 74, "pedestrian": 33, "vehicle": 2844}`
- Skip reasons: `{"actor_timestamp_gap": 14}`
- Raw / finite / removed-nonfinite points: 50803200 /
  24889568 / 25913632
- Maximum accepted actor / ego / TF gap: 29.463972 /
  29.736239 /
  28.379109 ms

## Timestamp policy

The PointCloud2 `header.stamp` is the sample timestamp. Actor GT, ego status,
and dynamic TF are independently selected by nearest source `header.stamp`,
with no interpolation. Maximum absolute gaps are actor
30 ms, ego
30 ms, and TF
30 ms. A missing or larger-gap input
rejects the entire frame; zero transforms are never substituted.

## Transform convention

All output points and boxes use `lidar_link`: +X forward, +Y left, +Z up.
The evaluated chain is `map -> odom -> base_link -> rear_axle_link ->
lidar_link`. The bag has no direct map-to-odom TF. For every cloud, the exporter
derives it from the nearest ego-status map pose (`map -> base_link`, as used by
the repository's zero-offset status-pose localizer) and recorded `odom ->
base_link`; static rear-axle and lidar transforms come from `/tf_static`.

## Box convention

Each label stores center `(x,y,z)`, `(length,width,height)`, and yaw in radians
in `lidar_link`. Yaw is counter-clockwise from lidar +X and normalized to
`[-pi, pi]`. MORAI bridge code converts ObjectInfo heading degrees to radians;
the map convention is ENU, heading zero at +X/East and positive CCW.

`ObjectStatus.size.x/y/z` is exported as length/width/height without swapping.
Vehicle length is validated against `overhang + wheelbase + rear_overhang`.
The repository defines vehicle/status origin as rear-axle center at ground, so
vehicle centers shift forward by `(wheelbase + overhang - rear_overhang)/2`
and upward by `height/2`. Pedestrian and obstacle scenario positions are their
ground-centered origins and shift only upward by `height/2`. Raw source fields
and the applied center policy remain in every label.

The bag-specific class mapping is supported by checkpoint14 scenario IDs:
UID 3 is in `pedestrianList` and arrives as type 0; UID 13 is in `vehicleList`
and arrives as type 1; UID 2 is in `objectList` and arrives as type 2. It is not
the gRPC ObjectType enum and must be revalidated for another simulator build.

Only centers within the configured lidar ROI are labeled. Visibility and
occlusion are not inferred in STEP 03 and require the STEP 04 visual audit.

## Splits

Splits are bag/scene-level, never random frame-level. This one-bag export puts
the complete `static_20260805_003151` scene in
`train`. Empty val/test files are intentional;
additional bags must be assigned as whole scenes before training.
