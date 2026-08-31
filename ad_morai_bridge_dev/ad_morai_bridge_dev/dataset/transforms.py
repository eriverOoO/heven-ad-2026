"""World <-> LiDAR transform reconstruction for one frame.

The chain is ``map -> odom -> base_link -> rear_axle_link -> lidar_link``,
identical to the offline exporter. MORAI publishes no direct ``map -> odom``
edge, so it is derived per frame from the simulator ego map pose
(``map -> base_link``) and the recorded ``odom -> base_link``. Static
rear-axle and LiDAR extrinsics come from ``/tf_static`` and are stored once
per run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ad_morai_bridge_dev.dataset.geometry import (
    RigidTransform,
    quaternion_from_rpy,
    transform_from_message,
)

STATIC_EDGES = (("base_link", "rear_axle_link"), ("rear_axle_link", "lidar_link"))
DYNAMIC_EDGE = ("odom", "base_link")


@dataclass(frozen=True)
class StaticExtrinsics:
    """``base_link -> lidar_link`` composed once from ``/tf_static``."""

    base_to_lidar: RigidTransform
    edges: dict[str, dict[str, list[float]]]

    @classmethod
    def from_tf_static(cls, transforms: list[Any]) -> "StaticExtrinsics":
        by_edge: dict[tuple[str, str], RigidTransform] = {}
        for item in transforms:
            key = (
                str(item.header.frame_id).lstrip("/"),
                str(item.child_frame_id).lstrip("/"),
            )
            by_edge[key] = transform_from_message(item)
        missing = [edge for edge in STATIC_EDGES if edge not in by_edge]
        if missing:
            raise ValueError(f"missing static TF edges: {missing}")
        composed = RigidTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        for edge in STATIC_EDGES:
            composed = composed.compose(by_edge[edge])
        edge_records = {
            f"{p}->{c}": {
                "translation": list(by_edge[(p, c)].translation),
                "quaternion_xyzw": list(by_edge[(p, c)].rotation),
            }
            for p, c in STATIC_EDGES
        }
        return cls(composed, edge_records)


def ego_map_pose(message: Any) -> RigidTransform:
    """``map -> base_link`` from an ``EgoVehicleStatus`` message."""

    return RigidTransform(
        (float(message.position.x), float(message.position.y), float(message.position.z)),
        quaternion_from_rpy(
            float(message.rpy.x), float(message.rpy.y), float(message.rpy.z)
        ),
    )


def build_map_to_lidar(
    map_to_base: RigidTransform,
    odom_to_base: RigidTransform,
    static: StaticExtrinsics,
) -> dict[str, Any]:
    """Return the ``map -> lidar_link`` transform and the derived edges."""

    map_to_odom = odom_to_base.compose(map_to_base.inverse())
    map_to_lidar = (
        static.base_to_lidar.inverse()
        .compose(odom_to_base.inverse())
        .compose(map_to_odom)
    )
    return {
        "map_to_lidar": map_to_lidar,
        "chain": ["map", "odom", "base_link", "rear_axle_link", "lidar_link"],
        "map_to_odom": {
            "translation": list(map_to_odom.translation),
            "quaternion_xyzw": list(map_to_odom.rotation),
        },
    }
