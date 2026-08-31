"""Assemble one canonical dataset sample from a LiDAR frame + matched sources.

Pure and ROS-free (it takes already-deserialised messages / values), so the
synchronisation, transform and validity logic is unit-testable without a
simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ad_morai_bridge_dev.dataset.boxes import actor_from_message, canonical_box
from ad_morai_bridge_dev.dataset.pointcloud import decode_pointcloud, field_schema
from ad_morai_bridge_dev.dataset.schema import (
    ACTOR_GT_TOPIC,
    EGO_GT_TOPIC,
    LIDAR_TOPIC,
    MAP_FRAME,
    SkewConfig,
    FrameRecord,
    frame_validity,
)
from ad_morai_bridge_dev.dataset.sync import Match
from ad_morai_bridge_dev.dataset.transforms import (
    StaticExtrinsics,
    build_map_to_lidar,
    ego_map_pose,
)
from ad_morai_bridge_dev.dataset.geometry import transform_from_message


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


@dataclass(frozen=True)
class AssembledFrame:
    record: FrameRecord
    point_arrays: dict[str, np.ndarray]
    gt_payload: dict[str, Any]
    ego_payload: dict[str, Any]
    tf_payload: dict[str, Any]


class FrameAssembler:
    def __init__(
        self,
        run_id: str,
        static: StaticExtrinsics,
        skew: SkewConfig,
        scenario_file: str,
        scenario_sha256: str,
    ) -> None:
        self._run_id = run_id
        self._static = static
        self._skew = skew
        self._scenario_file = scenario_file
        self._scenario_sha256 = scenario_sha256

    def assemble(
        self,
        frame_index: int,
        lidar_message: Any,
        ego_match: Match,
        actor_match: Match,
        tf_match: Match,
    ) -> AssembledFrame:
        stamp_ns = _stamp_ns(lidar_message.header.stamp)
        lidar_frame_id = str(lidar_message.header.frame_id).lstrip("/")
        arrays = decode_pointcloud(lidar_message)
        point_count = len(next(iter(arrays.values()))) if arrays else 0
        record = FrameRecord(
            frame_index=frame_index,
            sample_id=f"{self._run_id}_{stamp_ns}",
            lidar_header_stamp_ns=stamp_ns,
            lidar_frame_id=lidar_frame_id,
            lidar_filename=f"lidar/{frame_index:06d}.npz",
            point_count=point_count,
            point_fields=list(arrays.keys()),
        )

        # --- ego -------------------------------------------------------
        ego_payload: dict[str, Any] = {
            "schema_source_topic": EGO_GT_TOPIC,
            "status": ego_match.status,
            "source_stamp_ns": ego_match.source_stamp_ns,
            "skew_ns": ego_match.skew_ns,
        }
        record.ego_status = ego_match.status
        record.ego_source_stamp_ns = ego_match.source_stamp_ns
        record.ego_skew_ns = ego_match.skew_ns
        map_to_base = None
        if ego_match.status == "matched":
            ego = ego_match.value
            map_to_base = ego_map_pose(ego)
            ego_payload["frame_id"] = str(ego.header.frame_id).lstrip("/")
            ego_payload["is_simulator_ground_truth"] = True
            ego_payload["position_map"] = [
                float(ego.position.x),
                float(ego.position.y),
                float(ego.position.z),
            ]
            ego_payload["rpy_rad"] = [
                float(ego.rpy.x),
                float(ego.rpy.y),
                float(ego.rpy.z),
            ]
            ego_payload["linear_velocity"] = [
                float(ego.velocity.x),
                float(ego.velocity.y),
                float(ego.velocity.z),
            ]
            ego_payload["angular_velocity"] = [
                float(ego.angular_velocity.x),
                float(ego.angular_velocity.y),
                float(ego.angular_velocity.z),
            ]
            ego_payload["acceleration"] = [
                float(ego.acceleration.x),
                float(ego.acceleration.y),
                float(ego.acceleration.z),
            ]
            ego_payload["signed_velocity_mps"] = float(ego.signed_velocity)

        # --- dynamic tf ---------------------------------------------------
        tf_payload: dict[str, Any] = {
            "status": tf_match.status,
            "source_stamp_ns": tf_match.source_stamp_ns,
            "skew_ns": tf_match.skew_ns,
            "static_edges": self._static.edges,
        }
        record.tf_status = tf_match.status
        record.tf_source_stamp_ns = tf_match.source_stamp_ns
        record.tf_skew_ns = tf_match.skew_ns
        map_to_lidar = None
        if tf_match.status == "matched" and map_to_base is not None:
            odom_to_base = transform_from_message(tf_match.value)
            chain = build_map_to_lidar(map_to_base, odom_to_base, self._static)
            map_to_lidar = chain["map_to_lidar"]
            tf_payload["chain"] = chain["chain"]
            tf_payload["map_to_odom"] = chain["map_to_odom"]
            tf_payload["odom_to_base"] = {
                "translation": list(odom_to_base.translation),
                "quaternion_xyzw": list(odom_to_base.rotation),
            }

        # --- actor gt ---------------------------------------------------
        boxes: list[dict[str, Any]] = []
        gt_payload: dict[str, Any] = {
            "schema_source_topic": ACTOR_GT_TOPIC,
            "status": actor_match.status,
            "source_stamp_ns": actor_match.source_stamp_ns,
            "skew_ns": actor_match.skew_ns,
            "canonical_frame": MAP_FRAME,
            "class_map_scenario_file": self._scenario_file,
            "class_map_scenario_sha256": self._scenario_sha256,
            "boxes": boxes,
        }
        record.actor_status = actor_match.status
        record.actor_source_stamp_ns = actor_match.source_stamp_ns
        record.actor_skew_ns = actor_match.skew_ns
        if actor_match.status == "matched":
            message = actor_match.value
            gt_payload["frame_id"] = str(message.header.frame_id).lstrip("/")
            for item in message.objects:
                actor = actor_from_message(item)
                box = canonical_box(actor, map_to_lidar)
                boxes.append(box)
            record.actor_count = len(boxes)
            record.actor_ids = [int(b["actor_id"]) for b in boxes]
            record.lidar_frame_boxes = any("lidar_frame" in b for b in boxes)

        tracking, detection, flags = frame_validity(record, boxes)
        record.valid_for_tracking_gt = tracking
        record.valid_for_detection_gt = detection
        record.validation_flags = flags

        return AssembledFrame(
            record=record,
            point_arrays=arrays,
            gt_payload=gt_payload,
            ego_payload=ego_payload,
            tf_payload=tf_payload,
        )

    @staticmethod
    def lidar_topic() -> str:
        return LIDAR_TOPIC

    @staticmethod
    def field_schema(lidar_message: Any) -> list[dict[str, Any]]:
        return field_schema(lidar_message)
