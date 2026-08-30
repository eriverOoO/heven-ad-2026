"""Pure ROS <-> AB3DMOT-core adapter functions (T-1B, no rclpy dependency).

Message-type classes are always injected (never imported at module level),
following the same pattern already used by `centerpoint_ros.py` and
`morai_replay.py` -- keeps this module unit-testable with lightweight
fakes, with `ab3dmot_tracker_node.py` supplying the real
`autoware_perception_msgs`/`geometry_msgs`/`unique_identifier_msgs` classes.

Several small helpers (`stamp_to_ns`, `select_classification`,
`normalized_quaternion`, `yaw_from_quaternion`, `quaternion_from_yaw`) are
faithful ports of the equivalent validated helpers already in
`ad_lidar_perception/src/tracking/autoware_prediction_node.cpp`
(`stamp_to_ns`, `select_classification`, `normalized_orientation`,
`yaw_from_orientation`, `orientation_from_yaw`) -- reused for consistency
with HEVEN's existing tracked-object-handling conventions, not reinvented.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Sequence

import numpy as np

from ad_lidar_perception.ab3dmot_core import Detection, TrackedState

_NANOSECONDS_PER_SECOND = 1_000_000_000
_QUATERNION_NORM_TOLERANCE = 1.0e-6
# ObjectClassification labels run UNKNOWN=0 .. PEDESTRIAN=7 in this message
# definition; mirrors the exact bound `autoware_prediction_node.cpp` checks.
_MAX_CLASSIFICATION_LABEL = 7


class DetectedObjectsAdapterError(ValueError):
    """A `DetectedObjects` message failed validation and cannot be tracked."""


def stamp_to_ns(stamp: Any) -> int:
    """Validate and convert a `builtin_interfaces/Time` to integer nanoseconds.

    Port of `autoware_prediction_node.cpp::stamp_to_ns`.
    """
    if stamp.sec < 0 or stamp.nanosec >= _NANOSECONDS_PER_SECOND:
        raise DetectedObjectsAdapterError("stamp is malformed")
    nanoseconds = int(stamp.sec) * _NANOSECONDS_PER_SECOND + int(stamp.nanosec)
    if nanoseconds <= 0:
        raise DetectedObjectsAdapterError("stamp must be strictly positive")
    return nanoseconds


class TimestampDecision(Enum):
    """How `Ab3dmotTrackerNode` should react to one message's timestamp,
    relative to the last one it successfully processed."""

    PROCESS = "process"  # first frame, or a normal strictly-later stamp
    SKIP_DUPLICATE = "skip_duplicate"  # identical stamp; drop silently-safe
    REJECT_ROLLBACK = "reject_rollback"  # stamp went backwards


def classify_timestamp(stamp_ns: int, last_stamp_ns: int | None) -> TimestampDecision:
    """Classify ``stamp_ns`` against ``last_stamp_ns`` (``None`` on the
    first frame). Never returns a decision that would let a non-positive
    dt reach the tracker: duplicates and rollbacks are rejected, and
    everything else is a normal, strictly-increasing step.
    """
    if last_stamp_ns is None or stamp_ns > last_stamp_ns:
        return TimestampDecision.PROCESS
    if stamp_ns == last_stamp_ns:
        return TimestampDecision.SKIP_DUPLICATE
    return TimestampDecision.REJECT_ROLLBACK


def _finite_probability(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


def select_classification(classifications: Sequence[Any]) -> tuple[int, float]:
    """Pick the highest-probability classification (ties -> lower label).

    Port of `autoware_prediction_node.cpp::select_classification`.
    """
    if not classifications:
        raise DetectedObjectsAdapterError("object has no classification")
    selected_label = 0
    selected_probability = 0.0
    selected = False
    for classification in classifications:
        if classification.label > _MAX_CLASSIFICATION_LABEL or not _finite_probability(
            classification.probability
        ):
            raise DetectedObjectsAdapterError("invalid classification")
        if (
            not selected
            or classification.probability > selected_probability
            or (
                classification.probability == selected_probability
                and classification.label < selected_label
            )
        ):
            selected = True
            selected_label = classification.label
            selected_probability = classification.probability
    return selected_label, selected_probability


def normalized_quaternion(orientation: Any) -> tuple[float, float, float, float]:
    """Validate and normalize a quaternion; returns (x, y, z, w).

    Port of `autoware_prediction_node.cpp::normalized_orientation`.
    """
    components = (orientation.x, orientation.y, orientation.z, orientation.w)
    if not all(math.isfinite(v) for v in components):
        raise DetectedObjectsAdapterError("quaternion must be finite")
    norm_squared = sum(v * v for v in components)
    if not math.isfinite(norm_squared) or norm_squared <= 0.0:
        raise DetectedObjectsAdapterError("quaternion has invalid norm")
    norm = math.sqrt(norm_squared)
    if abs(norm - 1.0) > _QUATERNION_NORM_TOLERANCE:
        raise DetectedObjectsAdapterError("quaternion is not near unit length")
    return tuple(v / norm for v in components)


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Port of `autoware_prediction_node.cpp::yaw_from_orientation`."""
    sine = 2.0 * (w * z + x * y)
    cosine = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sine, cosine)
    if not math.isfinite(yaw):
        raise DetectedObjectsAdapterError("yaw is nonfinite")
    return yaw


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """Planar (roll=pitch=0) quaternion for ``yaw``; returns (x, y, z, w).

    Port of `autoware_prediction_node.cpp::orientation_from_yaw`.
    """
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def quaternion_multiply(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Hamilton product q1 * q2 (both (x, y, z, w)); returns (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quaternion_rotate_vector(
    q: tuple[float, float, float, float], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Rotate 3-vector ``v`` by quaternion ``q`` (x, y, z, w)."""
    qx, qy, qz, qw = q
    qv = np.array([qx, qy, qz])
    vec = np.array(v)
    uv = np.cross(qv, vec)
    uuv = np.cross(qv, uv)
    result = vec + 2.0 * (qw * uv + uuv)
    return float(result[0]), float(result[1]), float(result[2])


def transform_pose_z_up(
    x: float, y: float, z: float, yaw: float, transform: Any
) -> tuple[float, float, float, float]:
    """Apply a `geometry_msgs/TransformStamped`-shaped ``transform`` (with
    ``.transform.translation`` and ``.transform.rotation``) to a HEVEN z-up
    pose. Returns the transformed (x, y, z, yaw) in the transform's target
    frame.

    Uses full quaternion composition (not a naive yaw addition) so a
    transform with any roll/pitch component is still handled correctly;
    the resulting orientation's yaw is extracted the same way
    `yaw_from_quaternion` extracts it from a detection's own orientation.
    """
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    q_transform = normalized_quaternion(rotation)
    q_object = quaternion_from_yaw(yaw)
    rotated_x, rotated_y, rotated_z = quaternion_rotate_vector(q_transform, (x, y, z))
    new_x = rotated_x + translation.x
    new_y = rotated_y + translation.y
    new_z = rotated_z + translation.z
    q_result = quaternion_multiply(q_transform, q_object)
    new_yaw = yaw_from_quaternion(*q_result)
    return new_x, new_y, new_z, new_yaw


def detected_objects_to_detections(
    message: Any, transform: Any | None
) -> list[Detection]:
    """Convert a `DetectedObjects` message into a list of `Detection`s.

    Raises `DetectedObjectsAdapterError` on the first invalid object,
    rejecting the whole message (matches the existing whole-array
    rejection-with-diagnostics convention in `autoware_prediction_node.cpp`
    rather than silently dropping individual malformed objects). Pass
    ``transform=None`` only when ``message.objects`` is empty (there is
    nothing to transform); a non-empty message with no transform is a
    caller bug, not a data problem, and raises.
    """
    if message.objects and transform is None:
        raise DetectedObjectsAdapterError("a transform is required for a non-empty DetectedObjects")

    detections: list[Detection] = []
    for obj in message.objects:
        if obj.shape.type != 0:  # Shape.BOUNDING_BOX
            raise DetectedObjectsAdapterError("only bounding-box objects are supported")
        dims = obj.shape.dimensions
        if not (math.isfinite(dims.x) and math.isfinite(dims.y) and math.isfinite(dims.z)):
            raise DetectedObjectsAdapterError("dimensions must be finite")
        if dims.x <= 0.0 or dims.y <= 0.0 or dims.z <= 0.0:
            raise DetectedObjectsAdapterError("dimensions must be positive")
        if not _finite_probability(obj.existence_probability):
            raise DetectedObjectsAdapterError("existence probability is invalid")

        label, label_probability = select_classification(obj.classification)

        pose = obj.kinematics.pose_with_covariance.pose
        position = pose.position
        if not all(
            math.isfinite(value)
            for value in (position.x, position.y, position.z)
        ):
            raise DetectedObjectsAdapterError("position must be finite")
        qx, qy, qz, qw = normalized_quaternion(pose.orientation)
        yaw = yaw_from_quaternion(qx, qy, qz, qw)

        if transform is not None:
            x, y, z, yaw = transform_pose_z_up(
                pose.position.x, pose.position.y, pose.position.z, yaw, transform
            )
        else:
            x, y, z = pose.position.x, pose.position.y, pose.position.z
        if not all(math.isfinite(value) for value in (x, y, z, yaw)):
            raise DetectedObjectsAdapterError("transformed pose must be finite")

        detections.append(
            Detection(
                x=x,
                y=y,
                z=z,
                yaw=yaw,
                length=dims.x,
                width=dims.y,
                height=dims.z,
                label=label,
                label_probability=label_probability,
                existence_probability=obj.existence_probability,
            )
        )
    return detections


def track_id_to_uuid(track_id: int) -> np.ndarray:
    """Deterministic, purely mechanical int -> 16-byte UUID encoding.

    Not a real (random/content-derived) UUID -- a stable, collision-free
    (for any realistic track-id range) big-endian encoding of the track's
    own integer id into the low 8 bytes of a 16-byte array, per the
    "no research ambiguity, just an implementation detail" resolution in
    `docs/research/tracking_architecture.md` "AB3DMOT Integration
    Decisions" §4.
    """
    if track_id < 0:
        raise ValueError("track_id must be non-negative")
    uuid_bytes = np.zeros(16, dtype=np.uint8)
    uuid_bytes[8:16] = np.frombuffer(int(track_id).to_bytes(8, "big"), dtype=np.uint8)
    return uuid_bytes


def bound_position_covariance(
    covariance: np.ndarray, maximum_std_m: float
) -> tuple[np.ndarray, bool]:
    """Clip a 3x3 position covariance so no direction's standard deviation
    exceeds ``maximum_std_m``, preserving symmetry and positive
    semi-definiteness via eigenvalue clipping.

    This is a *published-uncertainty* bound applied only at the
    estimator -> ROS boundary (Option C in
    `docs/perception/competition_dynamic_object_pipeline.md`). The internal
    Kalman filter covariance (`LinearKFEstimator._kf.P`) is never touched;
    only the covariance written into the outgoing `TrackedObject` is
    bounded, so downstream consumers (HEVEN prediction, Dynamic OGM) do not
    interpret a track's transient one-coast-step position uncertainty
    (`P0_pos + dt^2 * P0_vel`, ~360 m^2 for a born-then-lost track under the
    reference AB3DMOT `P[7:,7:] *= 1000` prior) literally.

    ``maximum_std_m`` <= 0 or non-finite disables the bound and returns the
    input unchanged (identical to the pre-fix behaviour, which is the
    default for every non-competition AB3DMOT configuration).

    Returns ``(bounded_covariance, was_bounded)``. ``bounded_covariance`` is
    a fresh array whenever the bound activates and the original object
    otherwise.
    """
    if not math.isfinite(maximum_std_m) or maximum_std_m <= 0.0:
        return covariance, False
    maximum_variance = maximum_std_m * maximum_std_m
    symmetric = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(eigenvalues[-1]) <= maximum_variance:
        return covariance, False
    # Clip from above at the ceiling and from below at 0: the input block is
    # PSD by construction, but flooring at 0 makes the serialized matrix
    # unconditionally PSD even if numerical noise produced a tiny negative
    # eigenvalue (downstream OGM rejects any non-PSD covariance).
    clipped = np.clip(eigenvalues, 0.0, maximum_variance)
    bounded = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    return 0.5 * (bounded + bounded.T), True


def _set_covariance_block(
    covariance: np.ndarray, block: np.ndarray, row_offset: int, col_offset: int
) -> None:
    """Write a 3x3 ``block`` into the 6x6-flattened (36,) ``covariance``
    array at (row_offset, col_offset), matching
    `autoware_prediction_node.cpp`'s existing flattened-covariance
    convention (row-major 6x6)."""
    for i in range(3):
        for j in range(3):
            covariance[(row_offset + i) * 6 + (col_offset + j)] = float(block[i, j])


def world_velocity_to_object_local(
    yaw: float,
    velocity_world: Sequence[float],
    covariance_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate AB3DMOT's world-frame velocity into Autoware object axes.

    AB3DMOT's state is ``[x, y, z, ..., vx, vy, vz]`` in the tracker
    frame. Autoware's tracked-object contract uses object-local twist; its
    CV tracker performs this same ``R(-yaw)`` conversion before publishing.
    Apply the identical basis change to the covariance at this ROS boundary
    so the tracker core keeps its native world-frame state.
    """
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    world_to_local = np.array(
        [
            [cosine, sine, 0.0],
            [-sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    local_velocity = world_to_local @ np.asarray(velocity_world, dtype=float)
    local_covariance = world_to_local @ covariance_world @ world_to_local.T
    return local_velocity, local_covariance


def tracked_state_to_message(
    state: TrackedState,
    message_types: dict[str, Any],
    *,
    orientation_available: bool = True,
    maximum_position_std_m: float = 0.0,
) -> tuple[Any, bool]:
    """Map one `TrackedState` to a `TrackedObject`.

    Field mapping matches `docs/research/tracking_architecture.md`
    "AB3DMOT Integration Decisions" §4 exactly:
    - position/orientation/dimensions: direct from the KF state.
    - velocity: AB3DMOT world-frame m/s rotated into the object-local frame
      used by Autoware ``TrackedObject`` twist.
    - track_id: `track_id_to_uuid`.
    - classification/existence_probability: the passthrough fields T-1B
      added to `Detection`/`Track`/`TrackedState` (see `ab3dmot_core.py`).
    - pose covariance (position block + yaw variance) and twist covariance
      (velocity block): real KF `P` sub-blocks, not invented. The position
      block is passed through `bound_position_covariance` first when
      ``maximum_position_std_m`` > 0 (Option C; velocity covariance and
      yaw variance are left raw -- see
      `docs/perception/competition_dynamic_object_pipeline.md`).
    - acceleration_with_covariance, is_stationary: **left at the message's
      own zero/false default** -- AB3DMOT's 10-state KF tracks no
      acceleration state and applies no stationary/moving heuristic, so
      per this task's "use the message's appropriate unknown/default
      semantics" instruction, these are not populated at all (not
      invented, not guessed).

    Returns ``(tracked_object, position_covariance_bounded)``.
    """
    output = message_types["TrackedObject"]()
    output.object_id.uuid = track_id_to_uuid(state.track_id)
    output.existence_probability = float(state.existence_probability)

    classification = message_types["ObjectClassification"]()
    classification.label = int(state.label)
    classification.probability = float(state.label_probability)
    output.classification.append(classification)

    pose = output.kinematics.pose_with_covariance.pose
    pose.position.x, pose.position.y, pose.position.z = state.x, state.y, state.z
    qx, qy, qz, qw = quaternion_from_yaw(state.yaw)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw

    reported_position_covariance, position_covariance_bounded = bound_position_covariance(
        state.position_covariance, maximum_position_std_m
    )
    pose_covariance = np.zeros(36)
    _set_covariance_block(pose_covariance, reported_position_covariance, 0, 0)
    # geometry_msgs/PoseWithCovariance is a row-major 6x6 over
    # (x, y, z, roll, pitch, yaw): the yaw-yaw variance is flat index
    # 5*6+5 = 35, not 3*6+3 = 21 (which is roll-roll). Autoware's
    # convention and `autoware_prediction_node.cpp` both read index 35.
    # With yaw_measurement_mode="unobserved" the KF never corrects theta,
    # so P[3,3] is the uninformative birth prior plus accumulated process
    # noise (>= 10 rad^2, std > pi) -- publishing it at the correct slot
    # lets prediction see "yaw unknown" instead of falling back to its
    # 0.04 rad^2 default and treating the always-zero placeholder yaw as a
    # confident measurement. orientation_availability stays UNAVAILABLE.
    pose_covariance[5 * 6 + 5] = float(state.yaw_variance)
    output.kinematics.pose_with_covariance.covariance = pose_covariance.tolist()

    local_velocity, local_velocity_covariance = world_velocity_to_object_local(
        state.yaw,
        (state.vx_mps, state.vy_mps, state.vz_mps),
        state.velocity_covariance,
    )
    twist = output.kinematics.twist_with_covariance.twist
    twist.linear.x, twist.linear.y, twist.linear.z = local_velocity.tolist()
    twist_covariance = np.zeros(36)
    _set_covariance_block(twist_covariance, local_velocity_covariance, 0, 0)
    output.kinematics.twist_with_covariance.covariance = twist_covariance.tolist()

    availability_type = message_types["TrackedObjectKinematics"]
    output.kinematics.orientation_availability = (
        availability_type.AVAILABLE
        if orientation_available
        else availability_type.UNAVAILABLE
    )

    output.shape.type = message_types["Shape"].BOUNDING_BOX
    output.shape.dimensions.x, output.shape.dimensions.y, output.shape.dimensions.z = (
        state.length,
        state.width,
        state.height,
    )
    return output, position_covariance_bounded


def tracked_states_to_message(
    states: Sequence[TrackedState],
    stamp: Any,
    frame_id: str,
    message_types: dict[str, Any],
    *,
    orientation_available: bool = True,
    maximum_position_std_m: float = 0.0,
) -> tuple[Any, int]:
    """Serialize a list of `TrackedState` into one `TrackedObjects`.

    Returns ``(tracked_objects, position_covariance_bounded_count)`` -- the
    number of objects whose published position covariance was clipped by
    `bound_position_covariance` (0 unless ``maximum_position_std_m`` > 0).
    """
    output = message_types["TrackedObjects"]()
    output.header.stamp = stamp
    output.header.frame_id = frame_id
    bounded_count = 0
    for state in states:
        obj, bounded = tracked_state_to_message(
            state,
            message_types,
            orientation_available=orientation_available,
            maximum_position_std_m=maximum_position_std_m,
        )
        output.objects.append(obj)
        bounded_count += int(bounded)
    return output, bounded_count
