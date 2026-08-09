"""Deterministic ROS-free metrics for MORAI perception validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


_NANOSECONDS_PER_SECOND = 1_000_000_000
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _integer_stamp(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _positive(value: object, name: str) -> float:
    numeric = _finite(value, name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class MetricValue:
    value: float | int | None
    unavailable_reason: str | None
    support: int = 0
    expected: int = 0
    excluded: int = 0
    partial_reason: str | None = None

    def __post_init__(self) -> None:
        if self.value is None:
            _identifier(self.unavailable_reason, "unavailable_reason")
        elif self.unavailable_reason is not None:
            raise ValueError("available metric cannot have an unavailable reason")
        elif isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("metric value must be numeric")
        elif not math.isfinite(float(self.value)):
            raise ValueError("metric value must be finite")
        for name in ("support", "expected", "excluded"):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.expected != self.support + self.excluded:
            raise ValueError("expected must equal support plus excluded")
        if self.partial_reason is not None:
            _identifier(self.partial_reason, "partial_reason")
        if self.value is not None and self.excluded and self.partial_reason is None:
            raise ValueError("partial metric must have a partial reason")


@dataclass(frozen=True)
class ObjectSample:
    stamp_ns: int
    frame_id: str
    object_id: str
    x_m: float
    y_m: float
    vx_mps: float | None = None
    vy_mps: float | None = None

    def __post_init__(self) -> None:
        _integer_stamp(self.stamp_ns, "ObjectSample.stamp_ns")
        _identifier(self.frame_id, "ObjectSample.frame_id")
        _identifier(self.object_id, "ObjectSample.object_id")
        _finite(self.x_m, "ObjectSample.x_m")
        _finite(self.y_m, "ObjectSample.y_m")
        if (self.vx_mps is None) != (self.vy_mps is None):
            raise ValueError("velocity components must both be present or absent")
        if self.vx_mps is not None:
            _finite(self.vx_mps, "ObjectSample.vx_mps")
            _finite(self.vy_mps, "ObjectSample.vy_mps")


@dataclass(frozen=True)
class ObjectFrame:
    stamp_ns: int
    frame_id: str
    objects: tuple[ObjectSample, ...]

    def __post_init__(self) -> None:
        _integer_stamp(self.stamp_ns, "ObjectFrame.stamp_ns")
        _identifier(self.frame_id, "ObjectFrame.frame_id")
        identifiers = []
        for item in self.objects:
            if not isinstance(item, ObjectSample):
                raise ValueError("ObjectFrame objects must be ObjectSample values")
            if item.stamp_ns != self.stamp_ns:
                raise ValueError("ObjectFrame objects must have the frame stamp")
            if item.frame_id != self.frame_id:
                raise ValueError("ObjectFrame objects must have the frame id")
            identifiers.append(item.object_id)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("ObjectFrame contains duplicate object ids")


@dataclass(frozen=True)
class PredictionPoint:
    source_stamp_ns: int
    frame_id: str
    track_id: str
    horizon_s: float
    x_m: float
    y_m: float

    def __post_init__(self) -> None:
        _integer_stamp(self.source_stamp_ns, "PredictionPoint.source_stamp_ns")
        _identifier(self.frame_id, "PredictionPoint.frame_id")
        _identifier(self.track_id, "PredictionPoint.track_id")
        _positive(self.horizon_s, "PredictionPoint.horizon_s")
        _finite(self.x_m, "PredictionPoint.x_m")
        _finite(self.y_m, "PredictionPoint.y_m")


@dataclass(frozen=True)
class RateSummary:
    count: int
    rate_hz: float
    maximum_gap_s: float
    gap_count: int


@dataclass(frozen=True)
class Association:
    stamp_ns: int
    truth_id: str
    observation_id: str
    distance_m: float


@dataclass(frozen=True)
class AssociationResult:
    matches: tuple[Association, ...]
    unmatched_truth_ids: tuple[str, ...]
    unmatched_observation_ids: tuple[str, ...]
    unavailable_reason: str | None = None


@dataclass
class _ResidualEdge:
    target: int
    reverse_index: int
    capacity: int
    distance_cost: float
    identifier_cost: int


@dataclass(frozen=True)
class FrameExclusion:
    stamp_ns: int
    reason: str

    def __post_init__(self) -> None:
        _integer_stamp(self.stamp_ns, "FrameExclusion.stamp_ns")
        _identifier(self.reason, "FrameExclusion.reason")


@dataclass(frozen=True)
class DetectionMetrics:
    recall: MetricValue
    false_positives: MetricValue
    false_positives_per_frame: MetricValue
    excluded_frames: tuple[FrameExclusion, ...]


@dataclass(frozen=True)
class TrackingMetrics:
    initialization_delay_s: dict[str, MetricValue]
    missed_actor_frames: MetricValue
    drop_episodes: MetricValue
    completeness: MetricValue
    id_switches: MetricValue
    position_rmse_m: MetricValue
    velocity_rmse_mps: MetricValue
    excluded_frames: tuple[FrameExclusion, ...]


@dataclass(frozen=True)
class HorizonMetrics:
    ade_m: MetricValue
    fde_m: MetricValue


@dataclass(frozen=True)
class RunMetadata:
    git_commit: str
    profile_hashes: Mapping[str, str]
    bag_metadata: Mapping[str, Any]
    actor_preset: str

    def __post_init__(self) -> None:
        if not isinstance(self.git_commit, str) or _COMMIT.fullmatch(self.git_commit) is None:
            raise ValueError("git_commit must be a full lowercase Git commit")
        _identifier(self.actor_preset, "actor_preset")
        if not self.profile_hashes:
            raise ValueError("profile_hashes must not be empty")
        for name, digest in self.profile_hashes.items():
            _identifier(name, "profile hash name")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError("profile hash must be a SHA-256 digest")
        try:
            json.dumps(self.bag_metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("bag_metadata must be finite JSON data") from error


@dataclass(frozen=True)
class ReportArtifacts:
    json_path: Path
    markdown_path: Path


def summarize_rate(
    stamps_ns: Sequence[int], *, maximum_gap_s: float
) -> RateSummary:
    if len(stamps_ns) < 2:
        raise ValueError("rate summary requires at least two stamps")
    threshold = _positive(maximum_gap_s, "maximum_gap_s")
    stamps = tuple(_integer_stamp(value, "stamp") for value in stamps_ns)
    if any(current <= previous for previous, current in zip(stamps, stamps[1:])):
        raise ValueError("stamps must be strictly increasing")
    gaps = tuple(
        (current - previous) / _NANOSECONDS_PER_SECOND
        for previous, current in zip(stamps, stamps[1:])
    )
    duration = (stamps[-1] - stamps[0]) / _NANOSECONDS_PER_SECOND
    return RateSummary(
        count=len(stamps),
        rate_hz=(len(stamps) - 1) / duration,
        maximum_gap_s=max(gaps),
        gap_count=sum(gap > threshold for gap in gaps),
    )


def associate_frame(
    truth: ObjectFrame,
    observations: ObjectFrame,
    *,
    maximum_distance_m: float,
) -> AssociationResult:
    gate = _positive(maximum_distance_m, "maximum_distance_m")
    if truth.stamp_ns != observations.stamp_ns:
        raise ValueError("association frames must have the same stamp")
    if truth.frame_id != observations.frame_id:
        return AssociationResult(
            matches=(),
            unmatched_truth_ids=tuple(item.object_id for item in truth.objects),
            unmatched_observation_ids=tuple(
                item.object_id for item in observations.objects
            ),
            unavailable_reason=(
                "common-frame association unavailable: "
                f"{truth.frame_id} != {observations.frame_id}"
            ),
        )

    truth_items = sorted(truth.objects, key=lambda item: item.object_id)
    observed_items = sorted(
        observations.objects, key=lambda item: item.object_id
    )
    truth_count = len(truth_items)
    observation_count = len(observed_items)
    source = 0
    truth_offset = 1
    observation_offset = truth_offset + truth_count
    sink = observation_offset + observation_count
    graph: list[list[_ResidualEdge]] = [
        [] for _ in range(sink + 1)
    ]

    def add_edge(
        start: int,
        target: int,
        distance_cost: float,
        identifier_cost: int,
    ) -> int:
        forward_index = len(graph[start])
        reverse_index = len(graph[target])
        graph[start].append(
            _ResidualEdge(
                target,
                reverse_index,
                1,
                distance_cost,
                identifier_cost,
            )
        )
        graph[target].append(
            _ResidualEdge(
                start,
                forward_index,
                0,
                -distance_cost,
                -identifier_cost,
            )
        )
        return forward_index

    for truth_index in range(truth_count):
        add_edge(source, truth_offset + truth_index, 0.0, 0)
    for observation_index in range(observation_count):
        add_edge(observation_offset + observation_index, sink, 0.0, 0)

    assignment_edges = {}
    identifier_base = observation_count + 1
    for truth_index, truth_item in enumerate(truth_items):
        identifier_weight = identifier_base ** (
            truth_count - truth_index - 1
        )
        for observation_index, observed_item in enumerate(observed_items):
            distance = math.hypot(
                truth_item.x_m - observed_item.x_m,
                truth_item.y_m - observed_item.y_m,
            )
            if distance <= gate:
                edge_index = add_edge(
                    truth_offset + truth_index,
                    observation_offset + observation_index,
                    distance,
                    (observation_index - observation_count)
                    * identifier_weight,
                )
                assignment_edges[(truth_index, observation_index)] = (
                    truth_offset + truth_index,
                    edge_index,
                    distance,
                )

    while True:
        costs: list[tuple[float, int] | None] = [None] * len(graph)
        predecessors: list[tuple[int, int] | None] = [None] * len(graph)
        costs[source] = (0.0, 0)
        for _ in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                if costs[node] is None:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity == 0:
                        continue
                    candidate = (
                        costs[node][0] + edge.distance_cost,
                        costs[node][1] + edge.identifier_cost,
                    )
                    if costs[edge.target] is None or candidate < costs[edge.target]:
                        costs[edge.target] = candidate
                        predecessors[edge.target] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if predecessors[sink] is None:
            break
        node = sink
        while node != source:
            predecessor = predecessors[node]
            if predecessor is None:
                raise RuntimeError("association residual path is incomplete")
            previous, edge_index = predecessor
            edge = graph[previous][edge_index]
            edge.capacity = 0
            graph[node][edge.reverse_index].capacity = 1
            node = previous

    matches = []
    used_truth = set()
    used_observations = set()
    for (truth_index, observation_index), (
        node,
        edge_index,
        distance,
    ) in assignment_edges.items():
        if graph[node][edge_index].capacity != 0:
            continue
        truth_id = truth_items[truth_index].object_id
        observation_id = observed_items[observation_index].object_id
        used_truth.add(truth_id)
        used_observations.add(observation_id)
        matches.append(
            Association(
                stamp_ns=truth.stamp_ns,
                truth_id=truth_id,
                observation_id=observation_id,
                distance_m=distance,
            )
        )
    matches.sort(key=lambda item: (item.truth_id, item.observation_id))
    return AssociationResult(
        matches=tuple(matches),
        unmatched_truth_ids=tuple(
            item.object_id
            for item in truth_items
            if item.object_id not in used_truth
        ),
        unmatched_observation_ids=tuple(
            item.object_id
            for item in observed_items
            if item.object_id not in used_observations
        ),
    )


def _frames_by_stamp(frames: Sequence[ObjectFrame], name: str) -> dict[int, ObjectFrame]:
    result = {}
    previous = None
    for item in frames:
        if not isinstance(item, ObjectFrame):
            raise ValueError(f"{name} must contain ObjectFrame values")
        if previous is not None and item.stamp_ns <= previous:
            raise ValueError(f"{name} stamps must be strictly increasing")
        result[item.stamp_ns] = item
        previous = item.stamp_ns
    return result


def _validated_frame_pairs(
    truth_frames: Sequence[ObjectFrame],
    observation_frames: Sequence[ObjectFrame],
    *,
    observation_name: str,
    expected_stamps_ns: Sequence[int] | None,
) -> tuple[tuple[ObjectFrame, ObjectFrame], ...]:
    truth_by_stamp = _frames_by_stamp(truth_frames, "truth_frames")
    observation_by_stamp = _frames_by_stamp(
        observation_frames, observation_name
    )
    if not truth_by_stamp:
        raise ValueError("truth_frames must not be empty")

    if expected_stamps_ns is None:
        expected = tuple(truth_by_stamp)
    else:
        expected = tuple(
            _integer_stamp(stamp, "expected stamp")
            for stamp in expected_stamps_ns
        )
        if not expected or any(
            current <= previous
            for previous, current in zip(expected, expected[1:])
        ):
            raise ValueError(
                "expected stamp set must be nonempty and strictly increasing"
            )

    expected_set = set(expected)
    truth_set = set(truth_by_stamp)
    observation_set = set(observation_by_stamp)
    if truth_set != expected_set:
        raise ValueError(
            "truth_frames stamp set does not match expected stamp set"
        )
    if observation_set != expected_set:
        raise ValueError(
            f"{observation_name} stamp set does not match expected stamp set"
        )
    return tuple(
        (truth_by_stamp[stamp], observation_by_stamp[stamp])
        for stamp in expected
    )


def _frame_exclusion_reason(
    truth: ObjectFrame,
    observations: ObjectFrame,
    expected_frame_id: str | None,
) -> str | None:
    if expected_frame_id is not None:
        expected = _identifier(expected_frame_id, "expected_frame_id")
        if truth.frame_id != expected or observations.frame_id != expected:
            return (
                f"frame mismatch at stamp {truth.stamp_ns}: expected "
                f"{expected}, got {truth.frame_id}/{observations.frame_id}"
            )
    if truth.frame_id != observations.frame_id:
        return (
            f"frame mismatch at stamp {truth.stamp_ns}: "
            f"{truth.frame_id} != {observations.frame_id}"
        )
    return None


def _coverage_reason(reasons: Sequence[str]) -> str | None:
    unique = tuple(dict.fromkeys(reasons))
    return "; ".join(unique) if unique else None


def _metric(
    value: float | int | None,
    *,
    support: int,
    expected: int,
    unavailable_reason: str | None = None,
    partial_reason: str | None = None,
) -> MetricValue:
    return MetricValue(
        value=value,
        unavailable_reason=unavailable_reason,
        support=support,
        expected=expected,
        excluded=expected - support,
        partial_reason=partial_reason,
    )


def score_detections(
    truth_frames: Sequence[ObjectFrame],
    detection_frames: Sequence[ObjectFrame],
    *,
    maximum_distance_m: float,
    expected_stamps_ns: Sequence[int] | None = None,
    expected_frame_id: str | None = None,
) -> DetectionMetrics:
    frame_pairs = _validated_frame_pairs(
        truth_frames,
        detection_frames,
        observation_name="detection_frames",
        expected_stamps_ns=expected_stamps_ns,
    )
    matched = 0
    evaluated_truth_total = 0
    expected_truth_total = sum(len(truth.objects) for truth, _ in frame_pairs)
    false_positives = 0
    evaluated_frames = 0
    exclusions = []
    for truth, detections in frame_pairs:
        reason = _frame_exclusion_reason(
            truth, detections, expected_frame_id
        )
        if reason is not None:
            exclusions.append(FrameExclusion(truth.stamp_ns, reason))
            continue
        association = associate_frame(
            truth, detections, maximum_distance_m=maximum_distance_m
        )
        evaluated_frames += 1
        matched += len(association.matches)
        evaluated_truth_total += len(truth.objects)
        false_positives += len(association.unmatched_observation_ids)
    coverage_reason = _coverage_reason(
        tuple(exclusion.reason for exclusion in exclusions)
    )
    if expected_truth_total == 0:
        recall = _metric(
            None,
            support=0,
            expected=0,
            unavailable_reason="recall denominator is zero",
        )
    elif evaluated_truth_total == 0:
        recall = _metric(
            None,
            support=0,
            expected=expected_truth_total,
            unavailable_reason="all truth samples were excluded",
        )
    else:
        recall = _metric(
            matched / evaluated_truth_total,
            support=evaluated_truth_total,
            expected=expected_truth_total,
            partial_reason=coverage_reason,
        )
    frame_count = len(frame_pairs)
    if evaluated_frames == 0:
        false_positive_rate = _metric(
            None,
            support=0,
            expected=frame_count,
            unavailable_reason="all detection frames were excluded",
        )
    else:
        false_positive_rate = _metric(
            false_positives / evaluated_frames,
            support=evaluated_frames,
            expected=frame_count,
            partial_reason=coverage_reason,
        )
    return DetectionMetrics(
        recall=recall,
        false_positives=_metric(
            false_positives,
            support=evaluated_frames,
            expected=frame_count,
            partial_reason=coverage_reason,
        ) if evaluated_frames else _metric(
            None,
            support=0,
            expected=frame_count,
            unavailable_reason="all detection frames were excluded",
        ),
        false_positives_per_frame=false_positive_rate,
        excluded_frames=tuple(exclusions),
    )


def _rmse(
    values: Sequence[float],
    unavailable_reason: str,
    *,
    expected: int | None = None,
    partial_reason: str | None = None,
) -> MetricValue:
    expected_count = len(values) if expected is None else expected
    if not values:
        return _metric(
            None,
            support=0,
            expected=expected_count,
            unavailable_reason=unavailable_reason,
        )
    return _metric(
        value=math.sqrt(sum(value * value for value in values) / len(values)),
        support=len(values),
        expected=expected_count,
        partial_reason=partial_reason,
    )


def score_tracking(
    truth_frames: Sequence[ObjectFrame],
    track_frames: Sequence[ObjectFrame],
    *,
    maximum_distance_m: float,
    expected_stamps_ns: Sequence[int] | None = None,
    expected_frame_id: str | None = None,
) -> TrackingMetrics:
    frame_pairs = _validated_frame_pairs(
        truth_frames,
        track_frames,
        observation_name="track_frames",
        expected_stamps_ns=expected_stamps_ns,
    )
    first_truth_stamp = {}
    actor_stamps: dict[str, list[int]] = {}
    evaluated_actor_stamps: dict[str, list[int]] = {}
    initialized = {}
    last_track = {}
    in_drop = {}
    drop_episodes = 0
    id_switches = 0
    position_errors = []
    velocity_errors = []
    matched_actor_frames = 0
    id_transition_support = 0
    id_transition_expected = 0
    previous_truth_ids: set[str] = set()
    evaluated_actor_frames = 0
    expected_actor_frames = sum(len(truth.objects) for truth, _ in frame_pairs)
    exclusions = []
    for truth, tracks in frame_pairs:
        stamp = truth.stamp_ns
        truth_ids = {item.object_id for item in truth.objects}
        id_transition_expected += len(previous_truth_ids & truth_ids)
        previous_truth_ids = truth_ids
        for item in truth.objects:
            first_truth_stamp.setdefault(item.object_id, stamp)
            actor_stamps.setdefault(item.object_id, []).append(stamp)
        reason = _frame_exclusion_reason(truth, tracks, expected_frame_id)
        if reason is not None:
            exclusions.append(FrameExclusion(stamp, reason))
            last_track = {}
            continue
        evaluated_actor_frames += len(truth.objects)
        for item in truth.objects:
            evaluated_actor_stamps.setdefault(item.object_id, []).append(stamp)
        association = associate_frame(
            truth, tracks, maximum_distance_m=maximum_distance_m
        )
        truth_items = {item.object_id: item for item in truth.objects}
        track_items = {item.object_id: item for item in tracks.objects}
        matched_truth = set()
        current_tracks = {}
        for match in association.matches:
            matched_truth.add(match.truth_id)
            matched_actor_frames += 1
            if match.truth_id not in initialized:
                initialized[match.truth_id] = stamp
            previous_track = last_track.get(match.truth_id)
            if previous_track is not None:
                id_transition_support += 1
                if previous_track != match.observation_id:
                    id_switches += 1
            current_tracks[match.truth_id] = match.observation_id
            in_drop[match.truth_id] = False
            truth_item = truth_items[match.truth_id]
            track_item = track_items[match.observation_id]
            position_errors.append(match.distance_m)
            if truth_item.vx_mps is not None and track_item.vx_mps is not None:
                velocity_errors.append(
                    math.hypot(
                        truth_item.vx_mps - track_item.vx_mps,
                        truth_item.vy_mps - track_item.vy_mps,
                    )
                )
        last_track = current_tracks
        for item in truth.objects:
            if item.object_id not in initialized or item.object_id in matched_truth:
                continue
            if not in_drop.get(item.object_id, False):
                drop_episodes += 1
                in_drop[item.object_id] = True

    coverage_reason = _coverage_reason(
        tuple(exclusion.reason for exclusion in exclusions)
    )
    delay_metrics = {}
    for actor_id, first_stamp in sorted(first_truth_stamp.items()):
        all_stamps = actor_stamps[actor_id]
        evaluated_stamps = evaluated_actor_stamps.get(actor_id, [])
        if actor_id not in initialized:
            expected = len(all_stamps)
            support = len(evaluated_stamps)
            if support == 0:
                reason = (
                    "initialization delay is unavailable because all "
                    f"{expected} actor frames were excluded"
                )
            elif support < expected:
                reason = (
                    "initialization delay is unavailable because "
                    f"{expected - support} of {expected} actor frames were "
                    "excluded"
                )
            else:
                reason = "actor was never initialized"
            delay_metrics[actor_id] = _metric(
                None,
                support=support,
                expected=expected,
                unavailable_reason=reason,
            )
        else:
            initialization_stamp = initialized[actor_id]
            expected = sum(stamp <= initialization_stamp for stamp in all_stamps)
            support = sum(
                stamp <= initialization_stamp for stamp in evaluated_stamps
            )
            if support < expected:
                delay_metrics[actor_id] = _metric(
                    None,
                    support=support,
                    expected=expected,
                    unavailable_reason=(
                        "initialization delay is unavailable because "
                        f"{expected - support} of {expected} "
                        "pre-initialization actor frames were excluded"
                    ),
                )
            else:
                delay_metrics[actor_id] = _metric(
                    (initialization_stamp - first_stamp)
                    / _NANOSECONDS_PER_SECOND,
                    support=support,
                    expected=expected,
                )
    missed_actor_frames = evaluated_actor_frames - matched_actor_frames

    def opportunity_reason(label: str, support: int) -> str | None:
        missing = expected_actor_frames - support
        if missing == 0:
            return None
        reason = (
            f"{label} unavailable for {missing} of "
            f"{expected_actor_frames} actor frames"
        )
        if coverage_reason is not None:
            reason = f"{reason}; {coverage_reason}"
        return reason

    def opportunity_unavailable_reason(label: str) -> str:
        if expected_actor_frames == 0:
            return f"{label} denominator is zero"
        reason = (
            f"{label} unavailable for all {expected_actor_frames} actor frames"
        )
        if coverage_reason is not None:
            reason = f"{reason}; {coverage_reason}"
        return reason

    if id_transition_expected == 0:
        id_switch_metric = _metric(0, support=0, expected=0)
    elif id_transition_support == 0:
        id_unavailable_reason = (
            "ID switches are unavailable for all "
            f"{id_transition_expected} actor transitions"
        )
        if coverage_reason is not None:
            id_unavailable_reason = (
                f"{id_unavailable_reason}; {coverage_reason}"
            )
        id_switch_metric = _metric(
            None,
            support=0,
            expected=id_transition_expected,
            unavailable_reason=id_unavailable_reason,
        )
    else:
        id_partial = None
        if id_transition_support < id_transition_expected:
            id_partial = (
                "ID-switch comparison unavailable for "
                f"{id_transition_expected - id_transition_support} of "
                f"{id_transition_expected} actor transitions"
            )
            if coverage_reason is not None:
                id_partial = f"{id_partial}; {coverage_reason}"
        id_switch_metric = _metric(
            id_switches,
            support=id_transition_support,
            expected=id_transition_expected,
            partial_reason=id_partial,
        )
    return TrackingMetrics(
        initialization_delay_s=delay_metrics,
        missed_actor_frames=_metric(
            missed_actor_frames,
            support=evaluated_actor_frames,
            expected=expected_actor_frames,
            partial_reason=coverage_reason,
        ),
        drop_episodes=_metric(
            drop_episodes,
            support=evaluated_actor_frames,
            expected=expected_actor_frames,
            partial_reason=coverage_reason,
        ),
        completeness=_metric(
            matched_actor_frames / evaluated_actor_frames
            if evaluated_actor_frames
            else None,
            support=evaluated_actor_frames,
            expected=expected_actor_frames,
            unavailable_reason=(
                "all truth samples were excluded"
                if evaluated_actor_frames == 0
                else None
            ),
            partial_reason=coverage_reason,
        ),
        id_switches=id_switch_metric,
        position_rmse_m=_rmse(
            position_errors,
            opportunity_unavailable_reason("position"),
            expected=expected_actor_frames,
            partial_reason=opportunity_reason("position", len(position_errors)),
        ),
        velocity_rmse_mps=_rmse(
            velocity_errors,
            opportunity_unavailable_reason("velocity"),
            expected=expected_actor_frames,
            partial_reason=opportunity_reason("velocity", len(velocity_errors)),
        ),
        excluded_frames=tuple(exclusions),
    )


def score_predictions(
    predictions: Sequence[PredictionPoint],
    truth_frames: Sequence[ObjectFrame],
    *,
    source_associations: Mapping[tuple[int, str], str],
    horizons_s: Sequence[float],
) -> dict[float, HorizonMetrics]:
    horizons = tuple(_positive(value, "horizon") for value in horizons_s)
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be nonempty and unique")
    if tuple(sorted(horizons)) != horizons:
        raise ValueError("horizons must be increasing")
    truth_by_key = {}
    for frame in _frames_by_stamp(truth_frames, "truth_frames").values():
        for item in frame.objects:
            truth_by_key[(frame.stamp_ns, item.object_id)] = (
                frame.frame_id,
                item,
            )

    predictions_by_source: dict[
        tuple[int, str], dict[float, PredictionPoint]
    ] = {}
    for prediction in predictions:
        if not isinstance(prediction, PredictionPoint):
            raise ValueError("predictions must contain PredictionPoint values")
        if prediction.horizon_s not in horizons:
            raise ValueError(
                f"prediction uses undeclared horizon {prediction.horizon_s}"
            )
        source_key = (prediction.source_stamp_ns, prediction.track_id)
        source_predictions = predictions_by_source.setdefault(source_key, {})
        if prediction.horizon_s in source_predictions:
            raise ValueError(
                "duplicate prediction key: "
                f"{prediction.source_stamp_ns}/{prediction.track_id}/"
                f"{prediction.horizon_s}"
            )
        source_predictions[prediction.horizon_s] = prediction

    for source_key, truth_id in source_associations.items():
        if (
            not isinstance(source_key, tuple)
            or len(source_key) != 2
            or isinstance(source_key[0], bool)
            or not isinstance(source_key[0], int)
            or source_key[0] < 0
        ):
            raise ValueError("source association keys must be (stamp, track_id)")
        _identifier(source_key[1], "source association track_id")
        _identifier(truth_id, "source association truth_id")

    source_keys = tuple(
        sorted(set(predictions_by_source) | set(source_associations))
    )

    output = {}
    for horizon in horizons:
        ade_values = []
        fde_values = []
        exclusion_causes = []
        required_horizons = tuple(
            declared for declared in horizons if declared <= horizon
        )
        for source_key in source_keys:
            truth_id = source_associations.get(source_key)
            if truth_id is None:
                exclusion_causes.append("source association")
                continue
            source_predictions = predictions_by_source.get(source_key, {})
            if any(
                required not in source_predictions
                for required in required_horizons
            ):
                exclusion_causes.append("prediction horizon")
                continue
            source_errors = []
            truth_exclusion_cause = None
            for required in required_horizons:
                prediction = source_predictions[required]
                target_stamp = prediction.source_stamp_ns + int(
                    round(required * _NANOSECONDS_PER_SECOND)
                )
                truth_entry = truth_by_key.get((target_stamp, truth_id))
                if truth_entry is None:
                    truth_exclusion_cause = "future truth"
                    break
                truth_frame_id, truth = truth_entry
                if truth_frame_id != prediction.frame_id:
                    truth_exclusion_cause = "common-frame/TF truth"
                    break
                source_errors.append(
                    math.hypot(
                        prediction.x_m - truth.x_m,
                        prediction.y_m - truth.y_m,
                    )
                )
            if truth_exclusion_cause is not None:
                exclusion_causes.append(truth_exclusion_cause)
                continue
            ade_values.extend(source_errors)
            fde_values.append(source_errors[-1])

        support = len(fde_values)
        expected = len(source_keys)
        if support:
            partial_reason = None
            if exclusion_causes:
                cause = (
                    exclusion_causes[0]
                    if len(set(exclusion_causes)) == 1
                    else "prediction input"
                )
                partial_reason = (
                    f"{cause} unavailable for {len(exclusion_causes)} "
                    f"of {expected} trajectories"
                )
            ade = _metric(
                sum(ade_values) / len(ade_values),
                support=support,
                expected=expected,
                partial_reason=partial_reason,
            )
            fde = _metric(
                sum(fde_values) / len(fde_values),
                support=support,
                expected=expected,
                partial_reason=partial_reason,
            )
        else:
            unique_causes = set(exclusion_causes)
            if unique_causes == {"source association"}:
                reason = "source association is unavailable"
            elif unique_causes == {"prediction horizon"}:
                reason = "prediction horizon is incomplete"
            elif unique_causes == {"future truth"}:
                reason = "future truth is unavailable"
            elif unique_causes == {"common-frame/TF truth"}:
                reason = "common-frame/TF truth is unavailable"
            else:
                reason = "no complete prediction trajectories are available"
            ade = _metric(
                None,
                support=0,
                expected=expected,
                unavailable_reason=reason,
            )
            fde = _metric(
                None,
                support=0,
                expected=expected,
                unavailable_reason=reason,
            )
        output[horizon] = HorizonMetrics(
            ade_m=ade,
            fde_m=fde,
        )
    return output


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("report contains a nonfinite value")
    return value


def write_reports(
    run_directory: Path,
    *,
    metadata: RunMetadata,
    metrics: Mapping[str, Any],
) -> ReportArtifacts:
    root = Path(run_directory)
    if not root.is_absolute():
        root = root.resolve()
    if not root.is_dir():
        raise ValueError("run_directory must exist")
    document = {
        "schema_version": 1,
        "metadata": _json_value(metadata),
        "metrics": _json_value(metrics),
    }
    encoded = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    json_path = root / "metrics.json"
    temporary_json = root / ".metrics.json.tmp"
    temporary_json.write_text(encoded, encoding="utf-8")
    os.replace(temporary_json, json_path)

    lines = [
        "# MORAI classical perception validation",
        "",
        f"Actor preset: `{metadata.actor_preset}`",
        "",
        "## Metrics",
        "",
    ]
    for name, value in sorted(metrics.items()):
        if isinstance(value, MetricValue):
            rendered = (
                str(value.value)
                if value.value is not None
                else f"unavailable: {value.unavailable_reason}"
            )
        else:
            rendered = json.dumps(_json_value(value), sort_keys=True)
        lines.append(f"- `{name}`: {rendered}")
    markdown_path = root / "summary.md"
    temporary_markdown = root / ".summary.md.tmp"
    temporary_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary_markdown, markdown_path)
    return ReportArtifacts(json_path=json_path, markdown_path=markdown_path)
