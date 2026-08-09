"""mm-2025 traffic-light classes and framework-independent evaluation logic."""

from collections import Counter, deque
from dataclasses import dataclass
from math import hypot
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class SignalAspects:
    """Independent illuminated aspects represented by one detector class."""

    red: bool = False
    yellow: bool = False
    straight_green: bool = False
    left_green: bool = False

    @property
    def illuminated(self) -> bool:
        """Return whether at least one known aspect is illuminated."""
        return self.red or self.yellow or self.straight_green or self.left_green


MM2025_CLASS_ASPECTS = {
    "1300": SignalAspects(straight_green=True),
    "1301": SignalAspects(red=True),
    "1302": SignalAspects(yellow=True),
    "1303": SignalAspects(red=True, left_green=True),
    "1305": SignalAspects(left_green=True),
    "1400": SignalAspects(straight_green=True),
    "1401": SignalAspects(red=True),
    "1402": SignalAspects(yellow=True),
    "1403": SignalAspects(red=True, left_green=True),
    "1404": SignalAspects(red=True, yellow=True),
    "1405": SignalAspects(straight_green=True, left_green=True),
    "1406": SignalAspects(yellow=True, straight_green=True),
}


def aspects_for_class(class_name: str) -> Optional[SignalAspects]:
    """Map an mm-2025 class name to illuminated aspects."""
    return MM2025_CLASS_ASPECTS.get(str(class_name).strip())


@dataclass(frozen=True)
class TrafficDetection:
    """Small evaluator-facing representation of one detector bbox."""

    detection_id: str
    class_name: str
    confidence: float
    center_x: float
    center_y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        """Return the non-negative bbox area."""
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True)
class EvaluatedSignal:
    """One selected and temporally confirmed signal observation."""

    valid: bool
    confidence: float = 0.0
    aspects: SignalAspects = SignalAspects()
    source_class: str = ""
    detection_id: str = ""


class TargetSelector:
    """Select the ego-relevant supported detection with a stable score."""

    def __init__(
        self,
        image_width: int,
        image_height: int,
        target_roi_normalized: Sequence[float],
        confidence_weight: float = 0.60,
        area_weight: float = 0.25,
        center_weight: float = 0.15,
    ) -> None:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("expected image dimensions must be positive")
        if len(target_roi_normalized) != 4:
            raise ValueError("target_roi_normalized must contain four values")
        x1, y1, x2, y2 = (float(value) for value in target_roi_normalized)
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("target ROI must be ordered within normalized image bounds")
        weights = (confidence_weight, area_weight, center_weight)
        if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
            raise ValueError("target selection weights must be non-negative and non-zero")

        self._width = float(image_width)
        self._height = float(image_height)
        self._roi = (x1, y1, x2, y2)
        total = sum(weights)
        self._weights = tuple(value / total for value in weights)

    def _score(self, detection: TrafficDetection) -> Optional[float]:
        x = detection.center_x / self._width
        y = detection.center_y / self._height
        x1, y1, x2, y2 = self._roi
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return None
        if aspects_for_class(detection.class_name) is None:
            return None

        roi_center_x = (x1 + x2) / 2.0
        roi_center_y = (y1 + y2) / 2.0
        half_diagonal = max(hypot(x2 - x1, y2 - y1) / 2.0, 1.0e-9)
        center_score = max(
            0.0,
            1.0 - hypot(x - roi_center_x, y - roi_center_y) / half_diagonal,
        )
        area_score = min(1.0, detection.area / (self._width * self._height * 0.05))
        confidence_score = min(1.0, max(0.0, detection.confidence))
        confidence_weight, area_weight, center_weight = self._weights
        return (
            confidence_weight * confidence_score
            + area_weight * area_score
            + center_weight * center_score
        )

    def select(self, detections: Iterable[TrafficDetection]) -> Optional[TrafficDetection]:
        """Return the highest scoring supported detection inside the target ROI."""
        candidates = []
        for detection in detections:
            score = self._score(detection)
            if score is not None:
                candidates.append((score, detection.confidence, detection.area, detection))
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (item[0], item[1], item[2], item[3].detection_id),
        )[3]


class SignalVoteFilter:
    """Confirm detector classes with a bounded sliding-window vote."""

    def __init__(self, window_frames: int, minimum_vote_frames: int) -> None:
        if window_frames <= 0:
            raise ValueError("window_frames must be positive")
        if not 1 <= minimum_vote_frames <= window_frames:
            raise ValueError("minimum_vote_frames must be within the window")
        self._window = deque(maxlen=int(window_frames))
        self._minimum_votes = int(minimum_vote_frames)

    def update(self, selected: Optional[TrafficDetection]) -> EvaluatedSignal:
        """Add one frame and return a valid signal only after enough exact votes."""
        self._window.append(selected)
        supported = [
            item for item in self._window
            if item is not None and aspects_for_class(item.class_name) is not None
        ]
        if not supported:
            return EvaluatedSignal(valid=False)

        current_aspects = (
            aspects_for_class(selected.class_name)
            if selected is not None
            else None
        )
        current = EvaluatedSignal(
            valid=False,
            confidence=selected.confidence if current_aspects else 0.0,
            aspects=current_aspects or SignalAspects(),
            source_class=selected.class_name if current_aspects else "",
            detection_id=selected.detection_id if current_aspects else "",
        )

        counts = Counter(item.class_name for item in supported)
        winning_class, winning_votes = max(
            counts.items(), key=lambda item: (item[1], item[0])
        )
        if winning_votes < self._minimum_votes:
            return current

        representative = max(
            (item for item in supported if item.class_name == winning_class),
            key=lambda item: (item.confidence, item.area, item.detection_id),
        )
        return EvaluatedSignal(
            valid=True,
            confidence=representative.confidence,
            aspects=aspects_for_class(winning_class) or SignalAspects(),
            source_class=winning_class,
            detection_id=(
                selected.detection_id
                if selected is not None
                and aspects_for_class(selected.class_name) is not None
                else ""
            ),
        )

    def clear(self) -> None:
        """Forget all prior observations after an upstream stale timeout."""
        self._window.clear()
