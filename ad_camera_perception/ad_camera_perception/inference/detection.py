"""Framework-independent detection value objects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One axis-aligned object detection in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        """Return the non-negative bounding-box width."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Return the non-negative bounding-box height."""
        return max(0.0, self.y2 - self.y1)
