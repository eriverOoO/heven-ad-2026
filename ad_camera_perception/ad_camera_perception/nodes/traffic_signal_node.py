"""Compatibility entry point for the mm-2025 traffic-light detector node."""

from .traffic_light_detector_node import TrafficLightDetectorNode, main


TrafficSignalNode = TrafficLightDetectorNode


__all__ = ["TrafficLightDetectorNode", "TrafficSignalNode", "main"]


if __name__ == "__main__":
    main()
