#!/usr/bin/env python3
"""RViz-only text label for the opt-in multi-pipeline study demo."""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


class StudyPipelineLabel(Node):
    def __init__(self):
        super().__init__("ad_study_pipeline_label")
        self.declare_parameter("label", "HEVEN STUDY PIPELINE")
        self.declare_parameter("frame_id", "base_link")
        self.publisher = self.create_publisher(MarkerArray, "/ad/study/pipeline_label", 1)
        self.timer = self.create_timer(0.5, self.publish_label)

    def publish_label(self):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = str(self.get_parameter("frame_id").value)
        marker.ns = "study_pipeline_label"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 8.0
        marker.pose.position.z = 5.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.85
        marker.color.b = 0.1
        marker.color.a = 1.0
        marker.text = str(self.get_parameter("label").value)
        self.publisher.publish(MarkerArray(markers=[marker]))


def main():
    rclpy.init()
    node = StudyPipelineLabel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
