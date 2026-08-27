import unittest
from types import SimpleNamespace

from ad_lidar_perception.detection_recording import summarize_detected_objects

LABEL_NAMES = {0: "unknown", 1: "vehicle", 7: "pedestrian"}


def _object(label, score, position, dimensions):
    return SimpleNamespace(
        classification=[SimpleNamespace(label=label)],
        existence_probability=score,
        kinematics=SimpleNamespace(
            pose_with_covariance=SimpleNamespace(
                pose=SimpleNamespace(position=SimpleNamespace(x=position[0], y=position[1], z=position[2]))
            )
        ),
        shape=SimpleNamespace(
            dimensions=SimpleNamespace(x=dimensions[0], y=dimensions[1], z=dimensions[2])
        ),
    )


def _message(objects, *, stamp_sec, stamp_nanosec, frame_id="lidar_link"):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp_sec, nanosec=stamp_nanosec), frame_id=frame_id),
        objects=objects,
    )


class DetectionRecordingTest(unittest.TestCase):
    def test_summarizes_objects_and_latency(self):
        message = _message(
            [_object(1, 0.9, (1.0, 2.0, 0.0), (4.0, 2.0, 1.5))],
            stamp_sec=1, stamp_nanosec=0,
        )
        record = summarize_detected_objects(
            message, backend="centerpoint", received_wall_time_ns=1_500_000_000, label_names=LABEL_NAMES
        )
        self.assertEqual(record["schema"], "heven.ros_detection_comparison.v1")
        self.assertEqual(record["backend"], "centerpoint")
        self.assertEqual(record["object_count"], 1)
        self.assertEqual(record["objects"][0]["class_name"], "vehicle")
        self.assertAlmostEqual(record["latency_ms"], 500.0)

    def test_empty_objects(self):
        message = _message([], stamp_sec=0, stamp_nanosec=0)
        record = summarize_detected_objects(
            message, backend="euclidean", received_wall_time_ns=0, label_names=LABEL_NAMES
        )
        self.assertEqual(record["object_count"], 0)
        self.assertEqual(record["objects"], [])

    def test_unknown_label_falls_back_to_generic_name(self):
        message = _message(
            [_object(99, 0.5, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))], stamp_sec=0, stamp_nanosec=0
        )
        record = summarize_detected_objects(
            message, backend="euclidean", received_wall_time_ns=0, label_names=LABEL_NAMES
        )
        self.assertEqual(record["objects"][0]["class_name"], "label_99")

    def test_rejects_empty_backend(self):
        message = _message([], stamp_sec=0, stamp_nanosec=0)
        with self.assertRaises(ValueError):
            summarize_detected_objects(message, backend="", received_wall_time_ns=0, label_names=LABEL_NAMES)


if __name__ == "__main__":
    unittest.main()
