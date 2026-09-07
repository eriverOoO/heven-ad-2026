import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from validate_morai_audit_bag import REQUIRED_TOPICS, validate_metadata


def metadata(topics, paths=("data_0.mcap",), duration=180_000_000_000):
    return {"rosbag2_bagfile_information": {"duration": {"nanoseconds": duration}, "relative_file_paths": list(paths), "topics_with_message_count": [{"topic_metadata": {"name": name, "type": kind}, "message_count": 10} for name, kind in topics.items()]}}


class MoraiAuditBagValidationTest(unittest.TestCase):
    def test_valid_metadata_is_warn_until_deep_content_check(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data_0.mcap").write_bytes(b"mcap")
            result = validate_metadata(metadata(REQUIRED_TOPICS), root)
        self.assertEqual(result["status"], "WARN")
        self.assertFalse(any(item["status"] == "FAIL" for item in result["checks"]))

    def test_missing_topic_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data_0.mcap").write_bytes(b"mcap")
            topics = dict(REQUIRED_TOPICS)
            del topics["/clock"]
            result = validate_metadata(metadata(topics), root)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any(item["name"] == "/clock" and item["status"] == "FAIL" for item in result["checks"]))

    def test_missing_payload_fails(self):
        with TemporaryDirectory() as directory:
            result = validate_metadata(metadata(REQUIRED_TOPICS), Path(directory))
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any(item["name"] == "payload" and item["status"] == "FAIL" for item in result["checks"]))


if __name__ == "__main__":
    unittest.main()
