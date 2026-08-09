from ad_morai_bridge_dev.bridge.raw_streams import RawStreamSpec, read_raw_stream_specs


class FakeParameters:
    def __init__(self, values):
        self.values = values

    def declare_parameter(self, name, default):
        return type("Parameter", (), {"value": self.values.get(name, default)})()


def test_raw_stream_specs_read_declared_extra_streams():
    node = FakeParameters(
        {
            "extra_raw_streams.names": ["radar_legacy"],
            "extra_raw_streams.radar_legacy.bind_ip": "127.0.0.1",
            "extra_raw_streams.radar_legacy.port": 9401,
            "extra_raw_streams.radar_legacy.frame_id": "radar_front",
        }
    )

    assert read_raw_stream_specs(node) == (
        RawStreamSpec("radar_legacy", "127.0.0.1", 9401, "radar_front"),
    )


def test_raw_stream_specs_default_to_empty():
    assert read_raw_stream_specs(FakeParameters({})) == ()
    # ROS YAML may hand an unset list through as None.
    assert read_raw_stream_specs(
        FakeParameters({"extra_raw_streams.names": None})
    ) == ()
