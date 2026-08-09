
from ad_morai_bridge.arrival_time import ReceiptClockMapper


def test_receipt_clock_mapper_uses_one_stable_clock_offset():
    mapper = ReceiptClockMapper(
        sampled_monotonic=10.0,
        sampled_ros_ns=20_000_000_000,
    )

    assert mapper.stamp(9.75) == (19, 750_000_000)
    assert mapper.stamp(9.750_001) == (19, 750_001_000)


def test_receipt_clock_mapper_preserves_monotonic_packet_order():
    mapper = ReceiptClockMapper(
        sampled_monotonic=1_000_000.0,
        sampled_ros_ns=2_000_000_000_000_000,
    )

    first = mapper.stamp(999_999.123_456)
    second = mapper.stamp(999_999.123_457)
    first_ns = first[0] * 1_000_000_000 + first[1]
    second_ns = second[0] * 1_000_000_000 + second[1]
    assert second_ns > first_ns
