import pytest

from ad_morai_bridge.codecs.common import PacketFormatError
from ad_morai_bridge.codecs.gps import (
    GpsFixAccumulator,
    decode_nmea,
    rmc_epoch_stamp,
)
from ad_morai_bridge.protocol_records import GpggaRecord, GprmcRecord


def _sentence(body: str) -> bytes:
    checksum = 0
    for char in body:
        checksum ^= ord(char)
    return f"${body}*{checksum:02X}\r\n".encode()


def test_rmc_and_gga_decode_hemispheres_and_pair_checked_fix():
    rmc = decode_nmea(
        _sentence("GPRMC,123519,A,4807.038,S,01131.000,W,022.4,084.4,230394,003.1,W,A")
    )
    gga = decode_nmea(
        _sentence("GPGGA,123520,4807.038,S,01131.000,W,1,08,0.9,545.4,M,46.9,M,,")
    )

    assert isinstance(rmc, GprmcRecord)
    assert isinstance(gga, GpggaRecord)
    assert rmc.latitude == pytest.approx(-48.1173)
    assert rmc.longitude == pytest.approx(-11.5166667)
    accumulator = GpsFixAccumulator()
    assert accumulator.update(rmc) is None
    fix = accumulator.update(gga)
    assert fix is not None
    assert fix.altitude == pytest.approx(545.4)
    assert fix.speed_mps == pytest.approx(22.4 * 0.514444, rel=1e-5)
    assert fix.track_degrees == pytest.approx(84.4)
    assert accumulator.update(rmc) is None


def test_rmc_and_gga_preserve_every_semantic_field_and_checksum():
    rmc_packet = _sentence(
        "GPRMC,123519.25,A,4807.038,S,01131.000,W,22.4,84.4,230394,3.1,E,D"
    )
    gga_packet = _sentence(
        "GPGGA,123520.50,4807.038,N,01131.000,E,2,11,0.8,545.4,M,46.9,M,1.5,0042"
    )
    rmc = decode_nmea(rmc_packet)
    gga = decode_nmea(gga_packet)

    assert rmc.speed_knots == pytest.approx(22.4)
    assert rmc.track_degrees == pytest.approx(84.4)
    assert rmc.magnetic_variation_degrees == pytest.approx(3.1)
    assert rmc.magnetic_variation_direction == "E"
    assert rmc.mode_indicator == "D"
    assert rmc.checksum is not None
    assert gga.satellites == 11
    assert gga.geoid_separation == pytest.approx(46.9)
    assert gga.altitude_unit == "M"
    assert gga.geoid_unit == "M"
    assert gga.differential_age == pytest.approx(1.5)
    assert gga.station_id == "0042"
    assert gga.checksum is not None
    assert rmc.sentence == rmc_packet.decode().rstrip("\r\n")
    assert gga.sentence == gga_packet.decode().rstrip("\r\n")


def test_bad_checksum_unknown_sentence_and_truncated_sentence_fail():
    with pytest.raises(PacketFormatError):
        decode_nmea(b"$GPRMC,1,A,,,,,,,,,,*00\r\n")
    with pytest.raises(PacketFormatError):
        decode_nmea(_sentence("GPVTG,1,2,3"))
    with pytest.raises(PacketFormatError):
        decode_nmea(b"$GPGGA,1")
    with pytest.raises(PacketFormatError):
        decode_nmea(
            _sentence("GPGGA,123520,1e400,N,01131.000,E,1,08,0.9,1.0,M,0.0,M,,")
        )
    with pytest.raises(PacketFormatError):
        decode_nmea(
            _sentence("GPGGA,123520,9100.000,N,01131.000,E,1,08,0.9,1.0,M,0.0,M,,")
        )


@pytest.mark.parametrize(
    "body",
    [
        "GPRMC,123519,A,4807.038,N,01131.000,E,nan,084.4,230394,,,A",
        "GPRMC,123519,A,4807.038,N,01131.000,E,inf,084.4,230394,,,A",
        "GPRMC,123519,A,4807.038,N,01131.000,E,022.4,nan,230394,,,A",
        "GPRMC,123519,A,4807.038,N,01131.000,E,022.4,inf,230394,,,A",
        "GPGGA,123520,4807.038,N,01131.000,E,1,08,nan,545.4,M,46.9,M,,",
        "GPGGA,123520,4807.038,N,01131.000,E,1,08,inf,545.4,M,46.9,M,,",
        "GPGGA,123520,4807.038,N,01131.000,E,1,08,0.9,nan,M,46.9,M,,",
        "GPGGA,123520,4807.038,N,01131.000,E,1,08,0.9,inf,M,46.9,M,,",
    ],
)
def test_nonfinite_optional_nmea_fields_are_rejected(body):
    with pytest.raises(PacketFormatError):
        decode_nmea(_sentence(body))


@pytest.mark.parametrize("kind", ["GPRMC", "GPGGA"])
@pytest.mark.parametrize(
    "utc",
    ["", "12345", "12AB00", "240000", "126000", "125960", "120000."],
)
def test_nmea_utc_requires_valid_hhmmss_grammar_and_ranges(kind, utc):
    if kind == "GPRMC":
        body = f"GPRMC,{utc},A,4807.038,N,01131.000,E,1.0,2.0,230394,,,A"
    else:
        body = f"GPGGA,{utc},4807.038,N,01131.000,E,1,08,0.9,1.0,M,0.0,M,,"

    with pytest.raises(PacketFormatError):
        decode_nmea(_sentence(body))


@pytest.mark.parametrize(
    "rmc_utc,valid",
    [("120000", True), ("120010", False)],
)
def test_stale_or_invalid_rmc_is_not_attached_to_gga(rmc_utc, valid):
    accumulator = GpsFixAccumulator()
    accumulator.update(
        GprmcRecord(
            utc=rmc_utc,
            valid=valid,
            latitude=37.0,
            longitude=126.0,
            speed_knots=10.0,
            track_degrees=20.0,
            date="010101",
            magnetic_variation_degrees=None,
            magnetic_variation_direction="",
            mode_indicator="",
            checksum=None,
        )
    )

    fix = accumulator.update(
        GpggaRecord(
            utc="120010",
            latitude=37.0,
            longitude=126.0,
            fix_quality=1,
            satellites=8,
            hdop=1.0,
            altitude=10.0,
            altitude_unit="M",
            geoid_separation=None,
            geoid_unit="",
            differential_age=None,
            station_id="",
            checksum=None,
        )
    )

    assert fix is not None
    assert fix.speed_mps is None
    assert fix.track_degrees is None


@pytest.mark.parametrize("utc", ["999999", "not-a-time"])
def test_malformed_matching_utc_records_cannot_pair_as_fresh(utc):
    accumulator = GpsFixAccumulator()
    accumulator.update(
        GprmcRecord(
            utc=utc,
            valid=True,
            latitude=37.0,
            longitude=126.0,
            speed_knots=10.0,
            track_degrees=20.0,
            date="010101",
            magnetic_variation_degrees=None,
            magnetic_variation_direction="",
            mode_indicator="",
            checksum=None,
        )
    )

    fix = accumulator.update(
        GpggaRecord(
            utc=utc,
            latitude=37.0,
            longitude=126.0,
            fix_quality=1,
            satellites=8,
            hdop=1.0,
            altitude=10.0,
            altitude_unit="M",
            geoid_separation=None,
            geoid_unit="",
            differential_age=None,
            station_id="",
            checksum=None,
        )
    )

    assert fix is not None
    assert fix.speed_mps is None
    assert fix.track_degrees is None


def test_valid_fractional_utc_pairs_across_midnight_wrap():
    rmc = decode_nmea(
        _sentence("GPRMC,235959.500,A,4807.038,N,01131.000,E,10.0,20.0,230394,,,A")
    )
    gga = decode_nmea(
        _sentence("GPGGA,000000.500,4807.038,N,01131.000,E,1,08,0.9,1.0,M,0.0,M,,")
    )
    accumulator = GpsFixAccumulator()
    accumulator.update(rmc)

    fix = accumulator.update(gga)

    assert fix is not None
    assert fix.speed_mps == pytest.approx(10.0 * 0.514444)
    assert fix.track_degrees == pytest.approx(20.0)
    assert fix.source_stamp == (764467200, 0)


@pytest.mark.parametrize(
    "utc",
    ["123519", "123519.2", "123519.25", "123519.999999999999"],
)
def test_rmc_epoch_uses_date_and_whole_hhmmss_without_decimal_nanoseconds(utc):
    record = decode_nmea(
        _sentence(
            f"GPRMC,{utc},A,4807.038,N,01131.000,E,10.0,20.0,230394,,,A"
        )
    )

    assert rmc_epoch_stamp(record) == (764426119, 0)


@pytest.mark.parametrize("date", ["", "23039", "321399", "notday"])
def test_rmc_epoch_requires_an_unambiguous_calendar_date(date):
    record = GprmcRecord(
        utc="123519.25",
        valid=True,
        latitude=37.0,
        longitude=126.0,
        speed_knots=0.0,
        track_degrees=0.0,
        date=date,
        magnetic_variation_degrees=None,
        magnetic_variation_direction="",
        mode_indicator="",
        checksum=None,
    )

    assert rmc_epoch_stamp(record) is None


def test_zero_zero_gga_is_no_fix_even_when_morai_reports_fix_quality():
    accumulator = GpsFixAccumulator()

    fix = accumulator.update(
        GpggaRecord(
            utc="120010",
            latitude=0.0,
            longitude=0.0,
            fix_quality=1,
            satellites=8,
            hdop=1.0,
            altitude=0.0,
            altitude_unit="M",
            geoid_separation=None,
            geoid_unit="",
            differential_age=None,
            station_id="",
            checksum=None,
        )
    )

    assert fix is not None
    assert fix.status < 0
