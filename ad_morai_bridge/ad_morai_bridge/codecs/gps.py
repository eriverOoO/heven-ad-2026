from datetime import datetime, timezone
import math

from .common import PacketFormatError
from ..protocol_records import GpggaRecord, GprmcRecord, GpsFixRecord


def _optional_float(value: str, field: str) -> float | None:
    if not value:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PacketFormatError("gps", f"{field} is not finite")
    return parsed


def _utc_seconds(value: str) -> int:
    if not isinstance(value, str):
        raise ValueError("UTC time is not text")
    whole, separator, fraction = value.partition(".")
    if len(whole) != 6 or not whole.isdigit():
        raise ValueError("UTC time must use HHMMSS grammar")
    if separator and (not fraction or not fraction.isdigit()):
        raise ValueError("UTC fractional seconds are malformed")
    hour = int(whole[:2])
    minute = int(whole[2:4])
    second = int(whole[4:6])
    if hour >= 24 or minute >= 60 or second >= 60:
        raise ValueError("UTC time is outside HH/MM/SS ranges")
    # MORAI emits variable-width fractional text. It is preserved in the raw
    # message but is not interpreted as decimal nanoseconds.
    return hour * 3_600 + minute * 60 + second


def _validated_utc(value: str) -> str:
    try:
        _utc_seconds(value)
    except ValueError as exc:
        raise PacketFormatError("gps", str(exc)) from exc
    return value


def _coordinate(value: str, direction: str, stream: str) -> float:
    try:
        raw = float(value)
    except ValueError as exc:
        raise PacketFormatError(stream, "invalid coordinate") from exc
    if direction in ("N", "S"):
        maximum = 90
    elif direction in ("E", "W"):
        maximum = 180
    else:
        raise PacketFormatError(stream, f"invalid hemisphere {direction!r}")
    if not math.isfinite(raw) or raw < 0.0:
        raise PacketFormatError(stream, "coordinate is not finite and non-negative")
    degrees = int(raw / 100.0)
    minutes = raw - degrees * 100.0
    if minutes >= 60.0 or degrees > maximum or (degrees == maximum and minutes > 0.0):
        raise PacketFormatError(stream, "coordinate is outside latitude/longitude range")
    result = degrees + minutes / 60.0
    if direction in ("S", "W"):
        result = -result
    return result


def _sentence_fields(packet: bytes) -> tuple[list[str], int | None, str]:
    try:
        text = packet.rstrip(b"\0\r\n").decode("ascii")
    except UnicodeDecodeError as exc:
        raise PacketFormatError("gps", "sentence is not ASCII") from exc
    if not text.startswith("$"):
        raise PacketFormatError("gps", "missing NMEA prefix")
    body, marker, supplied = text[1:].partition("*")
    checksum = None
    if marker:
        if len(supplied) != 2:
            raise PacketFormatError("gps", "invalid checksum field")
        try:
            checksum = int(supplied, 16)
        except ValueError as exc:
            raise PacketFormatError("gps", "invalid checksum field") from exc
        calculated = 0
        for byte in body.encode("ascii"):
            calculated ^= byte
        if calculated != checksum:
            raise PacketFormatError(
                "gps", f"checksum {calculated:02X} != {checksum:02X}"
            )
    return body.split(","), checksum, text


def decode_nmea(packet: bytes) -> GprmcRecord | GpggaRecord:
    fields, checksum, sentence = _sentence_fields(packet)
    kind = fields[0] if fields else ""
    try:
        if kind == "GPRMC":
            if len(fields) < 10:
                raise PacketFormatError("gps", "truncated GPRMC sentence")
            fields.extend([""] * (13 - len(fields)))
            return GprmcRecord(
                utc=_validated_utc(fields[1]),
                valid=fields[2] == "A",
                latitude=_coordinate(fields[3], fields[4], "gps"),
                longitude=_coordinate(fields[5], fields[6], "gps"),
                speed_knots=_optional_float(fields[7], "speed"),
                track_degrees=_optional_float(fields[8], "course"),
                date=fields[9],
                magnetic_variation_degrees=_optional_float(
                    fields[10], "magnetic variation"
                ),
                magnetic_variation_direction=fields[11],
                mode_indicator=fields[12],
                checksum=checksum,
                sentence=sentence,
            )
        if kind == "GPGGA":
            if len(fields) < 10:
                raise PacketFormatError("gps", "truncated GPGGA sentence")
            fields.extend([""] * (15 - len(fields)))
            return GpggaRecord(
                utc=_validated_utc(fields[1]),
                latitude=_coordinate(fields[2], fields[3], "gps"),
                longitude=_coordinate(fields[4], fields[5], "gps"),
                fix_quality=int(fields[6] or 0),
                satellites=int(fields[7] or 0),
                hdop=_optional_float(fields[8], "HDOP"),
                altitude=_optional_float(fields[9], "altitude"),
                altitude_unit=fields[10],
                geoid_separation=_optional_float(
                    fields[11], "geoid separation"
                ),
                geoid_unit=fields[12],
                differential_age=_optional_float(
                    fields[13], "differential age"
                ),
                station_id=fields[14],
                checksum=checksum,
                sentence=sentence,
            )
    except (ValueError, IndexError) as exc:
        raise PacketFormatError("gps", f"invalid {kind} field: {exc}") from exc
    raise PacketFormatError("gps", f"unsupported sentence {kind!r}")


class GpsFixAccumulator:
    def __init__(self):
        self._rmc: GprmcRecord | None = None

    def update(self, sentence: GprmcRecord | GpggaRecord) -> GpsFixRecord | None:
        if isinstance(sentence, GprmcRecord):
            self._rmc = sentence
            return None
        rmc = self._rmc
        use_rmc = bool(rmc and rmc.valid and _utc_distance_sec(rmc.utc, sentence.utc) <= 1.5)
        has_position = not (
            sentence.latitude == 0.0 and sentence.longitude == 0.0
        )
        source_stamp = (
            paired_gga_epoch_stamp(rmc, sentence)
            if use_rmc and rmc is not None
            else None
        )
        return GpsFixRecord(
            latitude=sentence.latitude,
            longitude=sentence.longitude,
            altitude=sentence.altitude or 0.0,
            status=1 if sentence.fix_quality > 0 and has_position else -1,
            satellites=sentence.satellites,
            hdop=sentence.hdop,
            speed_mps=(
                rmc.speed_knots * 0.514444
                if use_rmc and rmc is not None and rmc.speed_knots is not None
                else None
            ),
            track_degrees=rmc.track_degrees if use_rmc and rmc is not None else None,
            source_stamp=source_stamp,
            source_rejected=bool(use_rmc and source_stamp is None),
        )


def _utc_distance_sec(first: str, second: str) -> float:
    try:
        delta = abs(_utc_seconds(first) - _utc_seconds(second))
    except (TypeError, ValueError):
        return float("inf")
    return min(delta, 86_400.0 - delta)


def rmc_epoch_stamp(record: GprmcRecord) -> tuple[int, int] | None:
    """Return an RMC date + whole-HHMMSS UTC epoch, never fraction-derived."""

    midnight = _rmc_midnight_epoch(record)
    if midnight is None:
        return None
    try:
        seconds = _utc_seconds(record.utc)
    except (TypeError, ValueError):
        return None
    return (midnight + seconds, 0)


def paired_gga_epoch_stamp(
    rmc: GprmcRecord,
    gga: GpggaRecord,
) -> tuple[int, int] | None:
    """Use the paired RMC date and GGA whole second, including day rollover."""

    midnight = _rmc_midnight_epoch(rmc)
    if midnight is None:
        return None
    try:
        rmc_seconds = _utc_seconds(rmc.utc)
        gga_seconds = _utc_seconds(gga.utc)
    except (TypeError, ValueError):
        return None
    delta = gga_seconds - rmc_seconds
    if delta < -43_200:
        midnight += 86_400
    elif delta > 43_200:
        midnight -= 86_400
    return (midnight + gga_seconds, 0)


def _rmc_midnight_epoch(record: GprmcRecord) -> int | None:
    date = record.date
    if len(date) != 6 or not date.isdigit():
        return None
    try:
        day = int(date[:2])
        month = int(date[2:4])
        year_2digit = int(date[4:6])
        # NMEA's conventional 1980..2079 pivot makes the two-digit date usable;
        # the bridge acceptance window still proves whether that epoch matches ROS.
        year = 1900 + year_2digit if year_2digit >= 80 else 2000 + year_2digit
        midnight = datetime(year, month, day, tzinfo=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None
    return int(midnight.timestamp())
