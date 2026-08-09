"""Typed records produced by the MORAI UDP protocol decoders."""

from dataclasses import dataclass


Vector3 = tuple[float, float, float]
Stamp = tuple[int, int]


@dataclass(frozen=True)
class CtrlCommandRecord:
    ctrl_mode: int = 2
    gear: int = 4
    long_cmd_type: int = 1
    velocity: float = 0.0
    acceleration: float = 0.0
    accel: float = 0.0
    brake: float = 0.0
    steering: float = 0.0


@dataclass(frozen=True)
class CollisionRecord:
    object_type: int
    object_id: int
    position: Vector3
    global_offset: Vector3


@dataclass(frozen=True)
class CollisionArrayRecord:
    stamp: Stamp
    collisions: tuple[CollisionRecord, ...]


@dataclass(frozen=True)
class EgoStatusRecord:
    stamp: Stamp | None
    ctrl_mode: int
    gear: int
    signed_velocity: float
    map_data_id: int
    accel: float
    brake: float
    size: Vector3
    overhang: float
    wheelbase: float
    rear_overhang: float
    position: Vector3
    rpy: Vector3
    velocity: Vector3
    angular_velocity: Vector3
    acceleration: Vector3
    steering: float
    link_id: str
    tire_metrics: tuple[float, ...] = ()


@dataclass(frozen=True)
class ImuRecord:
    stamp: Stamp | None
    orientation_xyzw: tuple[float, float, float, float]
    angular_velocity: Vector3
    linear_acceleration: Vector3


@dataclass(frozen=True)
class GprmcRecord:
    utc: str
    valid: bool
    latitude: float
    longitude: float
    speed_knots: float | None
    track_degrees: float | None
    date: str
    magnetic_variation_degrees: float | None
    magnetic_variation_direction: str
    mode_indicator: str
    checksum: int | None
    sentence: str = ""


@dataclass(frozen=True)
class GpggaRecord:
    utc: str
    latitude: float
    longitude: float
    fix_quality: int
    satellites: int
    hdop: float | None
    altitude: float | None
    altitude_unit: str
    geoid_separation: float | None
    geoid_unit: str
    differential_age: float | None
    station_id: str
    checksum: int | None
    sentence: str = ""


@dataclass(frozen=True)
class GpsFixRecord:
    latitude: float
    longitude: float
    altitude: float
    status: int
    satellites: int
    hdop: float | None
    speed_mps: float | None
    track_degrees: float | None
    source_stamp: Stamp | None = None
    source_rejected: bool = False


@dataclass(frozen=True)
class CameraFrameRecord:
    stamp: Stamp
    jpeg: bytes
    first_arrived: float


@dataclass(frozen=True)
class VelodynePacketRecord:
    stamp: Stamp
    data: bytes
    first_azimuth_hundredths: int
