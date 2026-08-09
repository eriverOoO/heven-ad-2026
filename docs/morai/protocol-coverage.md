# MORAI 2025 protocol coverage

이 문서는 `protocol_coverage.json`의 사람이 읽는 대응표다. UDP 바이트의 기준은
2025 대회 빌드에서 확인한 패킷과 이관된 `mm_udp/lib/define`이며, 공식 MORAI
24.R2 메시지 저장소의 커밋
`176610aeaecad28cc31e62e1cd2548ba14c6fde7`은 필드 의미를 확인하는 참고 자료다.
공식 ROS 메시지가 대회 빌드와 자동으로 **UDP-byte compatible**하다고 가정하지
않는다.

지원 수준은 다음 세 가지다.

- `typed`: 검증된 바이트 레이아웃을 ROS 메시지로 변환한다.
- `raw_only`: 레이아웃이 검증되지 않은 개발 센서를 원본 바이트로만 공개한다.
- `unsupported_build`: 공식 저장소에는 있으나 2025 대회 빌드 endpoint가 확인되지 않았다.

`raw` 열의 audit 토픽은 기능 목록일 뿐 기본 실행 토픽이 아니다.
competition/development profile 모두 `publish_raw_packets: false`이며, 이 값이
`true`일 때만 raw publisher와 raw 처리 경로가 생성된다.

MORAI Network Settings의 포트 쌍과 저장 순서는
[`network-persistence.md`](network-persistence.md)에 기록한다.

## Competition endpoints

| 방향 | endpoint | port | wire evidence | 지원 | raw (opt-in) | full | normalized |
|---|---|---:|---|---|---|---|---|
| 입력 | Competition Status | 1909 | `#MoraiInfo$`, 181/229 B | typed | `/ad/udp/raw/competition_status` | `/ad/vehicle/status` | 동일 |
| 입력 | CollisionData | 9092 | `#CollisionData$`, 181 B | typed | `/ad/udp/raw/collisions` | `/ad/safety/collisions` | 동일 |
| 입력 | Center camera (`camera_front`) | 9291 | `MOR`/`BOX` | typed | `/ad/udp/raw/camera_front` | `/ad/sensors/camera/front/compressed` | 동일 |
| 입력 | Left camera | 9293 | `MOR`/`BOX` | typed | `/ad/udp/raw/camera_left` | `/ad/sensors/camera/left/compressed` | 동일 |
| 입력 | Right camera | 9295 | `MOR`/`BOX` | typed | `/ad/udp/raw/camera_right` | `/ad/sensors/camera/right/compressed` | 동일 |
| 입력 | Traffic-light camera | 9307 | `MOR`/`BOX` | typed | `/ad/udp/raw/camera_traffic_light` | `/ad/sensors/camera/traffic_light/compressed` | 동일 |
| 입력 | GPS | 9297 | `$GPRMC`, `$GPGGA` | typed | `/ad/udp/raw/gps` | `/ad/sensors/gps/rmc`, `/ad/sensors/gps/gga` | `/ad/sensors/gps/fix` |
| 입력 | IMU | 9299 | `#IMUData$`, 107/115 B | typed | `/ad/udp/raw/imu` | `/ad/sensors/imu/full` | `/ad/sensors/imu/data` |
| 입력 | VLP-16 | 2368 | 1206 B Velodyne packet | typed | `/ad/udp/raw/velodyne` | `/ad/sensors/lidar/raw` | `/ad/sensors/lidar/points` (선택) |
| 출력 | CtrlCmd type 1 | 9093 | `#MoraiCtrlCmd$`, 55 B | typed | - | `/ad/control/command` | 동일 |

## Development endpoints

Development launch는 competition bridge도 포함한다. 카메라 BBox 변환은
`publish_raw_packets: true`와 `camera_bboxes.enabled: true`를 함께 지정했을 때만
competition raw 토픽의 `BOX` 패킷에서 만들며, 기본 profile에서는 실행되지 않는다.

| 방향 | endpoint | port | wire evidence | 지원 | raw (opt-in) | full/typed | normalized |
|---|---|---:|---|---|---|---|---|
| 입력 | Ego Status | 1911 | `#MoraiInfo$`, 181/229 B | typed | `/ad/dev/udp/raw/ego_status` | `/ad/dev/vehicle/ego_status/full` | `/ad/dev/vehicle/ego_status` |
| 입력 | ObjectInfo | 7505 | `#MoraiObjInfo$`, 2160 B | typed | `/ad/dev/udp/raw/objects` | `/ad/dev/objects` | 동일 |
| 입력 | 2D LiDAR | 9301 | `#Lidar2D$`, 1107 B | typed | `/ad/dev/udp/raw/lidar2d` | `/ad/dev/lidar2d/full` | `/ad/dev/lidar2d/scan` |
| 입력 | Traffic Light Status | 7502 | `#TrafficLight$`, 48 B | typed | `/ad/dev/udp/raw/traffic_light_status` | `/ad/dev/traffic_light/status` | 동일 |
| 입력 | Intersection Status | 9102 | `#IntStatus$`, 37/39 B | typed | `/ad/dev/udp/raw/intersection_status` | `/ad/dev/intersection/status` | 동일 |
| 입력 | NPC Collision | 9108 | `#VehicleCollision$`, 1156 B | typed | `/ad/dev/udp/raw/npc_collisions` | `/ad/dev/npc/collisions` | 동일 |
| 파생 | Center BBox | competition raw | `BOX` + 115 B records | typed | `/ad/udp/raw/camera_front` | `/ad/dev/camera/front/bboxes` | - |
| 파생 | Left BBox | competition raw | `BOX` + 115 B records | typed | `/ad/udp/raw/camera_left` | `/ad/dev/camera/left/bboxes` | - |
| 파생 | Right BBox | competition raw | `BOX` + 115 B records | typed | `/ad/udp/raw/camera_right` | `/ad/dev/camera/right/bboxes` | - |
| 파생 | Traffic-light BBox | competition raw | `BOX` + 115 B records | typed | `/ad/udp/raw/camera_traffic_light` | `/ad/dev/camera/traffic_light/bboxes` | - |

개발 출력은 모두 typed 토픽과 기존 JSON 호환 토픽을 함께 제공한다.

| command | port | typed topic |
|---|---:|---|
| Ego Ghost | 9095 | `/ad/dev/command/ego_ghost` |
| Traffic Light | 7607 | `/ad/dev/command/traffic_light` |
| Intersection | 9132 | `/ad/dev/command/intersection` |
| Sensor Position | 9103 | `/ad/dev/command/sensor_control` |
| Lamps | 9097 | `/ad/dev/command/lamp_control` |
| Scenario Load | 9099 | `/ad/dev/command/scenario_load` |
| Save Sensor Data | 9105 | `/ad/dev/command/save_sensor_data` |
| Multi Ego | 7604 | `/ad/dev/command/multi_ego` |
| NPC Ghost | 9101 | `/ad/dev/command/npc_ghost` |

`publish_raw_packets: true`일 때 `extra_raw_streams.names`로 추가하는 개발 입력은
`raw_only`이며
`/ad/dev/udp/raw/<name>`만 생성한다. 이름과 포트가 검증되지 않으면 socket을
열기 전에 시작을 거부한다.

## Official reference-only interfaces

다음 24.R2 이름은 현재 빌드에서 검증된 endpoint가 없으므로 모두
`unsupported_build`다: AutoDoorCtrlCmd, AutoDoorStatus, ElevatorCtrlCmd,
ERP42Info, MapSpec, PRCtrlCmd, PREvent, PRStatus, RadarDetections, RadarTracks,
SkidCtrlCmd, SkidCtrlReport, SpeedGateCtrlCmd, SpeedGateStatus, SyncModeAddObj,
SyncModeRemoveObj, VehicleSpec, WaitForTick, WheelRpm, WheelTorque.
