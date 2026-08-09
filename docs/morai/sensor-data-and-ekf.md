# MORAI GPS, IMU, Vehicle Status와 ESKF 입력 계약

기본 protocol 검증 기준일은 2026-07-23 KST이고, timestamp/localization 통합 계약은
2026-08-03 KST에 갱신했다. 대상 simulator는
`25.S4.MolitComp03`, 차량은 `2023_Hyundai_Ioniq5`이다. 이 문서는 MORAI
UDP에서 ROS 2 메시지를 거쳐 HEVEN localization 및 opt-in ESKF로 들어가는 값만
다룬다. 현재 기본 backend는 `gnss_imu`이고 ESKF는
`localization_backend:=eskf`를 명시해야 실행된다.

## 결론

- GPS 위치와 IMU 자세/각속도/가속도는 현재 ESKF 입력 계약과 호환된다.
- GGA의 `satellites`는 위치 해 계산에 **사용 중인 위성 수**이다. 전체 가시 위성
  수가 아니며, `/ad/sensors/gps/gga`에만 남고 localization에는 들어가지 않는다.
- RMC의 `speed_knots`는 위성과의 상대속도가 아니라 GPS 수신기, 즉 차량의
  **speed over ground**이다. 현재 `/ad/sensors/gps/rmc`에만 남으며 upstream의
  GNSS Doppler velocity 입력에는 연결되지 않는다. 현재 데이터 경로에는
  "relative satellite velocity"라는 필드가 없다.
- Competition Vehicle Status에서는 `velocity.x`와 `gear`만 휠 종방향 속도로
  변환되어 ESKF에 들어간다. 나머지 위치, 자세, 가속도, 조향, tire 값은 ESKF가
  사용하지 않는다.
- 2026-07-23 04:18 KST 관측에서 GPS fix는 약 18.9 Hz, IMU는 약 38.2 Hz였다.
  당시에는 변경 전 포트가 남아 Vehicle Status가 0 packet이었지만, 포트 수정 후
  `1909`에서 약 40 Hz 수신과 `/ad/vehicle/status` 발행을 확인했다.
- 과거 source-stamp repeat 억제 설정에서는 normalized IMU가 약 25.1 Hz가 되어
  upstream stationary initializer가 끝나지 않는 호환성 결함을 재현했다. 현재
  bridge는 repeat를 audit만 하고 모든 유효 sample을 발행한다. initializer의
  rate-independent production 계약은 그대로 `1.5 s / 25 samples`이며, 이는 정확한
  20 Hz에서 가능한 30개 중 5개 누락을 허용한다.

## 근거의 구분

이 문서에서 사실의 출처를 다음처럼 구분한다.

| 표기 | 의미 |
|---|---|
| **MORAI official** | MORAI SIM: Drive 24.R2 manual 또는 MORAI 공식 message repository |
| **NMEA reference** | GNSS 제조사 문서가 설명하는 NMEA 0183 field 의미 |
| **Repository contract** | 이 repository의 decoder, message conversion, config와 test로 확인한 구현 |
| **Capture inference** | 공식 문서에 없는 byte 순서/확장 형식을 packet capture와 decoder가 일관되게 설명하는 경우 |
| **Live observation** | 실행 중인 ROS graph와 bridge diagnostics를 read-only로 측정한 값 |
| **Upstream contract** | 고정된 ESKF source `fc1f4d39c942813ea83dc4f017eb0892756ea94d`의 동작 |

주요 외부 근거:

- [MORAI 24.R2 sensor communication protocol](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/-35)
- [MORAI 24.R2 sensor coordinate systems](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/-8)
- [MORAI 24.R2 UDP communication protocol](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/udp-1)
- [Official MORAI ROS messages, tag 24.r2](https://github.com/MORAI-Autonomous/MORAI-ROS_morai_msgs/tree/d1ce16301f1b50fccab17a1c31e2dbb016fe5e0d)
- [Official MORAI NetworkModule, revision 78e8855](https://github.com/MORAI-Autonomous/MORAI-NetworkModule/tree/78e88558588451bdf9a10baf04d575c9aa3e8587)
- [Official MORAI DriveExample UDP, revision 986f306](https://github.com/MORAI-Autonomous/MORAI-DriveExample_UDP/tree/986f3066a8b48c066d9fba60fddac2fb48e3e00e)
- [Official MORAI ROS2 example, revision 73ac9b6](https://github.com/MORAI-Autonomous/MORAI-Example-ROS2/tree/73ac9b66102f07c800445f17bb896f34fbbfdbe4)
- [NovAtel GPGGA field reference](https://docs.novatel.com/OEM7/Content/Logs/GPGGA.htm)
- [NovAtel GPRMC field reference](https://docs.novatel.com/oem7/Content/Logs/GPRMC.htm)
- [Pinned ESKF repository](https://github.com/rsasaki0109/kalman_filter_localization_ros2/tree/fc1f4d39c942813ea83dc4f017eb0892756ea94d)

MORAI의 공식 ROS message는 의미를 비교하는 보조 근거다. 예를 들어 24.r2의
[`GPSMessage.msg`](https://github.com/MORAI-Autonomous/MORAI-ROS_morai_msgs/blob/d1ce16301f1b50fccab17a1c31e2dbb016fe5e0d/msg/GPSMessage.msg)는
latitude/longitude/altitude와 east/north offset을 담고,
[`EgoVehicleStatus.msg`](https://github.com/MORAI-Autonomous/MORAI-ROS_morai_msgs/blob/d1ce16301f1b50fccab17a1c31e2dbb016fe5e0d/msg/EgoVehicleStatus.msg)는
ROS용 상태 구조를 정의한다. 둘 다 Competition UDP의 binary schema로 사용하면
안 된다.

## 전체 데이터 흐름

| UDP | Bridge ROS 출력 | Localization adapter | Pinned ESKF에서 실제 사용 |
|---|---|---|---|
| GPS NMEA RMC/GGA | `/ad/sensors/gps/rmc`, `/ad/sensors/gps/gga`, `/ad/sensors/gps/fix` (`NavSatFix`) | valid fix의 WGS84를 EPSG:32652로 투영하고 offset을 빼 `/ad/localization/input/gnss_pose` (`PoseStamped`, `odom`) 발행 | antenna position xyz, configured GNSS variance, body-frame lever arm |
| IMU 107/115-byte datagram | `/ad/sensors/imu/data` (`Imu`), `/ad/sensors/imu/full` (`ImuPacket`) | 첫 GNSS와 동기화된 IMU orientation으로 body initial pose 생성; ESKF 선택 시 frame/time을 검증해 `/ad/localization/input/eskf_imu`로 전달 | orientation, angular velocity, linear acceleration; `imu_link -> base_link` TF 적용 |
| Ego/Competition Vehicle Status 181/229-byte datagram | `/ad/vehicle/status` (`EgoVehicleStatus`) | `abs(velocity.x)`에 gear 방향을 적용하여 `/ad/localization/input/wheel_speed` 발행 | `twist.linear.x`만 사용; message covariance 대신 `var_wheel_speed` 사용 |

Adapter는 GNSS 안테나 위치에서 회전된 lever arm을 빼 첫 body pose를 만든다. 현재
mount 정본에서 계산된 ESKF lever arm은 `[0.0, 0.0, 1.5685] m`이다. 이후 GNSS
측정은 antenna position으로 보내며 ESKF가 같은 lever arm을 measurement model에서
처리한다. 선택된 backend는 private odometry만 발행하고 TF는 발행하지 않는다.
`localization_manager`만 canonical `/ad/localization/odometry`, 정적 identity
`map -> odom`, 동적 `odom -> base_link`를 발행한다.

### Timestamp 선택과 audit 계약

Bridge production 기본값은 `timestamp_mode: arrival`이다. 여기서 arrival은 publish
시각이 아니라 `recvfrom` 직후 캡처한 host monotonic receipt를 bridge 시작 시 얻은
ROS epoch에 매핑한 값이다. 따라서 decode/queue 지연은 header에 포함되지 않지만,
이 시각을 MORAI sensor acquisition time이라고 주장하지도 않는다. 공식 세 예제는
일부 packet의 `sec/nsec` 필드를 보여 줄 뿐 clock basis, 공통 clock, 센서 간 coherent
snapshot 또는 dynamic sensor ROS header 정책을 보장하지 않는다.
고정 monotonic-to-ROS mapping은 실행 중 `/clock` pause/reset을 따라갈 수 없으므로
competition/dev bridge는 `use_sim_time=true`를 fail-closed한다. Clock domain을
의도적으로 바꾸면 bridge와 downstream time state를 함께 재시작한다.

Source `(sec,nsec)`와 NMEA UTC는 full/audit metadata로 보존한다. 구조, arrival 대비
1초 창, stream 내 비역행 gate는 source plausibility 진단과 명시적 `TimeReference`에만
사용하며 normalized header로 승격하지 않는다. `source_preferred` mode는 과거 bag
재현용 opt-in 호환 경로일 뿐 MORAI production profile에서 사용하지 않는다.
`device_when_available`은 clock-domain 검증이 없으므로 계속 거부한다.

IMU, Vehicle Status, 완성된 Camera frame의 같은 accepted source stamp도 모든
normalized/full/audit 출력에 한 번씩 남는다. 공식 protocol에는 재전송 sequence나
discard 규칙이 없으므로 bridge가 이를 자동 제거하지 않는다. Camera receipt/audit
header는 JPEG 조립 완료 fragment가 아니라 해당 frame의 첫 fragment receipt다.
`/ad/sensors/timing` (`SensorTiming`)은 signed raw source pair, arrival header,
selected receipt stamp와 source 유효성/선택/거부, repeated-source-stamp 표시,
regression, publish 결과를
보존한다. `/ad/udp/statistics`도 stream별
`source_selected`, `arrival_fallback`, `source_rejected`, `duplicates`,
`stamp_regressions`를 decision event 기준으로 누적한다. 한 event가 selection과
duplicate에 동시에 집계될 수 있다. `duplicates`는 confirmed retransmission 수가
아니라 accepted source stamp repeat 수다. `arrival_fallback`은 production mode에서
receipt를 선택한 decision 수를 포함한다.

Development-only bridge도 같은 receipt 기본값과 audit gate를 재사용한다. Timestamp가
있는 development Ego Status, object array, camera bounding-box도 normalized header는
receipt이고 source repeat를 제거하지 않는다. Source timestamp가 없는 2D LiDAR,
traffic/intersection status, NPC collision도 receipt를 쓴다.
`/ad/dev/sensors/timing`은 signed raw source와 receipt/선택 결과를 보존하며, invalid
source를 `builtin_interfaces/Time`에 억지로 넣지 않는다.

## GPS UDP와 ROS 변환

### Wire 및 NMEA field

**MORAI official:** GPS UDP는 NMEA 0183을 따르며 RMC와 GGA sentence가 함께
들어온다. 위도/경도는 NMEA의 degrees/minutes를 decimal degrees로 변환한다.

| Sentence | Repository가 decode하는 field | 단위와 유효성 |
|---|---|---|
| GPRMC | UTC, A/V status, latitude, longitude, speed, track, date, magnetic variation/direction, mode, checksum | speed는 knot, track은 true course degree; A만 valid |
| GPGGA | UTC, latitude, longitude, fix quality, satellites, HDOP, altitude/unit, geoid separation/unit, differential age, station ID, checksum | altitude/geoid는 sentence의 unit, 현재 `M`; fix quality > 0만 ROS fix |

Checksum이 있으면 bridge가 XOR checksum을 검증한다. 잘못된 좌표, UTC grammar,
checksum, 비유한 값은 packet 단위로 버린다. GGA가 들어올 때 1.5초 이내의 valid
RMC가 있으면 내부 `GpsFixRecord`에는 m/s로 환산한 speed와 track도 만들어지지만,
`sensor_msgs/NavSatFix`에는 해당 필드가 없어서 `/ad/sensors/gps/fix`에는 실리지
않는다.

### Satellite count의 정확한 의미

GGA의 field 8(`# sats`)은 NMEA 의미상 **position solution에 사용 중인 위성
수**이다. 가시 위성 전체 수는 GSV field이며 MORAI bridge는 GSV를 받거나
publish하지 않는다. 현재 경로는 다음과 같다.

```text
GPGGA satellites -> ad_interfaces/GpsGga.satellites
                 -> /ad/sensors/gps/gga only
                 -X-> NavSatFix
                 -X-> localization adapter / ESKF
```

따라서 `satellites == 9`는 현재 해에 9개가 사용됐다는 관측값이지, ESKF가 9를
가중치로 사용한다는 뜻이 아니다. ESKF의 GNSS 신뢰도는 profile의
`var_gnss_xy/z = 0.25 m^2`로 고정된다.

### "Relative satellite velocity"가 아닌 것

RMC `speed_knots`는 receiver/vehicle의 ground speed magnitude이고
`track_degrees`는 이동 경로의 true course다. 개별 위성 line-of-sight range-rate,
Doppler shift, ENU velocity vector가 아니다. 정지 상태의 track은 위치 잡음으로
불안정할 수 있으므로 heading으로도 사용하면 안 된다.

Pinned upstream은 PR #20 이후 별도
`geometry_msgs/TwistWithCovarianceStamped` GNSS Doppler velocity를 지원하지만,
현재 HEVEN profile은 해당 기능을 enable하지 않고 bridge도 그런 topic을 만들지
않는다. RMC scalar speed/course를 Doppler vector로 가장해 연결하지 않는다.

### `NavSatFix`와 ESKF

- GGA fix quality가 0이면 `STATUS_NO_FIX`; adapter가 거부한다.
- RMC/GGA typed message는 checksum을 포함한 원래 NMEA sentence 문자열과 각 field를
  함께 보존한다. RMC date와 `HHMMSS`의 whole second만 UTC epoch로 만들어
  `/ad/sensors/gps/time_reference` (`TimeReference`)에 발행한다. MORAI의 가변 길이
  fractional text를 decimal nanosecond로 해석하지 않는다. RMC epoch도 공통 stateful
  source gate와 `SensorTiming` audit를 거쳐 regression이면 `TimeReference`를 발행하지
  않는다. Paired fix는 RMC date와 GGA whole second를 결합하고 자정 전후에는 날짜를
  명시적으로 보정한다. 그 full epoch가 공통 gate를 통과해도
  `NavSatFix.header.stamp`는 receipt를 유지한다. RMC의 `A/V`는 항법 해 유효성이므로
  `V`도 안전하게 parse된
  UTC/date의 `TimeReference`는 발행할 수 있지만, position/speed pairing에는 계속
  `A`만 쓴다. Timing audit label은 RMC relation이 `gps/rmc`, paired fix가
  `gps/fix`이고, 두 decision의 epoch가 같으면 GPS duplicate counter에 함께
  집계하되 GNSS 출력을 억제하지 않는다.
- HDOP가 있으면 bridge는 `xy=(HDOP*3 m)^2`, `z=4*xy`라는 repository heuristic로
  `NavSatFix.position_covariance`를 채운다.
- Adapter가 `PoseStamped`로 변환하므로 그 covariance는 사라진다. Pinned ESKF의
  PoseStamped 경로는 message covariance가 아니라 `var_gnss_xy/z`를 사용한다.
- MORAI official 좌표계는 WGS84 UTM 52N, EPSG:32652다. Adapter는 easting
  `302595.0 m`, northing `4124145.0 m`를 뺀 local `odom` position을 발행한다.

## IMU UDP와 ROS 변환

### 공식 107-byte packet

**MORAI official 24.R2.2:** total 107 bytes다.

| 구간 | 크기 | 의미 |
|---|---:|---|
| `#IMUData$` | 9 | header |
| data length | 4 | `80` |
| aux | 12 | 고정 0 |
| orientation | 32 | quaternion, 4 x float64 |
| angular velocity | 24 | xyz float64, rad/s |
| linear acceleration | 24 | xyz float64, m/s² |
| CRLF | 2 | tail |

MORAI 좌표축은 `x=forward`, `y=left`, `z=up`이며 양의 각속도는 각 축 기준
counter-clockwise다. 이는 ROS body convention과 맞는다.

### Repository/capture로 확인한 세부사항

- Decoder는 little-endian 10 x float64를 사용하고 wire quaternion을 `w,x,y,z`로
  읽어 ROS `x,y,z,w`로 재배열한다. 공식 HTML text는 component byte 순서를
  명시하지 않으므로 이 순서는 **capture inference**다.
- Decoder는 공식 107-byte packet 외에 8-byte `sec,nsec`가 IMU data 앞에 붙은
  115-byte packet도 허용한다. Live `/ad/sensors/imu/full.has_device_stamp == true`가
  이 확장을 확인했지만 24.R2.2 공식 text에는 115-byte variant가 없다.
- `/ad/sensors/imu/data`의 quaternion은 정규화된다. zero quaternion과 non-finite
  값은 packet 단위로 거부한다. 잘못된 source stamp만으로 payload를 버리지는 않고
  arrival header로 fallback한다.
- Normalized Imu header는 receipt다. `/ad/sensors/imu/full`은 모든 decoded sample의
  receipt header와 표현
  가능한 device stamp를 보존하고, `SensorTiming`은 구조적으로 잘못된 signed raw
  source pair까지 보존한다. 유효 source repeat도 normalized/full/audit에 모두 남는다.
- Bridge는 세 covariance 배열을 0으로 둔다. Pinned ESKF는 orientation covariance의
  세 대각값이 모두 양수일 때만 message 값을 사용하고, 0이면
  `var_imu_orientation_rpy=0.0001 rad²`로 fallback한다. Gyro/acceleration은 message
  covariance가 아니라 noisy SensorInfo의 per-square-root-hour 계수를 continuous
  density variance로 변환한 `var_imu_w=2.738777777777778e-9`와
  `var_imu_acc=4.669444444444444e-7`을 사용한다.

ESKF는 IMU TF를 조회해 vector와 orientation을 `imu_link`에서 `base_link`로
변환한다. 정지 초기화 시 acceleration norm이 gravity에 가까운지와 gyro/accel
표준편차를 검사하고 gyro bias 및 roll/pitch를 초기화한다.

## Competition Vehicle Status UDP와 ROS 변환

### 181-byte base packet

**MORAI official:** total 181 bytes이며 timestamp, control mode, gear, vehicle
state를 포함한다. Envelope는 11-byte header, 4-byte data length, 12-byte aux,
152-byte payload, 2-byte CRLF다. Repository는 `#MoraiInfo$`와 `#EgoStatus$`를 모두
허용한다.

아래 offset은 payload 시작을 0으로 둔 **repository binary layout**이다. Field와
단위는 공식 manual과 대조했다.

| Offset | Type | Field | UDP 단위 |
|---:|---|---|---|
| 0--3 | int32 | seconds | s |
| 4--7 | int32 | nanoseconds | ns |
| 8 | int8 | ctrl mode | 1 keyboard, 2 auto |
| 9 | int8 | gear | 0 M, 1 P, 2 R, 3 N, 4 D, 5 L |
| 10--13 | float32 | signed velocity | km/h |
| 14--17 | int32 | map data ID | - |
| 18--21 | float32 | accel pedal | 0--1 |
| 22--25 | float32 | brake pedal | 0--1 |
| 26--37 | 3 x float32 | vehicle size xyz | m |
| 38--49 | 3 x float32 | overhang, wheelbase, rear overhang | m |
| 50--61 | 3 x float32 | position xyz | m |
| 62--73 | 3 x float32 | roll, pitch, yaw | degree |
| 74--85 | 3 x float32 | velocity xyz | km/h |
| 86--97 | 3 x float32 | angular velocity xyz | degree/s |
| 98--109 | 3 x float32 | acceleration xyz | m/s² |
| 110--113 | float32 | steering | degree |
| 114--151 | char[38] | current link ID | UTF-8/C string |

Bridge는 signed/xyz velocity를 m/s로, RPY/angular velocity/steering을 radian 계열로
바꾼다. Position, dimensions, acceleration, pedals는 값과 SI 단위를 유지한다.
`/ad/vehicle/status`와 `/ad/vehicle/status/full` header는 모두 receipt를 사용하고
wire timestamp는 device/source metadata로 보존한다. Source stamp가 반복되어도
normalized status/wheel consumer까지 모든 유효 packet을 전달한다.

### 229-byte extension

Repository는 base payload 뒤의 12 x float32를 다음 순서로 허용한다.

1. tire lateral force FL, FR, RL, RR
2. side-slip angle FL, FR, RL, RR
3. tire cornering stiffness FL, FR, RL, RR

이 순서는 공식 ROS message의 field ordering 및 capture와 일치하지만, 24.R2 UDP
HTML text는 229-byte variant와 각 tire field의 단위를 명시하지 않는다. 따라서
**capture inference**이며 단위를 추정하지 않는다. ESKF는 이 값들을 전부 무시한다.

### ESKF로 들어가는 유일한 status 값

```text
EgoVehicleStatus.velocity.x [m/s]
  -> abs(value)
  -> gear 2: negative, gear 4/5: positive
  -> <= 0.05 m/s: zero regardless of gear
  -> moving in gear 0/1/3 or unknown: drop
  -> TwistWithCovarianceStamped.twist.linear.x
  -> ESKF body forward-speed observation
```

Adapter는 `signed_velocity`가 아니라 `velocity.x`를 쓴다. ESKF는 incoming wheel
covariance와 frame ID를 fusion variance로 사용하지 않고 profile의
`var_wheel_speed=0.04 (m/s)^2`를 사용한다. Wheel packet이 없어도 GNSS/IMU ESKF는
실행되지만 wheel/NHC 보정은 없다.

## Pinned upstream ESKF와 issue 조사

고정 버전은
[`fc1f4d39c942813ea83dc4f017eb0892756ea94d`](https://github.com/rsasaki0109/kalman_filter_localization_ros2/commit/fc1f4d39c942813ea83dc4f017eb0892756ea94d)이며
대규모 IMU timestamp gap에서 replay가 영구 정지하지 않도록
`patches/kalman-filter-localization-ros2/0001-large-imu-gap-recovery.patch`를 적용한다.
그 밖의 upstream 동작은 고정 revision을 유지한다.

- [README](https://github.com/rsasaki0109/kalman_filter_localization_ros2/blob/fc1f4d39c942813ea83dc4f017eb0892756ea94d/README.md)는 GNSS PoseStamped,
  Imu, optional wheel Twist 입력과 NavSatFix/Doppler/replay/lever-arm 기능을 설명한다.
- [Issue #18](https://github.com/rsasaki0109/kalman_filter_localization_ros2/issues/18)의
  첫 답변은 raw RTK NavSatFix를 local PoseStamped로 변환하라고 했다. 뒤의 답변은
  PR #19로 optional NavSatFix가 추가됐다고 정정한다. 현재 HEVEN은 의도적으로
  EPSG:32652 `PoseStamped` 경로를 사용하므로 양쪽 설명과 호환된다.
- [Issue #15](https://github.com/rsasaki0109/kalman_filter_localization_ros2/issues/15)는
  제목만 "Doppler Velocity"이고 본문/답변이 없다. 이것만으로 요구사항이나 bug를
  추론하지 않았다.
- [Issue #5](https://github.com/rsasaki0109/kalman_filter_localization_ros2/issues/5)는
  sample data만으로 정확도를 검증할 수 없고 truth가 있는 dataset이 필요하다고
  답한다. 현재 stationary live sample은 wiring sanity check이지 정확도 검증이 아니다.
- [PR #20](https://github.com/rsasaki0109/kalman_filter_localization_ros2/pull/20)은
  GNSS Doppler velocity와 평가기를 추가했다. HEVEN에는 이 velocity input이 없다.
- [PR #25](https://github.com/rsasaki0109/kalman_filter_localization_ros2/pull/25)는
  현재 pinned merge이며 stationary initialization, replay, wheel/NHC, evaluation을
  포함한다. PR의 upstream test 결과는 HEVEN sensor wiring의 검증을 대신하지 않는다.

## Live 관측: 2026-07-23

Read-only ROS 관측은 약 04:18 KST에 수행했다.

| 항목 | 관측 |
|---|---|
| Active localization | lifecycle `active`, backend `gnss_imu`, status topic `/ad/dev/vehicle/ego_status` |
| GPS UDP statistics | 38.32 packet/s; RMC와 GGA 합계, malformed/drop 0 |
| `/ad/sensors/gps/fix` | 약 18.86 Hz; lat `37.241605`, lon `126.7737867`, alt `30.7 m`, covariance diag `[9,9,36]` |
| `/ad/sensors/gps/gga` | fix quality 1, satellites 9, HDOP 1.0, altitude 30.7 M |
| `/ad/sensors/gps/rmc` | valid, speed 0 knot, track 179.3 degree; 정지 track은 heading 근거가 아님 |
| IMU UDP statistics | 39.68 packet/s cumulative; topic 측정 약 38.23 Hz; drop 0 |
| `/ad/sensors/imu/data` | quaternion norm 약 1, gyro 약 `[0.000623,0.000205,0.000091] rad/s`, acceleration 약 `[-0.0612,0.0406,9.8150] m/s²` |
| `/ad/sensors/imu/full` | `has_device_stamp=true`; arrival와 device stamp 차이 약 2.3 ms |
| IMU malformed | cumulative 413; log상 03:51:41부터의 zero-quaternion startup burst이고 이후 counter가 증가하지 않음 |
| Competition status | 04:18 포트 수정 전에는 packets 0; `1909` 적용 후 약 40 Hz와 `/ad/vehicle/status` sample 확인 |
| Competition status timestamps | CP14→CP15 bag 13,276개 중 wire/device stamp 연속 중복 1,890개, bridge arrival header 역행·중복 0개 |
| `/ad/localization/odometry` | finite output 확인; 현재 direct `gnss_imu` backend 결과이므로 ESKF live proof가 아님 |

Target profile은 GPS 20 Hz, IMU 50 Hz지만 ROS 도착 주기는 load와 scheduling으로
낮아질 수 있다. ESKF 초기화/timeout은 target maximum이 아니라 관측 가능한 실제
주기를 견뎌야 한다. Competition status의 wire/device stamp는 인접 packet에서
같을 수 있다. 위 표의 arrival header 관측은 2026-07-23 당시 구 profile 결과다.
현재 bridge는 source repeat를 audit/count만 하고 normalized wheel/status도
one-for-one으로 발행한다.

## 재현된 ESKF 결함과 최소 수정

Pinned stationary initializer는 `window_duration_sec`보다 오래된 sample을 제거한
뒤 다음 두 조건을 동시에 요구한다.

1. sample 수가 `minimum_samples` 이상
2. 첫 sample부터 마지막 sample까지의 span이 window의 95% 이상

초기 `1.0 s / 50 samples`는 38--40 Hz에서 실패했다. 이후 사용한 `1.5 s / 50
samples`도 당시 source timestamp 중복 억제 뒤의 normalized stream에서는 실패했다.
8초 wall-time probe는 167개 IMU를 받았고 선택된 stamp는 모두 증가했지만, 1.5초
sliding window의 sample 수는 median/min/max `39/37/41`이었다. 따라서 50개 하한은
현재 stream에서도 수학적으로 만족할 수 없다. Upstream은 initialization 전에도
IMU마다 현재 state odometry를 publish하므로, 단순히 odometry 존재 여부만 검사하면
이 결함을 놓친다. 초기 covariance `100`이 줄지 않고 propagation이 시작되는지를
함께 봐야 한다.

production profile은 window `1.5 s`를 유지하고 minimum을 `25`로 낮춘다. 시간
span 95% 조건은 그대로라서 표본 수만 모으면 즉시 초기화되는 설정이 아니다. 정확한
20 Hz clock에서는 1.45초에 30개가 가능하며, 25개 하한은 5개 누락 여유와 분산
계산의 24 자유도를 남긴다. 실측 1.5초 최소 37개에도 충분한 여유가 있다. `2.0 s /
30 samples`보다 초기화가 0.5초 빠른 대신 variance 표본은 적다는 선택이며, 플랫폼
고유 주파수를 하드코딩하지 않는다.

회귀 테스트는 다음 두 층으로 고정했다.

- core characterization: 20 Hz에서 기존 minimum 50은 2초 뒤에도 `collecting`이고,
  minimum 25는 예정된 5개 sample을 누락해도 1.45초에 초기화된다.
- actual ROS component: 정확히 20 Hz인 합성 raw 입력으로 production launch를
  실행해 stationary initialization, GNSS blackout 중 IMU propagation 및 GNSS
  recovery를 모두 확인한다.

동시에 integration launch에 `localization_backend:=eskf`를 명시했다. 그렇지 않으면
테스트 이름과 달리 기본 direct backend를 실행한다. Bridge, simulator, upstream
source는 수정하지 않았다.

## 남은 제한과 현장 검증 절차

1. MORAI Network 설정에서 Competition Vehicle Status 송신을 켜고 목적지가
   `127.0.0.1:1909`인지 확인한다. `1911`은 development Ego Status 전용이다.
2. `ros2 topic echo /ad/vehicle/status --once`와
   `ros2 topic hz /ad/vehicle/status`로 실제 181/229-byte variant, 값, rate를 확인한다.
3. D/R 전환과 0.05 m/s 경계에서
   `/ad/localization/input/wheel_speed`의 부호와 drop 진단을 확인한다.
4. ESKF live 검증 때는 현재 direct stack과 분리된 ROS domain에서
   `localization_backend:=eskf`를 명시한다.
5. 수정 후 live 정지 run은 35 samples로 초기화되고 Z 표준편차 `0.00582 m`를
   보였지만, 직진/좌우 회전/후진, 급가속/급제동, GNSS 단절과 복귀는 truth가 있는
   bag으로 별도 평가한다. 정지 noise 억제를 dynamic 정확도로 일반화하지 않는다.

Opt-in 회귀 테스트 명령:

```bash
cd ~/heven_ad_2026_ws
source install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 AD_RUN_ROS_INTEGRATION=1 \
  python3 -m pytest -q \
  ad_localization/test/test_eskf_integration.py \
  ad_localization/test/test_eskf_profile.py
```

## MORAI IMU rate provenance와 2026-08-04 passive baseline

MORAI의 sensor JSON은 operator가 full-stop/load/restart 절차에서 선택하는
provenance이며, 이 build에서 활성 profile/rate를 ROS 또는 evaluator가 readback할
근거는 없다. 따라서 `ad_morai_imu_timing_eval --sensor-profile`의 path/hash/period는
측정 당시 읽은 파일의 기록일 뿐 active-runtime claim이 아니다. 실행 중 JSON을
hot-edit하거나 evaluator로 rate/control/gRPC/scenario/sensor-control을 보내지 않는다.

읽기 전용 ID 7 source(`sensorPeriod=0.019999999552965164`, SHA-256
`df5b9771b8ec39f325a95cc2146b3b5168e4215536f059290ee48ab3461bb71a`)에서 immutable
20/30/50 clones를 만들었다. 상세 manifest와 12초 50 Hz-target passive report는
`$AD_DATA_DIR/experiments/morai_imu_timing/` 아래에 있다. 이 run은 arrival timestamp
bridge에서 normalized/full/timing 419개가 exact parity이고 harmful counter delta가
0이라 transport exit 0이었지만, receipt rate 34.9167 Hz, source repeat 104/419,
burst 106, `>2T` gap 204로 advisory 경고가 발생했다. 즉 packet contract는 지켰지만
50 Hz label이 uniform arrival timing을 보장하지는 않는다. 20 Hz baseline을 기준으로
50 -> 30 -> 20, 이후 20 -> 30 -> 50 순서의 full-stop A/B 반복 report를 비교한다.
