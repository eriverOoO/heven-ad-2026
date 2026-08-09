# ad_localization

대회 허용 입력인 GNSS, IMU, Competition Vehicle Status를 공통 형식으로 바꾸고
선택된 backend의 private odometry를 `localization_manager`가 canonical
odometry와 TF로 승격한다. 기본 backend인 `gnss_imu`는 GNSS 위치와 IMU
orientation을 직접 사용한다. `imu_quaternion_encoder`는 MORAI Competition
Vehicle Status와 IMU quaternion을 쓰는 simulator 전용 비교 backend다.
`quaternion_wheel_gnss_ekf`는 quaternion 자세, wheel speed, 저이득 GNSS XY
보정을 쓰는 3-state 비교 backend다. `hybrid`는
CP14→CP15에서 저장 PCD 기반 FastLIO를 사용하고 그 전후에는 GNSS+IMU를 사용한다.
ESKF는 `localization_backend:=eskf`로 명시했을 때만 실행한다. ESKF 자체는
`rsasaki0109/kalman_filter_localization_ros2`의 고정 커밋에 이 repository가
소유한 대규모 IMU gap 복구 overlay를 적용해 사용한다.
이 패키지는 입력 변환, backend 선택, 초기화, 진단과 프로젝트 출력 계약을
담당한다.

## 설치와 빌드

Patchwork++와 ESKF는 `dependencies.repos`에 고정된 workspace source
repository로 가져온다. workspace 루트에서 실행한다.

```bash
vcs import src --input src/heven_ad_2026/dependencies.repos --skip-existing
src/heven_ad_2026/scripts/apply_dependency_patches.sh src
git -C src/patchwork-plusplus rev-parse HEAD
git -C src/kalman-filter-localization-ros2 rev-parse HEAD
```

Patchwork++ SHA는 `3e6903a1d5537a4cc2ace897b0bbb98a92d6014c`, Kalman SHA는
`fc1f4d39c942813ea83dc4f017eb0892756ea94d`여야 한다. Kalman 코드는 upstream
BSD license를 유지하며, `patches/kalman-filter-localization-ros2/`의 고정 overlay만
fail-closed script로 적용한다.

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
colcon build --packages-up-to ad_localization ad_description --symlink-install
source install/setup.bash
```

단독 실행은 다음과 같다.

```bash
ros2 launch ad_localization localization.launch.py autostart:=true

# ESKF는 별도 검증 시에만 opt-in
ros2 launch ad_localization localization.launch.py \
  localization_backend:=eskf autostart:=true

# Competition Vehicle Status의 절대 pose를 사용하는 MORAI 전용 mode
ros2 launch ad_localization localization.launch.py \
  localization_backend:=imu_quaternion_encoder \
  imu_quaternion_encoder_mode:=status_pose autostart:=true

# 마지막 GNSS seed + IMU quaternion yaw + signed vehicle speed 적분 mode
ros2 launch ad_localization localization.launch.py \
  localization_backend:=imu_quaternion_encoder \
  imu_quaternion_encoder_mode:=dead_reckoning autostart:=true

# Quaternion orientation + wheel prediction + gated GNSS XY update
ros2 launch ad_localization localization.launch.py \
  localization_backend:=quaternion_wheel_gnss_ekf autostart:=true

# CP14→CP15 fixed-map handoff를 포함한 전 구간용 profile
ros2 launch ad_localization hybrid_localization.launch.py
```

기본 PCD는 패키지의 `maps/cp14_to_cp15.pcd`를 사용한다.

전체 스택에서는 `ad_bringup/bringup.launch.py`가 이 launch를 include한다.

공통 adapter, 각 backend와 canonical manager 구현은 같은 역할의 header와 source를
함께 찾을 수 있도록 아래 구조로 나눈다. `gnss_shadow`는 별도 hybrid graph로 기존
구조를 유지한다.

```text
include/ad_localization/       src/
  adapter/                       adapter/
  gnss_imu/                      gnss_imu/
  imu_quaternion_encoder/        imu_quaternion_encoder/
  quaternion_wheel_gnss_ekf/     quaternion_wheel_gnss_ekf/
  manager/                       manager/
  gnss_shadow/                   gnss_shadow/
```

일반 launch에서 estimator는 서로 배타적으로 하나만 실행되고 다음 private topic 중
하나만 발행한다.

```text
/ad/localization/backends/gnss_imu/odometry ───────────────┐
/ad/localization/backends/eskf/odometry ───────────────────┼─> localization_manager
/ad/localization/backends/imu_quaternion_encoder/odometry ─┤       ├─> /ad/localization/odometry
/ad/localization/backends/quaternion_wheel_gnss_ekf/odometry ─────┘       ├─> map -> odom
                                                                         └─> odom -> base_link
```

각 estimator의 TF 발행은 강제로 끄며 manager 하나만 canonical odometry와 TF를
소유한다. manager는 frame mismatch, 비유한 상태, 비단위 quaternion, 중복·역행
timestamp를 거부한다. timestamp는 `sec >= 0`, `nanosec < 1e9`인 ROS 구조만
허용하고 pose/twist covariance 72개 원소도 모두 유한해야 한다. backend 교체는
`localization_backend` 한 인자로 수행한다.

`gnss_shadow`의 GNSS↔FastLIO 전환 core와 handoff node는 아직 별도 hybrid graph다.
FastLIO 전환 정책을 확정할 때 manager의 upstream selector로 합치는 작업은 이번
backend source 정리 범위에 포함하지 않는다.

## 좌표와 TF 계약

- GPS WGS84는 EPSG:32652로 변환한 뒤 easting `302595.0 m`, northing
  `4124145.0 m`를 뺀다. 고도 offset 기본값은 `0.0 m`이다.
- 장착 위치의 단일 기준은
  `ad_description/config/sensor_mounts.yaml`이다. launch가 선택된 profile의 GPS
  위치와 IMU 회전을 adapter 및 선택된 backend에 전달해 URDF와 추정기가
  일치한다. 현재 확정 profile은 `(0.0, 0.0, 1.2) m`, 향후 후축 profile은
  `(0.0, 0.0, 0.7) m`이다.
- 일반 `localization.launch.py`에서는 adapter와 선택된 backend의 TF를 모두 끄고,
  `localization_manager`만 정적 `map -> odom` identity와 동적
  `odom -> base_link`를 발행한다.
- `hybrid_localization.launch.py`에서는 두 backend의 TF를 끄고 handoff node만
  동적 `odom -> base_link`와 `/ad/localization/odometry`를 발행한다.

주요 출력은 `/ad/localization/odometry`, `/ad/localization/pose2d`,
`/diagnostics`이다.

## 휠 속도와 초기화

`EgoVehicleStatus.velocity.x`는 m/s로 그대로 사용하며 절댓값에 기어 방향을
적용한다.

- gear `2`: 후진, 음수
- gear `4` 또는 `5`: 전진, 양수
- `0.05 m/s` 이하: 정지, 0
- 그 외 기어에서 이동 중인 값: 방향이 불명확하므로 폐기

기본 `gnss_imu` backend는 GNSS timestamp에 맞는 IMU orientation이 있으면 GNSS
주기로 pose/TF를 발행한다. 동기화된 휠 속도가 있으면 odometry twist를 채우지만,
없거나 오래됐다고 pose를 막지는 않는다. GNSS가 끊기면 dead reckoning하지 않고
새 출력을 멈춘다.

`hybrid` profile은 CP14 20 m 전에서 FastLIO initial pose를 한 번 발행해
prewarm하고, CP14 8 m 이내에서 두 backend의 위치/yaw가 각각 `2.0 m`,
`0.20 rad` 이내로 일치할 때 FastLIO로 전환한다. CP15 20 m 이내에서는 GNSS
복귀를 준비해 같은 조건을 만족하면 GNSS+IMU로 돌아간다. 전환 때의 위치/yaw
보정은 2초 동안 감쇠하며, CP15 이후에는 FastLIO로 재진입하지 않는다. 세부값은
`config/hybrid.yaml`에서 관리한다.

ESKF backend는 첫 유효 GNSS 위치와 IMU orientation으로 한 번만 초기화하고 이후
IMU와 휠 속도로 propagation한다. 이 동작과 ESKF 튜닝은 현재 기본 주행 경로가
아니며 별도 검증 대상으로 남겨 둔다.

`imu_quaternion_encoder`의 `status_pose` mode는 Competition Vehicle Status가 주는
map-frame `position+rpy`와 `signed_velocity`를 직접 odometry로 만든다. 센서 융합
정확도를 평가하는 입력이 아니며 MORAI 외 플랫폼에는 사용할 수 없다. 현재 live
Competition stream은 position을 항상 정확히 `(0, 0, 0)`으로 채우므로 production
profile의 `reject_zero_status_position: true`가 이를 invalid sentinel로 보고 출력을
거부한다. 실제 map 원점이 유효한 플랫폼만 이 guard를 명시적으로 끌 수 있다.
`dead_reckoning` mode가 받는 GNSS seed는 GPS 안테나 위치다. seed 시각보다 미래가
아닌 가장 가까운 IMU quaternion에 IMU mount 역회전과 K-City UTM 52N
grid-convergence yaw를 적용한 뒤, 회전된 3D `base_link -> gps_link` lever arm을
안테나 위치에서 빼서 base 위치를 만든다. 이후 timestamp가 증가하는 status
sample마다 같은 인과적 IMU yaw와 signed speed를 2D 적분하고 보정된 seed z를
유지한다. 이미 출력한 status 시각 이하의 늦은 seed는 좌표 rewind를 막기 위해
거부한다. status 간격이 `maximum_integration_dt_sec`를 넘으면 누락 거리를 0으로
취급하지 않고 DR 상태를 폐기하며, gap 이후 시각의 새 GNSS seed가 올 때까지
출력을 재개하지 않는다. 두 mode 모두 private odometry만 발행하며 canonical
output/TF는 manager가 담당한다. `dead_reckoning`은 status position을 사용하지
않으므로 zero-position guard와 무관하며, GNSS seed와 causal IMU가 유효하면 계속
사용할 수 있다.

`quaternion_wheel_gnss_ekf` 상태는 `[x, y, wheel_bias_mps]`다. IMU에서는
normalize 및 mount/grid 보정한 quaternion만 사용하고 acceleration과 angular rate는
추정이나 출력에 쓰지 않는다. GNSS antenna pose는 표본 시각의 causal corrected
quaternion으로 lever arm을 제거한 뒤 초기 평균과 2D Mahalanobis/Joseph update에
사용한다. Z는 상태와 update에서 제외하고 odom frame의 `0.0 m`로 고정하며 큰
unobserved covariance를 발행한다. 휠은 작지만 0이 아닌 covariance로 propagation하고,
GNSS는 의도적으로 큰 `9.0 m²` 분산을 사용한다. 큰 GNSS innovation 한 번은 버리며,
예측 위치에서 멀고 서로 일관된 세 표본만 checkpoint teleport로 확인해 XY, bias와
covariance를 새 seed로 초기화한다. clock regression과 과도한 wheel gap은 causal
history와 초기화를 다시 시작한다. 이 MORAI profile은 실차 covariance calibration을
대체하지 않는다.

ESKF를 선택한 경우에만 adapter가 raw IMU를
`/ad/localization/input/eskf_imu`로 다시 발행한다. orientation에는 K-City의 UTM
52N grid-convergence yaw `-0.02350724531030645 rad`를 world-frame에서 한 번
left-multiply하고, header와 angular velocity, linear acceleration, 세 covariance
배열은 그대로 보존한다. 같은 보정 orientation으로 ESKF initial pose를 만든다.
기본 `gnss_imu` backend는 계속 raw IMU를 구독하고 자체 `world_yaw_offset_rad`를
한 번만 적용한다.

내부 topic은 launch argument `eskf_imu_topic`으로 바꿀 수 있다. launch는 같은
값을 adapter의 publisher와 upstream ESKF subscriber 양쪽에 전달한다. ESKF는 이
단일 보정 경로를 보장하기 위해 `initial_orientation_source=imu`와
`initial_orientation_yaw_offset_rad=0.0`만 허용한다.

## ESKF 입력 스케줄링과 executor

아래 내용은 opt-in ESKF backend에만 해당한다. ESKF는 하나의 상태와 replay
history를 순차 갱신하므로 single-threaded executor를
유지한다. 기본 callback group은 mutually exclusive라
`MultiThreadedExecutor`로만 교체해도 병렬화되지 않으며, callback group을
reentrant로 바꾸면 필터 상태와 입력 순서에 별도 동기화가 필요하다.

대신 competition profile은 `input_qos_depth: 100`으로 짧은 executor backlog의
IMU를 DDS history에 보존한다. `input_reorder_window_sec`는 `0.0`으로 두어 별도
50 ms reorder queue가 늦게 전달된 IMU를 폐기하지 않게 하고, GNSS와 휠의 지연
측정 처리는 timestamp replay 엔진 하나가 담당한다. 전체 planning stack에서 정상
MORAI 입력도 드물게 100 ms를 조금 넘는 간격을 만들 수 있으므로
`max_imu_dt_sec: 0.5`를 사용한다. 이는 짧은 scheduling jitter는 허용하면서 0.5초를
넘는 실제 IMU 단절은 계속 검출하는 값이다.

정지 초기화는 upstream이 `stationary_initialization_window_sec`의 95% time span과
`stationary_initialization_min_samples`를 동시에 요구한다. Production은 `1.5 s /
25 samples`를 사용한다. 20 Hz에서는 1.45초에 30개가 가능하므로 5개 누락을
허용하고, sample 수만 채운 짧은 burst는 time-span gate가 계속 거부한다. 이전
`1.5 s / 50 samples`는 source duplicate 억제 뒤 실측 1.5초 최대 41개보다 커서
초기화가 불가능했다. 20 Hz integration fixture는 실제 ESKF component의 초기화,
GNSS blackout propagation, GNSS recovery를 검증한다.

MORAI bridge는 센서 포트별 수신 thread를 이미 사용한다. 카메라/LiDAR와
status/GPS/IMU의 프로세스 분리는 navigation stream의 bridge 처리 지연이나 queue
drop이 측정될 때 적용한다. 현재 loopback 실측에서는 bridge 처리 지연이 2 ms
미만이고 drop은 없으므로 ESKF 입력 보존을 우선한다.

## MORAI 측정 신뢰도

현재 MolitComp03에서 검증한 noisy sensor profile은 다음 파일이다.

```text
/home/heven/MoraiLauncher_Lin/MoraiLauncher_Lin_Data/SaveFile/Sensor/
25.S4.MolitComp03/noise_SensorInfo_2023_Hyundai_Ioniq5.json
```

2026-07-23에 확정한 이 파일은 GPS를 20 Hz, IMU를 50 Hz로 설정한다. GPS에는
축별 Gaussian `0.5 m` 표준편차가 적용되고, IMU에는 가속도 축별
`0.041 / sqrt(h)`, 각속도 roll/pitch/yaw 각각
`0.003/0.003/0.00314 / sqrt(h)`의 white-noise 계수가 적용된다. IMU random walk와
bias instability는 꺼져 있다. `var_*`의 단위는 표준편차가 아니라 분산이다.

| 파라미터 | 분산 | 표준편차 | 근거 |
|---|---:|---:|---|
| `var_gnss_xy` | `0.25` | `0.50 m` | simulator-profile 가정; noisy-profile rosbag으로 재검증 필요 |
| `var_gnss_z` | `0.25` | `0.50 m` | simulator-profile 가정; noisy-profile rosbag으로 재검증 필요 |
| `var_imu_w` | `2.738777777777778e-9` | `0.00314 / sqrt(h)` | `(0.00314 / 60)^2` continuous density |
| `var_imu_acc` | `4.669444444444444e-7` | `0.041 / sqrt(h)` | `(0.041 / 60)^2` continuous density |
| `var_imu_gyro_bias` | `0.0` | - | profile의 random walk/bias instability 비활성화 |
| `var_imu_acc_bias` | `0.0` | - | profile의 random walk/bias instability 비활성화 |
| `var_imu_orientation_rpy` | `0.0001` | `0.01 rad` | simulator orientation 실측보다 여유 있는 하한 |
| `var_wheel_speed` | `0.04` | `0.20 m/s` | 더 낮은 후보가 위치 RMSE를 악화시켜 기존값 유지 |

이 값들이 실제 융합 신뢰도다. GNSS는 adapter에서 `PoseStamped`로 변환되어
`NavSatFix` covariance가 전달되지 않고, MORAI IMU covariance 배열은 0이며,
upstream wheel update도 메시지 covariance 대신 `var_wheel_speed`를 사용한다.
`var_imu_w`와 `var_imu_acc`는 propagation process noise이며, `fast` propagation의
continuous-density 모드가 사용한다. MORAI의 per-square-root-hour 계수를 60으로
나눈 뒤 제곱한다. IMU orientation은 별도 noise가 적용되지 않으므로 기존의
보수적인 `0.0001 rad^2` 하한을 유지한다. GPS UI noise label만으로는 covariance
계약이 충분히 정밀하지 않으므로 GNSS `0.25 m^2` 값은 provisional assumption이다.

upstream overlay는 평균 specific force와 표준 중력 `9.80665 m/s^2`의 차이에서
한 자세로 관측 가능한 중력 방향 가속도계 bias를 초기화하는 옵션을 제공한다.
수평 bias와 roll/pitch 오차는 한 자세만으로 분리할 수 없으므로 이 기능은 upstream
기본값과 HEVEN production profile 모두 `false`다.
`initial_imu_acc_bias_covariance: 0.01`은 `(m/s^2)^2` 단위의 초기 상태 분산이며
1-sigma `0.1 m/s^2`를 뜻한다. GNSS와 wheel NHC update가 이 초기 불확실성의
cross-covariance를 통해 session bias를 추정한다. 이는 MORAI 중력에 맞춘 상수가
아니며, `var_imu_acc_bias: 0.0`인 현재 profile에 연속 random walk를 추가하지 않는다.

2026-08-02의 45초 same-topic simultaneous 2x2 정지 A/B에서 baseline / covariance-only /
radial-initializer-only / combined 후보의 body-z velocity RMSE는 각각
`0.04002 / 0.00418 / 0.02192 / 0.00418 m/s`였다. covariance-only와 combined가
동일했고 initializer-only는 정지 3초 뒤 `0.02297 m/s`로 기준 `0.02 m/s`를
넘었으므로 더 단순한 covariance-only 구성을 provisional production 후보로
선택했다. 이어서 최종 covariance-only 후보와 baseline을 같은 ROS 입력 topic에
동시에 연결한
저속 closed-loop pulse를 세 번 반복했다. 최고속도는 `0.1403~0.1550 m/s`, 실행당
누적 이동은 `0.0499~0.0537 m`였고, body-z velocity RMSE 평균은
`0.01156 -> 0.00425 m/s`로 `63.3%` 감소했다. 세 실행 모두 제어 모드 복원과
정지 cleanup을 검증했다. `initialization` phase 행을 제외한 z position RMSE 평균은
`0.06223 -> 0.06264 m`로 사실상 동일했고 실행별 방향도 섞여 있었으므로 위치
개선은 주장하지 않는다. 이 재집계와 달리 당시 생성된 각 `summary.json`은
`initialization` 행을 포함하므로 숫자가 직접 일치하지 않는다. 같은 집계에서
body-x velocity RMSE는 약 `3.3%`, body-y는 약 `13.9%` 증가했지만 body-y 절대
증가는 `0.00123 m/s`였다. 이 작은 수평 비용도 장시간 실차 검증 항목으로 남긴다.

세 실행에서 두 후보의 output header sequence는 일치했지만 raw IMU/GNSS/wheel
payload와 후보별 입력 수신 counter는 저장하지 않았다. 따라서 이는 same-topic
simultaneous A/B이지 bit-identical delivery 증명은 아니다. 이후 실행의 evaluator는
공통 output header의 두 후보를 반드시 같은 truth sample에 매칭한다. 더 강한
동일입력 증명은 input digest 또는 동일 rosbag replay로 수행한다.

안전 deadline과 공통-truth 매칭을 적용한 뒤 실행한 별도 검증 run에서도 최고속도
`0.1517 m/s`, 전체 truth 누적 이동 `0.0510 m` 안에서 body-z velocity RMSE가
`0.01293 -> 0.00399 m/s`로 `69.1%` 감소했다. keyboard -> AUTO -> keyboard
복원, pre-waveform/cleanup stop, post-restore stop, 외부 충돌 없음이 모두 검증됐고
두 후보의 공통 header `383`개만 같은 truth에 대칭적으로 평가했다. 이 한 번의
추가 run은 구현 안전성과 기존 결론을 재확인하지만 실차 production sign-off를
추가로 의미하지는 않는다.

여전히 가속도계 bias covariance와 estimator rejection, large-gap,
numerical-failure, output-continuity counter는 artifact에 없고, 주행 후 3초 지표를
만들 만큼 긴 정지 구간도 없었다. 따라서 covariance consistency와 3분 mixed
dynamic acceptance는 아직 검증되지 않았으며 covariance-only는 provisional
production 후보로 유지한다. 특히 `0.01` covariance와 `var_imu_acc_bias: 0.0`은
MORAI 전용 상수는 아니지만 실차 확정값도 아니므로 IMU Allan variance, 온도,
수신기 covariance를 사용해 플랫폼별로 다시 검증한다.

실차 포팅 전에는 GNSS 계약도 바꿔야 한다. 현재 `NavSatFix`를 UTM
`PoseStamped`로 바꾸면서 sample covariance가 소실되므로 `var_gnss_*`는 고정
fallback일 뿐이다. 실차 경로는 UTM 축으로 회전한 receiver covariance를 함께
전달하고, NMEA GGA의 MSL 고도와 `NavSatFix`의 ellipsoid 고도 datum을 명시적으로
일치시켜야 한다. 현재 MORAI HDOP 기반 covariance나 `0.25 m^2`를 실차 기본값으로
복사하면 안 된다.

튜닝 평가에서 `/ad/dev/vehicle/ego_status`는 ground truth 비교에만 사용했으며
localization 입력에는 연결하지 않았다. 이 값은 근정지 MORAI 결과이므로 주행,
회전, 후진, GNSS 단절 구간은 별도 rosbag으로 다시 검증해야 한다.

## 검증과 튜닝

일반 테스트는 기본 `gnss_imu` core와 두 backend의 launch 상호배제를 검증한다.
실시간 ESKF 통합 테스트는 명시적으로 skip되며, ESKF를 다시 다룰 때만 다음처럼
별도로 실행한다.

```bash
AD_RUN_ROS_INTEGRATION=1 colcon test --packages-select ad_localization \
  --ctest-args -R eskf_integration --output-on-failure
```

`config/eskf.yaml`의 measurement variance는 근정지 MORAI에서 사용해 본
provisional 값이며 실차 calibration은 아니다. MORAI에서 직진, 좌우 회전, 정지,
후진 전환, GPS 중단과 복귀를 rosbag으로 기록한 뒤 process noise와
non-holonomic constraint를 별도 변경으로 튜닝한다.
