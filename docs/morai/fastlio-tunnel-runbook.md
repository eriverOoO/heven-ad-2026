# Fast-LIO2 CP14→CP15 터널 운용

이 문서는 K-City CP14→CP15 GNSS 음영 구간의 mapping, 저장 PCD localization,
GNSS+IMU/Fast-LIO2 hybrid 전환을 재현하는 절차다. 주행 입력은 competition UDP의
LiDAR, IMU, GPS, Vehicle Status만 사용한다. MORAI 25.S4에서는 한 scan의 point가
동일 pose에서 만들어지므로 Fast-LIO2와 전처리의 motion deskew를 켜지 않는다.

## 준비 자산

필요한 경로와 시나리오는 저장소에 포함되어 있다.

- `ad_data/paths/cp14_to_cp15.txt`: CP14→CP15 터널 경로
- `ad_data/morai/SaveFile/Scenario/R_KR_PR_K-city_2025/checkpoint14_premapping.json`:
  움직이는 actor를 제거한 mapping 시나리오
- `ad_localization/maps/cp14_to_cp15.pcd`: fixed-map localization용 PCD

workspace에서 자산과 LFS payload를 먼저 검증한다.

```bash
cd ~/heven_ad_2026_ws
git -C src/heven_ad_2026 lfs pull
src/heven_ad_2026/.venv/bin/python \
  src/heven_ad_2026/scripts/verify_ad_data.py \
  --root src/heven_ad_2026

export AD_DATA_DIR="$PWD/src/heven_ad_2026/ad_data"
test -s "$AD_DATA_DIR/paths/cp14_to_cp15.txt"
test -s src/heven_ad_2026/ad_localization/maps/cp14_to_cp15.pcd
```

## MORAI 시나리오와 네트워크

MORAI의 시나리오 load는 network connection을 끊을 수 있으므로 gRPC로 자동 load하지
않는다. UI에서 `checkpoint14_premapping.json`을 불러온 뒤 Network Settings의
**Load**로 `NetworkInfo_2023_Hyundai_Ioniq5.json`을 다시 적용한다. 저장소의
`NetworkInfo_2023_Hyundai_Ioniq5.template.json`은 포트 구성을 공유하기 위한
loopback 템플릿이며 실행 중인 simulator 파일을 자동 덮어쓰지 않는다.

확인 항목은 다음과 같다.

- Cmd Control UDP host port: `9093`
- Competition Vehicle Status: `1908 -> 1909`
- `/ad/vehicle/status`가 갱신되고 gear와 signed velocity가 실제 차량과 일치
- GPS, IMU, VLP-16 토픽의 publisher가 각각 하나

포트가 다시 이전 값으로 돌아가는 경우
[`network-persistence.md`](network-persistence.md)의 순서를 따른다.

## Control-off 사전 점검

mapping과 localization은 동시에 실행하지 않는다. 먼저 control을 끈 상태로 각
구성을 확인한다.

```bash
cd ~/heven_ad_2026_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export AD_DATA_DIR="$PWD/src/heven_ad_2026/ad_data"

# 새 PCD를 만들 때
ros2 launch ad_bringup tunnel_mapping.launch.py control_enabled:=false

# 저장 PCD를 사용할 때
ros2 launch ad_bringup tunnel_localization.launch.py control_enabled:=false
```

다음 조건을 모두 확인한다.

- LiDAR, IMU, Vehicle Status가 지속 발행됨
- Fast-LIO diagnostics가 healthy이고 effective point가 0이 아님
- `map -> odom -> base_link`의 동적 TF authority가 하나뿐임
- mapping mode에만 `/ad/localization/fastlio/save_map` service가 존재함
- localization mode에서 PCD가 비어 있거나 Git LFS pointer인 경우 시작이 거부됨

## Mapping 주행

시나리오에서 Ego가 CP14에 정지한 상태로 launch를 다시 시작한다. 센서와 초기화가
정상임을 확인한 후에만 control을 켠다.

```bash
ros2 launch ad_bringup tunnel_mapping.launch.py control_enabled:=true
```

MORAI UI에서 pause를 해제한다. CP15에서 Profile Stanley의 terminal full-brake와
차량 정지를 확인한 다음 map을 저장한다.

```bash
ros2 service call \
  /ad/localization/fastlio/save_map \
  std_srvs/srv/Trigger '{}'

MAP_PATH="$(ros2 pkg prefix ad_localization)/share/ad_localization/maps/cp14_to_cp15.pcd"
test -s "$MAP_PATH"
sha256sum "$MAP_PATH"
```

검증된 PCD를 source tree에 반영할 때는 실제 point cloud인지 확인하고 Git LFS로
추적한다. bag, PCD snapshot과 실행 로그는 별도 검증 자산으로 관리하며 일반 Git
blob으로 추가하지 않는다.

## Fixed-map localization 주행

동일 시나리오를 다시 CP14에 load하고 network profile을 복구한 뒤 실행한다.

```bash
MAP_PATH="$(ros2 pkg prefix ad_localization)/share/ad_localization/maps/cp14_to_cp15.pcd"
sha256sum "$MAP_PATH"
ros2 launch ad_bringup tunnel_localization.launch.py control_enabled:=true
```

MORAI UI에서 pause를 해제하고 CP15까지 주행한다. 주행 전후 PCD checksum 불변,
endpoint error, route error, lost-localization 횟수, odometry gap, CPU 사용량과 최종
정지를 함께 기록한다. node가 살아 있다는 사실만으로 성공 처리하지 않는다.

## 전체 경로 hybrid 주행

CP14 전에는 GNSS+IMU, CP14→CP15에는 fixed-map Fast-LIO2, CP15 이후에는 다시
GNSS+IMU를 사용하는 구성이다.

```bash
cd ~/heven_ad_2026_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export AD_DATA_DIR="$PWD/src/heven_ad_2026/ad_data"

ros2 launch ad_bringup course_trial.launch.py \
  control_enabled:=false
```

먼저 control-off 상태에서 initial pose, Fast-LIO prewarm과 backend diagnostics를
확인한다. 주행할 때만 `control_enabled:=true`로 다시 시작하고 MORAI UI에서 pause를
해제한다. 성공 판정에는 다음을 사용한다.

- handoff phase가 `gnss_approach -> fastlio_ready -> fastlio_active ->
  gnss_recovery -> gnss_finish` 순서로 진행
- 전환 직후 canonical pose와 yaw가 연속
- canonical odometry timestamp 역행 없음
- 동적 `odom -> base_link` TF authority 하나
- CP15 이후 GNSS 복귀와 종점 full brake 확인

## 2026-07-23 기준 결과

당시 25.S4 MolitComp03에서 확인한 재현 기준은 다음과 같다. 새로운 코드·맵·센서
설정에서는 다시 측정해야 하며 성공 보장값으로 사용하지 않는다.

- route artifact 280행, 중복 제거 후 278점, 평면 호장 `138.089222 m`
- mapping endpoint `0.370 m`, route error 최대 `0.470 m`
- fixed localization endpoint `0.389 m`, route error 최대 `0.471 m`
- 보강 후 fixed localization route error 최대 `0.166 m`, RMS `0.041 m`
- arrival-time wheel 계약 적용 후 endpoint `0.122 m`, 정지 후 `0.078 m`
- 주행 중 LiDAR/odometry 최대 gap 약 `0.23~0.35 s`, timestamp 역행 0
- 최종 PCD `89,814` points, SHA-256
  `b498c8635ac16736f17c3a1fad76e1aeddad893dc499d61b34d6d35499ee0c55`

MORAI의 약 `8.8 Hz` wall cadence는 센서 설정 `10 Hz`와 simulator 실행률 차이가
포함된 값이다. Fast-LIO 로그의 throttle 주기를 실제 scan 주기로 해석하지 않는다.
