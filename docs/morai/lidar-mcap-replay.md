# LiDAR MCAP replay와 튜닝

이 절차는 이미 녹화된 perception 출력을 재주입하지 않고 MCAP의 센서 입력만
재생해 다음 graph를 다시 실행한다.

```text
/ad/sensors/lidar/points
  -> self crop
  -> Patchwork++ ground removal
  -> finite-point filter
  -> adaptive Euclidean clustering
  -> Autoware multi-object tracker
  -> HEVEN IMM prediction
```

MORAI cloud는 instantaneous scan으로 취급하므로 motion deskew는 끈다. 재생
wrapper는 모든 포함 node에 `use_sim_time=true`를 적용하고 rosbag clock을 100 Hz로
발행한다. 원본 bag에 기록된 crop/detection/tracking/prediction 토픽은 재생하지
않는다. 대형 PointCloud2가 Fast DDS best-effort 구간에서 유실되지 않도록 replay
wrapper는 raw LiDAR publisher와 첫 self-crop subscriber만 reliable QoS로 맞춘다.
실차/시뮬레이터 launch의 기본 best-effort QoS는 바꾸지 않는다.

## 준비

ROS 2 Humble의 공식 MCAP storage plugin이 필요하다.

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends -y \
  ros-humble-rosbag2-storage-mcap
```

archive checksum과 압축 무결성을 확인한 뒤, repository에서 ignore되는 실험
경로에 해제한다.

```bash
cd /home/didgang1203/heven-ad-2026

HEVEN_BAG_ARCHIVE=/home/didgang1203/Downloads/static_20260805_003151.tar.zst
HEVEN_BAG_PARENT="$PWD/ad_data/experiments/local_bag_replay"

sha256sum "$HEVEN_BAG_ARCHIVE"
# 52562c4e444d7e4af74c3be8971a00db31b24b6fcdf55c7169821d136c69aef1
zstd --test "$HEVEN_BAG_ARCHIVE"
tar --zstd --list --file "$HEVEN_BAG_ARCHIVE"
mkdir -p "$HEVEN_BAG_PARENT"
tar --zstd --extract \
  --file "$HEVEN_BAG_ARCHIVE" \
  --directory "$HEVEN_BAG_PARENT" \
  --no-same-owner \
  --no-same-permissions
```

정상 해제된 bag은 다음 구조다.

```text
static_20260805_003151/
├── metadata.yaml
└── static_20260805_003151_0.mcap
```

## 빌드와 재생

이 PC에서는 repository root가 아니라 아래의 canonical colcon workspace에서
빌드·실행한다.

```bash
cd /home/didgang1203/heven-ad-2026/ad_data/experiments/lidar_mcap_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-up-to ad_lidar_perception ad_morai_bridge_dev
source install/setup.bash
```

기존 shell의 ROS 1 Noetic 환경 오염을 피하려면 먼저 다음 블록으로 깨끗한 shell을
연다. 이후 이 문서의 ROS 명령은 그 shell에서 실행한다.

```bash
env -i \
  HOME=/home/didgang1203 \
  USER=didgang1203 \
  LOGNAME=didgang1203 \
  PATH=/home/didgang1203/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LANG=C.UTF-8 \
  ROS_DOMAIN_ID=87 \
  ROS_LOCALHOST_ONLY=1 \
  ROS_LOG_DIR=/tmp/heven_lidar_replay_ros_logs \
  bash --noprofile --norc
```

bag path는 절대경로로 넘긴다. node가 준비된 다음 player가 시작되도록 기본 2초
지연을 두며, 무거운 graph나 cold start에서는 늘린다.

```bash
HEVEN_BAG_PATH=/home/didgang1203/heven-ad-2026/ad_data/experiments/local_bag_replay/static_20260805_003151

ros2 launch ad_lidar_perception lidar_bag_replay.launch.py \
  bag_path:="$HEVEN_BAG_PATH" \
  startup_delay_sec:=3.0 \
  rate:=0.5
```

`rate:=0.5`가 이 PC의 검증된 기본값이다. QoS override 전에는 1.0배속에서
1,298/1,778 frame, 0.5배속에서도 1,248/1,778 frame만 첫 구간을 통과했다.
reliable replay 경계 적용 후 0.5배속 전체 run은 모든 단계에서 1,778/1,778
frame을 보존했다.

wrapper는 다음 source-only whitelist만 재생한다.

- `/ad/sensors/lidar/points`
- `/tf`, `/tf_static`
- `/ad/localization/odometry`
- `/ad/localization/input/wheel_speed`
- `/ad/sensors/imu/data`

`start_paused:=true`를 사용하면 rosbag player service로 재생을 시작한다.

```bash
ros2 service call /rosbag2_player/resume rosbag2_interfaces/srv/Resume '{}'
```

## RViz에서 객체 보기

`DetectedObjects`, `TrackedObjects`, `PredictedObjectArray`는 RViz 기본
display가 직접 그리지 못하는 custom message다. `ad_viz`의 marker node가 이를
`visualization_msgs/MarkerArray`로 변환하므로, RViz를 직접 실행하지 말고 아래
시각화 launch를 사용한다. 이 launch는 `use_sim_time=true`와 `odom` 기준 frame을
사용하고 `/ad/viz/perception/objects`를 발행한다.

터미널 1에서 replay를 일시정지 상태로 시작한다.

```bash
ros2 launch ad_lidar_perception lidar_bag_replay.launch.py \
  bag_path:="$HEVEN_BAG_PATH" \
  startup_delay_sec:=3.0 \
  start_paused:=true \
  rate:=0.5
```

터미널 2에서 marker 변환 노드와 RViz를 함께 시작한다.

```bash
source /opt/ros/humble/setup.bash
source /home/didgang1203/heven-ad-2026/ad_data/experiments/lidar_mcap_ws/install/setup.bash
ros2 launch ad_viz visualization.launch.py
```

RViz의 `Fixed Frame`은 `odom`으로 둔다. 기본 설정에는 다음이 포함되어 있다.

```text
/ad/perception/lidar/clusters
/ad/perception/lidar/nonground_finite
/ad/viz/perception/objects
```

RViz가 준비된 뒤 터미널 3에서 bag을 재개한다.

```bash
source /opt/ros/humble/setup.bash
ros2 service call /rosbag2_player/resume rosbag2_interfaces/srv/Resume '{}'
```

`/ad/viz/perception/objects`에는 검출·추적 box, 속도 화살표, object ID와
미래 예측 궤적이 MarkerArray로 표시된다. 장애물 점군만 확인하려면 RViz에서
`PointCloud2`로 `/ad/perception/lidar/clusters`를 추가한다.

## 파라미터 튜닝

원본 config를 수정하지 않고 ignored experiment directory에 후보를 만든다.

```bash
HEVEN_EXPERIMENT="$PWD/ad_data/experiments/lidar-replay-candidate"
mkdir -p "$HEVEN_EXPERIMENT"
cp ad_lidar_perception/config/clustering/adaptive_euclidean_cluster.yaml \
  "$HEVEN_EXPERIMENT/clustering.yaml"
cp ad_lidar_perception/config/preprocessing/ground_segmentation.yaml \
  "$HEVEN_EXPERIMENT/ground.yaml"
```

후보 config는 절대경로, 기존 YAML 일반 파일이어야 한다. crop clearance는
`0.0..2.0 m` 범위만 허용한다.

```bash
ros2 launch ad_lidar_perception lidar_bag_replay.launch.py \
  bag_path:="$HEVEN_BAG_PATH" \
  cluster_config:="$HEVEN_EXPERIMENT/clustering.yaml" \
  ground_config:="$HEVEN_EXPERIMENT/ground.yaml" \
  crop_clearance_m:=0.20 \
  rate:=0.5
```

현재 기본 clustering profile은 이 bag에서 원거리 VLP-16 희소성을 고려한 다음
값을 사용한다.

- neighbor tolerance: `0.45 -> 1.60 m`, 45 m에서 포화
- minimum points: `5 -> 2`, 별도 보간으로 45 m에서 포화
- maximum component points: `20,000`
- ROI: X `[-4, 100] m`, Y `[-25, 25] m`, Z `[-1, 3] m`
- XY 대각선이 12 m보다 큰 component는 cluster debug cloud에는 보존하지만 동적
  detection/tracker 입력에서는 제외

한 번에 한 축만 바꾸고 동일한 30초 구간을 후보 선택용과 확인용으로 분리한다.
최종 후보는 검증된 `rate:=0.5` 전체 replay에서 output loss와 처리 지연을 다시
확인한다. 1.0배속은 이 PC에서 실시간 처리 수용 기준이 아니라 별도의 성능
stress test로 취급한다.

## 정량 audit

원본에 기록된 baseline stage와 prediction diagnostic을 read-only로 점검한다.
결과 경로는 bag directory 밖이어야 한다.

```bash
HEVEN_AUDIT="$PWD/ad_data/experiments/lidar-replay-audit"
ros2 run ad_morai_bridge_dev ad_morai_perception_mcap_audit \
  "$HEVEN_BAG_PATH" \
  --output-dir "$HEVEN_AUDIT"
```

`mcap_replay_audit.json`과 `mcap_replay_audit.md`에는 stage별 message/point/object
수, header stamp 중복·역행, LiDAR input 대비 exact-stamp coverage, prediction
diagnostic level/message/rejection reason이 기록된다. 이 bag의 metadata 기준으로는
tracked 1,796 frame에 비해 predicted가 301 frame뿐이므로, 재생 후 predicted
coverage와 stale/clock/frame rejection을 우선 확인한다.

2026-08-09 최종 reliable replay 결과는 다음과 같다.

| 항목 | 원본 recorded output | 최종 재계산 output |
|---|---:|---:|
| LiDAR input | 1,778 | 1,778 |
| crop/nonground/finite/detected/tracked/predicted frame | 단계별 1,789--1,806, predicted 301 | 단계별 모두 1,778 |
| input 대비 predicted exact-stamp coverage | 16.65% | 100% |
| tracked object / predicted object | 8,074 / 0 | 10,817 / 10,817 |
| prediction diagnostic | ERROR 2,990 | OK 10,817, ERROR 0 |
| stamp duplicate / non-increasing | 0 / 0 | 0 / 0 |

로컬 결과 파일은 다음 위치에 있다.

```text
ad_data/experiments/lidar_replay_runs/baseline_audit_20260809/
ad_data/experiments/lidar_replay_runs/candidate_reliable_20260809/
ad_data/experiments/lidar_replay_runs/candidate_reliable_audit_20260809/
```

동일 stamp의 cropped/nonground/detected를 직접 그린 BEV 표본도 최종 audit
directory의 `bev_frame_*.png`로 보존했다. index 100/889/1200/1650에서는
1--3개 차량 크기 component가 분리되지만, index 500에서는 y 약 2.3 m의 연속
도로 경계와 먼 수직 pole이 여러 작은 UNKNOWN box로 분절된다. ground truth 없이
최소 폭/높이 gate를 추가하면 원거리 보행자·이륜차 recall을 훼손할 수 있으므로
이번 profile에서는 자동 제거하지 않는다. 다음 정성 튜닝에서는 이 구간을 우선
보고, 실제 객체 label과 비교한 뒤 collinear static-structure suppression 또는
shape gate를 결정한다.

## 정성 확인

RViz fixed frame은 bag TF에 맞게 `odom`으로 두고 raw, cropped, ground,
nonground, clusters, tracked/predicted marker를 함께 확인한다. 특히 다음 실패를
구간별로 기록한다.

- 차량 point가 ground로 제거되는지
- 중앙분리대나 guardrail에 차량 cluster가 흡수되는지
- 40--60 m, 60--80 m에서 cluster와 track이 끊기는지
- 인접 차량 merge, 한 차량 split, track ID switch가 발생하는지
- 회전·끼어들기에서 예측 궤적이 CV에 고정되지 않고 CTRV 쪽으로 전환되는지

처리율은 별도 terminal에서 확인한다.

```bash
ros2 topic hz /ad/perception/lidar/cropped --use-sim-time
ros2 topic hz /ad/perception/lidar/nonground_finite --use-sim-time
ros2 topic hz /ad/perception/objects/detected --use-sim-time
ros2 topic hz /ad/perception/objects/tracked --use-sim-time
ros2 topic hz /ad/perception/objects/predicted --use-sim-time
```

MORAI production/dev bridge는 receipt-time 계약 때문에 sim time을 거부하므로 이
replay와 함께 실행하지 않는다.
