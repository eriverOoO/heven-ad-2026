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

### 카메라와 트래킹 원클릭 동시 재생

`morai_cam4_20260813_163222`처럼 전방 압축 카메라와 LiDAR를 함께 담은
MCAP은 저장소 루트에서 아래 실행 파일 하나로 연다.

```bash
./scripts/run_camera_lidar_tracking.sh \
  /absolute/path/to/morai_cam4_20260813_163222
```

인자를 생략하면 저장소 안의 로컬 전용 기본 위치
`morai_cam4_20260813_163222/morai_cam4_20260813_163222`를 찾는다. 두 번째
인자는 재생 배속이며 기본값은 이 PC에서 안정적으로 검증한 `0.5`다.

```bash
./scripts/run_camera_lidar_tracking.sh /absolute/path/to/bag 1.0
```

### 10개 추적 프리셋

프리셋 목록은 ROS 실행이나 bag 없이 확인할 수 있다.

```bash
./scripts/run_camera_lidar_tracking.sh --list-modes
```

`--mode`와 `--bag`으로 한 조건을 선택한다. 기본 모드는 3번이며 기존
원클릭 실행과 같은 조합이다.

```bash
./scripts/run_camera_lidar_tracking.sh \
  --mode 6 \
  --bag /absolute/path/to/morai_cam4_20260813_163222 \
  --rate 0.5
```

| 모드 | Detector | Association | Matcher | Estimator |
| ---: | --- | --- | --- | --- |
| 1 | Euclidean | GIoU | Greedy | Linear KF |
| 2 | Euclidean | GIoU | Hungarian | Linear KF |
| 3 | Euclidean | Euclidean 3 m | Hungarian | Linear KF |
| 4 | Euclidean | Mahalanobis Hybrid 10 m | Hungarian | Linear KF |
| 5 | Euclidean | Euclidean 3 m | Hungarian | CTRV EKF |
| 6 | Euclidean | Euclidean 3 m | Hungarian | IMM |
| 7 | Euclidean | Euclidean 3 m | Hungarian | KalmanNet |
| 8 | CenterPoint | Euclidean 3 m | Hungarian | Linear KF |
| 9 | CenterPoint | GIoU | Hungarian | Linear KF |
| 10 | CenterPoint | Mahalanobis Hybrid 10 m | Hungarian | Linear KF |

모든 프리셋은 기존 `ab3dmot.yaml` lifecycle을 바꾸지 않고
`yaw_measurement_mode=unobserved`, Euclidean gate 3 m, Mahalanobis gate
11.62를 공유한다. 4번과 10번의 Hybrid만 추가로 절대거리 10 m cap을
사용한다. 따라서 프리셋 간 차이는 표에 기재한 변수로 제한된다.

7번은 체크포인트와 PyTorch 환경을 명시해야 한다.

```bash
./scripts/run_camera_lidar_tracking.sh \
  --mode 7 --bag /absolute/path/to/bag \
  --kalmannet-checkpoint /absolute/path/to/dense_kalmannet_v2.pt
```

8~10번은 CenterPoint 체크포인트, CUDA 가능한 PyTorch 환경이 필요하다.
OpenPCDet 소스는 `references/openpcdet` 서브모듈의 검증된 커밋으로
고정되며 `git submodule update --init --recursive` 또는 표준 bootstrap이
준비한다. 실행 전에 CenterPoint용 Python venv를 활성화한다.

```bash
source /absolute/path/to/centerpoint-venv/bin/activate
./scripts/run_camera_lidar_tracking.sh \
  --mode 10 --bag /absolute/path/to/bag \
  --centerpoint-checkpoint /absolute/path/to/checkpoint.pth \
  --centerpoint-device cuda:0
```

경로는 환경변수 `HEVEN_KALMANNET_CHECKPOINT`,
`HEVEN_CENTERPOINT_CHECKPOINT`, `HEVEN_OPENPCDET_ROOT`로도 제공할 수 있다.
옵션과 환경변수가 없으면 각각 저장소의 로컬 전용
`models/experimental/dense_kalmannet_v2.pt`,
`models/experimental/centerpoint_t14_reproduction.pth`를 찾는다.
체크포인트와 bag은 저장소에 커밋하지 않는다. 다른 PC에는 별도로 복사하고
해시와 provenance를 확인한다. 기존 10모드 연구 자산의 SHA-256은
CenterPoint `466c8181a377682e032bb32579c8ddb65807b5feebd8625a750b1d5538ddbc95`,
KalmanNet `956604975e5204b2c584c3e8fe16a7ba9346097980566ecbbb22c2001fbf7d48`이다.

CenterPoint는 설계된 cropped-only 입력을 사용하므로 해당 모드에서는 지면
분리를 자동으로 끈다. Euclidean 모드는 기존 지면 분리 경로를 그대로 쓴다.
각 조건을 공정하게 비교하려면 실행을 완전히 종료한 뒤 다른 모드로 같은
bag을 처음부터 재생한다. 이 프리셋 화면은 정성 비교이며 과거 여러 데이터와
오프라인 프로토콜에서 얻은 수치를 온라인으로 재현했다는 의미가 아니다.

이 실행은 한 ROS launch에서 다음을 함께 시작한다.

- MCAP `/clock`, front compressed camera, raw LiDAR, TF 재생
- 기존 MORAI classical LiDAR 경로와 기본 Autoware tracker (`A-`)
- 명시적으로 선택한 AB3DMOT 프리셋 (`B-`; 기본은 Linear KF,
  Euclidean 3 m, Hungarian, yaw unobserved)
- front camera, cropped LiDAR, detection, 두 tracker marker가 켜진 RViz
- localization TF가 없는 bag을 위한 replay-only identity
  `odom -> base_link` anchor

다른 PC에서는 먼저 저장소 표준 `./scripts/bootstrap_workspace.sh`를
완료해야 한다. `package.xml`의 `rosbag2_storage_mcap`과
`compressed_image_transport` runtime dependency도 bootstrap의 `rosdep`
단계에서 설치된다. workspace가 저장소의 표준 상위 경로가 아니라면
`HEVEN_AD_WS_PATH=/absolute/path/to/workspace`를 지정한다.

MCAP은 3.6 GB 대용량 실험 데이터라 Git/GitHub에 포함하지 않는다. 다른
PC에는 bag 폴더를 별도 복사한 뒤 위 실행 파일에 절대경로를 넘긴다.
`.mcap`과 `/morai_cam4_*/`는 `.gitignore`로 차단돼 있다.

이 보기는 정성적 cross-check 전용이다. 이 bag에는 camera annotation,
`CameraInfo`, perception output, dynamic localization TF가 없으며 기존
T-series와도 다른 moving-ego sequence다. 화면 일치를 accuracy나 기존
실험 수치의 재검증으로 해석하지 않는다.

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
