# Autoware LiDAR detector/tracker 연동

기준은 ROS 2 Humble, Autoware `1.8.0`, Universe `0.51.0`,
`autoware_perception_msgs` `1.13.0`이다. 기본 DWA 실행은 model-free
adaptive Euclidean detector와 Autoware tracker overlay를 사용하며 CUDA,
TensorRT, model은 필요 없다.

## 기본 구성

`ad_lidar_perception/config/lidar_perception.yaml` 한 파일에서 조합을
선택한다. 저장소 기본값은 다음과 같다.

```yaml
detector:
  backend: euclidean_cluster
tracker:
  backend: autoware
occupancy:
  static_enabled: true
  dynamic_enabled: true
  publish_combined: true
```

이 구성은 Patchwork++, VLP-16 거리 적응형 Euclidean clustering,
Autoware tracking, 6초 IMM prediction, static/dynamic/combined OGM을
실행한다. 딥러닝 detector가 놓치는 벽·기둥·낮은 장애물도 static OGM
경로에는 남는다. 추적은 도로 밖에서도 유지하지만 planning OGM에는
현재 footprint 중 drivable corridor와 만나는 부분만 반영한다. 미래
footprint는 OGM에 시간축 없이 합치지 않고 DWA가 예측 시각별로 별도
충돌 검사한다.

선택 가능한 detector는 `euclidean_cluster`, `centerpoint_tiny`,
`centerpoint`, `transfusion`, `bevfusion_lidar`이고 tracker는
`autoware`이다. tracker는 detector가
필요하고 dynamic OGM은 tracker가 필요하다. 잘못된 조합, 문자열 boolean,
중복·미지 key는 실행 전에 거부한다. `build_only: true`는 engine 생성
준비용 검증 모드이며 주행 graph를 실행하지 않는다.

## 데이터 흐름

```text
/ad/sensors/lidar/points
  ├─ FAST-LIO2 (MORAI: instantaneous scan, point warp 없음)
  └─ [real hardware에서만 optional motion deskew]
       └─ [optional IONIQ5 self crop]
            ├─ static occupancy → /ad/perception/occupancy/static
            ├─ PointXYZIRC adapter → learned detector
            └─ [optional gravity leveler] → Patchwork++ → non-ground
                 └─ finite-point filter
                      └─ [optional one-frame densifier]
                           └─ adaptive Euclidean detector

detector → /ad/perception/objects/detected
         └─ tracker → /ad/perception/objects/tracked
              └─ HEVEN IMM prediction (stationary + CV + CTRV)
                   ├─ /ad/perception/objects/predicted
                   └─ /ad/perception/occupancy/dynamic

static + dynamic → /ad/perception/occupancy/combined
                 → /ad/perception/occupancy_grid (compatibility alias)
```

top-level `lidar_perception.launch.py`의 전처리 선택 인자는 다음과 같다.

- `platform_profile:=morai`: 기본값이다. MORAI CH16 cloud는 한 pose의 snapshot으로
  취급하며 `deskew_enabled:=true` 조합을 launch 전에 거부한다.
- `platform_profile:=real_hardware deskew_enabled:=true`: 실제 firing time이 있는
  실차 LiDAR의 scan 내 3D motion deskew를 실행한다.
- `deskew_mode:=3d`: `3d` 또는 비교용 `2d`만 허용한다.
- `self_crop_enabled:=true`: IONIQ5 차체 return을 공통 branch에서 한 번 제거한다.
- `patchwork_leveling_enabled:=true`: Patchwork++ 직전에 roll/pitch를 제거하고
  실제 `lidar_leveled_frame` TF를 발행한다.
- `densifier_enabled:=false`: 기본 OFF이다. ON일 때도 classical
  `euclidean_cluster`의 finite non-ground edge에만 적용한다.

각 optional stage는 OFF일 때 node 자체가 생성되지 않고 직전 topic이 다음 stage의
입력으로 전달된다. 따라서 MORAI 기본 경로는 `raw -> self crop -> downstream`이고,
실차에서 deskew를 켜면 자동으로 `raw -> deskew -> self crop -> downstream`이 된다.
crop을 끄면 deskew의 실제 output 또는 raw가 static occupancy와 Patchwork++에
직접 전달된다. learned detector adapter도 동일한 공통 전처리 include가 공급하므로
두 번째 전처리 graph를 만들지 않는다. FAST-LIO2는 항상 raw topic을 별도
소비하되 MORAI profile에서는 scan duration을 0으로 두고 per-point warp를 하지
않는다.

MORAI Velodyne adapter는 모든 point time을 0으로 채우는 `zero` mode를 사용한다.
azimuth로 합성한 0--0.1초 point time을 MORAI motion deskew의 근거로 사용하지
않는다. `azimuth`/rolling timing과 deskew 구현은 실차 profile을 위해 코드에
남기되 MORAI launch에서는 활성화하지 않는다.

### Classical densifier 운용 계약

densifier는 OFF일 때 process 자체가 없고 clustering 입력은
`/ad/perception/lidar/nonground_finite`이다. ON이면
`/ad/perception/lidar/nonground_finite`를 받아
`/ad/perception/lidar/nonground_densified`를 발행하며, clustering 입력
edge만 후자로 바뀐다. occupancy, Patchwork++, learned detector,
FAST-LIO2에는 densified cloud를 연결하지 않는다.

ON 상태에서도 누적 범위는 직전 raw input 한 frame뿐이다. `odom`을 fixed
frame으로 사용해 직전 stamp의 `<actual input frame>`을 현재 stamp의 같은
frame으로 옮긴다. 현재 profile의 canonical frame은 `lidar_link`이지만 다른
유효한 profile frame도 동일하게 처리한다. 변환 뒤 현재 좌표계 X
`[20, 100] m`, Y `[-12, 12] m` ROI와 `0.30 m` 3D voxel을 과거 point에만
적용한다. 현재 record는 ROI 밖과 voxel 중복을 포함해 순서와 bytes를 모두
보존하고 voxel을 먼저 차지한다. 과거 record는 voxel당 최대 하나만 추가하며
XYZ 외 bytes는 원본 그대로 유지한다.

history age `0.25 s`, translation jump `5.0 m`, rotation jump `0.35 rad`,
TF timeout `0.05 s`를 초기 보수 한계로 사용한다. stamp 역행/중복, stale
history, schema/endian/frame 불일치, TF 부재, pose jump, malformed cloud,
비유한 수치 또는 안전한 float/voxel 변환 실패 시 node는 중단하지 않고 현재
cloud를 정확히 그대로 발행한다. malformed current는 history를 비우며, 그 외
유효한 current-only fallback은 현재 raw input으로 history를 교체한다.
`euclidean_cluster`를 선택하면서 ground segmentation을 끄는 구성과 모호한
boolean/deskew mode/topic feedback은 dependent node를 띄우기 전에 거부한다.

예측은 tracker가 낸 ID, pose, shape, covariance와 속도를 HEVEN message로
변환한다. 객체 ID별 IMM이 정지, 등속 직선(CV), 등속 회전(CTRV) 가정을
동시에 갱신하고 0.5초 간격으로 6.0초까지 융합 미래 상태를 만든다.

## 환경 준비

Autoware tracker와 muSSP는 `dependencies.repos`를 통해 같은 workspace에
고정 revision으로 준비한다. 기본 adaptive Euclidean detector는 HEVEN
패키지에 포함되어 있으며 model이나 license
acknowledgement가 필요 없다.
딥러닝 detector를 선택할 때만 model 재배포 권한을 검토한 뒤 다음
acknowledgement를 명시해야 한다.

```bash
cd ~/heven_ad_2026_ws
export AD_DATA_DIR="$PWD/ad_data"
# CenterPoint/TransFusion/BEVFusion을 선택한 경우에만:
export AD_AUTOWARE_MODEL_LICENSE_REVIEWED=1
```

model 경로는 `$AD_DATA_DIR/models/autoware/` 아래이며 정확한 파일명과
SHA-256은
`ad_lidar_perception/config/autoware_perception.lock.yaml`에 고정되어 있다.
lock 자체, 설치된 Autoware launch/config와 model artifact는 SHA-256까지
검증한다. locally 생성한 TensorRT engine은 GPU별 파일이라 upstream
hash가 없으며, preflight는 symlink·빈 파일·non-regular 파일만 거부한다.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 src/heven_ad_2026/scripts/check_autoware_perception.py \
  --selection \
    src/heven_ad_2026/ad_lidar_perception/config/lidar_perception.yaml
```

기본 build는 required prediction adapter를 함께 만든다. 정적 OGM 전용
구성을 명시적으로 빌드할 때만 `-DAD_WITH_AUTOWARE=OFF`를 사용한다.

```bash
colcon build --packages-up-to ad_lidar_perception ad_viz \
  --symlink-install \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DAD_WITH_AUTOWARE=ON
```

## 실행과 확인

YAML 조합을 저장한 뒤 평소처럼 실행한다.

```bash
ros2 launch ad_bringup bringup.launch.py control_enabled:=true
```

`ad_bringup`이 시작한 RViz에서 combined OGM과 raw LiDAR는 기본 ON이다.
PointXYZIRC, static/dynamic
개별 layer, MPPI 후보 trajectory는 디버깅할 때만 켠다.

실제 detector/tracker 평가는 동일 rosbag으로 recall, false positive,
track ID switch, velocity variance, end-to-end latency와 GPU 사용량을
기록한다. 현재 머신에는 호환 TensorRT 개발 환경과 검토 완료 model이
없으므로 CenterPoint/TransFusion의 실측 FPS나 정확도는 아직 확정값으로
기록하지 않는다.

MCAP의 원천 LiDAR와 TF만 재생해 crop부터 IMM prediction까지 다시 검증하고
clustering/ground config를 반복 튜닝하는 절차는
[LiDAR MCAP replay와 튜닝](lidar-mcap-replay.md)을 따른다.

optional 환경이 모두 준비된 뒤에만 live gate를 명시적으로 켠다. 결과
지표는 build 결과 디렉터리에 JSON으로 남긴다.

```bash
export AD_RUN_AUTOWARE_INTEGRATION=1
export AD_AUTOWARE_INTEGRATION_METRICS_FILE="$PWD/build/ad_lidar_perception/autoware_pipeline_metrics.json"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test \
  --packages-select ad_lidar_perception \
  --ctest-args -R '^test_autoware_pipeline_integration$' --output-on-failure
```

이 gate는 설치된 전체 graph, PointXYZIRC ABI와 stamped TF, 빈·손상 cloud,
tracker/prediction stale clear, detector 종료 뒤 static/combined OGM
생존 여부를 검사한다. 환경 변수만 켜고 overlay, 고정 model/engine 또는
GPU 계측이 빠진 경우에는 성공으로 건너뛰지 않고 실패한다.
