# 현재 LiDAR perception 파이프라인 계약

- 조사일: 2026-08-15 (Asia/Seoul)
- HEVEN 기준 commit: `7828357ae1d87f077ae6c4f39e2783f77fa1109d`
- Autoware Universe 기준 commit: `d4d260983d357e1b2b34291d91933f9f4b53bf94`
- Patchwork++ 기준 commit: `3e6903a1d5537a4cc2ace897b0bbb98a92d6014c`

## 1. 범위와 조사 기준

이 문서는 STEP 01의 코드 조사 결과다. 운영 코드나 parameter는 변경하지
않았으며, 아래 내용은 현재 repository의 source, launch, config만을 근거로
작성했다. 별도 표기가 없는 경로는 repository root 기준이다.

기본 실행은 `ad_bringup/config/components.yaml`에서 LiDAR perception과 MORAI
bridge를 모두 활성화하고, `ad_bringup/ad_bringup/bringup_stack.py`가
`lidar_perception.launch.py platform_profile:=morai`와
`bridge.launch.py enable_velodyne_points:=true`를 포함하는 경우로 정의한다.

현재 기본 선택 파일 `ad_lidar_perception/config/lidar_perception.yaml`은 다음을
선택한다.

- detector: `euclidean_cluster`
- tracker: `autoware`
- static occupancy: enabled
- dynamic occupancy: enabled
- combined occupancy: enabled

`lidar_perception_morai_classical.yaml`은 occupancy 세 레이어를 끄는 별도 구성이지
기본값이 아니다. CenterPoint, TransFusion, BEVFusion LiDAR backend도 이미 선택
가능하지만 기본 runtime graph에는 포함되지 않는다.

저장소 규칙이 요구하는 repo별 Notion hub는 현재 세션에서 접근할 수 없었다.
따라서 Notion에만 기록된 운영 정책은 이 문서에 추측해서 넣지 않았다.

## 2. 기본 runtime data flow

```text
MORAI UDP Velodyne
  -> ad_velodyne_adapter
  -> velodyne_transform_node
  -> /ad/sensors/lidar/points_with_synthetic_time
  -> ad_point_time_zero_boundary
  -> /ad/sensors/lidar/points                 PointCloud2, lidar_link
       |
       +-> ad_self_crop_filter
       |    -> /ad/perception/lidar/cropped  PointCloud2, lidar_link
       |         |
       |         +-> Patchwork++
       |         |    +-> /ad/perception/lidar/cloud
       |         |    +-> /ad/perception/lidar/ground
       |         |    `-> /ad/perception/lidar/nonground
       |         |          -> ad_finite_point_filter
       |         |          -> /ad/perception/lidar/nonground_finite
       |         |          -> ad_adaptive_euclidean_cluster
       |         |               +-> /ad/perception/lidar/clusters
       |         |               `-> /ad/perception/objects/detected
       |         |                    -> Autoware multi_object_tracker
       |         |                    -> /ad/perception/objects/tracked, odom
       |         |                    -> stateful IMM prediction
       |         |                    -> /ad/perception/objects/predicted, odom
       |         |                    -> dynamic occupancy, base_link
       |         |
       |         `-> static occupancy, base_link
       |
       `-> (deskew는 MORAI 기본 구성에서 금지/비활성)

/ad/planning/drivable_mask, base_link
  -> static/dynamic occupancy와 exact-stamp pairing

static occupancy + dynamic occupancy (동일 stamp/frame/geometry)
  -> combined occupancy
  -> /ad/perception/occupancy/combined
  -> /ad/perception/occupancy_grid (compatibility alias)
```

기본값에서는 point-layout adapter와 densifier가 실행되지 않는다. point-layout
adapter는 learned detector 선택 시 `/ad/perception/lidar/points_xyzirc`를 만들며,
densifier는 명시적으로 켰을 때만
`/ad/perception/lidar/nonground_densified`를 cluster 입력으로 사용한다.

## 3. 실제 topic graph와 메시지 계약

### 3.1 LiDAR 입력과 전처리

#### `/ad/sensors/lidar/points`

- message: `sensor_msgs/msg/PointCloud2`
- frame_id: MORAI 기본 경로에서 `lidar_link`
- timestamp: Velodyne scan header stamp. MORAI 경계 노드는 point별 `time` 필드만
  모두 `+0.0`으로 만들고 header는 변경하지 않는다.
- producer: MORAI에서는 `ad_point_time_zero_boundary`; real hardware에서는
  `velodyne_transform_node`가 직접 publish
- consumers: self crop, static occupancy, localization 등. 기본 perception
  composition에서는 self crop과 static occupancy 경로의 upstream source다.
- QoS: MORAI boundary publisher는 ROS 2 sensor-data profile
  (keep-last 5, best-effort, volatile). Bag replay용 override는 reliable,
  volatile, keep-last 10이다.

#### `/ad/perception/lidar/deskewed` (조건부)

- message: `sensor_msgs/msg/PointCloud2`
- producer: `ad_motion_deskew`
- consumer: self crop
- frame_id/timestamp: 입력 header 유지
- QoS: input/output 모두 `SensorDataQoS`
- 활성 조건: `deskew_enabled:=true`. `platform_profile:=morai`에서는 launch가
  이를 명시적으로 거부하므로 기본 MORAI graph에는 없다.

#### `/ad/perception/lidar/cropped`

- message: `sensor_msgs/msg/PointCloud2`
- frame_id/timestamp: 입력 header를 그대로 복사한다. 점 좌표 자체도 입력 frame에
  남는다. `base_link <- input frame`의 입력 timestamp TF는 차량 내부 영역 판정에만
  사용한다.
- producer: `ad_self_crop_filter`
- consumers: Patchwork++, static occupancy; learned detector가 선택되면
  point-layout adapter도 사용
- QoS: publisher `SensorDataQoS`; subscription은 기본 best-effort sensor-data이며
  `self_crop_input_reliable:=true`일 때만 reliable
- 기본 crop: vehicle geometry와 `crop_clearance_m=0.20`으로 launch에서 다시
  계산된다. 현재 차량 값은 x `[-0.990, 4.045]`, y `[-1.145, 1.145]`,
  z `[-0.200, 1.805]` m이고 경계 포함 영역의 점을 제거한다.
- 실패 동작: timestamp TF가 없거나 cloud/layout이 잘못되면 해당 입력 cloud를
  drop하며 출력하지 않는다.

### 3.2 Ground segmentation

#### `/ad/perception/lidar/cloud`

- message: `sensor_msgs/msg/PointCloud2`
- producer: Patchwork/Patchwork++ backend만 publish
- consumer: 직접적인 production consumer는 조사 범위에서 확인되지 않음
- frame_id/timestamp: 입력 header 그대로
- QoS: reliable, transient-local, RMW default depth

#### `/ad/perception/lidar/ground`

- message: `sensor_msgs/msg/PointCloud2`
- producer: 기본값은 `patchworkpp_node` (`algorithm: patchworkpp`)
- consumer: visualization/benchmark 경로; detector 입력은 아님
- frame_id: launch가 Patchwork++의 `base_frame`을 실제 segmentation 입력 frame으로
  설정한다. 기본값은 `lidar_link`. leveling 활성 시 `<sensor>_leveled_frame`.
- timestamp: Patchwork++은 입력 stamp 유지
- QoS: Patchwork++ 경로는 reliable, transient-local, RMW default depth
- 대체 RANSAC 경로: debug ground pointcloud를 remap한다. 기존 benchmark에서 이
  debug 출력의 stamp가 0으로 관측되었으므로 기본 timestamp 계약과 호환되지
  않는다.

#### `/ad/perception/lidar/nonground`

- message: `sensor_msgs/msg/PointCloud2`
- frame_id/timestamp: 기본 Patchwork++ 경로에서 입력 stamp를 유지하고 frame은
  실제 segmentation frame(`lidar_link`)으로 설정
- producer: `ad_ground_segmentation` 이름으로 실행되는 Patchwork++ 또는 RANSAC
- consumer: finite-point filter 또는, 해당 filter bypass 시 cluster node
- QoS: Patchwork++ publisher는 reliable + transient-local; finite filter
  subscription은 `SensorDataQoS`이므로 reliability는 호환되지만 subscriber가
  transient-local history를 요구하지는 않는다.

Ground segmentation 주요 기본 parameter는 `algorithm=patchworkpp`,
`sensor_height=1.7685 m`(rear-axle static z 0.3685 + lidar z 1.4),
`num_iter=3`, `num_lpr=20`, `num_min_pts=5`, `th_seeds=0.35`,
`th_dist=0.18`, `th_seeds_v=0.25`, `th_dist_v=0.1`, `min_range=3.0`,
`max_range=80.0`, `uprightness_thr=0.707`이다.

#### `/ad/perception/lidar/nonground_finite`

- message: `sensor_msgs/msg/PointCloud2`
- producer: `ad_finite_point_filter`
- consumer: 기본값에서는 adaptive Euclidean cluster
- frame_id/timestamp: 입력 header 그대로
- QoS: input/output 모두 `SensorDataQoS`
- 데이터 처리: little-endian cloud의 유일한 FLOAT32 x/y/z를 요구한다. non-finite
  XYZ record를 제거하고 unorganized height 1 cloud로 compact한다. 나머지 point
  fields와 record bytes는 보존한다.

#### `/ad/perception/lidar/nonground_densified` (조건부)

- message: `sensor_msgs/msg/PointCloud2`
- producer: `ad_pointcloud_densifier`
- consumer: adaptive Euclidean cluster
- frame_id/timestamp: 현재 frame의 header 사용. 이전 frame 점은 TF로 현재 frame에
  변환해 추가한다.
- QoS: input/output 모두 `SensorDataQoS`
- 활성 조건: `densifier_enabled:=true`; 기본값은 false
- 주요 parameter: fixed frame `odom`, voxel 0.30 m, history age 0.25 s,
  x `[20,100]`, y `[-12,12]` m ROI

### 3.3 Detection, tracking, prediction

#### `/ad/perception/lidar/clusters`

- message: `sensor_msgs/msg/PointCloud2`
- producer: `ad_adaptive_euclidean_cluster`
- consumer: debug/visualization
- frame_id/timestamp: cluster 입력 header 그대로
- QoS: keep-last 10, default reliable/volatile
- point layout: x/y/z/intensity FLOAT32. intensity는 1부터 시작하는 cluster index다.
  12 m diagonal filter에서 제외되는 큰 component의 점도 debug cloud에는 포함된다.

#### `/ad/perception/objects/detected`

- message: `autoware_perception_msgs/msg/DetectedObjects`
- producer: 기본값은 adaptive Euclidean cluster; learned backend 선택 시 해당
  Autoware detector
- consumer: Autoware `multi_object_tracker`
- frame_id/timestamp: Euclidean detector는 cluster 입력 header 전체를 복사한다.
  기본값은 `lidar_link`와 원 LiDAR scan stamp다.
- QoS: Euclidean publisher keep-last 10 reliable/volatile; tracker subscription
  keep-last 1 reliable/volatile

#### `/ad/perception/objects/tracked`

- message: `autoware_perception_msgs/msg/TrackedObjects`
- producer: `autoware_multi_object_tracker/multi_object_tracker`
- consumer: `ad_autoware_prediction`
- frame_id: tracker `world_frame_id`, 현재 `odom`
- timestamp: `enable_delay_compensation=false`이므로 마지막 처리 measurement stamp.
  출력은 measurement callback 처리 후 즉시 생성되며 `publish_rate=10 Hz` timer
  보상 경로는 사용하지 않는다.
- QoS: tracker publisher keep-last 1 reliable/volatile; prediction subscriber도
  keep-last 1 reliable/volatile
- TF 요구: tracker는 detected array의 frame에서 `odom`으로 입력 stamp 시점 TF를
  최대 0.5 s 기다려 pose와 pose covariance를 변환한다. TF가 없으면 그 array를
  처리하지 않는다.

#### `/ad/perception/objects/predicted`

- message: `ad_interfaces/msg/PredictedObjectArray`
- producer: `ad_autoware_prediction`의 stateful IMM adapter
- consumer: `ad_dynamic_occupancy_grid`와 planner
- frame_id/timestamp: tracked array header를 그대로 복사하므로 `odom`과 tracker
  measurement stamp 유지
- QoS: keep-last 1, reliable, volatile
- 입력 gate: frame은 정확히 `odom`, stamp는 양수/단조 증가, 미래가 아니고
  현재 시각 기준 0.5 s 이하의 age여야 한다.
- IMM: UUID별 상태를 유지하며 stationary, constant-velocity,
  coordinated-turn 3개 model을 사용한다. 기본 초기확률은 0.20/0.60/0.20,
  track retention 1.0 s, 최대 update interval 2.0 s다.
- horizon: 0.5 s부터 6.0 s까지 0.5 s 간격 12개 상태
- 좌표 convention: 입력 tracked twist는 object-local XY로 보고 pose yaw를 이용해
  world(`odom`) XY velocity로 회전한다. predicted `initial_twist`는 array header
  frame인 `odom`에 표현한다.

#### `/ad/perception/objects/prediction_debug`

- message: `diagnostic_msgs/msg/DiagnosticArray`
- producer: `ad_autoware_prediction`
- frame_id/timestamp: 입력 tracked header 유지
- QoS: keep-last 10, reliable, volatile
- 내용: UUID별 model probability, 선택 mode, reset/gating reason. array 또는 object
  validation 실패 시 prediction 대신 rejection diagnostic만 publish한다.

### 3.4 Occupancy outputs

#### `/ad/perception/occupancy/static`

- message: `nav_msgs/msg/OccupancyGrid`
- producer: `ad_lidar_perception` static grid node
- inputs: 기본값에서 cropped LiDAR와 `/ad/planning/drivable_mask`
- frame_id: `base_link`
- timestamp: 입력 cloud stamp. road gate가 켜져 있으므로 동일 stamp mask가 있어야
  planning static grid가 publish된다.
- QoS: input/output 모두 `SensorDataQoS`
- geometry: x `[-4,100]`, y `[-10,10]`, resolution 0.1 m,
  1040 x 200 cells, identity origin orientation
- 추가 동작: cloud를 input stamp에 `base_link`로 변환한다. 0.5 s 동안 최대 8개
  cloud를 `odom` fixed frame에 누적한 뒤 현재 `base_link`로 되돌린다.

#### `/ad/viz/perception/occupancy/static_ungated`

- message/frame/timestamp/QoS: static grid와 동일
- 용도: 동일 point set으로 만든 road-mask 적용 전 visualization 전용 layer

#### `/ad/perception/occupancy/dynamic`

- message: `nav_msgs/msg/OccupancyGrid`
- producer: `ad_dynamic_occupancy_grid`
- inputs: predicted objects와 동일 stamp drivable mask
- frame_id: `base_link`
- timestamp: 유효 prediction stamp. invalid/stale 입력을 clear할 때는 허용된 최신
  stamp 또는 현재 ROS time으로 빈 layer를 publish할 수 있다.
- QoS: output과 mask subscription은 `SensorDataQoS`; prediction subscription은
  keep-last 1 reliable
- geometry: static과 동일
- 처리: `odom -> base_link`를 prediction stamp에 조회한다. 각 object의 현재
  footprint만 rasterize하고 covariance 최대 고유값의 2 sigma, 최소 0.20 m로
  팽창한다. future states는 유효성/순서를 검사하지만 OccupancyGrid의 시간축
  부재 때문에 모두 rasterize하지 않는다.
- timeout: prediction age 0.50 s, TF timeout 0.05 s, stale check 0.10 s

#### `/ad/perception/occupancy/combined`

- message: `nav_msgs/msg/OccupancyGrid`
- producer: `ad_combined_occupancy_grid`
- consumers: planner/visualization 구성에 따라 사용
- frame_id/timestamp: static layer header를 사용하며 dynamic과 stamp가 정확히 같고
  frame/geometry가 호환될 때만 publish
- QoS: subscriptions/publisher 모두 `SensorDataQoS`
- 결합: static/dynamic cell별 cost layer 결합

#### `/ad/perception/occupancy_grid`

- message: `nav_msgs/msg/OccupancyGrid`
- producer: combined node
- frame_id/timestamp/data: combined 결과와 동일한 compatibility alias
- QoS: `SensorDataQoS`

## 4. Adaptive Euclidean detector 상세

### 4.1 입력 crop

node는 finite filter 이후에도 x `[-4,100]`, y `[-25,25]`, z `[-1,3]` m의
포함 경계 ROI를 다시 적용하고 non-finite XYZ를 제거한다. 이 좌표는 입력 cloud
frame 기준이다. 기본 graph에서는 `lidar_link` 기준이며 `base_link`로 변환하지
않는다.

### 4.2 거리 적응 tolerance와 최소점 수

점의 XY range를 `r`이라 하면 다음 선형 보간을 사용한다.

```text
tolerance(r) = 0.45 + clamp(r / 45, 0, 1) * (1.60 - 0.45) [m]
min_points(r) = max(2, ceil(5 + clamp(r / 45, 0, 1) * (2 - 5)))
```

component의 채택 여부에는 component 전체의 평균 XY range로 계산한
`min_points`를 쓴다. 최대 cluster size는 20,000점이다. 현재 `use_height=false`라
neighbor 거리와 bucket 모두 XY만 사용하고 z 차이는 clustering distance에
포함하지 않는다.

### 4.3 hash bucket과 BFS

- bucket cell size는 거리와 무관하게 최대 tolerance인 1.60 m다.
- key는 `floor(x/1.60)`, `floor(y/1.60)`, z=0이다.
- 모든 finite point를 `unordered_map<Cell, vector<index>>`에 넣는다.
- 각 seed에서 deque BFS를 수행하고 현재 cell 주변 3 x 3 XY bucket을 검색한다.
- 두 점의 연결 threshold는 두 점 각각의 거리 적응 tolerance 평균이다.
- 방문된 component는 size filter에 실패해도 다시 분리하거나 재방문하지 않는다.

### 4.4 AABB와 12 m filter

cluster마다 XYZ min/max를 구하고 각 dimension을 최소 0.10 m로 clamp한다.
XY AABB diagonal `hypot(dimension_x, dimension_y)`가 12.0 m보다 크면
`DetectedObject`에서 제외한다. z dimension은 이 12 m 판정에 포함되지 않는다.

박스는 입력 좌표축에 정렬된 AABB다. 중심은 각 축 min/max의 중점이며,
orientation quaternion은 identity다. 따라서 `shape.dimensions.x/y`는 실제 객체의
heading 기준 length/width가 아니라 `lidar_link` 축 기준 extent다.

## 5. Euclidean DetectedObject -> tracker 계약

현재 detector가 실제로 채우는 field는 다음과 같다.

| field | Euclidean detector 값 | tracker에서의 처리 |
|---|---|---|
| array header | cluster input header 복사 | stamp에 처리, frame을 `odom`으로 TF 변환 |
| existence_probability | `1.0` | 먼저 내부 변환에서 `0.999`로 clamp한 뒤, `lidar_clustering` 채널이 신뢰하지 않아 기본 `0.75`로 대체 |
| classification | `UNKNOWN: 1.0` 한 개 | unknown PolygonTracker 생성. 채널 classification 신뢰 flag도 false |
| pose.position | AABB 중심 | input frame에서 `odom`으로 변환 |
| pose.orientation | identity quaternion | availability가 UNAVAILABLE이고 채널 orientation 신뢰도 false |
| has_position_covariance | default `false` | Autoware가 UNKNOWN object model covariance 생성 |
| pose covariance | default all zero | 위 모델 covariance로 대체/정규화 |
| orientation_availability | `UNAVAILABLE` | yaw covariance를 크게 만들고 unknown tracking에 사용 |
| has_twist | default `false` | 초기 twist 값은 zero; unknown velocity estimation은 tracker config에서 enabled |
| has_twist_covariance | default `false` | Autoware가 model covariance를 채움 |
| twist/covariance | default all zero | 모델/unknown tracker가 추정 |
| shape.type | `BOUNDING_BOX` | unknown은 shape를 그대로 사용; BEV association area에 사용 |
| shape.dimensions | AABB dx/dy/dz, 각 최소 0.10 m | tracking 및 prediction dimension으로 전달 |
| shape.footprint | 비어 있음 | bounding box에서는 필요하지 않음 |

Tracker launch의 detection01 channel 이름은 detector backend와 무관하게 현재
`lidar_clustering`으로 고정돼 있다. 따라서 learned detector를 선택해도 tracker는
현 launch 그대로라면 learned detector의 existence probability, extension,
classification, orientation을 신뢰하지 않는다. 이는 메시지 wire format 문제가
아니라 input-channel semantic configuration 문제다.

Tracker는 검출마다 새 UUID를 생성하고 association 후 track UUID를 유지한다.
`/tracked` 출력은 confidence를 통과한 track만 포함한다. IMM은 이 UUID를 state
key로 사용하므로 UUID 안정성은 tracker 이후 계약이다.

## 6. CenterPoint adapter가 맞춰야 할 계약

현재 Autoware CenterPoint backend는 project의 `object_detection.launch.py`에서
직접 `/ad/perception/lidar/points_xyzirc`를 받아
`/ad/perception/objects/detected`를 publish하도록 연결된다. 별도 adapter를
구현하거나 다른 CenterPoint runtime을 붙일 경우 최소한 다음을 만족해야 한다.

### 6.1 반드시 맞아야 하는 wire/runtime 계약

1. topic은 `/ad/perception/objects/detected`, type은 정확히
   `autoware_perception_msgs/msg/DetectedObjects`여야 한다.
2. `header.stamp`는 해당 LiDAR input scan의 stamp를 유지해야 한다. tracker TF와
   association time의 기준이며, tracker 출력과 IMM 입력 stamp로 이어진다.
3. `header.frame_id`는 비어 있지 않은 상대 frame이어야 하고, 해당 stamp에서
   `odom <- frame_id` TF가 존재해야 한다. 현재 classical detector와 같은
   `lidar_link`가 가장 직접적인 호환점이다.
4. QoS는 tracker의 reliable keep-last 1 subscription과 호환되는 reliable,
   volatile publisher여야 한다.
5. 각 object는 비어 있지 않고 유효 probability를 가진 classification 배열,
   finite pose, 0이 아닌 정상 quaternion, finite positive dimensions를 제공해야 한다.
6. shape는 현재 downstream prediction이 요구하는 `BOUNDING_BOX`여야 한다.
   prediction adapter는 다른 shape type을 array 단위로 거부한다.
7. `existence_probability`와 classification probability는 `[0,1]` 범위여야 한다.
8. orientation을 제공하면 quaternion과 `orientation_availability`가 서로
   일치해야 한다. twist를 제공하면 Autoware message의 twist는 object-local
   좌표 convention을 따라야 하며 `has_twist`/covariance flag도 실제 값과
   일치해야 한다.

### 6.2 현재 launch 때문에 보존되지 않는 learned semantics

`tracking.launch.py`가 detection01 channel을 무조건 `lidar_clustering`으로
설정하므로 classification, orientation, existence probability, learned box
extension은 tracker에서 신뢰되지 않는다. CenterPoint의 semantics를 실제로
활용하려면 향후 별도 STEP에서 backend에 따라 channel을 `lidar_centerpoint`로
선택하는 launch/config 변경이 필요하다. STEP 01에서는 변경하지 않았다.

### 6.3 downstream prediction이 최종 tracked output에 요구하는 값

CenterPoint 출력 자체가 아닌 tracker 출력 기준으로 IMM adapter는 다음을
array 단위로 검증한다.

- header frame 정확히 `odom`, 양수/단조 증가/0.5 s 이내 stamp
- 중복되지 않는 UUID
- `[0,1]` existence probability
- classification 한 개 이상, label `UNKNOWN..PEDESTRIAN`, probability `[0,1]`
- `BOUNDING_BOX`와 양수 finite x/y/z dimension
- unit에 가까운 finite pose quaternion
- finite pose covariance와 finite twist state

하나라도 실패하면 해당 object만 빼는 것이 아니라 그 tracked array의 prediction을
publish하지 않고 rejection diagnostic을 publish한다.

## 7. Launch/config 연결 구조와 주요 파일

| 역할 | 파일 |
|---|---|
| 전체 조합 및 backend branch | `ad_lidar_perception/launch/lidar_perception.launch.py` |
| 선택 schema/default | `ad_lidar_perception/ad_lidar_perception/selection.py`, `ad_lidar_perception/config/lidar_perception.yaml` |
| MORAI LiDAR 생산 경로 | `ad_morai_bridge/launch/bridge.launch.py`, `ad_morai_bridge/ad_morai_bridge/point_time_zero_node.py` |
| self crop launch/source/config | `ad_lidar_perception/launch/preprocessing.launch.py`, `ad_lidar_perception/src/preprocessing/self_crop_filter*.cpp`, `ad_lidar_perception/config/preprocessing/self_crop.yaml` |
| ground backend 선택 | `ad_lidar_perception/launch/ground_segmentation.launch.py` |
| Patchwork++ ROS wrapper | `../patchwork-plusplus/ros/src/GroundSegmentationServer.cpp` |
| ground config | `ad_lidar_perception/config/preprocessing/ground_segmentation.yaml`, `ad_lidar_perception/config/preprocessing/ransac_ground_filter.yaml` |
| finite/densifier 연결 | `ad_lidar_perception/launch/euclidean_clustering.launch.py`, `ad_lidar_perception/src/preprocessing/finite_point_filter*.cpp`, `ad_lidar_perception/src/preprocessing/pointcloud_densifier*.cpp` |
| Euclidean core/node/config | `ad_lidar_perception/src/clustering/adaptive_euclidean_cluster.cpp`, `ad_lidar_perception/src/clustering/adaptive_euclidean_cluster_node.cpp`, `ad_lidar_perception/config/clustering/adaptive_euclidean_cluster.yaml` |
| learned detector include | `ad_lidar_perception/launch/object_detection.launch.py`, `ad_lidar_perception/config/detectors/*.yaml` |
| Autoware tracker include/config | `ad_lidar_perception/launch/tracking.launch.py`, `ad_lidar_perception/config/tracking/autoware.yaml` |
| tracker implementation | `../autoware_universe/perception/autoware_multi_object_tracker/` |
| IMM adapter/core/config | `ad_lidar_perception/src/tracking/autoware_prediction_node.cpp`, `ad_lidar_perception/src/tracking/imm_predictor.cpp`, `ad_lidar_perception/config/tracking/prediction.yaml` |
| project prediction messages | `ad_interfaces/msg/PredictedObject*.msg`, `PredictedState.msg` |
| static occupancy | `ad_lidar_perception/src/occupancy_grid/occupancy_grid_node.cpp`, `ad_lidar_perception/config/occupancy_grid/static.yaml` |
| dynamic occupancy | `ad_lidar_perception/src/occupancy_grid/dynamic_occupancy_grid_node.cpp`, `ad_lidar_perception/src/occupancy_grid/dynamic_grid_builder.cpp`, `ad_lidar_perception/config/occupancy_grid/dynamic.yaml` |
| combined occupancy | `ad_lidar_perception/src/occupancy_grid/combined_occupancy_grid_node.cpp`, `ad_lidar_perception/src/occupancy_grid/grid_combiner.cpp`, `ad_lidar_perception/config/occupancy_grid/combined.yaml` |
| build/dependencies | `ad_lidar_perception/CMakeLists.txt`, `ad_lidar_perception/package.xml` |
| Autoware version/artifact lock | `ad_lidar_perception/config/autoware_perception.lock.yaml` |

## 8. 외부 dependency와 version lock

- `autoware_perception_msgs` 정확히 1.13.0
- `autoware_multi_object_tracker` 0.51.0
- optional learned detector packages 0.51.0:
  `autoware_lidar_centerpoint`, `autoware_lidar_transfusion`,
  `autoware_bevfusion`
- optional RANSAC: `autoware_ground_segmentation` 0.51.0
- default ground: in-workspace `patchworkpp`
- project prediction message: in-repo `ad_interfaces`
- TF: `tf2`, `tf2_ros`, `tf2_geometry_msgs`, `tf2_sensor_msgs`

`AD_WITH_AUTOWARE=ON`이 기본이며, 이때 exact message version을 찾지 못하면 detector와
prediction adapter build가 실패한다. Learned model artifact는 lock file과
`AD_DATA_DIR/models/autoware` 아래 hash/provenance 검증을 통과해야 runtime에 포함된다.

현재 `ad_lidar_perception/package.xml`은 `autoware_perception_msgs`와
`autoware_ground_segmentation`은 선언하지만, runtime에 include하는
`autoware_multi_object_tracker` 및 learned detector package는 직접
`exec_depend`로 선언하지 않는다. 전체 workspace와 provenance lock에는 package가
있지만, package manifest만으로는 이 runtime dependency가 완결되어 있지 않다.

## 9. 좌표계 요약

- vehicle convention: x forward, y left, z up
- `base_link`: rear axle center at ground
- `rear_axle_link`: `base_link` 위 z=0.3685 m
- `lidar_link`: rear axle parent 기준 x=1.15, y=0, z=1.4 m, RPY=0
- raw/crop/ground/nonground/detected: 기본값 `lidar_link`
- tracked/predicted: `odom`
- occupancy/drivable mask: `base_link`

Classical AABB는 `lidar_link` 축 정렬이며 object heading을 나타내지 않는다.
Tracker에서 `odom`으로 변환된 뒤에도 orientation availability가 UNAVAILABLE인
의미는 유지된다.

## 10. 미해결 항목과 제한

1. repo별 Notion hub를 읽지 못했으므로 대회 운영에서 별도로 고정한 topic/QoS/TF
   정책이 있는지는 확인하지 못했다.
2. 이 STEP은 코드 정적 조사다. 현재 commit의 전체 default graph를 새로 실행해
   모든 topic의 live `ros2 topic info --verbose`를 캡처하지 않았다. 저장된 bag의
   일부 topic은 이전 실행 구성의 관측값이며 source 계약의 근거로 사용하지 않았다.
3. RANSAC debug ground의 zero timestamp는 기존 benchmark 문서의 실제 관측 결과다.
   RANSAC upstream 구현의 보장 계약은 아니다.
4. learned backend 모델 artifact 존재 여부와 GPU/TensorRT runtime 성공 여부는 이
   STEP의 범위가 아니다.
5. 현재 dynamic OccupancyGrid는 future trajectory를 rasterize하지 않고 현재
   footprint만 포함한다. 시간별 future states는 planner가 원 prediction message를
   직접 사용할 때만 보존된다.
6. detector debug clusters에는 12 m filter로 object에서 제외된 component도 남아
   있으므로 `/clusters` 점 수와 `/detected` object 수는 일대일 대응하지 않는다.
7. Autoware tracker/learned detector의 runtime dependency가 package manifest에
   직접 선언되지 않은 상태다. 이 STEP에서는 manifest를 변경하지 않았다.

## 11. Rollback

이 STEP의 유일한 repository 변경은 이 문서 추가다. rollback은
`docs/perception/current_pipeline_contract.md`를 삭제하면 된다. 운영 source,
launch, config, message 정의에는 변경이 없다.
