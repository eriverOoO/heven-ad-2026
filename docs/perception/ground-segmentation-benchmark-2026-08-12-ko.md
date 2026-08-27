# 지면 분할 알고리즘 비교 평가 보고서

평가일: 2026-08-12 (Asia/Seoul)

평가 대상:

- Classic Patchwork, `max_range=80 m`
- Patchwork++, `max_range=80 m`
- Patchwork++, `max_range=105 m` 실험안
- Autoware RANSAC ground filter

## 한 문장 결론

현재 확보된 평지·정차 MORAI bag만 기준으로 보면 **Classic Patchwork가
가장 빠르고 보수적인 기본값**이며, 현재 설정의 Patchwork++ 80 m는 실제
actor 회수율을 높이지 못했다. Patchwork++ 105 m는 작은 개선 가능성을
보였지만 추가 검출과 지연도 늘었고, RANSAC은 actor 회수율이 가장 높았지만
지연과 ground timestamp 계약 문제가 있다.

## 최종 비교표

| 항목 | Classic 80 m | Patchwork++ 80 m | Patchwork++ 105 m | RANSAC |
|---|---:|---:|---:|---:|
| 비교 프레임 | 264 | 261 | 264 | 261 |
| 출력률 | 6.376 Hz | 6.359 Hz | 6.376 Hz | 6.456 Hz |
| 유효 ground 비율 | 81.75% | 80.60% | 82.47% | 직접 비교 불가 |
| 유효 nonground 비율 | 15.22% | 19.40% | 17.53% | 20.11% |
| 입력점 분할 완전성 | 96.98% | 100.00% | 100.00% | 직접 비교 불가 |
| 프레임당 nonground 중앙값 | 1,516.5 | 2,106 | 1,851 | 2,547 |
| 분할 지연 p50 | 3.42 ms | 5.04 ms | 5.12 ms | 9.25 ms |
| 분할 지연 p95 | 4.31 ms | 6.63 ms | 6.71 ms | 18.83 ms |
| cluster 점 중앙값 | 428.5 | 450 | 450 | 450 |
| 프레임당 검출 평균 | 6.09 | 6.56 | 6.72 | 6.68 |
| actor 매칭 수 | 31/104 | 31/104 | 34/104 | 41/104 |
| actor recall proxy | 29.81% | 29.81% | 32.69% | 39.42% |
| 매칭 중심 거리 평균 | 0.623 m | 0.613 m | 0.612 m | 0.501 m |

이 표만 볼 때 주의할 점:

- 출력률은 알고리즘의 최대 처리 능력이 아니라 해당 재생 구간에서 실제로
  출력한 비율이다.
- `actor 매칭 수`는 고유 객체 수가 아니다. 각 프레임에 등장한 actor를 모두
  더한 actor-frame instance 수이다.
- RANSAC의 검출 지연은 cropped 입력부터 측정했고 나머지는 raw 입력부터
  측정했으므로 완전히 동등한 end-to-end 비교가 아니다.
- MORAI actor 목록에는 벽, 연석, 가드레일, 기둥 같은 정적 구조물이 포함되지
  않는다. LiDAR 군집이 이런 구조물을 검출하면 안전상 유용할 수 있지만
  actor precision proxy에서는 unmatched detection으로 계산된다.

## 무엇을 확인하기 위한 평가인가

이번 평가는 다음 질문을 분리해서 답하도록 설계했다.

1. 입력 LiDAR 프레임을 빠뜨리지 않고 처리하는가?
2. self-crop 이후 유효점을 ground와 nonground로 얼마나 분류하는가?
3. 지면 분할 자체가 얼마나 지연되는가?
4. 지면 분할 차이가 군집화 점 수와 검출 객체 수에 어떻게 전파되는가?
5. 늘어난 검출이 실제 MORAI actor 회수로 이어지는가?
6. 각 출력 메시지가 downstream이 요구하는 timestamp와 PointCloud2 계약을
   지키는가?

이 질문들을 하나의 숫자로 합치지 않았다. 지연, 분할 완전성, actor 회수율,
unmatched cluster는 서로 다른 성질의 지표이기 때문이다.

## 시험 환경

- OS: Ubuntu 22.04
- Kernel: Linux 6.8.0-136-generic
- CPU: Intel Core i5-1240P
- 물리 코어/논리 CPU: 12/16
- ROS: ROS 2 Humble
- HEVEN 기준 commit:
  `45cb3610f3c0e09c85296f406e940e15462e766c`
- Patchwork++ commit:
  `3e6903a1d5537a4cc2ace897b0bbb98a92d6014c`
- Patchwork++ upstream release: v1.4.1

현재 shell 환경에는 ROS 1 Noetic 경로도 섞여 있어 경고가 출력됐다. 실제
runtime은 빌드된 ROS 2 Humble workspace를 사용했지만, 논문 또는 공식 성능
수치로 만들 때에는 순수 Humble shell과 고정 CPU governor에서 다시 반복해야
한다.

## 데이터셋

사용한 bag:

```text
bags/static_20260805_003151
```

bag 특성:

- MCAP, 2.6 GiB
- 전체 길이 242.322초
- `/ad/sensors/lidar/points`: 1,778개
- 전체 평균 LiDAR 주기: 약 7.34 Hz
- LiDAR 모델: MORAI VLP-16
- LiDAR frame: `lidar_link`
- launch가 계산한 sensor height: 1.7685 m
- raw cloud 레코드 수: 프레임당 28,800개
- 평가 구간 raw finite XYZ 중앙값: 약 14,844개

28,800개와 14,844개의 차이는 대부분 NaN 또는 no-return 레코드다. 그래서
raw PointCloud2의 `width × height`만 세면 알고리즘 간 nonground 차이가 실제보다
과장된다. 최종 비율은 전부 finite XYZ만 세어 계산했다.

최종 평가는 약 41초 구간을 사용했다. 프로세스 기동 시점 차이 때문에 후보별
기록 프레임이 261~264개로 조금 달랐다. 직접 비교 수치는 가능한 경우 source
header timestamp가 같은 프레임의 교집합만 사용했다.

이 bag의 한계도 명확하다.

- 파일 이름과 vehicle status상 정차 또는 거의 정차 상태다.
- 평지 중심이다.
- 강한 pitch/roll, 경사로 정상부, 과속방지턱, 포트홀, 터널 진출입을 검증하지
  못한다.
- 실제 LiDAR가 아니라 시뮬레이션이다.

따라서 이번 결과는 “현재 평지 시뮬레이션에서의 기준선”이지 일반적인 우열
증명이 아니다.

## 실제 시험 파이프라인

Classic과 Patchwork++는 현재 production launch 경로를 그대로 사용했다.

```text
recorded raw LiDAR
  -> self-crop
  -> ground segmentation
  -> finite XYZ filter
  -> adaptive Euclidean clustering
  -> DetectedObjects
```

- MORAI instantaneous scan 정책에 따라 deskew는 껐다.
- 현재 기본값과 동일하게 gravity leveling도 껐다.
- 각 후보를 서로 다른 ROS domain에서 단독 실행했다.
- bag 재생률은 1.0이었다.
- 후보를 동시에 돌리지 않아 후보 간 CPU 경쟁을 피했다.

RANSAC은 bag에 이미 기록된 `/ad/perception/lidar/cropped`를 입력받고 이후
동일한 finite filter와 clustering을 사용했다. 입력점 내용은 production
self-crop 출력과 같지만 raw-to-crop 과정의 runtime 부하는 포함되지 않는다.

## 알고리즘별 설정

### Classic Patchwork 80 m

```yaml
algorithm: patchwork
sensor_height: 1.7685
num_iter: 3
num_lpr: 20
num_min_pts: 5
th_seeds: 0.35
th_dist: 0.18
min_range: 3.0
max_range: 80.0
uprightness_thr: 0.707
```

현재 local classic 구현은 `min_range..max_range` 밖의 점을 CZM에 넣지 않으며
ground와 nonground 어느 쪽에도 반환하지 않는다. 그래서 finite partition
completeness가 96.98%다.

### Patchwork++ 80 m

공통 파라미터에 다음이 추가된다.

```yaml
algorithm: patchworkpp
th_seeds_v: 0.25
th_dist_v: 0.1
max_range: 80.0
```

Patchwork++는 범위 밖 점도 nonground로 반환한다. 따라서 finite 입력점을
ground 또는 nonground 중 하나에 100% 보존한다. adaptive elevation/flatness,
temporal ground revert도 사용한다.

현재 ROS wrapper는 intensity를 지원하지 않아 reflected-noise removal을 강제로
끄고 있다. 즉 upstream Patchwork++ 기능을 완전히 활성화한 상태는 아니다.

### Patchwork++ 105 m

80 m 설정에서 `max_range`만 105 m로 바꿨다.

이 실험을 추가한 이유는 범위 계약 불일치 때문이다.

- ground segmentation: 반경 80 m까지만 분할
- clustering ROI: 전방 x=100 m, 좌우 y=25 m까지 허용

80 m 밖 점은 Patchwork++ 80 m에서 자동 nonground가 된 뒤 clustering으로
넘어갈 수 있다. 105 m는 ROI 모서리까지 고려하여 이 구간도 실제 ground fitting
대상으로 만든 값이다. 이 설정은 평가만 했으며 production YAML에는 적용하지
않았다.

### Autoware RANSAC

```yaml
unit_axis: z
max_iterations: 200
min_trial: 1000
min_points: 500
outlier_threshold: 0.15
plane_slope_threshold: 12.0
voxel_size_x: 0.10
voxel_size_y: 0.10
voxel_size_z: 0.10
height_threshold: 0.18
debug: true
```

RANSAC의 일반 출력은 nonground이고 ground는 debug용 voxel-downsampled inlier다.
따라서 ground+nonground가 원본 전체와 같아야 한다는 partition 비교를 적용할
수 없다.

## 지표를 계산한 방법

### 프레임 정렬

ROS 메시지를 MCAP에서 역직렬화한 뒤 다음 값을 key로 사용했다.

```text
header.stamp.sec * 1,000,000,000 + header.stamp.nanosec
```

같은 key가 있는 입력과 출력만 한 프레임으로 매칭했다. recorder가 메시지를
받은 순서나 wall-clock 근접성만으로 연결하지 않았다.

### 출력률

```text
output_rate = (N - 1) / (마지막 header stamp - 첫 header stamp)
```

이 값은 해당 재생에서의 유지율이다. 알고리즘 최대 Hz를 측정하려면 입력률을
단계적으로 올리는 별도 saturation benchmark가 필요하다.

### point 비율

```text
finite_ground_ratio = finite ground / finite cropped
finite_nonground_ratio = finite nonground / finite cropped
partition = (finite ground + finite nonground) / finite cropped
```

XYZ 중 하나라도 NaN 또는 inf면 finite point에서 제외했다.

### 지연

MCAP recorder receive timestamp를 사용했다.

```text
segmentation latency
  = max(ground 수신시각, nonground 수신시각) - cropped 수신시각

detection latency
  = DetectedObjects 수신시각 - raw LiDAR 수신시각
```

RANSAC ground stamp는 전부 0이라 segmentation latency는 nonground만 사용했다.
또한 RANSAC 검출 지연은 cropped부터 계산했다.

이 지연에는 순수 알고리즘 계산 외에 ROS executor scheduling, 직렬화, DDS 전송,
recorder scheduling도 포함된다. CPU 함수 실행시간 자체로 해석하면 안 된다.

### 검출 수와 안정성

- `objects/frame`: 해당 프레임 DetectedObjects 배열 길이
- `total detections`: 모든 프레임 배열 길이의 합
- `adjacent count delta`: 연속 두 프레임 object 수 차이의 절댓값 평균

count delta가 낮으면 수량 변화는 안정적이지만 같은 객체 identity가 유지된다는
뜻은 아니다. tracking 안정성은 별도 ID 기반 지표가 필요하다.

## MORAI actor와 검출을 매칭한 방법

ground truth source는 `/ad/dev/objects`다. 각 검출 timestamp에 가장 가까운
actor 메시지와 `odom -> base_link` transform을 선택했다. 이 bag에서
`map -> odom`은 identity다.

actor map 좌표를 base_link로 변환하고, rear axle 기준으로 전방 1.15 m에 있는
LiDAR offset을 적용해 `lidar_link` 좌표를 계산했다. 그 다음 clustering ROI에
있는 actor만 남겼다.

```text
x: -4 .. 100 m
y: -25 .. 25 m
```

검출 중심과 actor 중심의 2D 거리를 모두 계산한 뒤 가장 가까운 쌍부터
one-to-one greedy matching했다. 임계값은 1.5, 2.5, 4.0 m 세 가지를 계산했고
주 표는 2.5 m다.

### 왜 이것을 AP가 아니라 proxy라고 부르는가

- bounding-box 3D IoU가 아니라 중심 거리다.
- Euclidean detector가 generic/unknown object를 내므로 class 평가를 하지 않았다.
- occlusion과 실제 LiDAR visibility를 판정하지 않았다.
- MORAI actor 목록에 벽, 연석, 기둥 등 정적 구조물이 없다.
- ROI 안에 actor가 있어도 실제 scan에서 충분한 점이 보인다고 보장할 수 없다.

그래서 precision proxy가 약 2%라고 해서 실제 오검출률이 98%라는 뜻은 절대
아니다. 이 데이터에서는 recall proxy가 상대 비교에 더 유용하다.

## 지면 분할 결과 해석

### Classic 대 Patchwork++ 80 m

- ground Jaccard: 95.56%
- Classic ground 중 Patchwork++도 ground로 본 비율: 97.08%
- Patchwork++ ground 중 Classic도 ground로 본 비율: 98.39%

두 알고리즘의 ground 판단은 매우 비슷하다. 차이의 큰 부분은 범위 밖 점을
버리는가, nonground로 보존하는가에 있다.

Patchwork++ 80 m는 classic보다 finite nonground를 4.18 percentage point 더
남겼다. 하지만 actor 매칭은 둘 다 31개로 같았다. 즉 이 bag에서는 추가
nonground가 actor 회수로 이어졌다는 증거가 없다.

### Patchwork++ 105 m

105 m에서는 80~105 m 구간 일부를 실제 ground로 fitting하므로 80 m 버전보다
nonground 비율이 오히려 19.40%에서 17.53%로 내려갔다.

Classic 대비:

- actor-frame match: 31 -> 34
- recall proxy: 29.81% -> 32.69%
- total detection: 1,609 -> 1,774
- median segmentation latency: 3.42 -> 5.12 ms

세 개의 추가 match는 긍정적이지만, 165개의 추가 frame-level detection과 함께
발생했다. 시나리오를 늘리지 않고 이 값만으로 production 기본값을 바꾸기에는
근거가 약하다.

### RANSAC

RANSAC은 같은 actor 104 instance 중 41개를 매칭했다. 1.5 m에서도 38개를
매칭했고 2.5 m에서 41개, 4.0 m에서도 그대로 41개였다. 넓은 threshold를
우연히 선택해서 좋아 보인 결과는 아니다.

반면 지연은 다음과 같다.

- Classic p50/p95: 3.42/4.31 ms
- RANSAC p50/p95: 9.25/18.83 ms

RANSAC median은 classic의 약 2.70배, p95는 약 4.37배다. VLP-16의 약 100 ms
scan period 안에는 충분히 들어오지만 다른 perception 부하가 함께 있을 때의
여유는 작아진다.

또한 RANSAC debug ground 261개 모두 stamp가 0이었다. 현재 `/ground`를 RViz만
본다면 당장 nonground detection은 동작하지만, exact-time pair나 downstream
fusion에는 사용할 수 없다.

## downstream 영향

### 평균 검출 수

- Classic: 6.09/frame
- Patchwork++ 80 m: 6.56/frame
- Patchwork++ 105 m: 6.72/frame
- RANSAC: 6.68/frame

Patchwork++ 80 m는 common frame에서 classic보다 평균 0.487개 더 검출했지만
actor match는 증가하지 않았다. 두 알고리즘의 object count가 같은 프레임은
59.0%였다.

Patchwork++ 105 m는 classic보다 평균 0.625개 더 검출했고 actor match가 3개
증가했다. object count가 같은 프레임은 52.3%였다.

### 검출 지연

- Classic raw-to-detected p50: 7.53 ms
- Patchwork++ 80 m: 9.19 ms
- Patchwork++ 105 m: 9.90 ms
- RANSAC cropped-to-detected: 10.96 ms

p95가 모든 후보에서 65~77 ms로 높다. segmentation 자체 p95보다 훨씬 크므로
지면 분할보다는 전체 launch의 scheduling 또는 recorder timing 영향일 가능성이
크다. profiling trace 없이 특정 노드의 문제로 단정하면 안 된다.

## threshold 민감도

| 알고리즘 | 1.5 m match | 1.5 m recall | 2.5 m match | 2.5 m recall | 4.0 m match |
|---|---:|---:|---:|---:|---:|
| Classic | 27 | 25.96% | 31 | 29.81% | 31 |
| Patchwork++ 80 m | 27 | 25.96% | 31 | 29.81% | 31 |
| Patchwork++ 105 m | 30 | 28.85% | 34 | 32.69% | 34 |
| RANSAC | 38 | 36.54% | 41 | 39.42% | 41 |

어느 후보도 2.5 m에서 4.0 m로 넓힐 때 match가 늘지 않는다. 따라서 후보 순위가
2.5 m라는 특정 임계값 선택 때문에 만들어진 것은 아니다.

## 알고리즘별 판단

### Classic Patchwork

장점:

- 분할 지연이 가장 낮다.
- Patchwork++ 80 m와 actor recall이 같다.
- 추가 unmatched detection이 적다.
- ground/nonground timestamp가 정상이다.

단점:

- radial range 밖 finite point 약 3%가 양쪽 출력에서 사라진다.
- Patchwork++가 목표로 하는 경사·요철 환경은 이번 bag으로 검증하지 못했다.

### Patchwork++ 80 m

장점:

- finite point를 100% 보존한다.
- classic과 ground 판단이 95% 이상 일치한다.
- 5~7 ms 수준이라 VLP-16 실시간 budget에는 충분하다.

단점:

- 이 bag에서는 actor recall 개선이 없다.
- nonground와 detection은 늘지만 match는 늘지 않았다.
- median 분할 지연이 classic보다 약 47% 높다.
- clustering ROI 100 m와 segmentation range 80 m가 불일치한다.

### Patchwork++ 105 m

장점:

- 범위 계약 불일치를 줄인다.
- classic보다 actor-frame match가 3개 늘었다.
- finite point를 전부 보존한다.

단점:

- recall 절대 개선은 2.88 percentage point에 불과하다.
- common frame total detection이 classic보다 약 10.3% 많다.
- median 분할 지연이 classic보다 약 49% 높다.
- 한 개 정적 bag 결과뿐이다.

### RANSAC

장점:

- 이번 bag의 actor recall proxy가 가장 높다.
- matched center 거리도 가장 작다.
- production self-cropped 입력에서는 finite filter와 clustering까지 정상 처리됐다.

단점:

- 지연이 가장 크고 변동도 크다.
- debug ground stamp가 전부 0이다.
- ground가 downsampled debug cloud라 완전한 ground/nonground partition이 아니다.
- 단일 dominant plane 방식은 경사와 다중 노면에서 취약할 가능성이 있다.

## 목적별 선택

| 목적 | 선택 | 이유 |
|---|---|---|
| 현재 평지에서 가장 낮은 지연 | Classic | 같은 recall에 3.42 ms median |
| 입력 finite point 완전 보존 | Patchwork++ | partition 100% |
| 이 정적 bag에서 actor recall 최대화 | RANSAC | 41/104 match |
| 다음 Patchwork++ 튜닝 후보 | Patchwork++ 105 m | 범위 일치와 작은 recall 개선 |
| 지금 production 기본값 | Classic | 현재 증거상 가장 보수적인 균형 |
| 경사·요철 데이터 확보 후 재검증 | Patchwork++ 105 m | 알고리즘 설계 목적에 맞는 후보 |

## 권고 사항

### 지금 할 결정

현재 데이터만으로는 Patchwork++ 80 m를 기본값으로 채택할 이유가 부족하다.
평지 대회 환경을 우선한다면 classic으로 유지 또는 복귀하는 것이 타당하다.

Patchwork++를 계속 시험한다면 80 m보다 105 m를 실험 profile로 분리하는 편이
낫다. production YAML을 바로 105 m로 바꾸기보다 별도 config로 A/B replay를
누적해야 한다.

RANSAC은 ground header stamp를 입력 stamp로 복원하는 wrapper 또는 upstream
수정 전까지 `/ground` 계약을 요구하는 기본 backend로 승격하면 안 된다.

### 다음 데이터셋에 반드시 포함할 장면

- 평지 open road, 차량 없음/있음
- 오르막, 내리막, 정상부 전환
- banked turn
- 연석, 과속방지턱, 램프, 포트홀
- 터널 입구와 출구
- 정차와 주행
- 근거리 보행자/자전거
- 80 m 이상 원거리 차량
- 노이즈 또는 invalid return 증가 상황
- 실제 VLP-16 bag

### production 승격 조건 제안

- 목표 LiDAR rate에서 input:nonground가 1:1
- segmentation p95가 scan period보다 충분히 작음
- malformed PointCloud2와 stamp 0이 없음
- 거리 구간별·객체 종류별 recall 개선
- static obstacle annotation 또는 수동 검토를 통한 unmatched cluster 분석
- tracking ID continuity 평가
- 각 시나리오를 최소 3회 반복해 중앙값과 분산 보고

Patchwork++는 어려운 지형에서 classic보다 recall 또는 ground IoU를 반복적으로
개선하면서 unmatched cluster를 과도하게 늘리지 않을 때 승격하는 것이 좋다.

## 재현 자료

기계 판독용 결과:

```text
docs/perception/ground-segmentation-benchmark-2026-08-12.csv
```

상세 영문 기술 기록:

```text
docs/perception/ground-segmentation-benchmark-2026-08-12.md
```

평가 중 생성한 MCAP은 `/tmp/heven_*`에 약 1.7 GiB로 남아 있으며 repository에는
포함하지 않았다.

기록한 핵심 토픽:

```text
/ad/sensors/lidar/points
/ad/perception/lidar/cropped
/ad/perception/lidar/ground
/ad/perception/lidar/nonground
/ad/perception/lidar/nonground_finite
/ad/perception/lidar/clusters
/ad/perception/objects/detected
```

## 이 보고서로 설명할 수 없는 것

- 실제 ground/non-ground point accuracy
- 표준 3D detection AP
- 객체 종류별 분류 정확도
- 가려진 actor의 검출 가능 여부
- 주행 중 deskew 영향
- 경사와 터널에서의 일반화
- 실제 센서 노이즈와 날씨 영향
- 장시간 CPU thermal throttling
- run-to-run 분산과 confidence interval

따라서 이 자료의 올바른 용도는 “현재 default를 무엇으로 둘지”와 “다음 실험을
무엇으로 설계할지”를 결정하는 것이다. 알고리즘의 일반적 우월성을 주장하는
자료로 사용해서는 안 된다.
