# HEVEN Stanley / Profile Stanley / Optuna 상세 가이드

작성 기준일: 2026-07-30
대상 체크아웃: `src/heven_ad_2026`
대상 브랜치: `dev`

이 문서는 지금까지 논의한 경로 추종, 종방향 차량 프로파일링, 브레이크 PID,
ROS 2/MORAI Optuna 자동 튜닝을 현재 소스 코드 기준으로 한 번에 설명한다.
설계 아이디어와 실제 구현을 섞지 않기 위해 다음 세 가지를 구분한다.

- **현재 구현됨**: 지금 소스에서 실제 실행되는 로직
- **환경만 검증됨**: 설치, 네트워크, PostgreSQL 공유까지만 검증된 상태
- **아직 구현되지 않음**: 논의했지만 현재 코드에는 없는 기능

---

## 1. 가장 짧은 결론

현재 Stanley는 두 개의 ROS 백엔드 이름으로 유지된다.

| 백엔드 | 횡방향 조향 | 경로 곡률 속도 계획 | 종방향 가감속 모델 | 최종 페달 |
|---|---|---|---|---|
| `stanley` | Stanley | 사용 | 고정 상한 `2.0 / 2.0 m/s²` | throttle/brake 분리 PID |
| `profile_stanley` | 같은 Stanley | 사용 | 측정 테이블과 고정 상한 중 더 작은 값 | 같은 throttle/brake 분리 PID |

중요한 점은 다음과 같다.

1. `profile_stanley`는 별도의 횡제어 알고리즘이 아니다. 같은
   `StanleyController`에 측정된 IONIQ 5 종방향 프로파일을 넣은 파생
   백엔드다.
2. 일반 `stanley`도 단순히 58.5 km/h만 명령하지 않는다. 전역 경로 곡률,
   횡가속도 상한, 고정 가감속 상한을 사용해 경로 전체 속도 프로파일을
   미리 계산한다.
3. `profile_stanley`의 측정 데이터는 전역경로 전체에서 커브 전 감속과
   커브 후 재가속이 가능한 공간 기반 목표속도를 미리 계산하는 데 쓰인다.
   현재 코드에는
   `목표 가속도 -> IONIQ throttle/brake map -> 페달 명령` feed-forward가
   없다. 최종 `accel`과 `brake`는 여전히 PID가 만든다.
4. `acceleration_mps2=5.0`, `deceleration_mps2=2.0`은 실제 가속도를
   해당 값으로 고정하지 않는다. 경로 목표속도 계획에 허용하는 상한이며
   Profile Stanley에서는 측정값이 더 작으면 측정값을 쓴다.
5. Optuna는 Stanley 코드를 Python에 복사해 재현하지 않는다. 실제
   `ad_planner` C++ 노드의 ROS 파라미터를 변경하고 컨트롤러를 재생성한
   뒤, 실제 MORAI 주행 결과를 목적함수로 사용한다.
6. 3대 컴퓨터의 Python/Optuna/PostgreSQL 공유 환경과 동시 study 접근은
   검증됐다. 그러나 3대 MORAI를 동시에 사용한 실제 Profile Stanley
   최적화 trial은 아직 시작하지 않았다.

---

## 2. 지금까지의 작업 맥락

### 2.1 출발점

리팩터링된 HEVEN 스택에서 이미 확인한 전제는 다음과 같다.

- 로컬라이제이션은 동작했다.
- 전역 경로 추종도 기본적으로 동작했다.
- 다음 핵심 목표는 동적 장애물 회피와 DWA/Stanley 조합의 튜닝이었다.
- 기존 코드를 지우지 않고 비교 가능한 여러 제어 버전을 남기기로 했다.

장애물 회피는 DWA/OGM 쪽 문제이고, 이 문서에서 다루는 Stanley는 전역
경로 추종과 목표속도 생성을 담당한다. 실제 운용 스택에서는 둘이 함께
동작하지만, Stanley 파라미터를 공정하게 비교하는 Optuna 실행에서는
일시적으로 perception 개입을 끈다.

### 2.2 종방향 프로파일링을 추가한 이유

기존 곡률 속도 계획은 다음처럼 차량의 감속 능력을 하나의 숫자로
가정했다.

```text
어느 속도에서든 최대 감속 계획값 = 2.0 m/s²
```

하지만 MORAI IONIQ 5에서 동일한 페달 명령도 속도에 따라 가속도와
감속도가 달랐다. 그래서 다음 데이터를 측정하도록 프로파일러를 만들었다.

- 속도별 accelerator 명령과 실제 가속도
- 속도별 brake 명령과 실제 감속도
- 명령 echo 지연
- 감속 시작 지연
- 반복 측정 분산과 품질 플래그

그다음 기존 Stanley를 없애지 않고 다음 두 버전을 남겼다.

- `stanley`: 고정 가감속 상한을 사용하는 비교 기준
- `profile_stanley`: 실제 측정된 속도별 가감속 능력과 제동 지연을
  사용하는 버전

### 2.3 Optuna를 도입한 이유

이전 `AdaptiveSampler` 방식은 프로젝트 내부 규칙으로 다음 후보를
생성하는 커스텀 샘플러였다. 지금 구조는 Optuna의 study/trial/storage
모델로 바뀌었다.

Optuna를 쓰면 다음이 가능하다.

- trial과 파라미터, 결과, 상태를 표준 구조로 저장
- TPE sampler 사용
- PostgreSQL을 통한 여러 worker의 동시 trial 배정
- heartbeat로 죽은 worker의 stale trial 감지
- 같은 실험 조건인지 fingerprint로 확인
- 실행 중인 trial과 너무 비슷한 후보를 피하는 `constant_liar`

---

## 3. 코드 위치 지도

| 역할 | 파일 |
|---|---|
| Stanley 설정과 클래스 | [`ad_control/include/ad_control/lateral/stanley.hpp`](../ad_control/include/ad_control/lateral/stanley.hpp) |
| Stanley 매 주기 계산 | [`ad_control/src/lateral/stanley.cpp`](../ad_control/src/lateral/stanley.cpp) |
| 백엔드 선택과 ROS 파라미터 변환 | [`ad_control/src/lateral/path_tracking_controller_factory.cpp`](../ad_control/src/lateral/path_tracking_controller_factory.cpp) |
| 경로 진행 인덱스 추적 | [`ad_control/src/path/route_progress_tracker.cpp`](../ad_control/src/path/route_progress_tracker.cpp) |
| 곡률 및 경로 속도 프로파일 | [`ad_control/src/path/route_speed_profile.cpp`](../ad_control/src/path/route_speed_profile.cpp) |
| 종방향 PID | [`ad_control/src/longitudinal/pid.cpp`](../ad_control/src/longitudinal/pid.cpp) |
| 현재 Stanley 파라미터 | [`ad_planner/config/planner.yaml`](../ad_planner/config/planner.yaml) |
| 차량 치수와 제어점 | [`ad_description/config/vehicle_parameters.yaml`](../ad_description/config/vehicle_parameters.yaml) |
| planner의 hold/reset 튜닝 서비스 | [`ad_planner/src/planner/planner_node.cpp`](../ad_planner/src/planner/planner_node.cpp) |
| 프로파일 CSV를 YAML 배열로 변환 | [`scripts/export_longitudinal_profile.py`](../scripts/export_longitudinal_profile.py) |
| Optuna 탐색 공간과 TPE | [`ad_tuning/ad_tuning/search.py`](../ad_tuning/ad_tuning/search.py) |
| 실제 ROS/MORAI trial 실행 | [`ad_tuning/ad_tuning/ros_morai_runner.py`](../ad_tuning/ad_tuning/ros_morai_runner.py) |
| Optuna orchestration | [`ad_tuning/ad_tuning/tuner_node.py`](../ad_tuning/ad_tuning/tuner_node.py) |
| 목적함수 | [`ad_tuning/ad_tuning/objective.py`](../ad_tuning/ad_tuning/objective.py) |
| 로컬 결과 저장 | [`ad_tuning/ad_tuning/storage.py`](../ad_tuning/ad_tuning/storage.py) |
| 튜닝 실행 파라미터 | [`ad_tuning/config/tuning.yaml`](../ad_tuning/config/tuning.yaml) |
| 튜닝 전용 launch | [`ad_tuning/launch/profile_stanley_morai_optuna.launch.py`](../ad_tuning/launch/profile_stanley_morai_optuna.launch.py) |

---

## 4. 전체 제어 흐름

현재 경로 추종 한 주기의 흐름은 다음과 같다.

```text
MORAI / localization pose, vehicle speed
        |
        v
rear-axle base_link pose
        |
        +-- 차량 앞축 제어점으로 3.0 m 이동
        |
        v
전역 경로에서 진행 인덱스 검색
        |
        +-- 위치 jump / 경로 이탈 / 역방향 검사
        |
        v
속도 비례 조향 lookahead 계산
        |
        v
heading error + cross-track error
        |
        v
Stanley steering 계산, 조향각/조향속도 제한

전역 경로
        |
        +-- 거리 기준 곡률 계산
        +-- 횡가속도 제한으로 곡선 속도 상한 계산
        +-- backward pass로 미리 감속
        +-- forward pass로 가능한 만큼 재가속
        +-- Profile Stanley만 측정 지연만큼 감속 지점을 앞당김
        |
        v
경로 인덱스별 목표속도 상한
        |
        +-- 사용자/mission 목표속도
        +-- launch ramp 목표속도
        +-- 경로 곡률 목표속도
        +-- Profile Stanley만 매 주기 목표속도 변화율 제한
        |
        v
최종 target speed
        |
        v
분리된 throttle/brake PID
        |
        v
MORAI CtrlCmd accel / brake / steering
```

---

## 5. 백엔드 선택

`path_tracking.backend`가 다음 문자열 중 하나여야 한다.

```yaml
path_tracking.backend: "stanley"
path_tracking.backend: "profile_stanley"
```

Factory에서:

- `stanley`는 `StanleyController`를 생성한다.
- `profile_stanley`는 `ProfileStanleyController`를 생성한다.
- `ProfileStanleyController` 생성자는 측정 프로파일을
  `StanleyConfig.speed_profile.longitudinal_profile`에 넣고 같은
  `StanleyController` 생성자로 들어간다.

따라서 두 Stanley의 CTE 계산, heading error 계산, 조향식, 진행 인덱스,
PID 구현은 완전히 같다. 차이는 경로 속도 프로파일이 어떤 종방향
가감속 한계를 사용하느냐에 있다.

---

## 6. Stanley 한 주기의 상세 계산

### 6.1 입력

`StanleyController::update()`는 다음 값을 받는다.

| 입력 | 뜻 |
|---|---|
| `pose` | `base_link`의 현재 `x`, `y`, `yaw` |
| `speed_mps` | 현재 차량 속도 |
| `dt` | 이전 제어 주기부터 지난 시간 |
| `behavior_id` | planner behavior 식별자 |
| `gear_id` | 현재 기어 식별자 |
| `target_speed_mps` | mission 등이 요청한 선택적 목표속도 override |

pose, speed, dt, 목표속도가 유한수가 아니거나 속도가 음수이거나
`dt <= 0`이면 invalid result를 반환한다.

### 6.2 rear axle pose와 앞축 제어점

차량 파라미터에서 `base_link` 원점은 뒤 차축 중심이다.

```yaml
origin: rear_axle_center_at_ground
lateral_control_point_x_m: 3.000
```

Stanley가 실제 CTE를 계산하는 점은 앞 차축이다.

```text
control_x = base_x + 3.0 * cos(yaw)
control_y = base_y + 3.0 * sin(yaw)
```

Stanley는 앞바퀴/앞축 기준 오차를 사용하는 것이 자연스럽기 때문에
wheelbase 3.0 m만큼 앞의 점을 제어점으로 사용한다. Optuna의 CTE도 같은
3.0 m 앞축 제어점을 사용한다. 컨트롤러와 목적함수가 서로 다른 점을
평가하지 않도록 맞춘 것이다.

### 6.3 위치 jump와 재탐색

이전 앞축 제어점에서 현재 앞축 제어점까지 거리가 20 m를 넘으면
localization reset 또는 MORAI 순간이동으로 판단한다.

이때 다음 상태를 초기화한다.

- route progress tracker
- PID 상태
- 이전 조향각
- launch ramp 시간
- 이전 목표속도

경로 인덱스를 다시 찾았는데도 경로로부터 20 m 넘게 떨어져 있으면
`route localization mismatch`를 반환한다.

여기서 `ControllerResult`의 `PhysicalCommand` 기본값은
`accel=0`, `brake=1`, `steering=0`이다. 즉 이 mismatch 결과는
`valid=true`이지만 안전하게 전제동 명령을 담는다.

### 6.4 진행 인덱스와 역방향 방지

처음에는 전역 경로 전체에서 가장 가까운 후보를 찾는다. 그 이후에는
현재 인덱스부터 `forward_window=200` 포인트 앞까지만 찾는다. 정상 주행
중 인덱스가 과거 구간으로 갑자기 돌아가는 것을 막는다.

경로가 가까이 교차하거나 왕복 차선이 붙어 있을 때는 방향도 사용한다.

- 차량 yaw와 경로 heading 차이가 90° 이하인 후보를 찾는다.
- 그 후보가 공간적으로 가장 가까운 점보다 최대 5 m만 더 멀면 방향이
  맞는 후보를 선택한다.
- 방향이 맞는 후보가 지나치게 멀면 순수 거리 최단 후보를 사용한다.

Stanley 계산 직전 현재 경로 heading과 차량 yaw 차이가 120°를 넘으면
`route heading mismatch`로 전제동한다. 이는 역주행 중 가까운 반대편
경로를 잡고 계속 진행하는 문제를 막는 최종 guard다.

### 6.5 조향 lookahead

조향에 사용하는 preview 거리는 속도에 따라 바뀐다.

```text
steering_lookahead
  = clamp(speed_mps * lookahead_time_s,
          lookahead_min_m,
          lookahead_max_m)
```

현재 값은:

```yaml
lookahead_time_s: 0.16
lookahead_min_m: 1.5
lookahead_max_m: 5.0
```

현재 설정의 예시는 다음과 같다.

| 차량 속도 | `speed * 0.16` | 실제 조향 lookahead |
|---:|---:|---:|
| 0 km/h | 0.00 m | 1.50 m |
| 20 km/h | 0.89 m | 1.50 m |
| 40 km/h | 1.78 m | 1.78 m |
| 58.5 km/h | 2.60 m | 2.60 m |
| 60 km/h | 2.67 m | 2.67 m |

현재 60 km/h 범위에서는 최대 5 m clamp에 도달하지 않는다.
`lookahead_time_s`를 크게 올리면 고속에서 경로 heading을 더 멀리 보고
조향하므로 부드러워질 수 있지만, 가까운 급커브 반응이 늦어질 수 있다.

### 6.6 조향용 heading error

가까운 포인트에서 실제 거리로 `steering_lookahead`만큼 경로를 전진해
target index를 얻는다. target index의 경로 segment heading을 사용한다.

```text
heading_error = wrap(path_heading - vehicle_yaw)
```

`heading_error_gain`이 이 항에 직접 곱해진다.

### 6.7 cross-track error

CTE는 lookahead target이 아니라 **현재 nearest 경로 segment**와 앞축
제어점 사이의 signed 횡오차다.

```text
cross_track_error
  = cross(path_segment_vector, control_point_offset)
    / path_segment_length
```

부호가 있으므로 경로의 왼쪽과 오른쪽에서 조향 방향이 반대로 나온다.

### 6.8 Stanley 조향식

현재 식은 다음과 같다.

```text
raw_steering =
    heading_error_gain * heading_error
  + atan2(cross_track_gain * cross_track_error,
          speed_mps + speed_softening_mps)
```

변수 영향은 다음과 같다.

- `cross_track_gain` 증가:
  같은 CTE에서 경로 중심으로 더 강하게 복귀한다. 너무 크면 좌우 진동과
  조향 포화가 늘어난다.
- `speed_softening_mps` 증가:
  CTE 항의 분모가 커져 특히 저속에서 조향이 약해진다. 너무 크면
  경로 복귀가 느려진다.
- `heading_error_gain` 증가:
  차량 방향을 경로 접선 방향으로 더 강하게 맞춘다. 너무 크면 곡선에서
  과조향하거나 CTE 복귀와 충돌할 수 있다.

### 6.9 조향각과 조향속도 제한

계산된 조향각은 먼저 `maximum_steering_rad`로 clamp된다. 현재
`planner.yaml` 값은:

```yaml
maximum_steering_rad: 0.6981317  # 약 40 deg
```

그다음 이전 명령에서 초당 최대 120°만 변하도록 rate limit된다.

```text
maximum_change = 120 deg/s * dt
steering = clamp(raw,
                 previous_steering - maximum_change,
                 previous_steering + maximum_change)
```

제어 주기 0.05 s라면 한 주기에 약 6°까지 변할 수 있다.

주의할 현재 설정 차이가 하나 있다.

- `vehicle_parameters.yaml`의 초기 bicycle road-wheel 한계:
  `0.588 rad`, 약 33.7°
- `planner.yaml`의 Stanley 전역 조향 한계:
  `0.6981317 rad`, 약 40°
- launch는 `local_motion.maximum_steering_rad`에는 0.588을 주입하지만
  전역 `maximum_steering_rad`는 차량 YAML로 덮어쓰지 않는다.

따라서 **현재 Stanley 자체는 0.6981317 rad를 사용한다**. 이 값이 실제
MORAI road-wheel 한계와 일치하는지는 별도 검증 대상이다.

---

## 7. 경로 곡률과 목표속도 계획

### 7.1 조향 lookahead와 곡률 lookahead는 다르다

두 값을 혼동하면 안 된다.

| 이름 | 목적 | 현재 방식 |
|---|---|---|
| `lookahead_time_s/min/max` | Stanley heading target 선택 | 속도 비례 |
| `curvature_lookahead_m` | 각 경로점의 곡률 계산 | 고정 물리 거리 |

조향은 속도가 높을수록 더 멀리 보는 것이 유리하다. 반면 곡률 계산은
경로 포인트 간격이 바뀌어도 같은 기하학적 의미를 유지해야 하므로
고정 거리 `curvature_lookahead_m`를 사용한다.

### 7.2 곡률 계산

각 경로점 `i`에서 앞뒤로 실제 경로 거리 `curvature_lookahead_m`만큼
이동한 세 점을 찾는다.

```text
previous ---- current ---- next
```

세 점으로 만든 외접원의 역반지름을 곡률로 사용한다.

```text
curvature = 2 * twice_triangle_area / (a * b * c)
```

단위는 `1/m`이다. 직선에 가까우면 0, 반지름이 작을수록 커진다.

현재:

```yaml
curvature_lookahead_m: 1.0
curvature_window_radius: 5
```

`curvature_window_radius=5`는 현재 인덱스 주변 앞뒤 5개 경로점에서 가장
큰 곡률을 선택한다. 이름은 smoothing에 가깝지만 실제 구현은 평균이
아니라 **worst curvature max filter**다. 따라서 좁은 곡률 spike도 속도
상한에 반영된다.

### 7.3 횡가속도 제한

곡률 `κ`와 속도 `v`에서 횡가속도는 다음과 같다.

```text
a_y = v² * κ = v² / R
```

허용 횡가속도에서 곡선 속도 상한은:

```text
v_curve = sqrt(lateral_acceleration_mps2 / max(curvature, 1e-6))
```

그 결과를 `minimum_speed_mps`와 `maximum_speed_mps` 사이로 clamp한다.
현재 허용 횡가속도는 `6.0 m/s²`, 경로 속도 범위는 5–60 km/h다.

`6.0 m/s²`를 온전히 사용하려면 필요한 최소 곡선 반지름 예시는:

| 속도 | 필요한 반지름 `R >= v²/6` |
|---:|---:|
| 40 km/h | 약 20.6 m |
| 50 km/h | 약 32.2 m |
| 58.5 km/h | 약 44.0 m |
| 60 km/h | 약 46.3 m |

이는 타이어 마찰을 직접 추정한 값이 아니라 현재 프로젝트의 속도 계획
상한이다. MORAI 노면/차량 모델에서 실제 미끄러짐 한계와 동일하다고
가정하면 안 된다.

### 7.4 미리 감속하는 backward pass

곡선 지점의 속도 상한만 낮추면 곡선에 도착한 순간 갑자기 목표속도가
떨어진다. 그래서 경로 끝에서 시작 방향으로 역순 계산한다.

다음 점 속도 `v_next`, 현재 점에서 다음 점까지 거리 `ds`, 허용 감속도
`a_brake`가 있을 때 현재 점에서 허용되는 최대 속도는:

```text
v_reachable = sqrt(v_next² + 2 * a_brake * ds)
v_current = min(curve_cap_current, v_reachable)
```

즉, **앞에 있는 곡선에서 필요한 속도까지 실제로 감속 가능한 지점부터
목표속도를 낮춘다.** 고정 몇 미터 앞만 보는 단순 규칙이 아니다.

### 7.5 곡선 탈출 후 재가속하는 forward pass

경로 시작부터 순방향으로 다시 계산한다.

```text
v_reachable = sqrt(v_previous² + 2 * a_accel * ds)
v_current = min(existing_cap_current, v_reachable)
```

곡선을 벗어난 뒤 차량이 낼 수 있는 가속도 범위에서 목표속도가 다시
올라간다. 별도의 “curve exit speed” 변수를 만들지 않고 기존 목표속도
프로파일이 자연스럽게 회복되도록 한 구조다.

### 7.6 열린 경로와 닫힌 경로

- 열린 경로는 마지막 점 목표속도를 0으로 설정하고 backward/forward
  pass를 한 번씩 수행한다.
- 닫힌 경로는 시작/끝 경계 때문에 잘못된 속도 discontinuity가 생기지
  않도록 경로를 3번 이어 붙인 뒤 backward/forward pass를 수행하고
  가운데 한 바퀴 결과만 사용한다.
- `maximum_laps=1`이면 닫힌 경로 한 바퀴가 끝난 뒤 terminal full brake
  상태가 된다.

---

## 8. 일반 `stanley`

일반 Stanley의 경로 속도 계획은 다음 값을 사용한다.

```yaml
stanley.lateral_acceleration_mps2: 6.0
stanley.acceleration_mps2: 2.0
stanley.deceleration_mps2: 2.0
```

경로 전체에서:

1. 곡률과 횡가속도 6.0 m/s²로 곡선 속도 상한을 만든다.
2. 모든 속도 구간에서 가속 능력을 2.0 m/s²라고 가정한다.
3. 모든 속도 구간에서 감속 능력을 2.0 m/s²라고 가정한다.
4. backward/forward pass로 미리 감속하고 곡선 탈출 후 재가속한다.
5. 매 제어 주기에는 경로 프로파일 값을 바로 PID 목표속도로 사용한다.

즉 일반 Stanley는 “프로파일이 없는 버전”이라기보다 **측정 기반
longitudinal profile이 없고 고정 한계를 쓰는 버전**이다.

---

## 9. `profile_stanley`

### 9.1 일반 Stanley와 공통인 부분

다음은 완전히 같다.

- 앞축 3.0 m 제어점
- route progress tracker
- 역방향 guard
- 속도 비례 조향 lookahead
- Stanley 조향식
- 최대 조향각과 조향속도 제한
- 곡률 계산식
- 횡가속도 기반 곡선 속도 상한
- throttle/brake 분리 PID

### 9.2 추가되는 부분

Profile Stanley는 `LongitudinalProfile`을 추가한다.

```cpp
struct LongitudinalProfile
{
  std::vector<double> speed_mps;
  std::vector<double> acceleration_mps2;
  std::vector<double> deceleration_mps2;
  double braking_delay_s;
};
```

배열 조건은:

- 세 배열 길이가 같아야 한다.
- 최소 2개 속도점이 필요하다.
- 속도는 0 이상이고 엄격히 증가해야 한다.
- 가속도와 감속도는 양수여야 한다.
- 모든 값이 유한수여야 한다.

### 9.3 속도별 보간

현재 속도가 테이블 두 점 사이면 선형 보간한다.

```text
a(v) = a_low + ratio * (a_high - a_low)
```

테이블 범위 밖에서는:

- 첫 속도보다 낮으면 첫 값을 유지
- 마지막 속도보다 높으면 마지막 값을 유지

현재 테이블의 마지막 점이 55 km/h이므로 목표 58.5 km/h와 상한
60 km/h 구간에서는 55 km/h 측정값을 그대로 사용한다.

### 9.4 설정값이 상한이라는 뜻

실제 경로 계획에 사용하는 가감속 능력은:

```text
planning_acceleration(v)
  = min(profile_stanley.acceleration_mps2,
        measured_acceleration(v))

planning_deceleration(v)
  = min(profile_stanley.deceleration_mps2,
        measured_deceleration(v))
```

예를 들어 40 km/h에서:

```text
측정 가속 능력       = 3.5313 m/s²
설정 가속 상한       = 5.0000 m/s²
계획에 사용되는 값   = 3.5313 m/s²

측정 감속 능력       = 1.8378 m/s²
설정 감속 상한       = 2.0000 m/s²
계획에 사용되는 값   = 1.8378 m/s²
```

반대로 10 km/h에서 측정 감속 능력이 3.6172 m/s²여도 계획에는
`min(2.0, 3.6172) = 2.0 m/s²`를 쓴다. 측정상 더 강하게 감속할 수 있어도
현재 comfort/safety 상한보다 공격적인 계획은 하지 않는다.

이 값은 MORAI가 실제 감속도를 정확히 2.0으로 유지하게 만드는
closed-loop 가속도 제어기가 아니다. 목표속도 경사를 만들 때 사용하는
물리적 가능량/상한이다.

### 9.5 제동 지연 보상

Profile Stanley는 경로 speed cap을 제동 지연만큼 앞쪽으로 복사한다.

```text
delay_distance
  = maximum_speed_mps * braking_delay_s
  = 16.6667 * 0.11589156
  ≈ 1.93 m
```

현재 구현은 속도마다 다른 지연거리를 계산하지 않고, 최대속도를 사용한
약 1.93 m의 보수적 고정 거리를 사용한다. 따라서 감속 시작점을 원래
곡률 속도 제한보다 약 1.93 m 앞당긴다.

### 9.6 경로 프로파일을 직접 사용

v4부터는 경로 전체 backward/forward pass로 이미 계산한 공간 기반
목표속도를 매 주기 직접 사용한다. 이전 목표속도 기준 slew limit는 같은
가감속 프로파일을 중복 적용해 커브 탈출 가속을 늦췄기 때문에 제거했다.
급커브 전 감속 시작점은 backward pass와 제동 지연 보상이 담당하고,
커브 후 재가속은 100% throttle 측정값을 사용한 forward pass가 담당한다.

---

## 10. 현재 사용 중인 측정 프로파일

### 10.1 원본 위치

프로파일 원본은 소스 트리 안에 복제하지 않고 workspace 데이터 디렉터리에
있다.

```text
$AD_DATA_DIR/vehicle_dynamics/
├── 20260727-ioniq5-accelerator-map-v1/
└── 20260727-ioniq5-brake-map-v2/
```

`planner.yaml`에는 다음 조합을 export한 배열이 들어 있다.

- accelerator map v1의 100% accelerator
- brake map v2의 20% brake
- 생성 스크립트: `scripts/export_longitudinal_profile.py`
- 생성일: 2026-07-27

### 10.2 export 규칙

export 스크립트는:

1. `status == complete`인 행만 사용한다.
2. accelerator는 `median_acceleration_mps2`를 사용한다.
3. brake는
   `coast_normalized_brake_deceleration_mps2`를 우선 사용하고, 없으면
   `median_mean_deceleration_mps2`를 사용한다.
4. brake delay는 각 행의 command echo delay와 감속 onset delay 중 큰
   값을 사용한다.
5. accelerator와 brake가 동시에 존재하는 속도 범위의 합집합을 만들고
   빠진 지점은 선형 보간한다.
6. 최종 `braking_delay_s`는 선택된 brake 점 전체에서 가장 큰 delay다.

### 10.3 현재 배열

| 속도 | accelerator 100% 측정 | brake 20% coast-normalized 측정 |
|---:|---:|---:|
| 5 km/h | 4.4221 m/s² | 3.6703 m/s² |
| 10 km/h | 4.0032 m/s² | 3.6172 m/s² |
| 15 km/h | 3.6880 m/s² | 3.5869 m/s² |
| 20 km/h | 3.5940 m/s² | 3.4646 m/s² |
| 25 km/h | 3.5641 m/s² | 2.1184 m/s² |
| 30 km/h | 3.5580 m/s² | 1.8324 m/s² |
| 35 km/h | 3.5330 m/s² | 1.8296 m/s² |
| 40 km/h | 3.5313 m/s² | 1.8378 m/s² |
| 45 km/h | 3.5035 m/s² | 1.8350 m/s² |
| 50 km/h | 3.5059 m/s² | 1.8458 m/s² |
| 55 km/h | 3.4879 m/s² | 1.8255 m/s² |

최종 제동 지연은:

```yaml
profile_stanley.longitudinal_profile.braking_delay_s: 0.11589156
```

### 10.4 현재 상한을 적용한 실제 계획값

| 속도 | 계획 가속 상한 | 계획 감속 상한 |
|---:|---:|---:|
| 5 km/h | 4.4221 | 2.0000 |
| 10 km/h | 4.0032 | 2.0000 |
| 15 km/h | 3.6880 | 2.0000 |
| 20 km/h | 3.5940 | 2.0000 |
| 25 km/h | 3.5641 | 2.0000 |
| 30 km/h | 3.5580 | 1.8324 |
| 35 km/h | 3.5330 | 1.8296 |
| 40 km/h | 3.5313 | 1.8378 |
| 45 km/h | 3.5035 | 1.8350 |
| 50 km/h | 3.5059 | 1.8458 |
| 55–60 km/h | 3.4879 | 1.8255 |

### 10.5 프로파일 완성 상태에 대한 주의

전체 0–185 km/h × 0–100% grid가 모두 완성된 것은 아니다.

현재 `profile.json` 기준:

| 데이터셋 | 전체 cell | complete | incomplete | limiter/attempt 제한 |
|---|---:|---:|---:|---:|
| accelerator map v1 | 429 | 418 | 0 | limiter-bound 11 |
| brake map v2 | 429 | 123 | 302 | attempt-limit 4 |

다만 현재 Profile Stanley가 사용하는 **5–55 km/h, accelerator 100%,
brake 20% 행은 각각 valid trial 3개로 complete**다.

또한 선택된 행에는 `simulator_acceleration_field_stuck` 품질 플래그가
있다. accelerator 값은 고정된 simulator acceleration field가 아니라
velocity-derived acceleration을 사용했다. 이 프로파일은 현재 MORAI
실험에서 얻은 실용적인 seed이지만 차량의 절대적인 물리 정답으로
간주하면 안 된다.

---

## 11. 목표속도 결정 순서

매 제어 주기 최종 목표속도는 다음 제한의 최솟값이다.

```text
selected_target
  = mission override가 있으면 override
  = 아니면 configured target 58.5 km/h

launch_target
  = launch_speed
  + (selected_target - launch_speed)
    * min(launch_elapsed / launch_ramp, 1)

raw_governed_target
  = min(selected_target,
        launch_target,
        route_speed_profile[nearest_index])
```

현재:

```yaml
target_speed_mps: 16.25            # 58.5 km/h
maximum_speed_mps: 16.6666666667   # 60 km/h
launch_speed_mps: 1.3888888889     # 5 km/h
launch_ramp_s: 3.0
```

시작 직후에는 목표속도를 5 km/h에서 출발시켜 3초 동안 selected target을
향해 선형으로 올린다. 이후에는 현재 경로 위치에 미리 계산된 공간 기반
속도 프로파일을 직접 사용한다.

`target_speed_mps=58.5 km/h`와 `maximum_speed_mps=60 km/h`는 역할이 다르다.

- `target_speed_mps`: 직선에서 추구하는 운용 목표
- `maximum_speed_mps`: 경로 속도 프로파일이 절대 넘지 못하는 상한
- Optuna `overspeed_kph=60`: 실제 차량 속도가 넘으면 대회식 패널티를
  주는 기준

---

## 12. 종방향 PID 상세

### 12.1 속도 오차 단위

컨트롤러 입력 속도는 m/s지만 Stanley factory는 `error_scale=3.6`을
설정한다.

```text
raw_error = (target_speed_mps - current_speed_mps) * 3.6
```

따라서 PID gain에 들어가는 속도 오차 단위는 사실상 km/h다.

### 12.2 throttle, coast, brake 모드

현재 brake deadband는 1.0 km/h다.

```text
raw_error > 0              -> throttle mode
-1 km/h <= raw_error <= 0  -> coast mode
raw_error < -1 km/h        -> brake mode
```

brake mode에서는 deadband를 제거한 오차를 사용한다.

```text
brake_effective_error = raw_error + 1.0
```

예를 들어 목표보다 1.2 km/h 빠르면 brake PID에 들어가는 유효 오차는
약 -0.2다. 목표보다 5 km/h 빠르면 약 -4.0이다.

### 12.3 throttle과 brake gain 분리

현재 파라미터:

```yaml
speed_pid.kp: 1.08
speed_pid.ki: 0.0
speed_pid.kd: 0.036

brake_pid.kp: 0.20
brake_pid.ki: 0.0
brake_pid.kd: 0.01
```

모드에 따라:

```text
throttle mode -> speed_pid Kp/Ki/Kd
brake mode    -> brake_pid Kp/Ki/Kd
coast mode    -> accel=0, brake=0
```

스로틀과 브레이크는 같은 페달 특성을 가지지 않으므로 gain을 분리한 것이
타당하다. 특히 MORAI brake 명령은 작은 값에서도 감속이 강할 수 있어
throttle Kp를 그대로 쓰면 급제동이 발생하기 쉽다.

### 12.4 PID 상태 reset

다음 중 하나가 바뀌면 integral을 0으로 만들고 derivative kick을
방지한다.

- 첫 실행
- behavior ID 변경
- gear ID 변경
- throttle/coast/brake actuation mode 변경

오차 부호와 integral 부호가 반대가 되어도 integral을 0으로 만든다.

### 12.5 anti-windup과 출력

PID 출력:

```text
output = Kp * error + Ki * integral + Kd * derivative
```

출력이 ±1을 넘고 새 integral이 포화를 더 악화시키면 직전 integral로
되돌린다. 최종 출력은:

```text
throttle mode: accel = clamp(output, 0, 1), brake = 0
brake mode:    accel = 0, brake = clamp(-output, 0, 1)
coast mode:    accel = 0, brake = 0
```

accel과 brake가 동시에 양수가 되는 경로는 없다.

### 12.6 현재 구현의 한계

Profile Stanley가 측정 가속도/감속도 테이블을 알고 있어도 현재 PID
출력은 다음처럼 동작한다.

```text
target speed - current speed
        |
        v
PID
        |
        v
0~1 throttle 또는 0~1 brake
```

아직 다음 구조는 아니다.

```text
원하는 종가속도
        |
        v
속도별 inverse IONIQ pedal map
        |
        v
feed-forward pedal + feedback PID correction
```

따라서 프로파일은 “이 속도에서 이만큼 감속 가능하니 언제부터 목표속도를
낮출지”를 개선하지만, “정확히 몇 % brake를 주어야 그 감속도가 나오는지”
를 직접 역변환하지 않는다.

---

## 13. Stanley 파라미터 사전

아래 현재 값은 일반 Stanley와 Profile Stanley가 동일하다. 프로파일
배열만 Profile Stanley에 추가된다.

### 13.1 횡제어와 진행 추적

| 파라미터 | 현재 값 | 단위 | 커지면 | 지나칠 때 |
|---|---:|---|---|---|
| `cross_track_gain` | 0.91 | gain | CTE 복귀 강해짐 | 좌우 진동, 포화 |
| `speed_softening_mps` | 2.57 | m/s | CTE 조향 약해짐 | 경로 복귀 지연 |
| `heading_error_gain` | 1.0 | gain | 경로 방향 정렬 강해짐 | 곡선 과조향 |
| `lookahead_time_s` | 0.16 | s | 고속 preview 증가 | 급커브 반응 지연 |
| `lookahead_min_m` | 1.5 | m | 저속 조향이 더 먼 점을 봄 | 저속 코너 cutting 가능 |
| `lookahead_max_m` | 5.0 | m | 최대 preview 증가 | 복잡 경로 반응 둔화 |
| `control_point_x_m` | 3.0 | m | 더 앞쪽 오차 평가 | 차량 geometry 불일치 |
| `forward_window` | 200 | point | 한 주기 앞쪽 탐색 범위 증가 | 계산량과 잘못된 branch 가능성 |
| `maximum_laps` | 1 | lap | 닫힌 경로 반복 수 증가 | 원치 않는 반복 |
| `maximum_steering_rad` | 0.6981317 | rad | 더 큰 조향 허용 | 실제 차량 한계 불일치 |

### 13.2 곡률과 속도 계획

| 파라미터 | 현재 값 | 단위 | 역할 |
|---|---:|---|---|
| `target_speed_mps` | 16.25 | m/s | 직선 운용 목표 58.5 km/h |
| `minimum_speed_mps` | 1.3889 | m/s | 곡률 프로파일 최저 5 km/h |
| `maximum_speed_mps` | 16.6667 | m/s | 경로 프로파일 상한 60 km/h |
| `lateral_acceleration_mps2` | 6.0 | m/s² | 곡률별 속도 상한 |
| `curvature_lookahead_m` | 1.0 | m | 곡률을 만드는 앞뒤 물리 거리 |
| `curvature_window_radius` | 5 | point | 주변 worst-curvature 범위 |
| `acceleration_mps2` | 5.0 | m/s² | forward pass 가속 상한 |
| `deceleration_mps2` | 2.0 | m/s² | backward pass 감속 상한 |
| `launch_speed_mps` | 1.3889 | m/s | 컨트롤러 시작 목표 5 km/h |
| `launch_ramp_s` | 3.0 | s | 시작 목표속도 ramp 시간 |

`minimum_speed_mps`는 열린 경로 마지막 점에는 적용되지 않는다. 마지막
점은 정지를 위해 0 m/s로 강제된다.

### 13.3 PID

| 파라미터 | 현재 값 | 의미 |
|---|---:|---|
| `speed_pid.kp` | 1.08 | 가속 속도 오차 비례 gain |
| `speed_pid.ki` | 0.0 | 가속 정상상태 오차 integral gain |
| `speed_pid.kd` | 0.036 | 가속 오차 변화 damping |
| `speed_pid.integral_limit` | 100.0 | integral 절댓값 제한 |
| `speed_pid.derivative_limit` | double max | 사실상 별도 derivative 제한 없음 |
| `brake_pid.kp` | 0.20 | 감속 속도 오차 비례 gain |
| `brake_pid.ki` | 0.0 | 감속 integral gain |
| `brake_pid.kd` | 0.01 | 감속 오차 변화 damping |
| 코드 고정 `error_scale` | 3.6 | m/s 오차를 km/h 오차로 변환 |
| 코드 고정 `brake_deadband` | 1.0 | 1 km/h까지 coast |

### 13.4 Profile Stanley 전용

| 파라미터 | 역할 |
|---|---|
| `longitudinal_profile.speed_mps` | 측정 속도 축 |
| `longitudinal_profile.acceleration_mps2` | 속도별 가속 가능량 |
| `longitudinal_profile.deceleration_mps2` | 속도별 감속 가능량 |
| `longitudinal_profile.braking_delay_s` | 제동 시작 지연의 보수적 최댓값 |

### 13.5 내부 상태 변수

`StanleyController` 내부에는 다음 상태가 유지된다.

| 변수 | 역할 |
|---|---|
| `route_` | 로드된 전역 경로 |
| `config_` | 생성 시 읽은 Stanley 설정 |
| `progress_` | 현재 route index, lap, terminal 상태 |
| `pid_` | integral, 이전 오차, mode, behavior, gear |
| `route_speed_profile_` | 미리 계산된 점별 속도와 곡률 |
| `previous_steering_rad_` | 조향 rate limit 기준 |
| `launch_elapsed_s_` | 시작 속도 ramp 진행 시간 |
| `previous_control_pose_` | 20 m pose jump 검출 기준 |
| `last_valid_` | 마지막 유효 결과와 디버그 상태 |

ROS 파라미터 값만 변경하면 이미 생성된 `config_`는 자동으로 바뀌지
않는다. Optuna가 파라미터 변경 후 `/ad/planner/reset_path_tracking`을
호출하는 이유가 이것이다.

---

## 14. Optuna가 실제 ROS 코드를 튜닝하는 방식

### 14.1 복제 코드가 아니다

Optuna용 Python 패키지 안에 Stanley 계산식을 다시 복사하지 않았다.
실제 trial은:

1. 실행 중인 C++ `ad_planner`에 ROS atomic parameter service로 후보값을
   넣는다.
2. planner의 `/ad/planner/reset_path_tracking`을 호출한다.
3. planner가 현재 ROS 파라미터를 다시 읽어 실제 C++ controller를
   재생성한다.
4. MORAI 차량을 주행시킨다.
5. odometry, 차량 속도, planner target speed, 실제 brake command를
   측정한다.

따라서 Optuna 목적함수에 들어가는 결과는 별도 Python Stanley 모델이
아니라 현재 ROS 패키지와 MORAI의 실제 실행 결과다.

### 14.2 튜닝 전용 stack

튜닝 launch는 다음을 켠다.

- vehicle description
- GNSS/IMU localization
- planner
- MORAI bridge
- MORAI MultiEgo reset용 dev bridge
- Optuna tuner

다음은 끈다.

- LiDAR perception
- camera perception
- RViz

`local_motion.backend=dwa` 설정 자체는 유지되지만
`perception.enabled=false`로 장애물/OGM/DWA intervention이 path controller
비교 점수에 섞이지 않게 한다.

이는 최종 운용 상태가 아니다. Optuna winner는 반드시
`perception.enabled=true`인 일반 DWA 스택에서 다시 검증해야 한다.

---

## 15. 한 Optuna trial의 정확한 순서

각 trial은 fail-closed 방식으로 진행된다.

1. 필요한 ROS service와 path, odometry, vehicle status, planner readiness를
   기다린다.
2. 나머지 graph를 기다리기 전에 `/ad/planner/hold_control=true`를
   요청해 planner를 전제동 상태로 만든다.
3. 차량 속도가 8초 안에 0.5 m/s 이하가 되는지 확인한다.
4. Optuna 후보 9개와 고정 파라미터를
   `/ad_planner/set_parameters_atomically`로 한 번에 적용한다.
5. MORAI `MultiEgoSetting` reset 명령을 0.1초 간격으로 8회 전송한다.
6. reset은 최대 2회 시도한다.
7. Ego 위치는 전역 경로 첫 점, yaw는 첫 경로 방향, 속도 0, gear 4,
   `ctrl_mode=2`로 설정한다.
8. reset 이후의 fresh odometry/status가 들어오고, 시작점 3 m 이내이며,
   속도 0.5 m/s 이하인지 확인한다.
9. hold 상태에서 `/ad/planner/reset_path_tracking`을 호출해 실제
   controller를 재생성한다.
10. planner 입력 준비 상태를 확인한다.
11. hold를 해제한다.
12. 약 20 Hz로 주행을 측정한다.
13. 완주, 이탈, 역방향, stall, timeout 중 하나로 trial을 끝낸다.
14. 어떤 결과든 trial 뒤 planner를 다시 full-brake hold한다.

튜닝 종료 시 global best 파라미터를 planner에 적용하고 controller를
재생성하지만, 차량은 계속 hold 상태로 남긴다.

---

## 16. Optuna 탐색 공간

현재 Optuna가 바꾸는 값은 9개다.

| 파라미터 | 범위 | 분포 | 목적 |
|---|---:|---|---|
| `profile_stanley.cross_track_gain` | 0.5–1.5 | uniform | CTE 복귀 강도 |
| `profile_stanley.speed_softening_mps` | 1.0–4.0 | uniform | 저속 CTE 항 완화 |
| `profile_stanley.heading_error_gain` | 0.7–1.3 | uniform | heading 정렬 강도 |
| `profile_stanley.lookahead_time_s` | 0.08–0.30 | uniform | 속도 비례 조향 preview |
| `profile_stanley.curvature_lookahead_m` | 0.75–3.0 | uniform | 곡률 계산 물리 거리 |
| `profile_stanley.speed_pid.kp` | 0.5–1.5 | uniform | throttle 비례 gain |
| `profile_stanley.speed_pid.kd` | 0.0–0.10 | uniform | throttle damping |
| `profile_stanley.brake_pid.kp` | 0.02–0.60 | **log** | brake 비례 gain |
| `profile_stanley.brake_pid.kd` | 0.0–0.05 | uniform | brake damping |

brake Kp는 0.02와 0.60 사이의 크기 차이가 크고 작은 값 영역을 더 세밀히
탐색해야 해서 log distribution을 사용한다.

현재 탐색에서 제외한 값:

- target speed
- 최대속도
- 횡가속도 상한
- scalar 가감속 상한
- measured longitudinal arrays
- `Ki`
- lookahead min/max
- 곡률 window radius
- launch ramp
- steering limit
- forward window

---

## 17. Optuna 고정 파라미터

매 trial에 다음 값이 강제로 함께 적용된다.

| 파라미터 | 고정값 |
|---|---:|
| `lookahead_min_m` | 1.5 m |
| `lookahead_max_m` | 5.0 m |
| `target_speed_mps` | 58.5 km/h |
| `maximum_speed_mps` | 60 km/h |
| `lateral_acceleration_mps2` | 6.0 m/s² |
| `acceleration_mps2` | 5.0 m/s² |
| `deceleration_mps2` | 2.0 m/s² |
| `speed_pid.ki` | 0.0 |
| `brake_pid.ki` | 0.0 |

나머지 planner YAML 값도 실행 중에는 사실상 고정이지만, 위 목록은
Optuna 코드가 candidate와 항상 합쳐 명시적으로 적용하고 experiment
fingerprint에 넣는 고정값이다.

---

## 18. warm start와 TPE sampler

study 시작 시 세 후보를 먼저 queue에 넣는다.

### 18.1 baseline seed

```text
cross_track=0.91
speed_softening=2.57
heading=1.0
lookahead_time=0.16
curvature_lookahead=1.0
speed Kp/Kd=1.08/0.036
brake Kp/Kd=0.20/0.010
```

### 18.2 smoother seed

```text
cross_track=0.75
speed_softening=2.8
heading=0.9
lookahead_time=0.20
curvature_lookahead=1.5
speed Kp/Kd=0.85/0.025
brake Kp/Kd=0.10/0.005
```

### 18.3 aggressive seed

```text
cross_track=1.1
speed_softening=2.2
heading=1.1
lookahead_time=0.12
curvature_lookahead=2.0
speed Kp/Kd=1.25/0.050
brake Kp/Kd=0.35/0.015
```

`skip_if_exists=True`라서 같은 shared study에 이미 같은 queued/finished
후보가 있으면 다시 넣지 않는다.

TPE 설정:

```python
TPESampler(
    seed=worker_specific_seed,
    n_startup_trials=18,
    multivariate=True,
    constant_liar=True,
)
```

- `n_startup_trials=18`: 충분한 초기 관측 전에는 TPE 모델보다 초기 탐색을
  우선한다.
- `multivariate=True`: 파라미터별 독립 분포가 아니라 파라미터 조합의
  상관관계를 함께 모델링한다.
- `constant_liar=True`: 다른 worker가 현재 실행 중인 후보 주변을 또
  선택하는 일을 줄인다.
- 각 worker seed는 공통 base seed와 worker ID hash를 XOR해 만든다.
  공유 study를 보면서도 세 worker가 똑같은 난수 흐름을 사용하지 않는다.

현재 pruner는 설정하지 않았다. `trial.report()`와 중간 pruning도 없다.
한 trial은 runner의 완주/이탈/stall/timeout 조건까지 실행된다.

---

## 19. 목적함수

### 19.1 완주 trial

완주 시 raw cost는:

```text
cost =
    elapsed_time
  + 30 * distance_weighted_mean(front_CTE²)
  + 30 * distance_weighted_mean(rear_CTE²)
  + competition_overspeed_penalty
  + 1 * integral(target_overspeed² dt)
  + 5 * integral(unnecessary_brake² dt)
  + 1 * brake_saturation_time
```

Optuna는 minimize한다. 낮을수록 좋다.

### 19.2 각 항의 의미

#### `elapsed_time`

코스를 빠르게 끝낼수록 작다. 단위는 초다.

#### 앞축·뒤축 `30 * distance_weighted_mean(CTE²)`

앞축 CTE는 Stanley 제어점에서, 뒤축 CTE는 뒤차축 중앙에서 측정한다.
각각의 거리 가중 MSE에 30을 곱해 더한다. sample 개수에 따른 worker
편향을 피하려고 경로 진행거리로 사다리꼴 적분한 뒤 실제 진행거리로
나눈다.

```text
distance_CTE_MSE =
  integral(CTE² ds) / evaluated_distance
```

sample 산술 평균과 simulator-time 가중 평균도 진단값으로 저장하지만,
목적함수에는 거리 가중값만 사용한다.

앞축 또는 뒤축 하나에만 일정한 CTE가 발생한 예:

| 일정한 CTE | CTE cost |
|---:|---:|
| 0.1 m | 0.3 |
| 0.2 m | 1.2 |
| 0.5 m | 7.5 |
| 0.8 m | 19.2 |

앞축과 뒤축이 같은 CTE라면 위 값의 두 배다.

#### competition overspeed

실제 속도가 60 km/h를 넘은 시간에 적용한다.

```text
overspeed > 0초, <3초 -> 15초
3초 이상, <6초       -> 30초
6초 이상, <9초       -> 45초
...
```

코드가 `floor(overspeed_time / 3)`를 사용하므로 정확히 3.0초부터 다음
15초가 추가된다.

#### target overspeed

실제 속도가 controller target보다 1 km/h 이상 빠른 부분만 계산한다.

```text
target_overspeed
  = max(0, actual_speed - target_speed - 1 km/h)
```

이를 제곱해 시간 적분한다. 곡선 진입 목표속도가 내려갔는데 차량이 계속
빠른 경우를 벌점으로 잡는다.

#### unnecessary brake

현재 속도가 `target + 1 km/h` 이하인데 brake를 사용하면:

```text
integral(brake_command² dt)
```

를 누적한다. 목표속도를 이미 만족했거나 더 느린데 계속 브레이크를 밟는
급제동/끌림을 줄이기 위한 항이다.

#### brake saturation

`brake >= 0.95`인 시간을 누적한다. 전제동에 가까운 명령이 오래 유지되는
후보를 직접 벌점으로 만든다.

### 19.3 미완주 trial

미완주 raw cost:

```text
raw_cost =
    2000
  + max(0, course_length - progress)
  + control_costs
```

이 `raw_cost`가 그대로 Optuna value다. 별도의 2000점은 다시 더하지
않는다. 대신 TPE `constraints_func`에 완주 여부와 최대 CTE 위반량을
전달하므로 sampler도 feasible 후보를 직접 구분한다.

완주했어도 앞축 또는 뒤축의 `max_CTE > 0.8 m`이면 constraint
violation으로 기록한다. 0.8 m 안에서도 CTE가 작은 후보를 선호하는
압력은 앞축·뒤축 거리 가중 CTE cost가 계속 담당한다.

### 19.4 feasible 조건

다음을 모두 만족해야 한다.

- 코스를 완주함
- reset 실패가 아님
- 입력 연결 끊김이 아님
- tuner shutdown abort가 아님
- 앞축 최대 CTE가 0.8 m 이하
- 뒤축 최대 CTE가 0.8 m 이하

최종 best export는 Optuna의 단순 `study.best_trial`을 바로 쓰지 않는다.
현재 experiment fingerprint가 같고, state가 COMPLETE이고,
`feasible=true`인 trial 중 raw `cost`가 가장 작은 trial만 선택한다.

---

## 20. 측정값과 중단 조건

### 20.1 매 sample 기록

각 worker의 trajectory CSV에는 다음이 기록된다.

```text
t_s
base_x_m
base_y_m
control_x_m
control_y_m
speed_mps
target_speed_mps
brake
cte_m
progress_m
segment
```

`progress_m`은 순간 projection이 뒤로 튀어도 감소하지 않도록
지금까지의 최댓값을 유지한다.

### 20.2 완주 기준

`course_length_m=0`이면:

```text
target = route_length - completion_margin
       = route_length - 5 m
```

양수를 주면 route length와 설정 course length 중 작은 값까지 평가한다.

### 20.3 trial 중단

| 조건 | 현재 값 | 처리 |
|---|---:|---|
| trial timeout | 420 s | 미완주 |
| CTE divergence | 12 m 초과 | 미완주 |
| route 대비 heading | 120° 초과 | 미완주 |
| localization pose jump | 20 m 초과 | 미완주 |
| stall 검사 시작 | 시작 8 s 후 | 조건 평가 |
| stall 속도 | 1 km/h 미만 | 4 s 지속 시 미완주 |
| odom/status/target/command stale | 2 s | 2 s 지속 시 infrastructure failure |
| reset 시작점 허용 오차 | 3 m | 벗어나면 reset 재시도/실패 |
| reset 전 정지 속도 | 0.5 m/s 이하 | 8 s 안에 필요 |

CTE divergence, 역방향, stall, timeout은 차량/파라미터 결과로 보고
Optuna COMPLETE 상태의 infeasible trial로 남는다.

reset 실패, 연결 끊김, tuner shutdown은 비교할 수 없는 인프라 실패로
보고 `InfrastructureTrialFailure`를 발생시켜 Optuna FAIL로 남긴다. 이런
trial은 `maximum_trials`의 COMPLETE 개수에 포함되지 않는다.

---

## 21. study 식별과 재현성

현재 search version:

```text
profile_stanley_ros_morai_v2
```

study 이름:

```text
profile_stanley_ros_morai_v2_<20-character experiment fingerprint>
```

fingerprint에는 다음이 들어간다.

- search space version
- route 좌표 digest
- route length
- 실제 평가 course length
- 명시적 fixed controller parameters
- `path_tracking.backend=profile_stanley`
- `perception.enabled=false`
- 목적함수 설정 전체
- scenario
- weather
- MORAI version
- 선언된 code revision
- vehicle profile ID
- timing source
- 설치되어 실행 중인 `ad_tuning/*.py` source digest

조건이 하나라도 바뀌면 다른 fingerprint와 study 이름이 생긴다. 같은
study 이름에서 metadata가 다르면 worker는 실행을 거부한다.

`scenario`, `weather`, `code_revision`이 `unspecified`여도 로컬 실행은
가능하지만 PostgreSQL 분산 실행은 거부한다. 세 worker의 점수를 직접
비교하려면 세 값이 실제로 같아야 한다.

---

## 22. PostgreSQL 분산 Optuna

### 22.1 역할

현재 구조는:

| 장비 | worker ID | ROS domain | 역할 |
|---|---|---:|---|
| 현재 우측 PC | `heven-right` | 21 | MORAI worker + PostgreSQL server |
| 좌측 PC | `heven-left` | 22 | MORAI worker |
| 노트북 | `heven-laptop` | 23 | MORAI worker |

PostgreSQL:

```text
server: 192.168.0.191:5432
database: heven_optuna
role: ad_tuning
```

비밀번호는 문서나 launch/YAML에 넣지 않고 각 PC의 mode-600 `.pgpass`에
있다. worker 설정은 `~/.config/heven/optuna-worker.env`에서 읽는다.

### 22.2 왜 ROS domain이 달라야 하는가

세 PC가 같은 Wi-Fi/LAN에 있고 같은 ROS domain과 같은 topic 이름을 쓰면
DDS discovery를 통해 서로의 planner, localization, command topic이
섞일 수 있다.

그래서:

- right는 domain 21
- left는 domain 22
- laptop은 domain 23

으로 분리한다. PostgreSQL study만 공유하고 각 MORAI/ROS graph는 서로
독립이다. 각 worker의 MORAI UDP endpoint도 자기 PC의 localhost를
사용한다.

### 22.3 현재 설치 버전

2026-07-28 재확인 결과 세 PC가 동일하다.

```text
Ubuntu 22.04 / ROS 2 Humble
Python 3.10.12
Optuna 4.9.0
psycopg 3.3.4
SQLAlchemy 2.0.51
```

### 22.4 환경 smoke test 결과

실제 MORAI 주행이 아닌 synthetic shared-study test:

```text
study: heven_3host_env_smoke_1785177808260
total trials: 12 COMPLETE
heven-right: 4
heven-left: 4
heven-laptop: 4
trial numbers: 0..11, duplicate 없음
peak concurrency: 3
wall span: 3.378 s
summed trial duration: 8.056 s
PostgreSQL database deadlocks: 0
```

이 결과가 증명하는 것:

- 세 PC가 같은 PostgreSQL Optuna study를 읽고 쓸 수 있다.
- trial 번호 할당이 충돌하지 않는다.
- 세 worker가 동시에 실행될 수 있다.
- 현재 Python dependency가 호환된다.

이 결과가 증명하지 않는 것:

- 세 MORAI가 동시에 reset/주행되는지
- 세 PC의 시나리오와 센서 설정이 완전히 동일한지
- 실제 Profile Stanley objective가 정상 수렴하는지
- 최적 파라미터가 장애물 포함 운용 스택에서도 좋은지

---

## 23. RDB heartbeat와 장애 처리

현재 값:

```yaml
storage.heartbeat_interval_sec: 15
storage.heartbeat_grace_period_sec: 90
storage.heartbeat_retry_count: 1
storage.connection_retry_attempts: 12
storage.connection_retry_interval_sec: 5.0
```

- 정상 실행 중 worker는 15초마다 heartbeat를 남긴다.
- 90초 넘게 heartbeat가 없으면 stale trial로 판단할 수 있다.
- stale trial은 기본 1회 재시도 queue에 들어간다.
- DB 연결 실패 시 planner를 full-brake hold하고 5초 간격으로 재연결한다.
- 기본은 12회 시도다.
- `connection_retry_attempts=0`이면 ROS가 종료될 때까지 계속 재시도한다.

PostgreSQL connection은:

- `pool_pre_ping=True`
- connection recycle 300초
- connect timeout 10초
- application name `ad_tuning_<worker-id>`

를 사용한다.

---

## 24. `maximum_trials`의 정확한 의미

기본:

```yaml
maximum_trials: 30
```

이는 각 worker가 30개씩 수행한다는 뜻이 아니다. shared study 전체의
COMPLETE trial 목표다.

- `30`: 세 worker 합계 COMPLETE 30개가 목표
- `0`: Ctrl+C까지 계속
- infrastructure FAIL trial: 개수에 포함하지 않음
- 이미 주행 중인 worker가 동시에 끝날 수 있어 최종 COMPLETE 수가 active
  worker 수 정도 초과할 수 있음

현재 loop는 한 번에 `study.optimize(..., n_trials=1)`을 호출하고 shared
COMPLETE 개수를 다시 확인한다.

---

## 25. 로컬 결과와 PostgreSQL 결과

기본 출력 위치:

```text
$AD_DATA_DIR/tuning/profile_stanley/
└── workers/
    └── <worker-id>/
        ├── profile_stanley_trials.jsonl
        ├── profile_stanley_trajectories/
        │   └── trial_XXXX.csv
        ├── pending_database_results/
        ├── best_profile_stanley_optuna.yaml
        └── best_profile_stanley_optuna.json
```

PostgreSQL에는:

- candidate parameters
- worker ID
- experiment fingerprint
- metrics
- feasible 여부
- raw cost
- Optuna value
- failure reason
- trajectory 파일 경로

가 trial attribute로 저장된다.

trajectory 전체 sample은 DB 용량을 키우지 않도록 worker 로컬 CSV에만
저장한다.

DB commit 전에 trial 결과를
`pending_database_results/trial_XXXXXXXX.json`에 먼저 기록한다. Optuna
callback이 RDB state commit을 확인한 뒤 이 pending 파일을 삭제한다.
네트워크가 끊기면 local JSONL, trajectory, pending JSON이 감사용으로
남는다.

---

## 26. 실행 방법

### 26.1 일반 Stanley 운용 스택

```bash
cd ~/heven_ad_2026_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export AD_DATA_DIR="$PWD/ad_data"

ros2 launch ad_bringup bringup.launch.py \
  path_tracking_backend:=stanley
```

### 26.2 Profile Stanley 운용 스택

```bash
cd ~/heven_ad_2026_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export AD_DATA_DIR="$PWD/ad_data"

ros2 launch ad_bringup bringup.launch.py \
  path_tracking_backend:=profile_stanley
```

비교할 때는 동일 경로, 동일 시나리오, 동일 weather, 동일 localization
초기조건을 사용해야 한다.

### 26.3 단일 PC Optuna

`OPTUNA_STORAGE_URL`을 설정하지 않으면 output directory의 SQLite를
사용한다.

```bash
cd ~/heven_ad_2026_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export AD_DATA_DIR="$PWD/ad_data"
unset OPTUNA_STORAGE_URL

ros2 launch ad_tuning profile_stanley_morai_optuna.launch.py \
  morai_ip:=127.0.0.1 \
  maximum_trials:=30
```

SQLite는 단일 worker fallback이다. 여러 PC를 SQLite 파일 공유로 묶으면
안 된다.

### 26.4 3대 PC PostgreSQL Optuna

각 PC에서:

```bash
cd ~/heven_ad_2026_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
source ~/.config/heven/optuna-worker.env
export AD_DATA_DIR="$PWD/ad_data"
```

그리고 **세 PC에서 동일하게** 실제 조건을 지정한다.

```bash
export AD_TUNING_SCENARIO='<동일한 실제 scenario ID>'
export AD_TUNING_WEATHER='<동일한 weather>'
export AD_TUNING_MORAI_VERSION='S4.251001'
export AD_TUNING_CODE_REVISION='<세 PC에 배포한 동일 revision>'
export AD_TUNING_VEHICLE_PROFILE_ID='20260727-ioniq5-accelerator40-brake20-v1'
```

각 PC에서 자기 MORAI가 준비된 뒤:

```bash
ros2 launch ad_tuning profile_stanley_morai_optuna.launch.py \
  morai_ip:=127.0.0.1 \
  maximum_trials:=120
```

`maximum_trials=120`은 세 PC 합계 목표다.

### 26.5 동시에 실행하면 안 되는 것

Optuna trial 중에는 다음을 같이 실행하지 않는다.

- vehicle dynamics profiler
- profiler loop guard
- 같은 MORAI에 명령을 보내는 다른 control publisher
- 같은 ROS domain의 다른 planner/control stack

MORAI GUI에서 시나리오, 센서, 차량을 바꾸면 기존 study 점수와 비교할 수
없다. 변경했다면 experiment context를 바꾸어 새 fingerprint study를
사용해야 한다.

---

## 27. 현재 튜닝 범위에 대한 해석

### 27.1 왜 target speed를 Optuna에서 제외했는가

target speed까지 자유롭게 두면 느리게 달리는 후보가 CTE와 overspeed에서
유리해질 수 있다. 현재 목표는 58.5 km/h 운용 조건에서 가장 빠르고
안정적인 controller response를 찾는 것이므로 target을 고정했다.

### 27.2 왜 횡가속도 6.0을 제외했는가

횡가속도 상한을 같이 튜닝하면:

- 낮은 값은 곡선에서 느려져 CTE가 좋아질 수 있고
- 높은 값은 elapsed time이 좋아질 수 있다.

그러면 controller gain 비교와 속도 정책 비교가 섞인다. 현재 Optuna는
제어 응답 파라미터를 먼저 찾기 위해 6.0으로 고정했다. 실제 미끄러짐
한계 검증은 별도 안전 실험으로 다루는 편이 낫다.

### 27.3 왜 `Ki=0`인가

현재 trial은 reset을 반복하고 곡선/직선 목표속도가 계속 변한다. integral
항은 긴 정상상태 오차를 줄일 수 있지만:

- throttle/brake mode 전환
- 목표속도 급변
- 포화

에서 windup과 overshoot를 만들 수 있다. 우선 Kp/Kd와 물리 기반 목표속도
프로파일을 튜닝하고 Ki는 0으로 고정했다.

### 27.4 brake 목적함수는 급브레이크를 반영하는가

현재는 세 경로로 반영한다.

1. 목표속도보다 실제 속도가 높으면 `target_overspeed²` 증가
2. 이미 목표속도 이하인데 브레이크를 쓰면 `unnecessary_brake²` 증가
3. brake 0.95 이상이 지속되면 `brake_saturation_time` 증가

따라서 brake Kp가 너무 작아 감속하지 못하는 경우와 너무 커서 전제동을
남발하는 경우를 둘 다 구분할 수 있다.

다만 현재 목적함수에는 jerk, 승차감, brake command 변화율은 없다.
사용자 요구에 따라 의도적으로 제외한 상태다.

---

## 28. 권장 분석 순서

실제 Optuna를 시작한 뒤에는 best cost 하나만 보면 안 된다.

1. feasible trial 수를 확인한다.
2. 앞축·뒤축 max CTE가 0.8 m 근처에 몰리는지 확인한다.
3. elapsed time 감소가 overspeed 증가로 얻어진 것인지 확인한다.
4. target overspeed integral을 확인한다.
5. unnecessary brake integral과 saturation time을 함께 확인한다.
6. trajectory CSV에서 곡선 진입 전 목표속도가 부드럽게 내려가는지 본다.
7. 곡선 탈출 후 target speed와 실제 speed가 얼마나 빨리 회복되는지 본다.
8. 같은 후보를 1회가 아니라 반복 검증해 MORAI 변동성을 확인한다.
9. winner를 perception/DWA가 켜진 스택에서 동적 장애물과 함께 검증한다.

특히 Profile Stanley 효과는 다음 그래프에서 가장 잘 보인다.

- route progress 대 target speed
- route progress 대 actual speed
- route progress 대 curvature speed cap
- 시간 대 brake command
- 시간 대 speed error
- route progress 대 CTE

---

## 29. 현재 구현되지 않은 것

다음은 논의했지만 현재 소스에 구현되지 않았다.

- MPC 종방향 제어
- 원하는 가속도를 직접 추종하는 acceleration controller
- 속도별 throttle/brake inverse map feed-forward
- feed-forward 페달 + PID feedback 결합
- Optuna intermediate report와 pruner
- jerk/승차감 목적함수
- 동일 파라미터의 여러 MORAI 반복 결과를 하나의 trial로 평균내는 구조
- weather에 따라 다른 차량 동역학 프로파일 자동 선택
- 55 km/h 초과 구간의 직접 측정 프로파일

현재 구조는 다음 중간 단계다.

```text
곡률 + 측정 가감속 능력
        |
        v
미래 경로 목표속도 계획
        |
        v
throttle/brake 분리 PID
        |
        v
MORAI
```

향후 feed-forward를 추가한다면:

```text
경로 목표속도
        |
        v
목표 가속도 생성
        |
        +-- 속도별 inverse pedal map -> feed-forward
        +-- 속도 오차 PID            -> feedback correction
        |
        v
최종 throttle/brake
```

순서가 적절하다. full MPC는 constraint와 예측 모델이 더 필요하므로,
현재 프로파일 기반 속도 계획과 feed-forward/PID를 먼저 검증한 뒤 비교하는
편이 구현 복잡도 대비 효과를 판단하기 쉽다.

---

## 30. 현재 확인된 것과 아직 확인하지 않은 것

### 현재 소스/데이터에서 확인됨

- `stanley`와 `profile_stanley` backend가 둘 다 존재한다.
- 두 backend는 같은 Stanley 횡제어와 PID를 사용한다.
- Profile Stanley는 측정 종방향 envelope와 제동 지연을 사용한다.
- 일반 Stanley도 고정 한계를 사용한 경로 속도 계획을 한다.
- 현재 목표 58.5 km/h, 최대 60 km/h, 횡가속도 상한 6.0 m/s²다.
- 조향 lookahead는 속도 비례이고 곡률 lookahead는 별도 고정 거리다.
- throttle/brake PID gain이 분리되어 있다.
- brake 관련 세 목적함수가 존재한다.
- Optuna는 실제 ROS planner 파라미터를 바꾸고 실제 controller를
  재생성한다.
- 세 PC의 동일 dependency와 PostgreSQL shared-study 동시 접근이
  동작한다.

### 아직 live MORAI 결과로 확인되지 않음

- 현재 Profile Stanley가 일반 Stanley보다 실제 코스에서 더 빠르면서
  CTE가 낮은지
- 3대 MORAI 실제 병렬 trial이 끝까지 안정적으로 반복되는지
- 현재 9개 탐색 범위가 충분한지
- 현재 목적함수 weight가 원하는 brake 감각과 정확히 일치하는지
- Optuna winner가 perception/DWA 동적 장애물 환경에서도 좋은지

즉 현재 상태는 **코드 경로, 데이터 주입, 목적함수, 분산 저장 환경까지
준비된 상태**이며, 실제 MORAI 최적화 결과가 이미 나왔다고 표현하면
안 된다.
