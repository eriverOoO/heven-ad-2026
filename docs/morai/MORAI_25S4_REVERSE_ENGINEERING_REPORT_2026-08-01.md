# MORAI 25.S4.MolitComp03 센서·통신·충실도 리버스 엔지니어링 보고서

- 최초 작성일: 2026-08-01 (Asia/Seoul)
- 최종 갱신일: 2026-08-04 (LiDAR scan-time/deskew, GNSS RMC, IMU 원천·noise·UDP scheduler·3호스트 주기/중복, bridge latency, wheel/encoder, IMU 20Hz·camera 4대 전체 stream timing 및 Linux/Windows scheduler 분석 반영)
- 분석 대상: `Simulator_v.S4.251001.MolitComp03_Linux`
- 분석 목적: MORAI가 실제 센서 물리까지 재현하는 고충실도 시뮬레이터인지 판단하고, 개발 시 반드시 고려해야 할 구현 특성·오류·비현실적 요소를 식별한다.
- 분석 방식: 설치 파일과 설정의 정적 분석, C# IL 역분석, Unity 자산·셰이더 참조 분석, 실행 중 UDP/ROS 데이터 측정, 대회 내부 명세와 MORAI 공식 문서 교차검증

## 1. 핵심 결론

MORAI 25.S4.MolitComp03는 다음 용도에는 유용하다.

- 지도와 차량 및 센서 장착 위치를 포함한 시스템 통합 시험
- 맑은 날의 기본적인 카메라·LiDAR 인지 알고리즘 개발
- Velodyne, NMEA, IMU, 차량 상태, 제어 UDP 프로토콜 연동
- 대회 시나리오와 제한된 fault/blackout 대응 시험
- 반복 가능한 SIL(Software-in-the-Loop) 시험

그러나 다음 용도의 근거로 사용하기에는 충실도가 부족하다.

- 실제 안개·비 환경에서의 LiDAR 성능 검증
- 실차 GNSS multipath/NLOS 및 RTK 성능 검증
- 실제 MEMS IMU의 온도·bias·scale factor·진동 특성 검증
- 실제 카메라의 rolling shutter, 노출, ISP 및 전자 잡음 검증
- 실시간 센서 동기화 및 네트워크 지연의 정밀 검증
- 실제 Ioniq 5 동역학 또는 액추에이터 응답의 검증

이번 심층 갱신에서 추가로 확정한 핵심 사실은 다음과 같다.

- **LiDAR는 rolling scan이 아니라 snapshot에 가깝다.** CH16 360° capture가 한 Unity frame 안에서 끝나며, sector 사이에 `yield`, physics step 또는 ego-pose 갱신이 없다. 완성된 75개 packet은 뒤이어 짧은 burst로 송신된다.
- **현재 브리지가 0~0.1초 point time을 인공 생성한다.** packet 도착시각과 azimuth로 시간을 펼치므로, 이 시간을 사용한 deskew는 MORAI에 없던 운동왜곡을 새로 만든다. MORAI profile에서는 point time을 전부 0으로 두고 외부 deskew와 FAST-LIO2 내부 undistortion을 우회하는 것이 맞다.
- **GNSS 속도는 존재하지만 Doppler 측정이 아니다.** RMC speed는 Unity `Rigidbody.velocity.magnitude`, course는 실제 진행방향이 아니라 sensor Transform yaw다. 위성 위치·반송파·Doppler·multipath 계산은 없다.
- **실시간 UDP RMC에는 속도 단위변환 버그가 있다.** m/s→knot에 `0.00053995`를 곱한다. 올바른 계수 `1.94384`보다 3,600배 작아서 일반 주행속도는 소수점 1자리 NMEA에서 `0.0 knot`가 된다.
- **IMU는 encoder에서 만들지 않는다.** orientation은 Unity Transform 자세, angular velocity는 Rigidbody angular velocity, acceleration은 Rigidbody velocity 차분과 gravity 보정으로 생성한다. 여기에 설정된 noise generator를 더한다.
- **IMU 50Hz는 보장 주기가 아니라 요청값이다.** 동일 binary와 `sensorPeriod=0.02s`를 사용한 세 호스트에서 raw UDP datagram은 약 34.0/47.2/34.0Hz, 서로 다른 device timestamp는 약 25.2/38.0/18.0Hz로 달랐다. MORAI의 coroutine scheduler는 늦어진 시간을 보상하려고 0초 wait의 catch-up 송신을 하므로 긴 gap 뒤 2ms 미만 burst와 같은 timestamp 반복이 생긴다.
- **IMU device timestamp는 physics sample time이 아니다.** realtime UDP serializer가 송신 직전에 host wall clock의 seconds와 millisecond를 따로 읽어 만든 1ms 해상도 값이다. 따라서 고유 timestamp 수는 새 상태의 근사 지표일 뿐 정확한 physics update 수로 단정할 수 없다.
- **현재 heven22의 주기 저하는 ROS bridge나 localhost UDP 병목으로 설명되지 않는다.** application/kernel drop은 0이었고 UDP receipt→ROS callback 지연은 median 약 1.55ms, p95 약 2.51ms였다. 병목은 MORAI 내부 frame/coroutine scheduling과 host별 scene/render/worker 부하 쪽에 있다.
- **2026-08-04 현재 camera는 4대가 모두 활성이다.** IMU 20Hz, camera 4대 20Hz 조건의 90초 동시 계측에서 camera는 각 19.84~19.85Hz였고, IMU 19.83Hz, GNSS fix 18.87Hz, status/collision 약 40.5Hz, LiDAR cloud 8.62Hz였다. 주기 미달과 jitter는 IMU 하나에 국한되지 않는다.
- **Target Frame Rate 60은 실측 FPS 보장이 아니다.** 현재 `TargetFrameRate=60`, time scale 1, quality 2이지만 pause 화면에서 X11 drawable 갱신은 약 25.07event/s였다. 이는 compositor가 합칠 수 있는 window damage event이므로 정확한 Unity present/physics FPS는 아니며, 동적 주행 FPS로도 해석하면 안 된다.
- **현재 `wheel_speed`는 encoder가 아니다.** 대회 UDP `EgoVehicleStatus.velocity.x`는 차체 Rigidbody의 body-forward 속도다. 팀 localization은 이 값을 `wheel_speed`라는 topic으로 다시 포장한다. tick, pulse/rev, 양자화, encoder 전자잡음은 없다.

종합 판정은 다음과 같다.

> MORAI는 기하·시나리오·프로토콜 중심의 SIL 환경으로는 실용적이지만, 실제 센서의 대기·광학·전자·위성·통신 물리까지 검증된 고충실도 센서 시뮬레이터는 아니다.

특히 안개나 비가 LiDAR 거리, intensity, dropout, false return에 물리적으로 반영되는 구현은 찾지 못했다. 따라서 현재 대회 빌드에서 안개 대응만을 이유로 LiDAR Statistical Outlier Removal을 상시 적용할 필요는 없다.

이 보고서에서 말하는 "정확도"는 둘로 분리한다.

1. **시뮬레이터 내부 truth에 대한 수치 정밀도**: NMEA 자릿수, Velodyne 거리 count, float32 직렬화처럼 코드로 계산할 수 있는 값
2. **실제 센서/실차에 대한 정확도**: 제조사 센서 및 Ioniq 5 실차와 비교 validation이 필요한 값

MORAI는 첫 번째는 일부 정량화할 수 있지만, 두 번째를 보장할 validation 자료는 현재 설치물에서 찾지 못했다.

## 2. 증거 범위와 신뢰도

이 보고서에서는 사실의 근거를 다음과 같이 구분한다.

| 등급 | 의미 |
|---|---|
| 코드 확인 | 현재 빌드의 C# IL, 필드, 메서드 호출 또는 직렬화 코드를 확인함 |
| 설정 확인 | 현재 저장된 센서 JSON에 존재하는 값 |
| 실행 관측 | 현재 실행 중인 동일 빌드의 UDP 또는 ROS 출력에서 측정함 |
| 문서 확인 | MORAI 공식 문서 또는 대회 내부 명세에서 확인함 |
| 추론 | 코드와 실행 현상에 가장 잘 부합하지만 내부 설계 문서로 확정하지 못함 |
| 미확인 | 구현 또는 검증 자료를 찾지 못했으며 존재하지 않는다고 단정할 수 없음 |

분석한 관리형 어셈블리는 다음과 같다.

- 파일: `/home/heven/MoraiLauncher_Lin/MoraiLauncher_Lin_Data/Project/Simulator/Simulator_v.S4.251001.MolitComp03_Linux/Simulator_Data/Managed/Assembly-CSharp.dll`
- 크기: `34,745,856 bytes`
- SHA-256: `a2ad7d12af07e2b3173c9c3e66a4c499af9fe70d2d91d48d5671db8a2547d669`
- 형태: Mono/.NET C# 관리형 어셈블리. IL2CPP가 아니므로 메서드 IL과 타입 참조를 분석할 수 있다.
- 제한: 타입과 메서드 이름 상당수가 난독화되어 있어 일부 의미는 필드 타입, 호출 대상, 상수, 직렬화 결과를 조합해 판정했다.

공개 MORAI 문서는 주로 24.R1/24.R2이고 분석 빌드는 25.S4이므로, 현재 실행 결과와 C# 구현을 우선 근거로 사용했다.

### 2.1 핵심 IL 증거 지도

아래 RVA는 위 SHA-256의 `Assembly-CSharp.dll`에만 유효하다.

| 기능 | 타입/경로 | 핵심 RVA 및 확인 내용 |
|---|---|---|
| LiDAR one-frame capture | `CLidar3D` active CH16 delegate | `0x986314`: 360° sector loop 안에 render/readback은 있으나 yield/physics/pose advance 없음 |
| LiDAR packet burst | `CLidar3D` UDP path | `0x986AB4`, `0x986DEC`, `0x987B68`: 완성 buffer를 75×1206 bytes로 절단·연속 send |
| GNSS position/noise | `CGPS.SimXYZ2LLH` | `0x93F82C`: Transform.position XYZ에 Box-Muller noise 후 LLH 투영 |
| GNSS realtime UDP | `CGPS` commType 1 delegate | `0x9431C4`: GGA/RMC 생성, 고정 quality/satellites/HDOP 확인 |
| GNSS RMC speed | `CGPS` speed helper | `0x9421C8`: Rigidbody speed magnitude와 잘못된 0.00053995 계수 확인 |
| IMU UDP | `CIMU` commType 1 delegate | `0x9733D4`: `GetIMUData` 후 quaternion/gyro/accel 10 doubles 송신 |
| IMU UDP scheduler | `CIMU.SensorStart` commType 1 / iterator | method token `0x0601389B`가 iterator `0x060138C6`을 시작. `Stopwatch`로 지연·overshoot를 계산하고 음수 wait를 0으로 clamp한 뒤 `WaitForSeconds` 사용 |
| IMU realtime timestamp | `CIMU` UDP serializer | method token `0x0601385A`: `DateTime.Now` 기반 seconds와 별도 `DateTime.UtcNow.Millisecond×1,000,000`을 직렬화 |
| IMU data assembly | `CIMU.GetIMUData` | `0x975368`; orientation `0x9743D4`, gyro `0x973020`, accel `0x972A30` |
| IMU physics source | `CIMU` physics update clones | Rigidbody velocity 차분/fixedDeltaTime/gravity, angularVelocity, InverseTransformDirection 확인 |
| IMU white noise | `MORAI.Noise.WhiteNoiseGenerator` | `0x1342ED1`, scale helper `0x1344BD0`: `stdev×(1/60)/sqrt(dt)` |
| Ego signed speed | `MoraiObjectBase.GetSignedVelocity` | `0x2CF0E`: body-frame `Rigidbody.velocity.z×3.6` |
| Ego status serializer | `MoraiInfoPublisher` | 예: `0x2A9D60`: `GetSignedVelocity`, converted angular velocity, local acceleration 직접 사용 |
| 선택형 per-wheel speed | `NaverInfoPublisher_ObdWheelSpeeds` | 예: `0x27745C`: 네 wheel collider `angularVelocity`와 radius 계열 변환, encoder tick 없음 |

난독화 때문에 동일 기능의 commType/ROS/동기모드 clone이 여러 개 존재한다. 판정은 이름 하나가 아니라 `SensorStart`의 delegate 선택, field dataflow, serializer와 실행 packet을 함께 추적했다.

## 3. 현재 센서 구성

기본 설정 파일:

`/home/heven/MoraiLauncher_Lin/MoraiLauncher_Lin_Data/SaveFile/Sensor/25.S4.MolitComp03/SensorInfo_2023_Hyundai_Ioniq5.json`

별도 노이즈 프리셋:

`/home/heven/MoraiLauncher_Lin/MoraiLauncher_Lin_Data/SaveFile/Sensor/25.S4.MolitComp03/noise_SensorInfo_2023_Hyundai_Ioniq5.json`

2026-08-03 기준 두 파일은 SHA-256 `8349697afaebe21d3c4bf0f85de961b8ae208a2f83780ae3783120a85fdda1c3`로 **byte-identical**하다. 따라서 현재 설치 상태에서는 "기본=noise off, noise preset=on" 구분이 없다. 아래 활성 상태가 두 파일에 동일하게 들어 있다.

현재 `GroundTruthList`는 비어 있다. 아래 센서들이 ground-truth 전용 sensor message를 구독하는 구조가 아니라, 각 센서 구현이 Unity scene/Transform/Rigidbody state를 읽어 sensor 형식으로 바꾼다.

### 3.1 카메라

| ID | 위치 m | 자세 deg | 해상도 | FOV | 설정 주기 | UDP |
|---:|---|---|---:|---:|---:|---|
| 1 | (1.90, 0.00, 1.20) | (0, 2, 0) | 1280×720 | 90° | 0.05s, 20Hz | 9290 → 9291 |
| 2 | (1.15, 0.65, 1.20) | (0, 10, 70) | 640×480 | 130° | 0.05s, 20Hz | 9292 → 9293 |
| 3 | (1.15, -0.65, 1.20) | (0, 10, 290) | 640×480 | 130° | 0.05s, 20Hz | 9294 → 9295 |
| 4 | (1.90, 0.00, 1.20) | (0, 330, 0) | 1920×1080 | 120° | 0.05s, 20Hz | 9306 → 9307 |

모든 카메라의 설정 JPEG quality는 90이고 generic Gaussian noise는 비활성이다. 2026-08-01 초기 실행에서는 1~3번 카메라만 패킷이 관측됐고 4번은 설정에만 존재했지만, 2026-08-04 full competition bridge 실행에서는 4번 traffic-light camera까지 활성화되어 네 stream 모두 약 19.85frame/s로 관측됐다. 따라서 4번 camera가 simulator에서 본질적으로 송신 불가능한 것은 아니며, 초기 미관측은 당시 launch/enable 상태에 한정한다.

### 3.2 3D LiDAR

- ID: 5
- 위치: `(1.15, 0.00, 1.40)m`
- 자세: `(0, 0, 0)°`
- 설정 주기: 0.1s, 10Hz
- UDP: 2369 → 2368
- 설정 최대거리: 100m
- 타입: `lidar3DType=0`, CH16/VLP-16 packet profile
- 한 scan 구성: 16 channel, 12 block/packet, 75 packet/scan
- Gaussian range noise: 비활성 (`gnGenerator.isActivated=false`, `noiseRatio=0`)
- intensity 사용
- VLP-16/Velodyne 호환 패킷 형태

### 3.3 GPS

- ID: 6
- 위치: `(0, 0, 1.2)m`
- 설정 주기: 0.05s, 20Hz
- UDP: 9296 → 9297
- **현재 기본 설정 noise: 활성**
- Gaussian mean 0, standard deviation 0.5
- 설정 `gpsStatus=2`
- 실제 UDP GGA fix quality: 코드에 1로 고정
- 실제 UDP satellites: 9 고정, HDOP: 1.0 고정, geoid separation: 0.0 고정

즉 GGA quality 1이 관측된 것은 noise가 꺼졌다는 뜻이 아니다. 현재 설정에서는 위치 Gaussian noise가 켜져 있지만, active UDP serializer가 `gpsStatus=2`를 GGA quality에 반영하지 않고 1을 고정 출력한다.

### 3.4 IMU

- ID: 7
- 위치: `(0, 0, 1.2)m`
- 설정 주기: 0.02s, 50Hz
- UDP: 9298 → 9299
- generic Gaussian generator: 비활성
- random walk: 6축 모두 비활성, 저장 stdev 0.001
- bias instability: 6축 모두 비활성, 저장 stdev 0.001, correlation time 0.02s
- **white noise: 현재 기본 설정에서 6축 모두 활성**
  - acceleration 설정 stdev: x/y/z 각각 0.041
  - angular 설정 stdev: roll 0.003, pitch 0.003, yaw 0.00314

이 수치는 출력 sample의 그대로의 표준편차가 아니다. 구현이 `설정 stdev × (1/60) / sqrt(Time.deltaTime)`로 다시 스케일한다.

## 4. 충실도 평가표

| 분야 | 평가 | 근거 |
|---|---:|---|
| 지도·좌표·센서 외부 파라미터 | 중상 | ENU, UTM52N, 차량축, 센서 장착 위치가 명확함 |
| 맑은 날 LiDAR 기하 | 중간 | VLP-16 패킷, 채널, 회전, material intensity 근사 |
| 안개·비 LiDAR | 매우 낮음 | 대기 감쇠·산란·강우 입자·dropout 연결 없음 |
| GNSS 물리 | 낮음 | ground truth 좌표 투영 + 단순 Gaussian + blackout + NMEA 양자화 |
| GNSS 속도/course | 매우 낮음 | Rigidbody speed/yaw 대입, Doppler 없음, realtime RMC knot 변환 버그 |
| IMU 물리 | 중하 | Rigidbody/Transform truth 기반 + 단순 noise, orientation은 거의 ground truth |
| wheel/encoder | 매우 낮음 | 현재 wheel_speed는 차체 Rigidbody 속도이며 실제 encoder 모델이 아님 |
| 카메라 장면 시각성 | 중간 | Unity 렌더링과 Enviro 장면 효과 |
| 카메라 센서 물리 | 낮음 | rolling shutter, 노출, ISP, 전자 noise 모델 근거 없음 |
| 시간·동기화 | 낮음 | FPS 의존, 목표 주기 미달, 동일 sample 중복 |
| UDP 프로토콜 호환성 | 중상 | 대회 인터페이스 개발에는 충분하나 integrity/recovery가 없음 |
| 차량 동역학 | 중간 또는 미검증 | Vehicle Physics/Pacejka 구성은 있으나 실차 validation 자료 없음 |

### 4.1 현재 출력별 truth·noise·수치 정밀도 요약

| 출력 | 원천 | 현재 자동 noise | 코드로 정량화 가능한 정밀도 | 실제 센서 정확도로 해석 가능한가 |
|---|---|---|---|---|
| Camera | Unity render target | Gaussian off, JPEG Q90 | 해상도와 8-bit JPEG 압축 수준 | 아니오. lens/rolling shutter/exposure/ISP validation 없음 |
| LiDAR range | 한 frame의 GPU depth/reflectivity capture | off (`noiseRatio=0`) | Velodyne 거리 count 2mm 단위, 각 point 최대 양자화 약 ±1mm | 아니오. mesh·depth·material 모델 및 대기물리 오차가 별개 |
| LiDAR time | simulator snapshot 뒤 packet burst | simulator가 유효한 rolling point time을 제공하지 않음 | 현재 팀 bridge가 0~약 0.1s를 합성 | 아니오. 합성 시간은 물리 측정시각이 아님 |
| GNSS position | sensor Transform.position | 로컬 XYZ 독립 Gaussian σ=0.5m 활성 | 위경도 약 14.8~18.5cm grid, altitude 0.1m | 내부 truth+noise일 뿐 실제 GNSS RF 정확도 아님 |
| GNSS speed | Rigidbody speed magnitude | 없음 | NMEA 0.1knot이나 realtime 변환 버그로 보통 0.0 | Doppler 속도가 아니므로 사용 금지 권고 |
| GNSS course | sensor Transform yaw | 없음 | NMEA 0.1° | COG가 아니며 후진·slip에서 틀림 |
| IMU orientation | sensor/vehicle Transform quaternion | orientation noise 없음 | Unity float quaternion을 double로 직렬화 | 거의 ground truth 자세. 실제 AHRS 정확도 아님 |
| IMU gyro | Rigidbody angularVelocity의 body 변환 | white noise 활성 | dt=1/60s일 때 설정상 σ 약 0.000387/0.000387/0.000405rad/s | 실제 MEMS bias/온도/대역폭을 대표하지 않음 |
| IMU accel | Rigidbody velocity 차분, gravity 보정 | white noise 활성 | dt=1/60s일 때 설정상 축별 σ 약 0.00529m/s² | physics 차분/FPS 영향이 있고 실제 MEMS validation 없음 |
| Ego speed | body-frame Rigidbody.velocity.z | 별도 noise 없음 | float32, 30m/s 부근 수치 간격 약 2µm/s | simulator truth에 가까우나 실차 wheel encoder가 아님 |
| 선택형 OBD wheel speed | wheel collider angularVelocity×radius 계열 | encoder noise/quantization 없음 | Unity physics float/time-step 수준 | 현재 대회 UDP 미사용, 실제 encoder 정확도 아님 |

위 표의 LiDAR 2mm, float32 µm/s 같은 값은 **직렬화 분해능**이지 장면이나 차량 모델의 절대 정확도가 아니다. 작은 숫자를 실차 정확도로 오해하면 안 된다.

## 5. GNSS 상세 분석

### 5.1 NMEA 위경도 양자화

현재 C# 구현은 위도와 경도를 다음 형식으로 출력한다.

```text
위도: ddmm.mmmm
경도: dddmm.mmmm
```

최소 단위는 다음과 같다.

```text
0.0001 arc-minute = 1 / 600000 degree
```

현재 K-City 부근 위도 약 37.2409916667°에서 WGS84 곡률을 적용하면:

| 방향 | NMEA 좌표 1 LSB |
|---|---:|
| 남북 | 약 18.497027cm |
| 동서 | 약 14.788328cm |

반올림에 따른 이론상 최대 오차는:

- 남북: 약 ±9.2485cm
- 동서: 약 ±7.3942cm
- 두 축 결합 최대: 약 11.8410cm

따라서 “GPS가 항상 15~18cm 틀린다”가 아니라 “좌표 표현 격자가 축별 약 14.8~18.5cm 간격이고, 양자화 오차가 반 격자 범위 내에 존재한다”가 정확한 표현이다.

### 5.2 다른 NMEA 해상도

- GGA altitude: 소수점 1자리, 0.1m 간격, 반올림 최대 약 ±5cm
- GGA/RMC UTC format: `HHMMSS.{millisecond:D1}` 계열
- 소수부가 고정 3자리가 아니다. 예를 들어 `.4`는 0.4초가 아니라 코드상 4ms 값을 문자열로 붙인 결과일 수 있다.
- timestamp 정렬에 NMEA UTC 문자열을 직접 decimal second로 해석하면 안 된다.
- RMC speed: 0.1 knot 간격 = 약 0.0514444m/s
- speed 반올림 오차: 최대 약 ±0.0257222m/s
- RMC course: 0.1° 간격, 최대 약 ±0.05°
- HDOP: 소수점 1자리
- satellites: 정수, 기본 전송값 9
- NMEA checksum은 구현되어 있음
- 한 센서 tick에서 GGA와 RMC를 별도 UDP send로 전송

UDP 전송 자체 때문에 cm 단위 오차가 생기는 것은 아니다. 위치 분해능을 제한하는 것은 현재 NMEA 자릿수와 serializer이며, 실제 오차 분산은 그 위에 더해진 Gaussian 설정과 simulator physics에 의해 결정된다.

### 5.3 GNSS noise의 실제 의미

GPS noise는 위경도에 직접 더해지는 것이 아니다. `CGPS.SimXYZ2LLH`는 **GPS sensor Transform.position**의 로컬 Cartesian x/y/z 각 축에 독립적인 Box-Muller Gaussian sample을 더한 뒤 WGS84 좌표로 투영한다. 시간상관, 축간상관, 위성 공통오차는 없다.

현재 **기본 설정 자체**의 standard deviation 0.5는 코드상 로컬 거리값에 직접 사용되므로 축별 약 0.5m로 해석해야 한다. 구형 공식 문서에서 standard deviation을 `%`처럼 설명하는 것과 구현이 일치하지 않는다.

2026-08-03 동일 빌드 정지 220 sample 관측은 다음과 같았다.

| 항목 | sample 표준편차 |
|---|---:|
| north | 약 0.481m |
| east | 약 0.525m |
| altitude | 약 0.537m |

설정 σ=0.5m와 일치한다. 등방성 σ=0.5m로 근사하면 수평 CEP50은 약 0.59m, R95는 약 1.22m다. 이 값은 현재 설정의 정지 분포이지 실제 GNSS 수신기의 환경별 정확도 보증값은 아니다.

### 5.4 GNSS 속도·course와 Doppler 판정

RMC에는 speed와 course 필드가 나오지만 생성 원천은 다음과 같다.

```text
speed source  = Unity Rigidbody.velocity.magnitude
course source = GPS sensor Transform yaw
```

따라서 다음이 아니다.

- 위성 신호의 carrier Doppler
- pseudorange rate least squares
- 시간에 따른 satellite ephemeris
- GNSS receiver clock drift를 포함한 속도해
- 실제 velocity vector로 계산한 course over ground

특히 course는 속도 벡터의 방향이 아니라 차체/sensor heading이므로 후진, 횡미끄럼, 정지상태에서도 실제 COG와 다를 수 있다. 정지 관측에서도 track은 약 214.6°로 고정돼 있었다.

실시간 UDP helper는 다음 변환을 사용한다.

```text
reported_knots = speed_mps × 0.00053995
correct_knots  = speed_mps × 1.94384
```

`0.00053995`는 사실상 m/s가 아니라 m/h 계열 입력에 맞는 값이며 올바른 계수보다 3,600배 작다. RMC가 소수점 1자리이므로 약 92.6m/s 미만은 대부분 `0.0 knot`로 반올림된다. sync-mode의 다른 코드 경로에는 올바른 1.94384가 보이므로 realtime UDP 경로의 고유 결함으로 판정한다.

개발 결론:

- 이 RMC speed를 velocity update로 사용하지 않는다.
- 이 RMC course를 일반 COG 또는 sideslip 관측으로 사용하지 않는다.
- 현재 팀 localization의 `gps_course_enabled: false`는 이 빌드에는 올바른 방어다.
- 속도는 현재 대회 인터페이스에서는 `EgoVehicleStatus.velocity`를 쓰되, 그것도 encoder가 아니라 simulator truth라는 점을 명시한다.

### 5.5 구현되지 않았거나 근거가 없는 실제 GNSS 현상

- GPS/Galileo/GLONASS/BeiDou 별 위성 궤도
- 시간에 따른 가시 위성 수와 위성 배치
- 위성 기하로 계산되는 DOP
- 이온층·대류권 지연
- 도심 협곡 multipath 및 NLOS
- 안테나 phase center와 차체 차폐
- 수신기 clock drift
- 반송파 위상과 RTK ambiguity
- RTK float/fix 전이와 baseline 의존성
- 위성 공통오차 및 축간 상관관계
- carrier Doppler와 pseudorange-rate 속도

MORAI GNSS는 현실적인 위성 수신기라기보다 ground truth 좌표를 투영하고 단순 noise와 blackout/fault를 추가하는 모델에 가깝다.

`EnviroSatellite` 타입은 하늘 시각환경용이며 `CGPS` 위치 계산 경로와 연결되지 않는다. 어셈블리의 `DopplerVelocity` 문자열은 Apollo protobuf field이고, `dopplerLevel`은 Unity AudioSource 참조다. GNSS RF Doppler 구현의 증거가 아니다.

GNSS 음영영역도 건물 geometry로 위성 LOS를 ray-trace하는 방식이 아니다. scenario JSON에 사용자가 배치한 box/volume의 `noiseType`으로 상태를 바꾸며, type 1 경로는 zero 좌표/status 계열을 반환한다. 따라서 multipath가 자연스럽게 발생하는 것이 아니라 수동 blackout/fault volume이다.

### 5.6 개발 주의사항

- 10cm 이내의 localization 차이를 UDP NMEA GPS만으로 평가하지 않는다.
- GGA fix quality와 blackout 상태를 반드시 확인하되, GGA quality 1을 noise-off 또는 높은 정확도의 증거로 해석하지 않는다.
- `latitude == 0` 또는 `longitude == 0`만으로 invalid 판정을 끝내지 않는다.
- blackout/fault 시 마지막 좌표 또는 zero 계열 값이 나올 가능성에 대비한다.
- HDOP와 satellites를 실제 위성기하 기반 품질 지표로 간주하지 않는다.
- RMC speed/course를 Doppler velocity/COG로 간주하지 않는다.
- ROS `NavSatFix` covariance는 실측 정지·주행 로그로 별도 추정한다.

현재 active UDP GGA는 fix quality 1, satellites 9, HDOP 1.0, geoid separation 0.0을 고정 출력한다. 설정의 `gpsStatus=2`와 active UDP 출력이 불일치한다.

현재 팀 브리지는 `ad_morai_bridge/ad_morai_bridge/message_conversion.py:279-284`에서 `horizontal sigma = HDOP × 3m`, vertical variance는 그 4배로 구성한다. HDOP가 1.0 고정이므로 ROS covariance는 수평 9m², 수직 36m²가 된다. 이는 MORAI의 실제 noise σ=0.5m에서 유도한 값이 아니라 브리지의 휴리스틱이다.

## 6. IMU 상세 분석

### 6.1 측정값 생성

코드상 IMU 출력은 다음과 같이 생성된다.

- sensor/vehicle Rigidbody의 현재·이전 `velocity` 차이를 `Time.fixedDeltaTime`으로 나눠 가속도 계산
- Unity gravity를 보정
- Rigidbody Transform으로 `InverseTransformDirection`하여 body/sensor local acceleration 계산
- Rigidbody `angularVelocity`를 같은 방식으로 local 축에 변환
- quaternion은 sensor Transform rotation에서 만든 Unity 자세를 좌표변환해 사용
- angular velocity와 linear acceleration에 random walk, bias instability, white noise를 축별로 더함

**wheel collider, wheel speed, encoder tick을 읽는 호출은 CIMU 경로에 없다.** IMU는 encoder 값으로부터 vehicle motion을 역산하는 것이 아니라 Unity physics state를 직접 읽는다.

active `GetIMUData` 경로의 축 매핑은 다음과 같다.

```text
angular x = -localRigid.z + roll noise
angular y =  localRigid.x + pitch noise
angular z = -localRigid.y + yaw noise

accel x =  localAcc.z + noise
accel y = -localAcc.x + noise
accel z =  localAcc.y + noise
```

정지 상태에서는 위쪽 축으로 약 +g가 출력되는 형태다.

orientation에는 현재 별도 noise가 더해지지 않는다. UDP payload가 double이어도 원천 quaternion과 Rigidbody vector는 Unity float이므로 double 직렬화가 물리 정확도를 높이지 않는다. 따라서 orientation은 실제 IMU/AHRS 측정보다 **ground-truth attitude**에 가깝고, gyro/accel은 **physics truth 계열 값 + 설정 noise**에 가깝다.

### 6.2 noise 모델

축별로 다음 구조가 존재한다.

- White noise
- Random walk
- Bias instability/Gauss-Markov

현재 기본 JSON의 실제 활성 상태는 다음과 같다.

| 항목 | accel x/y/z | gyro roll/pitch/yaw |
|---|---|---|
| generic Gaussian | off | off |
| random walk | off, stored stdev 0.001 | off, stored stdev 0.001 |
| bias instability | off, stdev 0.001, tau 0.02s | off, stdev 0.001, tau 0.02s |
| white noise | **on, 0.041/0.041/0.041** | **on, 0.003/0.003/0.00314** |

하지만 실제 구현에는 중요한 한계가 있다.

1. Noise 함수에 sensor period나 fixedDeltaTime이 아니라 `Time.deltaTime`이 전달된다.
2. 따라서 동일 noise 설정이라도 GPU/CPU 부하, 렌더링 FPS, frame jitter에 따라 통계가 달라질 수 있다.
3. Gauss-Markov 상태 전이는 정확한 `exp(-dt/tau)`가 아니라 `(1-dt/tau)` Euler 근사를 사용한다.
4. `dt`가 correlation time과 비슷하거나 더 크면 부정확하거나 비정상적일 수 있다.

Random walk는 코드상 이전 값에 다음 계열 값을 누적한다.

```text
Normal(mean, stdev × 4.629e-6 × sqrt(dt))
```

White noise는 다음 계열이다.

```text
Normal(mean, stdev × (1/60) / sqrt(dt))
```

Gauss-Markov bias는 다음 형태다.

```text
(1 - dt/tau) × previous
+ Normal(mean, stdev × (1/3600) × sqrt(1 - exp(-2dt/tau)))
```

예를 들어 `Time.deltaTime=1/60s`가 안정적으로 유지된다고 가정하면 현재 white-noise sample 표준편차는 다음과 같다.

| 출력 | 계산된 sample σ |
|---|---:|
| accel x/y/z | 약 0.00529m/s² |
| gyro roll/pitch | 약 0.000387rad/s |
| gyro yaw | 약 0.000405rad/s |

이는 가정한 frame dt에서 noise 함수만 계산한 값이다. 실제 출력 분산에는 velocity finite difference, physics jitter, 동일 sample 반복, frame-rate 변화가 더해진다. 또한 dt가 1/30s이면 white-noise sample σ가 더 작아지는 등, 설정값만 보고 고정 covariance를 부여할 수 없다.

### 6.3 IMU 정확도 해석

| 성분 | simulator 내부 truth에 대한 성격 | 실제 IMU와의 차이 |
|---|---|---|
| orientation | Transform 자세를 거의 직접 제공 | 실제 AHRS의 drift, 자기장 교란, 추정오차 없음 |
| angular velocity | Rigidbody 각속도 + 현재 white noise | sensor bandwidth, scale factor, saturation, thermal bias 없음 |
| linear acceleration | velocity finite difference + gravity 보정 + white noise | 진동 전달, mounting resonance, ADC/anti-alias filter 없음 |

따라서 FAST-LIO2/ESKF에서 orientation covariance를 작게 잡으면 MORAI에서는 좋아 보여도 실차 일반화가 나빠질 수 있다. 반대로 gyro/accel noise covariance는 JSON의 0.041/0.003을 그대로 variance로 넣어서는 안 되며, 실제 유효 dt와 정지 log에서 output 단위 분산을 다시 추정해야 한다.

### 6.4 빠진 실제 IMU 현상

- orientation 자체의 추정오차
- scale factor error
- cross-axis sensitivity
- sensor misalignment
- 온도에 따른 bias 변화
- 진동·공진·mounting 특성
- ADC quantization
- saturation 및 clipping
- 센서 bandwidth와 anti-aliasing filter
- coning/sculling correction
- 자기계

### 6.5 UDP와 timestamp 문제

현재 25.S4 IMU UDP envelope는 `#IMUData$`이며 전체 길이는 115 bytes다.

현재 payload는 다음을 포함한다.

- seconds: 4 bytes
- nanoseconds: 4 bytes
- quaternion w, x, y, z: little-endian double 4개
- angular velocity x, y, z: little-endian double 3개
- linear acceleration x, y, z: little-endian double 3개

구형 24.R2 공식 문서는 timestamp가 없는 107 bytes로 설명하므로 현재 구현과 다르다.

realtime UDP serializer의 timestamp 구현은 다음과 같다.

- seconds: `DateTime.Now`에서 1970 epoch 기준 차이를 계산
- nanoseconds: 별도의 `DateTime.UtcNow.Millisecond × 1,000,000`
- 해상도: 실제로는 1ms
- 측정 의미: physics step 또는 IMU 상태를 획득한 시각이 아니라 **UDP 직렬화 시점의 host wall clock**

seconds와 milliseconds를 서로 다른 `DateTime` 호출로 샘플링하므로 초 경계에서 두 필드 조합이 모순될 가능성이 있다. 또한 NTP 보정이나 host wall-clock jump의 영향을 받을 수 있고, 같은 millisecond 안의 catch-up 송신은 같은 device timestamp를 가질 수 있다.

세 호스트 모두 현재 IMU가 `127.0.0.1:9298 → 9299`로 송신되므로 LAN/Tailscale을 지나지 않으며 host 간 NTP offset은 현재 packet 전달에는 영향을 주지 않는다. 2026-08-03 `systemd-timesyncd` 확인에서는 세 호스트 모두 synchronized였지만 추정 offset은 heven22 약 -6.9ms, heven-right 약 -3.2ms, heven-laptop 약 -17.9ms로 서로 달랐다. heven-right와 heven-laptop은 `LocalRTC=yes`였는데 이는 현재 loopback 문제의 원인은 아니지만 Linux 기반 다중 host 시간관리에는 덜 이상적인 설정이다. 향후 MORAI와 bridge를 서로 다른 PC에 둘 때 device timestamp를 직접 ROS header로 사용하려면 chrony/PTP 수준의 별도 동기화 검증이 필요하다.

보존된 IMU 저장 샘플 중 다음 이상값이 하나 발견됐다.

```text
TimeStamp.nSecs : 3074316224
```

이는 1초 범위인 0~999,999,999를 넘는다. 실시간 UDP에서 지속적으로 재현하지 못했으므로 시스템 전체의 확정 버그가 아닌 단일 보존 이상으로 분류한다.

### 6.6 주기와 중복

설정의 `sensorPeriod=0.02s`는 50Hz 목표값이지만, 현재 coroutine은 정확한 50Hz hardware clock처럼 동작하지 않는다. commType 1 sensor iterator는 `Stopwatch`로 처리시간과 이전 overshoot를 재고 대략 다음 wait를 계산한다.

```text
wait = sensorPeriod - elapsed - previous_overshoot
wait < 0 이면 0으로 clamp
yield WaitForSeconds(wait)
```

Unity frame이 늦으면 다음 wait가 0이 되어 catch-up datagram을 즉시 이어 보낸다. 이 때문에 수십 ms의 긴 gap 뒤 2ms 미만 packet이 붙는 burst가 생긴다. 과거에는 이 현상을 scheduler 추론으로만 적었지만, 현재는 iterator IL과 세 호스트 raw UDP 관측이 함께 뒷받침한다.

2026-08-03 동일 build, Target Frame Rate 60, `sensorPeriod=0.02s`의 약 12초 raw UDP 측정은 다음과 같았다.

| 호스트 | raw datagram rate | 서로 다른 device timestamp rate | 연속 timestamp 중복 | arrival gap median / p95 / max | 2ms 미만 gap |
|---|---:|---:|---:|---:|---:|
| heven22 | 약 34.0Hz | 약 25.2Hz | 104 / 407쌍 | 12.05 / 63.4 / 91.9ms | 106 |
| heven-right | 약 47.2Hz | 약 38.0Hz | 110 / 566쌍 | 20.88 / 49.72 / 68.77ms | 112 |
| heven-laptop | 약 34.0Hz | 약 18.0Hz | 192 / 409쌍 | 25.12 / 64.8 / 72.69ms | 204 |

여기서 `서로 다른 device timestamp rate`는 **새 physics sample rate의 정확한 측정값이 아니다.** timestamp가 physics clock이 아니라 1ms wall-clock serialization time이기 때문이다. 다만 같은 timestamp와 같은 orientation/state가 반복되고, 긴 gap 뒤 burst가 나타나는 사실은 설정 50Hz만큼 독립적인 새 측정이 균일하게 도착하지 않는다는 강한 증거다.

중복의 형태도 host/run에 따라 달랐다.

- heven-right와 heven-laptop에서는 같은 timestamp인 인접 pair의 payload도 전부 같았다.
- heven22에서는 같은 timestamp pair의 orientation은 매번 같았지만 gyro/accel noise가 다시 생성돼 payload hash는 달라질 수 있었다.
- 따라서 payload 전체가 같은 경우만 제거하면 heven22의 반복 상태를 놓친다. MORAI profile에서는 device timestamp 중복을 우선 기준으로 사용해야 한다.

한 개의 ROS publisher만 존재하고 bridge가 UDP datagram마다 한 번 publish하므로, 이 반복은 ROS multi-publisher가 아니라 simulator UDP 단계에서 이미 존재한다. 상세한 host별 성능 및 bridge 분리 측정은 9절에 정리한다.

### 6.7 개발 주의사항

- 수신 packet count를 IMU의 유효 update count로 보지 않는다.
- MORAI에서는 동일 device timestamp의 후속 packet을 ESKF/deskew에 다시 적분하지 않는다. payload hash는 보조 진단으로만 사용한다.
- 적분 dt를 설정값 0.02초로 고정하지 않는다.
- monotonic arrival time과 raw device timestamp를 함께 기록한다.
- `timestamp_mode: arrival`에서 ROS header가 서로 다르더라도 raw device timestamp가 같으면 같은 상태의 catch-up 송신일 수 있음을 고려한다.
- zero quaternion과 invalid nsec를 명시적으로 reject한다.
- orientation covariance를 0 또는 미상으로 처리하고 ground-truth quaternion의 과도한 신뢰를 피한다.
- JSON stdev를 ROS covariance에 그대로 복사하지 않는다. output sample 통계를 사용한다.
- simulator FPS를 바꾸면서 noise Allan deviation이 변하는지 검증한다.

## 7. 3D LiDAR 상세 분석

### 7.1 구현된 요소

- Velodyne VLP-16 호환 1206-byte packet
- CH16 firing geometry와 10Hz 형식
- material/reflectivity 계열 GPU shader
- intensity output
- ring을 복원할 수 있는 packet 구조
- 설정 가능한 거리 Gaussian noise
- 설정 가능한 최대거리

LiDAR shader 관련 설정에는 `_optEq`, `_gamma`, `_tenPercentDistance`, `_reflectivity`, `_uBackWidth`, `_intensity` 같은 변수가 존재한다. 이는 표면 기반 intensity 근사를 지원하지만 실제 수신기 전자회로나 대기전파 모델을 의미하지는 않는다.

### 7.2 360° 생성 방식과 deskew 판정

현재 설치 DLL의 active CH16 경로를 IL까지 추적한 결과는 다음과 같다.

1. sensor period 0.1s가 되면 capture delegate가 실행된다.
2. **한 Unity frame 안에서** horizontal sector를 360° 순회한다.
3. 각 sector에서 LiDAR Transform의 방향을 바꾸고 `Camera.Render`/GPU readback을 수행한다.
4. sector loop 사이에 coroutine `yield`, physics simulation step, `Time.deltaTime` 누적 또는 ego/object pose 갱신이 없다.
5. 결과 buffer를 75개의 1206-byte packet으로 잘라 연속 UDP send한다.
6. 화면에 보이는 회전 animation은 이후 `Time.deltaTime × 360 × rotationFrequency`로 갱신되며 point의 물리 측정시각을 뜻하지 않는다.

즉 한 scan 안의 모든 방향은 서로 다른 beam direction을 갖지만 **동일한 ego/object pose snapshot**을 본다. 실제 회전형 LiDAR처럼 scan 시작부터 끝까지 약 0.1초 동안 차량과 물체가 계속 움직이며 측정된 것이 아니다.

현재 팀 bridge/Velodyne 경로는 반대로 다음 시간을 만든다.

- raw UDP packet에 ROS header가 없으므로 bridge가 각 datagram의 socket arrival time을 붙인다 (`morai_bridge_node.py:512-519`).
- adapter 기본값 `point_timing_mode=azimuth`가 첫 azimuth부터의 phase를 0.1초 scan period에 선형 매핑한다 (`velodyne_adapter_node.py:220-246`).
- stock Velodyne converter의 firing offset까지 합쳐 PointCloud2 `time`이 실제로 약 0~0.09997s가 된다.

이 시간은 **복원된 실제 acquisition time이 아니라 합성값**이다. 따라서 deskew는 다음과 같은 인공 변형을 만든다.

```text
10m/s 직진 × 0.1s synthetic scan time ≈ 최대 1m 인공 shear
20m/s 직진 × 0.1s synthetic scan time ≈ 최대 2m 인공 shear
회전 중에는 0°/360° 경계 및 ring에 인공 seam 가능
```

사용자가 관측한 "deskew를 켜면 ring 원이 끊기고, 끄면 고속에서도 닫힌다"는 현상과 코드 구조가 일치한다. 단, 일반적인 실물 spinning LiDAR에서는 원이 닫히는지만으로 deskew 정오를 판단할 수 없고, **이번 판정은 현재 MORAI DLL의 한-frame capture IL이 핵심 근거**다.

권장 profile:

- MORAI 25.S4: 모든 point relative time을 0으로 설정
- 외부 motion deskew: off 또는 instantaneous-scan bypass
- FAST-LIO2: raw cloud를 받더라도 내부 `ImuProcess::UndistortPcl`을 수행하므로, 입력 point time을 모두 0으로 만들거나 MORAI profile에서 내부 undistortion을 bypass
- 실차 Velodyne: 실제 packet/firing time을 유지하고 deskew on

`point_timing_mode=zero`는 packet phase만 0으로 만들 뿐 stock converter의 packet 내부 firing offset이 남을 수 있다. 정확한 MORAI truth profile은 최종 PointCloud2의 `time` field 전부를 0으로 강제하는 방식이다.

외부 deskew를 끄는 것만으로 FAST-LIO2가 자동으로 안전해지는 것도 아니다. 현재 FAST-LIO2는 `/ad/sensors/lidar/points` raw 입력을 받고 `IMU_Processing.hpp:213`의 `UndistortPcl`을 내부 호출하므로 이 경로도 별도로 처리해야 한다. 현재 raw 입력 구조상 외부 deskew+FAST-LIO2의 이중 deskew는 아니지만, FAST-LIO2 내부에서 synthetic time을 신뢰하는 한 한 번의 잘못된 deskew는 남는다.

### 7.3 날씨와의 연결

`CLidar3D`의 251개 메서드 바디와 호출 참조에서 weather, fog, rain, Enviro와 직접 연결되는 동작을 찾지 못했다. 카메라에는 환경 재초기화 참조가 있지만 LiDAR에는 없다.

LiDAR replacement shader에서도 물리적 fog extinction 또는 particle backscatter에 해당하는 키워드를 확인하지 못했다.

따라서 안개·비가 시각적 환경에 나타나더라도 다음 현상은 LiDAR에 재현되지 않는 것으로 판단한다.

- 안개에 의한 거리별 감쇠
- 안개 입자의 근거리 backscatter
- 강우 입자의 일시적 false return
- 물방울에 의한 point dropout
- 젖은 표면의 반사율 변화
- receiver threshold 이하 신호 소멸
- beam divergence와 대기 산란
- multi-return
- 수신기 saturation, temperature, crosstalk

카메라에는 안개가 보이지만 point cloud는 맑은 날과 동일한 센서 간 불일치가 가능하다.

### 7.4 Gaussian distance noise 구현

거리 noise 함수의 핵심 동작은 다음과 같다.

1. raw distance가 0이면 0을 반환한다.
2. 특정 shaded/fault 영역 유형이면 0을 반환할 수 있다.
3. noise가 꺼져 있으면 raw distance를 그대로 반환한다.
4. noise mean은 `gaussianMean × 1000 × velodyneUnit` 형태로 count에 환산한다.
5. sigma는 `gaussianStdev × 0.01 × rawRange` 형태로 현재 거리에 비례한다.
6. Box-Muller Gaussian sample을 만든다.
7. sample 부호에 따라 raw count에 더하거나 뺀다.

### 7.5 UInt16 wrap 버그

중요한 구현 결함이 있다.

- noise 크기를 `UInt16`으로 변환한다.
- raw에 더하거나 뺀 값도 clamp보다 먼저 `UInt16`으로 변환한다.
- 변환 후 값은 unsigned이므로 뒤의 `>= 0` 검사는 항상 참이다.

그 결과 큰 음수 perturbation이 0으로 포화되지 않고 65535 부근으로 wrap될 수 있다. 큰 양수 perturbation도 낮은 값으로 wrap될 수 있다.

가능한 증상:

- 정상 point가 갑자기 최대거리 근처로 이동
- 원거리 point가 비정상적으로 근거리로 이동
- noise를 강하게 설정할수록 MORAI 고유 outlier 증가

현재 기본 설정에서는 LiDAR noise가 꺼져 있으므로 즉시 발생하지 않는다. 향후 대회 업데이트나 noise preset 변경 시 반드시 재확인해야 한다.

### 7.6 실행 관측

30 scan 관측 결과:

- 실제 scan rate: 약 7.5~7.7Hz
- 설정 scan rate: 10Hz
- point count: scan당 약 13,912~13,913
- point step: 22 bytes
- `is_dense = true`
- NaN/Inf 관측 없음
- 거리 최솟값: 약 0.958m
- 거리 median: 약 12.502m
- 설정 최대거리: 100m
- 관측 최대거리: 약 100.442~100.444m
- intensity: 0~156
- 관측 ring: 0~10
- PointCloud2 `time`: 0~약 0.0999707초. 위 역분석에 따라 simulator truth가 아니라 현재 bridge/converter 합성값

상단 ring이 없는 것은 현재 장면에서 반사점이 없기 때문일 수 있으므로 센서가 11채널이라는 의미로 해석하면 안 된다.

최대거리는 100m에서 엄격히 절단되지 않는다. 따라서 point가 100m를 조금 넘는다는 이유만으로 손상으로 판정하지 않는다.

### 7.7 Outlier Filter 권고

현재 기본 설정에서 권장 순서:

1. packet length와 Velodyne block marker 검사
2. zero/no-return 제거
3. NaN/Inf 제거
4. 설정 100m보다 0.5~1m 큰 margin을 둔 range gate
5. ring별 거리 불연속 검사
6. 이전 scan과의 temporal consistency
7. noise가 실제 활성화되거나 outlier가 관측될 때만 adaptive ROR/SOR 적용

상시 강한 SOR/ROR는 VLP-16의 희소한 유효 point를 삭제할 수 있다.

- 보행자 팔·다리
- 표지판과 신호등 기둥
- 멀리 있는 차량 모서리
- 얇은 연석 및 가드레일
- 상단 ring의 희소 point

따라서 현재 빌드에서는 “안개 때문에 Outlier Filter가 필요하다”가 아니라 “noise wrap이나 UDP 손상 등 실제 관측된 outlier를 방어하기 위한 제한적 필터”로 설계하는 것이 맞다.

## 8. 카메라 상세 분석

### 8.1 영상 생성의 성격

카메라는 Unity Camera로 장면을 렌더링하고 GPU readback 후 JPEG로 인코딩한다. 활성 continuous callback은 `LightBuzz.Jpeg.JpegEncoder.Encode`를 사용하고 현재 `compressedRatio=90`을 적용한다.

별도의 helper 경로에는 JPEG quality 100을 hard-code한 함수도 있지만 현재 연속 송신 경로와 구분해야 한다.

카메라 설정 구조에는 다음 필드가 존재한다.

- 해상도와 FOV
- 압축률
- camera/model/type
- radial lens distortion k1, k2, k3
- physical/fisheye/semantic/depth 계열 설정
- 2D/3D bounding box 설정

현재 lens distortion 값은 0이고 bounding box ground truth 출력은 비활성이다.

### 8.2 확인되지 않은 실제 카메라 물리

- row별 readout time과 rolling shutter
- exposure time
- ISO와 gain
- aperture
- shot noise 및 read noise
- sensor saturation과 blooming
- Bayer mosaic 및 demosaic
- 실제 ISP sharpening/denoising/HDR
- 온도 의존성
- 렌즈별 MTF와 chromatic aberration
- 카메라 표면의 빗물·오염

따라서 카메라는 실제 광학·전자 sensor라기보다 Unity가 생성한 RGB 장면을 JPEG로 전달하는 rendered camera로 보는 것이 적절하다.

### 8.3 날씨 영향

카메라 클래스에는 환경 재초기화 참조가 있어 Enviro의 안개·조명·시간대 효과가 rendered image에 나타날 수 있다. 하지만 이것은 물리적으로 보정된 extinction coefficient, exposure response, 빗방울 렌즈 효과가 구현됐다는 뜻은 아니다.

대회 내부 환경 명세에서는 Sunny/Foggy와 11시/13시/15시 변형을 사용한다. 현재 대회 조건에는 rain이 명시되지 않았다.

### 8.4 UDP packet 구조

카메라는 항상 65,000-byte UDP datagram을 만든다.

| Offset | 내용 |
|---:|---|
| 0~2 | ASCII `MOR` |
| 3~6 | seconds, uint32 |
| 7~10 | nanoseconds, uint32 |
| 11~14 | fragment index |
| 15~18 | 실제 payload size |
| 19~64997 | JPEG payload, 최대 64,979 bytes 및 padding |
| 64998~64999 | tail marker, 마지막 fragment는 별도 끝 marker |

동일 frame의 모든 fragment는 같은 simulator timestamp를 사용한다.

프로토콜에는 다음이 없다.

- frame CRC
- fragment CRC
- retransmission
- forward error correction
- ACK
- 누락 fragment 복구

### 8.5 IP fragmentation 문제

65KB UDP datagram은 Ethernet MTU 1500에서 약 44개의 IPv4 fragment로 분할된다. 그중 하나만 유실돼도 전체 UDP datagram이 폐기된다.

실측 JPEG 크기:

| 카메라 | JPEG 크기 범위 | median | UDP datagram/frame |
|---|---:|---:|---:|
| 전방 | 142,255~142,944 bytes | 142,611.5 | 3 |
| 좌측 | 52,032~52,389 bytes | 52,194.5 | 1 |
| 우측 | 52,601~53,007 bytes | 52,821.5 | 1 |

20Hz 기준 세 카메라 합계:

- 초당 약 100개의 65,000-byte UDP datagram
- UDP payload 약 6.5MB/s
- 약 52Mbps
- IPv4 fragment 약 4,400개/s

localhost에서는 손실이 없더라도 실제 Ethernet, NIC ring, kernel receive buffer, switch를 통과하면 frag loss 위험이 증가한다.

### 8.6 카메라 개발 요구사항

- fragment index와 size 검증
- frame timestamp별 조립
- 최대 JPEG/frame 크기 제한
- incomplete frame timeout
- 동시에 유지할 incomplete frame 수 제한
- 중복 또는 충돌 fragment 검출
- JPEG SOI/EOI 검사
- frame completion/drop 통계
- socket receive buffer와 kernel IP fragment 통계 계측

현재 팀 브리지 assembler는 timeout 0.5초, 최대 16MiB, 최대 incomplete frame 32개, 최대 fragment index 1024 제한을 가진다. 이는 MORAI 자체 기능이 아니라 브리지의 방어 구현이다.

공식 문서에는 같은 카메라 fragment를 설명하면서 64,987과 64,979가 혼재한다. 현재 코드의 실제 최대 payload는 64,979 bytes다.

## 9. 시간, 주기, 중복 및 동기화

### 9.1 설정 주기와 실제 주기

MORAI 공식 센서 문서는 IMU data rate를 5~100Hz로 설정할 수 있지만 machine performance의 영향을 받을 수 있다고 명시한다. 네트워크 설정 문서도 지정한 Frame Hz가 simulator FPS 때문에 목표보다 낮게 전송될 수 있다고 설명한다. 따라서 JSON의 50Hz는 guaranteed sampling clock이 아니라 target이다.

세 호스트의 환경을 맞춰 확인한 조건은 다음과 같다.

- 동일 `Assembly-CSharp.dll` SHA-256: `a2ad7d12af07e2b3173c9c3e66a4c499af9fe70d2d91d48d5671db8a2547d669`
- 동일 `Simulator.x86_64` SHA-256: `95aa60b3bead39f7f6e86d1189664d420d8323243c63a45766601cd8fadeba88`
- Target Frame Rate 60, time scale 1, quality 2
- IMU `sensorPeriod=0.02s`, UDP `127.0.0.1:9298 → 9299`
- heven22만 팀 ROS2 bridge가 실행 중이었고, heven-right/heven-laptop은 임시 raw UDP probe로 simulator 출력을 직접 측정

약 12초 raw IMU 측정 결과는 다음과 같다.

| 호스트 | raw IMU datagram | 서로 다른 device timestamp | 비고 |
|---|---:|---:|---|
| heven22 | 약 34.0Hz | 약 25.2Hz | 카메라 off 상태 |
| heven-right | 약 47.2Hz | 약 38.0Hz | 카메라 active |
| heven-laptop | 약 34.0Hz | 약 18.0Hz | 카메라 active |

별도의 약 6초 coarse probe에서 다른 simulator UDP stream도 host별로 함께 느려졌다.

| 호스트 | status | collision | GPS datagram | IMU datagram | LiDAR packet |
|---|---:|---:|---:|---:|---:|
| heven22, camera off | 약 33.8Hz | 약 33.9Hz | 약 31.1Hz | 약 31.8Hz | 약 559packet/s |
| heven-right | 약 50.0Hz | 약 50.1Hz | 약 40.0Hz | 약 47.7Hz | 약 700packet/s |
| heven-laptop | 약 35.5Hz | 약 35.5Hz | 약 35.5Hz | 약 35.5Hz | 약 659packet/s |

GPS는 한 sensor tick에서 GGA와 RMC를 각각 send하므로 위 GPS 숫자는 위치 update rate가 아니라 datagram rate다. LiDAR도 scan rate가 아니라 Velodyne packet rate다.

카메라를 끈 heven22에서는 직전 약 25Hz 수준이던 IMU datagram이 약 32.8~34Hz로 올라가 부하 영향은 확인됐다. 그러나 50Hz에 도달하지 않았고 서로 다른 timestamp는 약 25.2Hz였으므로 camera off만으로 timing 문제가 해결된 것은 아니다.

동시 resource 관측은 다음과 같았다. CPU 100%는 논리 core 하나를 뜻한다.

| 호스트 | MORAI CPU | MORAI thread | GPU | 관측 해석 |
|---|---:|---:|---:|---|
| heven22 | 약 182% | 96 | 약 87~93% | GPU/render 또는 현재 scene/frame 경로가 우세한 병목 후보 |
| heven-right | 약 644% | 95 | 약 35~48% | 100% worker 4개가 있어도 status는 약 50Hz 유지 |
| heven-laptop | 약 556% | 128 | 약 50~57% | 100% worker 4개와 함께 전체 stream이 약 35.5Hz로 제한 |

따라서 “전체 CPU가 남는다” 또는 “GPU가 남는다”만으로 정상 주기를 보장할 수 없다. Unity main/render/physics/coroutine의 frame 의존 경로와 host별 scene·camera·worker 부하가 함께 rate를 결정한다.

### 9.2 동일 sample 중복

초기 5초 단일호스트 probe에서는 다음 결과가 나왔다.

| 데이터 | 수신 수 | 수신률 | 고유 payload/stamp | 고유률 | 연속 완전중복 |
|---|---:|---:|---:|---:|---:|
| IMU | 204 | 40.851Hz | 108 | 21.533Hz | 96 |
| GPS GGA | 90 | 18.263Hz | 48 | 9.644Hz | 42 |
| Ego status | 206 | 41.601Hz | 115 | 23.134Hz | 91 |

이 초기 probe의 `고유 payload/stamp`는 payload hash와 timestamp를 묶은 관측 지표이므로 정확한 physics rate로 재해석하면 안 된다. 이후 세 호스트 측정에서는 device timestamp 중복을 별도로 세었고, heven22 104/407쌍, heven-right 110/566쌍, heven-laptop 192/409쌍이었다. 6.6절의 arrival gap 분포와 IL scheduler 분석까지 합치면 긴 frame 지연 뒤 catch-up burst가 반복의 주된 원인이다.

GPS UTC 문자열의 소수부는 millisecond 값을 고정폭 없이 붙이는 비표준 형태이므로 이를 0.1초 timestamp로 해석하면 안 된다. payload까지 완전히 동일한 연속 sample이 많다는 점은 formatter 자릿수만으로 설명되지 않는다.

### 9.3 TimeManager 구현

TimeManager의 timestamp는 모드에 따라 달라진다.

- sync/simulation mode: 내부 simulator time을 tick으로 변환
- real-time factor 1: `DateTime.UtcNow - Unix epoch`
- 그 외: 내부 simulation time 누적

Simulation Time 모드는 `/clock`을 지원하지만 모드 변경 시 simulation time이 0으로 reset될 수 있다.

중요한 예외는 현재 realtime UDP IMU다. 6.5절에서 확인한 `CIMU` serializer는 TimeManager의 physics/simulation time이 아니라 host `DateTime`을 직접 사용한다. 따라서 TimeManager가 존재한다는 사실만으로 IMU device stamp가 simulation state와 정확히 동기화됐다고 보면 안 된다.

현재 팀 bridge는 `use_sim_time=false`이고 `/clock`을 사용하지 않는다. `timestamp_mode: arrival`은 raw IMU device stamp 대신 UDP 수신시각을 ROS header로 쓴다. `UdpReceiver.recvfrom()` 직후 `time.monotonic()`을 기록하고, bridge 시작 때 한 번 계산한 monotonic↔ROS wall-clock offset으로 변환한다. 이 방식은 한 process 실행 중 timestamp regression을 줄이지만, 시작 후 NTP wall-clock 보정을 계속 추적하지는 않는다.

### 9.4 개발 요구사항

#### 현재 bridge가 병목인지 분리한 결과

heven22에서 확인한 팀 bridge 상태는 다음과 같다.

- application counter: dropped 0, malformed 0, bind error 0
- kernel UDP socket: `Recv-Q=0`, socket drop counter `d0`
- raw UDP receipt → ROS subscriber callback: median 약 1.55ms, p95 약 2.51ms, max 약 3.32ms
- bridge ROS header stamp − MORAI device stamp: median 약 0.708ms, p95 약 3.89ms, max 약 8.47ms
- localization IMU republish 추가 지연: median 약 0.253ms, p95 약 0.351ms, max 약 0.395ms
- localization diagnostics: `imu_dropped=0`

heven-right와 heven-laptop에는 팀 ROS bridge가 없는데도 각각 rate 저하와 burst/duplicate가 raw localhost UDP에서 재현됐다. 따라서 현재 주기 문제의 root cause는 network 또는 bridge가 아니라 MORAI 송신 이전의 frame/coroutine scheduling이다.

#### 수신·필터 구현 요구사항

- monotonic arrival time을 항상 기록한다.
- simulator/device timestamp도 원본 그대로 별도 기록한다.
- MORAI profile에서는 같은 device timestamp의 후속 IMU packet을 filter prediction과 LiDAR deskew에 다시 적분하지 않는다.
- payload hash가 다르더라도 orientation/state가 같고 noise만 재생성될 수 있으므로 payload 완전일치만 dedup 조건으로 쓰지 않는다.
- timestamp reset, regression, wrap을 탐지한다.
- 설정 Hz가 아니라 유효 sample 간 dt를 사용한다.
- sensor별 jitter, duplicate, dropped, malformed counter를 노출한다.
- 카메라·LiDAR·IMU를 timestamp 하나만으로 강제 동기화하지 않는다.
- simulation time 모드 변경 또는 scenario reset을 epoch 전환으로 처리한다.

현재 팀 브리지의 `timestamp_mode: arrival`은 비정상 device time을 ROS header에 직접 사용하지 않는 장점이 있지만 OS, UDP queue, scheduler 지연을 측정시간으로 포함한다. arrival와 device 중 하나만 남기지 말고 둘 다 기록하는 것이 안전하다.

#### 50Hz, 30Hz, 20Hz 설정 해석

- 공식 문서 근거는 “IMU 5~100Hz 설정 가능, 실제 rate는 machine performance의 영향을 받을 수 있음”이다. 50Hz 입력이 50Hz 출력을 보장한다는 의미가 아니다.
- Synchronous Mode 문서의 fixed time step **20ms는 50Hz**다. 이를 “20Hz 목표”와 혼동하면 안 된다.
- 별도 운영 안내의 “최소 20Hz”가 있다면 자연스러운 해석은 성능 acceptance floor다. 공식 sensor 문서에서 IMU를 반드시 20Hz로 설정하라는 근거는 찾지 못했다.
- heven-right처럼 raw datagram이 약 47Hz 나오는 환경은 50Hz 설정을 유지할 가치가 있다.
- heven22/heven-laptop의 Real Time MORAI에 한해 30Hz 설정은 catch-up burst를 줄이는 A/B 후보가 될 수 있다. 변경 전후에는 raw rate, 고유 stamp rate, 2ms 미만 gap, ESKF residual을 같이 비교해야 한다.
- 20Hz는 마지막 성능 fallback으로만 본다. rate를 낮춰도 wall-clock timestamp 의미와 frame-dependent scheduler 자체는 바뀌지 않는다.
- 실차 IMU 설정은 MORAI workaround를 복사하지 말고 실제 IMU datasheet, bandwidth, timestamp source 및 Allan variance로 별도 결정한다.

### 9.5 heven 실시간 IMU rate 및 camera 3대 A/B (2026-08-04)

`heven`의 동일 MORAI 25.S4 실행에서 IMU 설정률과 camera 부하를 바꾸어 추가 측정했다. 모든 구간에서 차량은 정지 상태였으므로 이 결과는 packet timing과 scheduler 동작을 비교하는 자료이며 급가속·급정거 중 IMU 물리값의 정확도 검증으로 해석하면 안 된다.

기록 정책은 다음과 같다.

- IMU, GPS, competition vehicle status, bridge statistics 및 diagnostics는 rosbag에 원문을 보존했다.
- camera 3대 조건에서는 front/left/right compressed image가 각각 약 19.9frame/s로 실제 발행되는 것을 별도 subscriber로 확인했다.
- camera image payload는 rosbag에 넣지 않았다. 따라서 MORAI rendering, UDP 전송, JPEG fragment 조립 및 ROS publish 부하는 포함하지만 image bag 기록에 따른 DDS·disk 부하는 제외된다.
- `timestamp_mode: arrival`이므로 ROS header는 수신 시각이고 raw MORAI timestamp는 `*/full.device_stamp`로 별도 보존된다.
- 아래 `고유 stamp rate`는 연속 device timestamp가 증가한 packet 수를 arrival duration으로 나눈 진단값이다. 6.5절에서 확인했듯 device stamp는 physics sample time이 아니므로 실제 physics update rate와 동일시하지 않는다.
- `상대 burst`는 arrival interval `< 0.25 × 요청주기`, `상대 gap`은 `> 2 × 요청주기`다. 요청률마다 임계값이 다르므로 서로 다른 설정 간 절대 jitter 비교에는 p50/p95도 함께 사용해야 한다.

| IMU 설정 | camera | bag | 구간 | IMU datagram | 고유 stamp rate | stamp 반복 | 상대 burst | 상대 gap | GPS fix | status datagram | status 고유 stamp |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20Hz | off | `/home/heven/rosbags/imu_20hz_20260804_020534` | 120.5s | 19.16Hz | 15.30Hz | 20.2% | 35.4% | 23.1% | 15.25Hz | 40.95Hz | 33.31Hz |
| 30Hz | off | `/home/heven/rosbags/imu_30hz_20260804_015431` | 100.4s | 25.67Hz | 17.02Hz | 33.7% | 39.3% | 19.1% | 16.83Hz | 42.93Hz | 약 34.96Hz |
| 50Hz | off | `/home/heven/rosbags/imu_50hz_nocam_20260804_021629` | 71.9s | 32.33Hz | 24.11Hz | 25.4% | 26.5% | 46.5% | 15.22Hz | 40.75Hz | 32.84Hz |
| 50Hz | front/left/right 3대 | `/home/heven/rosbags/imu_50hz_cam3_20260804_023607` | 68.9s | 38.75Hz | 24.42Hz | 37.0% | 39.6% | 29.7% | 18.99Hz | 40.19Hz | 26.71Hz |
| 30Hz | front/left/right 3대 | `/home/heven/rosbags/imu_30hz_cam3_20260804_024401` | 68.3s | 28.53Hz | 23.76Hz | 16.7% | 21.2% | 7.8% | 18.89Hz | 40.16Hz | 26.42Hz |

camera 3대가 실제로 동작한 30Hz와 50Hz 구간은 GPS, status 및 camera rate가 거의 같아 이 표에서 가장 비교 가능성이 높은 A/B다.

| camera 3대 조건 | raw IMU p50 / p95 | 동일 device stamp 제거 후 p50 / p95 / p99 / max | 고유 stamp rate |
|---|---:|---:|---:|
| IMU 30Hz | 37.88 / 82.11ms | 39.86 / 84.08 / 107.42 / 133.72ms | 23.76Hz |
| IMU 50Hz | 31.61 / 75.49ms | 39.85 / 82.69 / 88.46 / 107.47ms | 24.42Hz |

검증된 해석은 다음과 같다.

- camera 3대 조건에서 IMU 요청률을 30Hz에서 50Hz로 올리면 raw datagram은 28.53Hz에서 38.75Hz로 늘지만 고유 stamp rate는 23.76Hz에서 24.42Hz로 거의 변하지 않았다.
- 동일 device stamp를 제거하면 30Hz와 50Hz의 중앙 interval은 각각 39.86ms와 39.85ms이고 p95도 84.08ms와 82.69ms로 거의 같다. 현재 조건에서는 50Hz 추가 packet의 대부분이 독립적인 physics sample이 아니라 catch-up 반복이라는 가설을 강하게 지지한다.
- camera 부하는 모든 stream rate를 일률적으로 낮추지 않았다. 총 IMU datagram은 늘기도 했지만 반복 비율이 증가했고, status는 총 rate가 유지되는 동안 고유 stamp rate가 감소했다. raw Hz만으로 센서 정보량을 판단하면 안 된다.
- 모든 활성 stream에서 bridge `dropped`, `malformed`, `stamp_regressions`는 0이었다. 현재 A/B 차이는 bridge packet loss보다 MORAI 송신 scheduler 변화와 일치한다.
- 30Hz camera 3대 조건은 50Hz와 거의 같은 고유 stamp rate를 더 적은 반복·burst와 낮은 bridge 처리량으로 제공했다. 다만 device stamp가 physics time이 아니고 정지 시험이므로, 최종 ESKF 설정 선택 전에는 동적 주행 A/B가 필요하다.

다음 20Hz 시험에서는 rate 선택보다 cross-stream scheduler 결합을 확인하는 데 목적을 둔다. localization 저대역 입력은 rosbag 원문으로 보존하고 대용량 입력은 payload bag 기록이 만드는 부하를 피하기 위해 arrival metadata를 전수 계측한다.

- IMU raw/full: receipt interval, device stamp 반복·회귀, payload/state 반복, burst cluster
- GPS RMC/GGA/fix: datagram interval, pair completion interval, UTC 양자화 및 동일 fix 반복
- competition vehicle status: datagram interval, device stamp 반복, `signed_velocity` 변화. 현재 competition 입력에는 실제 per-wheel encoder가 없으므로 이는 차체 종방향 속도 proxy이지 물리 encoder 측정으로 부르지 않는다.
- camera front/left/right/traffic-light: UDP fragment rate, completed JPEG frame interval, incomplete/timeout/drop
- LiDAR: UDP packet interval과 completed cloud interval을 분리하고 scan rate와 packet rate를 혼동하지 않는다.
- collision 및 diagnostics: event stream 특성상 고정 Hz를 요구하지 않고 event timestamp, duplicate 및 지연만 확인한다.
- 모든 stream: bridge application drop/malformed, kernel socket drop, source/arrival timestamp 선택 결과를 함께 기록한다.

### 9.6 IMU 20Hz + camera 4대 + 전체 stream 동시 계측 (2026-08-04)

#### 조건과 보존 자료

이번 계측은 사용자가 IMU를 20Hz로 설정하고 camera 4대를 모두 활성화한 현재 `heven` 실행을 대상으로 했다.

- simulator: `TargetFrameRate=60`, time scale 1, quality 2, X11 window 1850×1016
- sensor target: IMU 20Hz, camera 4대 각각 20Hz, GNSS fix 20Hz, LiDAR scan 10Hz, status/collision network frame 50Hz
- bridge: full competition bridge, camera front/left/right/traffic-light와 IMU/GNSS/status/collision/LiDAR 모두 활성
- 상태: simulator 화면에 `Pause / Press ESC key` overlay가 표시된 정지 조건. 동적 physics·급가감속 시험이 아님
- rosbag: `/home/heven/rosbags/morai_20hz_cam4_all_20260804_025721`, 159.181s, 1.7GiB, 189,424 messages
- 90초 전수 arrival metadata: 같은 bag의 `jitter_metadata.json`
- 90초 window-update/resource profile: 같은 bag의 `system_fps_resources.json`

이번 bag은 이전 camera 3대 A/B와 달리 네 camera의 압축 image, raw LiDAR packet 및 완성 cloud까지 실제 저장했다. 따라서 DDS serialization과 약 1.7GiB의 disk write 부하도 포함한다. 이전 metadata-only 시험과 수치를 직접 동일 조건으로 비교하면 안 된다.

아래 interval은 subscriber callback에서 기록한 monotonic arrival 기준이다. `상대 burst`는 `<0.25×target period`, `상대 gap`은 `>2×target period`다.

| stream | target | 실효률 | arrival p50 / p95 / p99 / max | 상대 burst | 상대 gap | source/timestamp 관측 |
|---|---:|---:|---:|---:|---:|---|
| IMU | 20Hz | 19.826Hz | 47.37 / 84.83 / 109.38 / 145.27ms | 5.83% | 1.18% | 1,784개 중 device stamp 반복 85회, 증가 proxy 18.881Hz |
| GNSS fix | 20Hz | 18.872Hz | 48.72 / 85.73 / 112.19 / 147.72ms | 1.30% | 1.36% | NMEA source time은 다른/불완전한 clock domain이라 normalized stamp로 채택하지 않음 |
| vehicle status | 50Hz | 40.470Hz | 31.08 / 55.00 / 65.32 / 82.18ms | 38.00% | 30.81% | 3,643개 중 device stamp 반복 1,271회, 증가 proxy 26.347Hz |
| collision | 50Hz | 40.504Hz | 30.78 / 55.31 / 64.51 / 82.33ms | 37.37% | 30.10% | full bag의 timing audit에서 source duplicate 2,433/6,441 |
| camera front | 20Hz | 19.847Hz | 46.23 / 84.04 / 93.63 / 119.95ms | 0% | 0.22% | source duplicate 0 |
| camera left | 20Hz | 19.844Hz | 45.37 / 83.21 / 90.74 / 103.38ms | 0% | 0.11% | source duplicate 0 |
| camera right | 20Hz | 19.850Hz | 45.71 / 82.75 / 91.75 / 114.11ms | 0% | 0.11% | source duplicate 0 |
| camera traffic-light | 20Hz | 19.849Hz | 46.92 / 84.65 / 92.47 / 115.06ms | 0% | 0.17% | source duplicate 0 |
| LiDAR raw packet | packet stream | 647.110packet/s | 0.228 / 0.528 / 80.07 / 152.65ms | N/A | N/A | packet burst 뒤 scan gap이 생기는 snapshot 송신 구조 |
| LiDAR scan/cloud | 10Hz | 8.619Hz | 120.08 / 147.37 / 162.41 / 273.59ms | 0% | 0.13% | scan과 cloud가 동일 776개 |

full 159초 bag의 ROS message 수와 rate를 다시 계산해도 camera 네 대는 각각 3,160~3,161개, IMU 3,156개, GNSS fix 3,010개, status 6,432개, collision 6,441개, LiDAR scan/cloud 각각 1,373개였다. bridge 누적 counter에서는 모든 활성 stream의 application `drop`, `malformed`, timestamp regression이 0이었다. camera UDP fragment rate는 front 약 59.58/s, left/right 각각 약 19.86/s, traffic-light 약 39.71/s로 합계 약 139.0datagram/s, payload 약 8.62MiB/s였다.

#### 해석

- IMU를 20Hz로 낮추면 raw rate는 목표에 가까워지고 device-stamp 반복은 4.8% 수준으로 줄지만, 0.22ms 최소 interval, p95 84.8ms, p99 109.4ms의 catch-up/gap 구조는 남는다. 즉 20Hz가 scheduler 문제를 제거하는 것은 아니다.
- 네 camera는 모두 평균 20Hz에 매우 가깝지만 p95 frame interval은 82.8~84.6ms다. 평균 rate가 정상이어도 균일한 50ms camera clock은 아니다.
- GNSS, status/collision, LiDAR scan도 각 target보다 낮고 jitter가 있다. 따라서 현재 현상은 IMU packet parser만의 결함이 아니라 MORAI의 frame/coroutine 및 host load에 걸친 cross-stream timing 특성이다.
- status/collision은 약 40.5 datagram/s지만 고유 source-state proxy는 더 낮다. datagram 수를 vehicle physics update 수로 사용하면 안 된다.
- raw LiDAR의 647packet/s와 0.228ms median interval은 고주파 센서 clock이 아니라 한 snapshot scan의 packet을 몰아서 보내는 구조다. 완성 cloud는 8.62Hz로 따로 판단해야 한다.
- bridge가 packet을 버린 증거는 없다. 다만 이번 full-payload rosbag 자체가 camera/LiDAR DDS 및 disk 부하를 추가하므로, simulator만의 무부하 upper bound로 해석하면 안 된다.

### 9.7 Target FPS, 현재 Linux 자원 및 Windows 11 목표 PC 전망

#### 현재 `heven` 실측

현재 host는 Core i9-14900K 24core(8P+16E)/32thread, GeForce RTX 3070 8GiB, RAM 32GiB, Ubuntu 22.04/X11이다. simulator preference에서 `TargetFrameRate=60`, time scale 1, quality 2를 확인했다. Unity의 `Application.targetFrameRate`는 목표/상한이며 rendering이 무거우면 실제 frame rate가 그보다 낮을 수 있다.

Linux에서 Unity의 정확한 swap/present counter를 attach하려 했지만 `perf_event_paranoid=4`로 권한 없이 사용할 수 없었다. 대신 simulator X11 window의 XDamage event를 90초 계측했다.

| 항목 | 90초 관측 |
|---|---:|
| X11 drawable damage/update | 25.07event/s, 2,257 events |
| update interval | p50 39.21ms, p95 58.48ms, p99 64.97ms, max 96.82ms |
| interval CV | 0.241 |
| 33.33ms 초과 interval | 90.29% |
| simulator CPU | 평균 547.5%, p95 583.8%, 최대 591.9% (`100%=logical core 1개`) |
| GPU utilization | 평균 62.9%, p95 75.7%, 최대 87% |
| GPU VRAM / power / temperature | 3,481MiB / 평균 166.6W / 평균 68.4°C |
| simulator RSS / swap | 평균 약 6.63GiB / 약 650MiB |
| system available RAM / free swap | 평균 약 10.79GiB / 약 502MiB |

XDamage는 실제 drawable 변경을 계측하지만 compositor가 여러 render를 하나의 damage event로 합칠 수 있어 정확한 swapchain present FPS와 일대일 대응하지 않는다. 또한 화면이 pause 상태였으므로 이 25.07event/s를 동적 주행 physics FPS라고 부를 수 없다. 확인 가능한 결론은 현재 pause/full-record 조건의 표시 갱신이 60event/s가 아니었다는 것과, `TargetFrameRate=60` 자체가 실효 60FPS 보장이 아니라는 점까지다.

10초 per-thread snapshot에서는 simulator가 약 552% CPU를 사용했고 네 compute thread가 각각 약 100%였다. snapshot 마지막 CPU 기준으로 main thread, `UnityGfxDeviceW`, Vulkan submission thread는 E-core CPU 21/23/22에서 관측됐고, 포화된 네 compute thread는 P-core logical CPU에서 관측됐다. worker들은 P/E core 양쪽에 분산됐다. 이는 Linux가 main/render thread를 항상 P-core에 고정한다고 가정할 수 없다는 증거지만, 한 snapshot의 last-CPU 값이 전체 residency 비율을 뜻하지는 않는다. 강제 affinity나 E-core 비활성화는 다른 scene에서 throughput을 낮출 수 있으므로 기본 대책으로 권고하지 않는다.

#### 목표 PC: i5-13600KF + RTX 4060 Ti + 32GiB + Windows 11

공식 사양상 i5-13600KF는 6P+8E, 20thread, 최대 5.1GHz이고, 현재 i9-14900K는 8P+16E, 32thread, 최대 6.0GHz다. 목표 PC는 CPU core/thread 수와 single-thread peak 모두 현재 host보다 낮다. RTX 4060 Ti는 8GiB 또는 16GiB GDDR6 모델이 있고 current RTX 3070과 architecture·memory subsystem이 달라 CUDA core 수만으로 MORAI 상대 FPS를 산출할 수 없다.

Windows 11은 Intel Thread Director가 제공하는 instruction mix와 core-state hint를 scheduler가 활용해 hybrid CPU의 P/E core 배치를 결정한다. Microsoft의 heterogeneous scheduling policy도 thread QoS와 system configuration을 반영한다. 따라서 Windows 11이 Unity foreground/main/render thread를 현재 Linux snapshot보다 P-core에 더 안정적으로 배치할 가능성은 있다. 그러나 이는 다음을 고치지 못한다.

- MORAI coroutine의 frame 지연 및 0-wait catch-up 구조
- IMU device stamp가 physics time이 아니라 serialization wall clock인 구조
- camera 65KB UDP/IP fragmentation과 JPEG encode/readback 비용
- 한 frame에 snapshot을 만든 뒤 packet을 몰아서 보내는 LiDAR 구조
- CPU/GPU가 실제로 부족한 경우의 frame-time 초과

반대로 Windows에는 DWM/WDDM, driver, Defender, update/background service에 의한 별도 jitter가 있고, 전원 모드·창 모드·driver·Hardware-accelerated GPU scheduling 상태에 따라서도 결과가 바뀐다. Windows scheduler가 항상 Linux보다 빠르거나 jitter가 작다고 단정할 수 없다. ROS를 WSL2에서 실행하면 Windows는 VM/vCPU를 스케줄하고 그 안에서 Linux guest scheduler가 다시 ROS thread를 배치하므로, MORAI native-Windows 단독 결과와도 별도 조건이다.

현재 결과에 근거한 보수적 전망은 다음과 같다.

- 동일 pause/static scene에서 camera 4대와 IMU 20Hz는 목표에 가깝게 유지될 가능성이 높지만 보장은 아니다.
- status/collision 50Hz와 LiDAR 10Hz는 현재 더 강한 CPU에서도 각각 약 40.5Hz와 8.62Hz였으므로 목표 PC에서 자동 개선될 근거가 없다.
- current GPU 평균 62.9%, p95 75.7%라 pause 조건에는 GPU headroom이 있었지만, dynamic traffic·weather·고해상도 camera readback에서는 달라질 수 있다.
- CPU는 목표 PC가 명확히 약하므로 Thread Director의 배치 이득이 있더라도 60FPS를 사양만으로 보장할 수 없다. 숫자로 예측하기보다 같은 map/vehicle/camera/quality에서 실측해야 한다.
- RAM 용량은 같지만 current run에서도 simulator가 약 6.6GiB RSS와 650MiB swap을 사용했고 system 전체 free swap이 약 0.5GiB뿐이었다. Windows에서도 page-file activity가 생기면 긴-tail jitter가 커질 수 있으므로 committed memory와 hard fault를 함께 봐야 한다.

Windows 목표 PC의 acceptance test는 강제 affinity/E-core off 같은 튜닝 없이 기본 상태부터 시작한다. 동일한 5분 주행에서 PresentMon 또는 ETW/WPA로 actual presented FPS와 p50/p95/p99 frame time, CPU별 thread residency, GPU busy, hard fault를 기록하고, 동시에 bridge의 sensor별 arrival p50/p95/p99/max, duplicate, drop을 기록한다. 그 뒤에만 foreground priority/power mode 같은 한 변수를 A/B한다. 이 순서는 작은 평균 성능 향상을 위해 다른 scene의 안정성을 희생하는 위험을 줄인다.

## 10. 좌표계 및 단위

### 10.1 Map 좌표계

- ENU
- +X: East
- +Y: North
- +Z: Up
- heading 0°: East
- heading 양의 방향: counter-clockwise

### 10.2 차량 좌표계

- origin: rear axle center
- +X: forward
- +Y: left
- +Z: up

### 10.3 GPS

- Datum: WGS84
- Projected CRS: UTM zone 52N
- EPSG:32652

### 10.4 LiDAR

MORAI raw Velodyne 좌표:

- +Y: forward
- +X: right
- +Z: up

ROS 변환 후 REP-103:

- +X: forward
- +Y: left
- +Z: up

raw packet을 ROS 좌표로 착각하면 point cloud가 90° 회전하거나 좌우 반전될 수 있다. 변환은 한 곳에서만 수행하고, vehicle 전방·좌측·상단의 기준물체를 이용한 unit/integration test가 필요하다.

## 11. 차량 동역학

어셈블리에는 다음 구성요소가 존재한다.

- Unity Rigidbody와 PhysicsModule
- Vehicle Physics 계열 ground vehicle controller
- VP wheel collider
- electric vehicle controller
- suspension 및 progressive suspension
- anti-roll bar
- aerodynamic surface
- rolling friction
- tire friction modifier
- Pacejka 계열 처리 문자열

이는 단순 kinematic bicycle model보다 복잡한 모델임을 보여준다. 하지만 다음 실차 validation 자료는 찾지 못했다.

- Ioniq 5 speed별 step steering 비교
- steering actuator delay 및 deadband
- throttle/brake actuator response
- brake distance 및 ABS 동작 비교
- 실제 tire μ-slip curve
- 하중이동과 횡가속도 비교
- ESC/TCS 동작 검증
- 회생제동과 SOC/temperature 영향
- 젖은 노면 및 노면 재질별 검증

따라서 Vehicle Physics/Pacejka가 포함되어 있다는 사실만으로 실제 Ioniq 5 고충실도를 주장할 수 없다.

권장 식별 시험:

1. 속도별 steering step response
2. steering sine sweep
3. throttle step과 coast-down
4. brake step 및 정지거리
5. 일정 원 선회와 lateral acceleration
6. 속도별 actuator command-to-response delay
7. frame rate와 fixed timestep을 변경한 반복성 시험

제어기 gain은 실차 물성값이 아니라 MORAI 실행환경의 실제 응답에 맞춰 별도 식별·튜닝해야 한다.

## 12. Radar와 2D LiDAR

현재 대회 센서 설정에는 radar와 2D LiDAR가 없으며 대회 허용 인터페이스에도 포함되지 않는다. 따라서 정적 분석 결과만 기록한다.

Radar 구조에는 다음 필드가 있다.

- range/angle resolution
- distance/angle Gaussian noise
- min/max range
- horizontal/vertical FOV
- ray count
- rotation frequency
- RCS 계열 출력

그러나 다음 실제 radar 물리의 근거는 찾지 못했다.

- chirp/waveform
- range-Doppler FFT
- multipath 및 ghost target
- ground clutter
- sidelobe
- rain/fog attenuation
- micro-Doppler
- CFAR와 receiver noise floor

현재 대회 범위 밖이고 실행 A/B 검증을 하지 않았으므로 radar 평가는 낮은 신뢰도의 참고사항이다.

## 13. UDP 제어 및 차량 상태

### 13.1 Ego control command

현재 팀 codec 기준 전체 길이는 55 bytes다.

payload 구조:

```text
ctrlMode: int8
gear: int8
longCmdType: int8
velocity: float32
acceleration: float32
accel: float32
brake: float32
steering: float32
```

envelope는 `#MoraiCtrlCmd$`와 data length, auxiliary integer 세 개, payload, CRLF 형태다.

프로토콜에는 다음이 없다.

- command timestamp
- sequence number
- ACK
- retransmission
- CRC
- stale command 표시

따라서 packet이 유실되거나 순서가 바뀌었을 때 simulator와 송신 측이 이를 명시적으로 판별할 수 없다.

### 13.2 차량 상태와 collision

- status payload는 tire extension 여부에 따라 181 또는 229 bytes
- collision payload는 181 bytes
- collision object는 최대 5개
- 다수 값이 float32
- status timestamp도 IMU와 유사한 wall-clock seconds/nanoseconds 구조
- status 역시 설정 50Hz보다 낮고 동일 payload 중복이 관측됨

### 13.3 wheel speed와 encoder 정확도

현재 대회 pipeline에는 물리 encoder packet이 없다. 이름 때문에 혼동되는 경로는 다음과 같다.

```text
Unity Rigidbody.velocity
  -> MoraiObjectBase.GetSignedVelocity()
     body transform으로 변환한 velocity.z × 3.6 [km/h]
  -> EgoVehicleStatus.velocity.x / signed_velocity (float32)
  -> 팀 bridge에서 /3.6 [m/s]
  -> localization_adapter가 /ad/localization/input/wheel_speed로 재포장
```

DLL의 `GetSignedVelocity`는 다음 의미다.

```text
GetRealTransform().InverseTransformDirection(Rigidbody.velocity).z × 3.6
```

따라서 현재 `/ad/localization/input/wheel_speed`는 다음을 포함하지 않는다.

- wheel encoder tick 또는 pulse/rev
- ABS wheel-speed sensor의 tooth count
- sample-window quantization
- 저속 zero-speed threshold나 missed pulse
- 타이어 유효반경 오차
- encoder bias, electronic noise, latency
- 네 바퀴 평균 또는 좌우 차동으로 계산한 속도

팀 adapter는 `EgoVehicleStatus.velocity.x`의 절댓값에 gear 방향을 붙이고, `wheel_speed_variance=0.04`를 코드/설정에서 **임의로 부여**한다. 이 0.04m²/s²는 MORAI가 제공한 encoder covariance가 아니다. `gps_course_enabled=false`일 때 lateral speed도 0으로 둔다.

수치 직렬화는 float32다. 30m/s 부근의 float32 간격은 약 2×10^-6m/s라서 serializer 양자화는 사실상 무시할 수 있다. 정지상태에서 관측된 약 10^-4m/s급 미세 변화는 encoder noise가 아니라 Rigidbody/physics jitter다. 실제 정확도를 제한하는 것은 float 분해능보다 physics model, fixed timestep, tire/contact 및 frame scheduling이다.

어셈블리에는 별도의 `NaverInfoPublisher_ObdWheelSpeeds`가 존재한다. 이 선택형 publisher는 네 wheel collider의 `angularVelocity`를 읽고 wheel radius 계열 변환을 적용해 FL/FR/RL/RR wheel speed를 만든다. 그러나:

- 현재 MolitComp03 대회 sensor JSON/UDP bridge에는 연결되어 있지 않다.
- tick encoder가 아니라 wheel collider의 연속 각속도다.
- 코드에서 encoder quantization/noise/fault 모델을 찾지 못했다.
- wheel slip은 physics에서 wheel angular speed와 차체 speed가 달라지는 범위만 반영될 수 있지만, 실제 타이어/ABS sensor accuracy validation은 없다.

결론적으로 현재 localization에서 말하는 wheel speed는 **가상 encoder조차 아니라 body velocity ground truth에 가까운 값**이다. MORAI에서 이 값을 강하게 신뢰해 얻은 성능을 실차 encoder fusion 성능으로 해석하면 안 된다.

#### GNSS 음영구간에서의 ground-truth 누출

일반 competition bridge만 사용해도 `/ad/vehicle/status.velocity.x`에서 차체 Rigidbody 전진속도를 받고, 일반 IMU packet의 `orientation`에서는 gyro 적분값이 아니라 Unity Transform 자세에 가까운 quaternion을 받는다. 마지막 정상 GNSS 위치를 초기점으로 두고 이 두 값을 결합하면 yaw drift와 실제 encoder 누적오차가 사실상 제거되므로, GNSS 음영구간의 dead reckoning이 실차보다 비현실적으로 잘된다. 이때 선택형 per-wheel speed는 필요하지 않다. 다만 전진속도만 적분하면 횡방향 slip은 직접 관측하지 못하며, timestamp 중복·지연과 수치 적분 오차는 남는다.

더 직접적인 누출도 있다. 일반 `/ad/vehicle/status`에는 map-frame `position`과 `rpy`가 함께 포함된다. localization이 이 값을 계속 사용하면 GNSS 음영 여부와 무관하게 simulator vehicle pose를 직접 받을 수 있으므로, localization 검증 자체를 우회하게 된다. `ad_morai_bridge_dev`의 ego status만 차단해서는 충분하지 않다.

의미 있는 GNSS 단절 시험에서는 `/ad/vehicle/status.position`, `/ad/vehicle/status.rpy`와 IMU `orientation`을 localization 입력에서 제외하고, IMU `angular_velocity`·`linear_acceleration` 및 규정상 허용된 차량 속도만 사용해야 한다. 차량 속도를 허용하더라도 현재 값은 encoder가 아니므로 타이어 유효반경, slip, pulse 양자화, bias, latency를 별도 모델링해야 실차 dead reckoning에 가까워진다. Competition Status packet이 허용 endpoint라는 사실과 그 안의 모든 ground-truth field를 localization에 사용할 수 있다는 규정 해석은 구분해야 한다.

### 13.4 제어 안전 요구사항

- 송신 측 heartbeat
- command sequence를 애플리케이션 레벨에서 별도 관리
- 마지막 유효 명령의 monotonic time 기록
- stale command timeout 시 throttle 0, brake 적용
- process death 및 network disconnect watchdog
- gear/steering/throttle/brake range validation
- 송신률과 실제 차량 status update률을 분리 계측

현재 브리지의 control 기능이 비활성이라면 브리지 내부 watchdog도 동작하지 않는다. 외부 제어 노드가 별도 안전정책을 가져야 한다.

## 14. MORAI 고유 결함·불일치 목록

### 확인된 문제

1. LiDAR 날씨 비연동: 안개·비가 LiDAR 감쇠·산란·dropout에 반영되지 않는다.
2. LiDAR snapshot/deskew 불일치: simulator는 360°를 한 frame에 capture하지만 bridge는 0~0.1s point time을 합성한다.
3. LiDAR synthetic deskew artifact: 합성 시간을 사용하면 고속/회전에서 ring seam과 최대 `speed×0.1s` 규모의 인공 shear가 생긴다.
4. LiDAR UInt16 wrap: distance noise 활성 시 음수/양수 overflow가 인공 outlier를 만들 수 있다.
5. GNSS RMC speed 변환 버그: realtime UDP가 m/s에 0.00053995를 곱해 knot 값을 3,600배 작게 만든다.
6. GNSS course 의미 오류: velocity vector가 아니라 sensor Transform yaw를 COG field에 넣는다.
7. GNSS 품질 상수화: active UDP GGA가 fix quality 1, satellites 9, HDOP 1.0, geoid 0.0을 고정한다.
8. GNSS 설정/출력 불일치: 현재 `gpsStatus=2`지만 active UDP GGA quality는 1이다.
9. GNSS UTC 비표준 가변폭: millisecond 값을 `D1` 계열로 붙여 `.4` 같은 문자열의 decimal-second 의미가 모호하다.
10. GPS noise 단위 설명 불일치: 구형 문서의 `%` 설명과 현재 로컬 XYZ 거리 직접 적용 코드가 다르다.
11. IMU noise의 FPS 의존성: sensor period가 아니라 `Time.deltaTime`을 사용한다.
12. IMU Gauss-Markov 근사: exact exponential 대신 Euler 근사를 사용한다.
13. IMU orientation 비현실성: Transform quaternion에 orientation measurement noise가 없다.
14. wheel-speed 명칭 불일치: 현재 localization wheel topic은 encoder가 아니라 Rigidbody body velocity다.
15. 동일 sample 반복: IMU, GPS, 차량 상태에서 동일 timestamp/state가 반복된다. IMU는 host/run에 따라 같은 상태에서 noise만 다시 생성돼 payload hash가 달라질 수도 있다.
16. 설정 Hz 미달과 host 편차: 동일 binary·50Hz 설정에서 raw IMU가 heven22/right/laptop 약 34.0/47.2/34.0Hz, 고유 device stamp는 약 25.2/38.0/18.0Hz였다.
17. IMU protocol 문서 불일치: 공개 107 bytes, 현재 구현 115 bytes.
18. 카메라 fragment 문서 불일치: 공식 페이지 안에서 64,987과 64,979가 혼재한다.
19. HDOP validation 불일치: 오류 문자열은 12 이하라고 하지만 코드상 99.9까지 허용한다.
20. LiDAR range overshoot: 100m 설정에서 약 100.44m point가 나온다.
21. 대형 카메라 UDP: 65KB datagram이 MTU 1500에서 심각한 IP fragmentation을 만든다.
22. 카메라 복구 수단 부재: CRC, retransmission, FEC가 없다.
23. 네 번째 카메라 초기 활성화 불일치: 2026-08-01 실행에서는 설정에만 존재했지만, 2026-08-04 full bridge에서는 traffic-light camera까지 정상 관측됐다. 당시 launch/enable 상태의 차이로 범위를 제한한다.
24. 비정상 IMU nsec 보존 사례: 1e9를 넘는 단일 저장값이 존재한다.
25. Control protocol 무결성 부재: timestamp, sequence, ACK, CRC가 없다.
26. IMU timestamp 의미 불일치: device stamp는 physics sample time이 아니라 1ms 해상도의 host wall-clock serialization time이다.
27. IMU catch-up burst: coroutine 지연 시 wait를 0으로 clamp해 긴 gap 뒤 2ms 미만 연속 송신이 발생한다.

### 물리 충실도가 낮거나 근거가 없는 부분

1. 실제 위성 기반 GNSS 오차
2. 실제 MEMS IMU thermal/vibration/scale factor 특성
3. 카메라 rolling shutter와 sensor electronics
4. LiDAR atmospheric optics와 multi-return
5. Radar waveform, clutter, ghost, Doppler processing
6. 실제 차량 actuator와 tire validation
7. 실시간 센서 clock synchronization
8. 실제 Ethernet queueing 및 센서별 hardware latency
9. wheel encoder pulse/quantization/electronics 및 ABS sensor 오차

### 아직 확정하지 못한 부분

1. 2026-08-01 초기 실행에서만 네 번째 카메라가 비활성이었던 정확한 launch/enable 원인. 현재 simulator/bridge에서는 정상 활성임
2. 고유 device timestamp rate와 실제 Unity physics state update rate의 정확한 일대일 관계
3. 보존된 IMU invalid nsec의 생성 경로
4. 대회 당일 sensor JSON이 현재와 달라질 가능성
5. 빌드 업데이트 시 protocol 변경 가능성

## 15. 개발 우선순위 권고

### P0: 반드시 구현

- 모든 raw UDP의 packet length와 envelope 검증
- arrival monotonic time과 raw device time 동시 기록
- duplicate, dropped, malformed, jitter counter
- GPS fix quality 및 blackout 처리
- MORAI realtime RMC speed/course를 velocity/COG update에서 제외
- IMU raw device timestamp 기반 duplicate 적분 방지. `timestamp_mode: arrival`의 ROS header만으로 중복을 판정하지 않음
- LiDAR 좌표계 변환 unit test
- MORAI LiDAR point relative time을 전부 0으로 만들고 외부/FAST-LIO2 내부 deskew bypass
- 카메라 fragment timeout 및 incomplete frame 폐기
- control stale-command watchdog
- 빌드·센서 설정 파일의 hash 기록

### P1: 대회 전 검증

- Sunny/Foggy에서 카메라 pixel histogram 및 detection 성능 A/B
- Sunny/Foggy에서 LiDAR range/intensity/dropout A/B
- 현재 두 JSON이 동일하므로 별도 복사본에서 noise를 명시적으로 off/on한 GPS 정지 분산 측정
- 별도 복사본에서 IMU noise를 명시적으로 off/on한 Allan deviation 비교
- `EgoVehicleStatus.velocity`와 선택형 per-wheel collider speed의 slip 상황 A/B. 현재는 encoder 검증이 아님을 유지
- FPS와 GPU load를 바꾸면서 sensor noise·rate 비교. 2026-08-04 pause/full-record 조건의 XDamage 약 25.07event/s는 확보했으나 dynamic present/physics FPS는 미측정
- MORAI Real Time에서 IMU target 50Hz와 30Hz를 A/B하고 raw/stamp rate, burst, duplicate, ESKF residual을 함께 비교
- Windows 11 목표 PC에서 PresentMon/ETW 기반 actual present FPS, frame-time p99, P/E core residency, GPU busy 및 page fault를 같은 4-camera 동적 주행으로 측정
- localhost와 실제 Ethernet에서 camera fragment loss 비교
- simulator reset 및 TimeManager mode 변경 시 timestamp 처리 검증
- 100m range margin과 선택적 outlier filter 검증

### P2: 제어 및 localization 튜닝

- MORAI 차량의 speed별 steering/throttle/brake system identification
- GNSS covariance 실측 추정
- IMU와 vehicle status의 유효 update rate 및 실제 dt 기반 filter tuning. 단순 datagram count를 sample rate로 사용하지 않음
- MORAI instantaneous-scan profile과 실차 rolling-scan profile을 분리하고 regression test
- arrival timestamp 사용 시 network jitter 민감도 평가

## 16. 권장 로그 스키마

모든 센서 로그에 최소 다음을 보존한다.

```text
host_monotonic_ns
host_wall_time_ns
device_seconds
device_nanoseconds
sensor_id
source_ip
source_port
destination_port
packet_length
payload_hash
sequence_or_fragment_index
parse_status
drop_reason
simulator_build_hash
sensor_config_hash
```

추가 센서별 필드:

- GPS: sentence type, UTC, fix quality, HDOP, satellite count
- IMU: quaternion validity, duplicate flag, effective dt
- LiDAR: packet azimuth, scan ID, point count, max range, invalid count
- Camera: frame stamp, fragment count, JPEG size, completion latency
- Control: application sequence, command age, watchdog state

## 17. 최종 판단

MORAI 25.S4.MolitComp03의 강점은 지도, 차량 배치, 센서 기하, 시나리오, 제한된 noise/fault, 대회 UDP protocol을 하나의 실행환경에서 제공한다는 점이다.

그러나 고충실도라는 표현을 다음과 같이 제한해서 사용해야 한다.

- 기하·시나리오 충실도: 비교적 높음
- 프로토콜 호환성: 비교적 높음
- 맑은 날 perception 개발 유용성: 중간 이상
- 실제 센서 물리 충실도: 낮음에서 중간
- 악천후 LiDAR 충실도: 매우 낮음
- GNSS 현실성: 위치 truth+독립 Gaussian+NMEA, 위성/Doppler/multipath 없음
- IMU 현실성: Unity physics truth+설정 noise, orientation은 ground truth에 가까움
- wheel encoder 현실성: 현재 대회 pipeline에는 encoder 모델이 없고 body velocity를 사용
- 실시간성과 센서 동기화: PC 성능 및 FPS 의존, catch-up burst, device stamp가 physics time이 아님

따라서 MORAI에서 성능이 좋다는 사실은 “현재 MORAI 모델과 설정에서 잘 작동한다”는 뜻이지, 실제 안개·비·GNSS 음영·센서 열잡음·네트워크 지연 환경에서도 동일하게 작동한다는 증거가 아니다.

대회 개발에서는 MORAI 고유의 timestamp 반복, host wall-clock device stamp, catch-up burst, FPS 의존 주기, NMEA 양자화, realtime RMC speed 버그, 가짜 wheel-speed 명칭, LiDAR synthetic point time/noise wrap, 카메라 IP fragmentation을 명시적으로 방어해야 한다. 반대로 snapshot LiDAR에 deskew를 적용하거나 악천후 모델이 없는 cloud에 과도한 outlier filter를 추가하면 simulator가 생성하지 않은 왜곡을 만들거나 정상 희소 point만 손실할 수 있다.

실무적으로 센서별 취급은 다음 한 줄로 요약된다.

- Camera: rendered scene + JPEG, 실제 imager noise가 아님
- LiDAR: one-frame geometric snapshot + material intensity, 현재 noise off, MORAI에서는 deskew off
- GNSS: Transform position + σ0.5m white Gaussian + NMEA quantization, Doppler/multipath 없음
- IMU: Transform/Rigidbody truth + 활성 white noise, encoder와 무관
- Vehicle/wheel: body Rigidbody velocity truth, 현재 encoder sensor가 아님

## 18. 참고 자료

### 현재 설치 및 로컬 구현

- C# 어셈블리: `/home/heven/MoraiLauncher_Lin/MoraiLauncher_Lin_Data/Project/Simulator/Simulator_v.S4.251001.MolitComp03_Linux/Simulator_Data/Managed/Assembly-CSharp.dll`
- 기본 센서 설정: `/home/heven/MoraiLauncher_Lin/MoraiLauncher_Lin_Data/SaveFile/Sensor/25.S4.MolitComp03/SensorInfo_2023_Hyundai_Ioniq5.json`
- 노이즈 프리셋: `/home/heven/MoraiLauncher_Lin/MoraiLauncher_Lin_Data/SaveFile/Sensor/25.S4.MolitComp03/noise_SensorInfo_2023_Hyundai_Ioniq5.json`
- 팀 브리지 설정: `ad_morai_bridge/config/competition.yaml`
- GPS ROS 변환: `ad_morai_bridge/ad_morai_bridge/message_conversion.py`
- raw LiDAR arrival stamp: `ad_morai_bridge/ad_morai_bridge/morai_bridge_node.py`
- LiDAR azimuth 시간 합성: `ad_morai_bridge/ad_morai_bridge/velodyne_adapter_node.py`
- FAST-LIO2 내부 undistortion: `third_party/fast_lio/src/IMU_Processing.hpp`
- wheel-speed 재포장: `ad_localization/src/adapter/localization_adapter.cpp`
- 카메라 assembler: `ad_morai_bridge/ad_morai_bridge/codecs/camera.py`

### MORAI 공식 문서

- [센서 설정 및 출력주기](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/-34)
- [Sensors: IMU/GNSS 5~100Hz 및 machine-performance 주의](https://help-morai-sim-en.scrollhelp.site/morai-sim-standard-en/sensors)
- [네트워크 설정 UI: 목표 Frame Hz가 simulator FPS 때문에 낮아질 수 있음](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R1.0/ui)
- [Synchronous Mode: fixed time step 20ms](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R1.0/synchronous-mode-2)
- [센서 UDP 프로토콜](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/-35)
- [좌표계](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/-8)
- [Sensor Noise](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R1.0/sensor-noise)
- [Time Manager](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/morai-sim-time-management-function)
- [Network/Control Protocol](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R1.0/ros-1)
- [MORAI SIM: Drive 소개](https://help-morai-sim.scrollhelp.site/ko/morai-sim-drive/24.R2/morai-sim-drive-3)

### Frame rate, CPU/GPU 및 OS scheduler 공식 자료

- [Unity `Application.targetFrameRate`: target을 달성하지 못할 수 있음](https://docs.unity3d.com/2023.2/Documentation/ScriptReference/Application-targetFrameRate.html)
- [Intel desktop processor comparison: i5-13600KF 사양](https://cdrdv2-public.intel.com/841923/Intel-Core-Desktop-Boxed-Processors-Comparison-Chart.pdf)
- [Intel 14th Gen processor guide: i9-14900K 사양](https://cdrdv2-public.intel.com/813072/14th%20Gen%20Workstation%20Quick%20Reference%20Guide_Intel%20Core%20Desktop%20Processors.pdf)
- [Intel Thread Director와 Windows 11 scheduler 설명](https://www.intel.com/content/www/us/en/support/articles/000097053/processors/intel-core-processors.html)
- [Microsoft heterogeneous scheduling policy](https://learn.microsoft.com/en-us/windows-hardware/customize/power-settings/configuration-for-hetero-power-scheduling-schedulingpolicy)
- [NVIDIA GeForce RTX 4060 Ti 공식 사양](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/)
- [NVIDIA GeForce RTX 3070 공식 사양](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3070-3070ti/)
- [Linux scheduler CPU capacity model](https://docs.kernel.org/scheduler/sched-capacity.html)
- [Linux `intel_pstate` 정책](https://docs.kernel.org/admin-guide/pm/intel_pstate.html)

### 대회 내부 명세

- [Project Info](https://app.notion.com/p/39c3bf06830081afa218d31de3473305)
- [Rules](https://app.notion.com/p/39c3bf0683008188ab34c3a9de6920d2)
- [Environment](https://app.notion.com/p/39c3bf06830081158405d8495c50cd48)
- [Coordinate and Unit](https://app.notion.com/p/39c3bf06830081c78a7ce2122ed5056f)
- [Sensor and Network](https://app.notion.com/p/39c3bf06830081318746e722d97f24e3)
- [Sources](https://app.notion.com/p/39c3bf0683008102a802e8a3d909abe5)

## 19. 재검증이 필요한 변경 조건

다음 중 하나라도 바뀌면 이 보고서의 수치와 protocol을 다시 검증해야 한다.

- MORAI 25.S4 이후 simulator update
- `Assembly-CSharp.dll` hash 변경
- sensor JSON 변경
- noise preset 적용
- TimeManager 모드 변경
- localhost에서 Ethernet 분산 구성으로 전환
- GPU/CPU 또는 Target FPS 변경
- 대회 운영진의 sensor rate/noise/weather 공지 변경
- 카메라 수 또는 해상도 변경
- LiDAR type, frequencyIndex, velodyneUnit 변경

이 보고서는 현재 보존된 25.S4 빌드와 2026-08-01~2026-08-04 실행 관측 및 IL 역분석에 대한 분석이며, 향후 업데이트된 MORAI 버전에 자동으로 일반화되지 않는다.

## 20. 개발할 때 반드시 알아야 할 MORAI 요약

- LiDAR는 한 frame의 360° snapshot이다. MORAI에서는 point time을 0으로 두고 외부·FAST-LIO2 deskew를 끈다.
- LiDAR 안개·비 감쇠, 산란, dropout, false return은 없다. 악천후 검증용으로 믿지 않는다.
- LiDAR noise는 현재 꺼져 있다. 켜면 UInt16 wrap으로 가짜 근·원거리 outlier가 생길 수 있다.
- 강한 SOR/ROR는 얇고 먼 정상 point를 지운다. 실제 outlier가 확인될 때만 제한적으로 쓴다.
- GNSS는 Transform 위치에 축별 σ=0.5m Gaussian을 더한 값이다. 위성·Doppler·multipath·NLOS·RTK 모델은 없다.
- GNSS 위경도 격자는 약 14.8~18.5cm, 고도는 0.1m다. UDP가 아니라 NMEA 자릿수의 한계다.
- GGA quality=1, satellites=9, HDOP=1.0은 고정값이다. 실제 정확도나 위성기하로 믿지 않는다.
- realtime RMC 속도는 3,600배 작아 보통 0.0knot이다. course도 COG가 아니라 차체 yaw이므로 쓰지 않는다.
- IMU는 encoder가 아니라 Transform/Rigidbody에서 만든다. 자세는 ground truth에 가깝고 gyro·accel에만 noise가 붙는다.
- IMU white noise는 현재 켜져 있고 `Time.deltaTime` 의존이다. JSON stdev를 covariance로 그대로 쓰지 않는다.
- IMU·GNSS·차량 상태는 설정 Hz보다 느리고 동일 sample이 반복된다. IMU 50Hz는 target일 뿐이며 세 호스트 raw rate는 약 34/47/34Hz, 고유 device stamp는 약 25/38/18Hz였다.
- IMU realtime device stamp는 physics time이 아니라 송신 직전 host wall clock의 1ms 값이다. arrival time과 함께 기록하고, MORAI에서는 같은 device stamp를 다시 적분하지 않는다.
- 긴 gap 뒤 2ms 미만 catch-up burst가 생긴다. 30Hz는 Real Time MORAI의 A/B 후보일 뿐이고 20Hz는 마지막 fallback이며, 실차 IMU rate와 분리한다.
- IMU 20Hz와 camera 4대 동시 pause 시험에서도 IMU 19.83Hz, GNSS 18.87Hz, status/collision 약 40.5Hz, LiDAR cloud 8.62Hz였다. camera 네 대만 각각 약 19.85Hz로 target에 근접했다. jitter는 IMU 하나가 아니라 여러 MORAI stream의 공통 특성이다.
- simulator target은 60FPS지만 현재 pause/full-record Linux window update는 XDamage 기준 약 25.07event/s였다. 이는 exact present/physics FPS가 아니므로 Windows 목표 PC도 사양으로 추정하지 말고 PresentMon/ETW와 sensor arrival을 동시에 측정한다.
- 현재 `wheel_speed`는 encoder가 아니라 body-frame Rigidbody 속도다. encoder tick·양자화·전자잡음은 없다.
- 선택형 per-wheel 속도도 wheel collider 각속도 기반이다. MORAI 성능을 실차 encoder fusion 성능으로 해석하지 않는다.
- IMU quaternion GT와 차체 속도 GT를 결합하면 GNSS 음영 dead reckoning이 비현실적으로 정확해진다. 현실성 시험에서는 IMU `orientation`을 쓰지 않는다.
- 일반 `/ad/vehicle/status.position·rpy`도 pose GT 누출이다. localization 입력에서 차단하고 dev GT와 함께 평가 전용으로 격리한다.
- 카메라는 Unity render+JPEG Q90이다. 현재 front/left/right/traffic-light 4대가 모두 활성이다. rolling shutter·노출·ISP·전자잡음·렌즈 빗물 모델은 없다.
- 카메라 UDP는 65KB datagram이라 IP fragmentation에 취약하다. fragment timeout·완전성 검사·drop 통계를 둔다.
- UDP 센서·제어에는 신뢰할 sequence/CRC/ACK/retransmission이 부족하다. malformed 검사와 control watchdog을 둔다.
- raw LiDAR는 +Y forward/+X right이고 ROS는 +X forward/+Y left다. 좌표변환은 한 번만 한다.
- 센서 위치 기준은 rear axle 중심이다. `base_link`, axle, sensor TF와 지면 높이를 명시적으로 분리한다.
- 기본 JSON과 noise JSON은 현재 동일하다. 파일명으로 noise 상태를 추정하지 말고 값과 hash를 확인한다.
- MORAI 결과는 현재 geometry·physics·protocol 통합시험 결과일 뿐 실차 센서 정확도나 악천후 성능 증거가 아니다.
