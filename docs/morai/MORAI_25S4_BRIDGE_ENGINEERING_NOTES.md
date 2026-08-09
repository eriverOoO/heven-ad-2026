# MORAI 25.S4 브리지 엔지니어링 노트

이 문서는 MORAI 25.S4 MolitComp03 환경에서 확인한 센서·UDP·차량 제어 특성을
팀 코드에 반영할 때 필요한 판단만 간추린다. 수치와 동작은 현재 보존된 simulator
binary, sensor 설정과 2026-08-01~2026-08-04 실행 관측에 한정된다. 상세한 역분석
근거와 재현 조건은
[`MORAI_25S4_REVERSE_ENGINEERING_REPORT_2026-08-01.md`](MORAI_25S4_REVERSE_ENGINEERING_REPORT_2026-08-01.md)를
참고한다.

## 시나리오와 제어 관측

- 신호등 인식 결과를 이용한 정지는 동작했으나, 단순 제동 명령만으로 정지선을
  맞추면 차량이 정지선을 넘는 구간이 있었다. 신호 상태 판정과 별개로 목표 정지점,
  남은 거리와 현재 속도를 사용하는 종방향 감속 제어가 필요하다.
- 원형 교차로 진입 전 좌회전 신호를 직진 진행으로 해석한 사례가 있었다. 이는
  신호등 인식 성공 여부와 경로의 진행 방향 의미를 분리해 검증해야 함을 뜻한다.
- UDP 제어 패킷에는 신뢰할 수 있는 timestamp, sequence, ACK와 CRC가 없다.
  제어 송신 측에서 명령 유효시간, 값 범위와 stale-command fallback을 관리해야 한다.
- 이 관측은 현재 MORAI 차량 동역학에 대한 결과다. 실차의 제동 응답이나 정지거리로
  일반화하지 않는다.

## GNSS 출력의 의미

- 위치는 Unity sensor Transform 좌표에 설정된 Gaussian noise를 더한 뒤 NMEA로
  변환한다. 실제 위성 배치, multipath, NLOS, RTK와 Doppler 측정 모델은 확인되지
  않았다.
- 현재 위경도 표현의 격자는 K-City 부근에서 축별 약 14.8~18.5 cm이고, 고도는
  0.1 m 단위다. 이는 UDP 전송 오차가 아니라 NMEA 문자열의 수치 해상도다.
- 현재 설정에서는 위치 noise의 축별 표준편차가 약 0.5 m다. GGA의 fix quality,
  satellite count와 HDOP는 실제 위성 기하에서 계산된 품질 지표가 아니다.
- realtime RMC speed는 단위 변환 결함이 있고 course는 진행 속도 벡터가 아닌 sensor
  heading이다. 따라서 두 필드는 일반적인 GNSS 속도와 COG 관측값으로 사용하지 않는다.

## IMU 출력과 주기

- orientation은 Unity Transform 자세에 가깝다. angular velocity와 linear
  acceleration은 Rigidbody 상태에서 계산한 값에 설정된 noise를 더한다. 실제 9축
  장치 내부 자세 추정기의 drift, 온도 변화와 자기장 교란을 재현한 출력은 아니다.
- 설정의 50 Hz는 요청 주기이며 보장되는 hardware sampling rate가 아니다. 동일
  binary에서도 host 부하에 따라 실제 UDP 수신률과 고유 device timestamp 비율이
  달랐다.
- coroutine이 늦어진 주기를 보상하면서 긴 gap 뒤 짧은 간격으로 packet을 연속
  송신할 수 있다. 이때 같은 device timestamp와 같은 상태가 반복될 수 있으며,
  일부 실행에서는 상태가 같아도 noise가 다시 생성되어 payload 전체는 달라졌다.
- 20 Hz 설정은 평균 수신률과 중복 비율을 개선한 관측이 있지만 scheduler의 jitter를
  제거하지는 않았다. 설정률은 host와 전체 sensor 부하를 포함한 A/B 결과로 선택하고,
  실차 sensor 설정과 분리한다.
- 선택형 compatibility boundary는 동일 device timestamp의 반복 측정을 제거할 수
  있다. 기본 대회 경로에서는 비활성으로 두며, 실행 관측으로 필요성이 확인된 경우에만
  켠다.

## 차량 속도 출력

- 대회용 vehicle status의 종방향 속도는 wheel encoder tick이 아니라 body-frame
  Rigidbody velocity다.
- pulse/revolution, 타이어 유효반경 오차, 저속 양자화, missed pulse와 전자 잡음은
  포함되지 않는다. simulator 내부에서는 매우 안정적인 속도 입력으로 사용할 수
  있지만, 이를 실차 encoder 성능의 근거로 해석하지 않는다.

## LiDAR snapshot과 timestamp

- 현재 CH16/VLP-16 경로는 360도를 한 Unity frame에서 생성한 뒤 75개 UDP packet을
  짧은 burst로 전송한다. 실제 회전형 LiDAR처럼 scan 동안 차량 pose가 계속 변하는
  rolling acquisition이 아니다.
- MORAI profile에서는 point relative time을 0으로 만든다. azimuth를 0~0.1 s로
  펼치면 simulator에 없던 차량 운동을 point cloud에 인위적으로 적용할 수 있다.
- raw packet에는 ROS header가 없으므로 bridge가 각 datagram의 socket receipt time을
  기록한다. snapshot scan의 기준 timestamp는 scan 조립이 끝난 마지막 packet의
  receipt time을 사용한다. 첫 packet 시각을 기준으로 잡으면 약 한 scan 주기만큼
  최신 차량 자세와 어긋날 수 있다.
- 현재 설정의 LiDAR 거리 noise는 꺼져 있다. 강한 outlier 제거를 기본 적용하면
  VLP-16의 얇고 먼 정상 point를 잃을 수 있으므로, 실제 이상점이 관측된 경우에만
  제한적으로 적용한다.
- 날씨가 LiDAR 거리 감쇠, 산란, dropout과 false return으로 연결되는 구현은 현재
  binary에서 확인하지 못했다. 안개 화면만으로 실제 악천후 LiDAR 성능을 검증했다고
  판단하지 않는다.

## Camera UDP

- 카메라는 Unity render 결과를 JPEG로 전달하며 실제 imager의 rolling shutter,
  노출, ISP와 전자 noise 모델을 보장하지 않는다.
- 한 UDP datagram이 약 65 KB이므로 일반 Ethernet MTU에서는 IP fragmentation이
  발생한다. bridge는 fragment 순서와 크기, frame timestamp, timeout, JPEG 경계와
  incomplete-frame 수를 제한한다.
- 현재 전체 입력 계측에서는 전방·좌측·우측·신호등 카메라 네 stream이 모두
  동작했다. 특정 실행의 미수신을 simulator의 영구 제약으로 일반화하지 않는다.

## ROS timestamp 정책

- 현재 운영 경로에서는 모든 sensor가 공유하는 신뢰 가능한 측정 clock을 제공한다고
  가정하지 않는다. MORAI가 포함한 device timestamp도 physics sample time이 아니라
  직렬화 시점의 host wall clock이거나, sensor마다 의미가 다른 값일 수 있다.
- bridge는 `recvfrom()` 직후의 monotonic receipt time을 기록하고, process 시작 시
  계산한 ROS wall-time epoch로 변환해 기본 `header.stamp`로 사용한다.
- 원본 device timestamp가 있는 message는 이를 별도 field와 timing diagnostic에
  보존한다. source timestamp를 선택하는 모드는 값의 유효성, arrival과의 차이,
  중복과 회귀를 검사한 뒤에만 사용한다.
- camera는 완성 frame의 첫 fragment receipt time을 사용한다. LiDAR는 snapshot이
  완성된 마지막 packet receipt time을 사용한다. 서로 다른 packet 조립 규칙을 하나의
  공통 규칙으로 오해하지 않는다.
- 수신률, malformed packet, queue drop, source duplicate, stamp regression과 jitter를
  stream별 diagnostic으로 확인한다. 단순 topic 발행률만으로 sensor의 독립 update
  수를 판단하지 않는다.

## TF 기준점

- MORAI 차량 좌표의 기준은 실제 rear axle 중심이다. 이를 `rear_axle_link`로 둔다.
- 일반적인 차량 알고리즘이 사용하는 `base_link`는 rear axle의 지면 투영점을
  기대하므로, 현재 모델에서는 wheel radius 절반만큼 낮춘 정적 근사로 정의한다.
- suspension에 따른 순간 높이를 정확히 반영하려면 차량 동역학 상태에서 dynamic
  transform을 복원해야 한다. 현재 simulator 입력과 개발 효익을 고려해 이 복원은
  적용하지 않는다.
- sensor extrinsic은 `ad_description`에서 정적 TF로 한 번만 발행한다. bridge 또는
  개별 sensor launch에서 같은 transform을 중복 발행하지 않는다.

## 대회용과 개발용 경계

- `ad_morai_bridge`는 대회에서 허용된 sensor와 vehicle-status UDP, 선택형 control
  출력만 담당한다. 기본 control 송신은 비활성이다.
- simulator actor, scenario와 비대회용 데이터 접근은 `ad_morai_bridge_dev`에
  격리한다. 대회 bringup에서 개발용 package를 의존하거나 실행하지 않는다.
- simulator binary, sensor JSON, host 사양, target FPS, camera 구성이나 통신 경로가
  바뀌면 이 문서의 rate와 timestamp 가정을 다시 검증한다.
