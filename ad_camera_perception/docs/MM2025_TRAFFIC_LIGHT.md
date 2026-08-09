# mm-2025 신호등 인식 파이프라인

이 파이프라인은 MORAI Camera-4 영상을 mm-2025 YOLOv7 모델로 추론하고, 복합
신호를 보존한 상태로 Planner에 전달하기 위한 ROS 2 구성이다. 정지·출발·가속
판단은 하지 않는다.

## 데이터 흐름

1. `traffic_light_detector`가
   `/ad/sensors/camera/traffic_light/compressed`를 구독한다.
2. YOLOv7 검출 결과를 `/vision/traffic_light/detections`에 발행한다.
3. `traffic_light_evaluator`가 target ROI 안에서 대상 신호등을 선택하고 5프레임 중
   3프레임 투표로 안정화한다.
4. `/vision/traffic_light/status`에 `red`, `yellow`, `straight_green`,
   `left_green`, `valid`, `confidence`, 원본 class와 bbox ID를 발행한다.
5. `traffic_light_visualizer`가 같은 timestamp의 영상·검출·상태를 합쳐 독립적인
   OpenCV 창에 표시한다.

## OpenCV 시각화

시각화 노드는 RViz, rqt, Foxglove 또는 별도 웹 UI를 사용하지 않는다.
`cv2.imshow()` 창 하나에 다음 정보를 직접 표시한다.

- target ROI
- 모델이 검출한 모든 bbox와 class/confidence
- evaluator가 선택한 bbox
- 최종 `valid` 상태와 독립 신호 aspect

창에서 `q` 또는 `Esc`를 누르면 시각화 창만 닫힌다. 창을 띄우려면 실행 터미널에
`DISPLAY` 또는 `WAYLAND_DISPLAY`가 설정되어 있어야 한다.

## 실행 준비

모델 weight와 외부 YOLOv7 source는 저장소에 포함하지 않는다. 다음 두 절대 경로를
준비한다.

- `model_path`: mm-2025의 `yolov7_best.pt`
- `yolov7_repository_path`: `models/experimental.py`와 `utils/general.py`가 있는
  YOLOv7 checkout

```bash
ros2 launch ad_camera_perception traffic_signal.launch.py \
  model_path:=/absolute/path/to/yolov7_best.pt \
  yolov7_repository_path:=/absolute/path/to/mm_yolov7_ros/src/yolov7 \
  enable_visualizer:=true \
  show_window:=true
```

실제 Camera-4 영상에서 target ROI, confidence, IoU, voting 값은 별도의 실시간
검증에서 조정한다.
