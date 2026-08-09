# Vision model files

모델 weight는 크기가 크고 배포 정책 확인이 필요하므로 Git에 직접 commit하지 않는다.
노드는 package share 또는 환경 변수로 지정한 외부 경로에서 모델을 읽어야 한다.

모델을 도입할 때 다음 정보를 함께 기록한다.

- 파일명과 SHA256
- 모델 종류와 입력 크기
- class ID/name mapping
- dataset 및 dataset YAML 버전
- Ultralytics 버전
- 학습 명령과 validation 결과

동적 장애물 baseline 파일명은 `yolo26s.pt`다.

## mm-2025 신호등 모델

mm-2025의 `yolov7_best.pt`는 이 저장소에 commit하지 않는다. 파일을 설치된
`ad_camera_perception/models/`에 복사·symlink하거나 launch의 `model_path`에 절대
경로로 전달한다.

이 checkpoint는 원본 YOLOv7 Python source가 필요하다. `models/experimental.py`와
`utils/general.py`가 들어 있는 디렉터리를 `yolov7_repository_path`로 전달한다.
mm-2025 저장소에서는 일반적으로 `mm_yolov7_ros/src/yolov7`이다.

```bash
ros2 launch ad_camera_perception traffic_signal.launch.py \
  model_path:=/path/to/mm_yolov7_ros/models/yolov7_best.pt \
  yolov7_repository_path:=/path/to/mm_yolov7_ros/src/yolov7
```

Camera-4 입력은 `/ad/sensors/camera/traffic_light/compressed`다. 전용 visualizer는
기본적으로 `cv2.imshow()` 창을 열며 `q` 또는 `Esc`로 창만 닫을 수 있다.

현재 replay 기준 Python dependency는 `ultralytics==8.4.47`이다. `yolo26s.pt`를 단순
파일명으로 지정하면 Ultralytics가 공식 weight를 처음 한 번 내려받아 cache한다.

2026-07-24 replay에 사용한 공식 COCO weight:

- 파일명: `yolo26s.pt`
- 크기: `20,422,725 bytes`
- SHA256: `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`
- Ultralytics: `8.4.47`
