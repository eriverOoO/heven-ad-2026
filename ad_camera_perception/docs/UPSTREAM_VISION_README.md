# ad_camera_perception

HEVEN의 동적 장애물 및 신호등 인지를 담당하는 ROS 2 Humble Python 패키지다.

## 설계 경계

- detector, evaluator, visualizer는 각각 독립 노드로 구현한다.
- detector는 `vision_msgs/msg/Detection2DArray`를 발행한다.
- evaluator는 `ad_interfaces`의 상세 상태 메시지를 직접 발행한다.
- visualizer는 제어 경로에서 선택 사항이며 동일 구현을 서로 다른 설정으로 두 번 실행한다.
- ROS 노드는 입출력과 파라미터 처리만 담당하고 판단 로직은 순수 Python 모듈에 둔다.
- 카메라 callback은 최신 프레임만 보관하고 timer callback에서 추론한다.

## Python 모듈

| 모듈 | 역할 |
|---|---|
| `nodes` | 얇은 ROS 2 node wrapper |
| `inference` | 모델 backend와 detection 변환 |
| `dynamic_obstacle` | ROI 판정과 시간 필터 |
| `traffic_light` | 대상 신호 선택과 상태 필터 |
| `visualization` | 공용 debug overlay |
| `utils` | 파라미터 및 공통 유틸리티 |

## 동적 장애물 bbox replay

현재 detector는 COCO-pretrained `yolo26s.pt`로 candidate bbox를 만들고, 초기 제외 class
ID `[9, 10, 11, 12]`를 제거한 뒤 원본 이미지 좌표의
`vision_msgs/msg/Detection2DArray`를 발행한다. 입력 callback은 최신 프레임만 유지한다.

```bash
python3 -m pip install -r ad_camera_perception/requirements.txt
mkdir -p "${XDG_CACHE_HOME:-$HOME/.cache}/ad_camera_perception/models"
cd "${XDG_CACHE_HOME:-$HOME/.cache}/ad_camera_perception/models"
python3 -c "from ultralytics import YOLO; YOLO('yolo26s.pt')"
cd -
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to ad_camera_perception
source install/setup.bash
ros2 launch ad_camera_perception dynamic_obstacle_detection.launch.py
```

`device:=auto`가 기본값이며 Ultralytics가 사용 가능한 장치를 선택한다. GPU driver를
사용할 수 없는 replay 환경에서는 명시적으로 `device:=cpu`를 줄 수 있다.
모델 기본 디렉터리는 `${XDG_CACHE_HOME:-$HOME/.cache}/ad_camera_perception/models`이며
`AD_CAMERA_PERCEPTION_MODEL_DIR`로 바꿀 수 있다.

로컬 OpenCV 창을 시각화 노드에서 바로 띄우려면 다음처럼 실행한다. 이 경우
`rqt_image_view`는 필요하지 않다.

```bash
ros2 launch ad_camera_perception dynamic_obstacle_detection.launch.py \
  device:=cuda:0 show_window:=true
```

기본 `show_window:=false`에서는 GUI 없이 debug image topic만 발행하므로 headless 실행과
rosbag 기록이 가능하다.

다른 터미널에서 녹화 영상을 재생한다.

```bash
source /opt/ros/humble/setup.bash
ros2 bag play <bag-directory>
```

debug image는 `/ad/viz/perception/camera/dynamic_obstacle`에 발행된다.

```bash
rqt_image_view /ad/viz/perception/camera/dynamic_obstacle
```

기본 입력은 `/ad/sensors/camera/front/compressed`의
`sensor_msgs/msg/CompressedImage`다. 다른 입력은 launch argument로 바꿀 수 있다.

```bash
ros2 launch ad_camera_perception dynamic_obstacle_detection.launch.py \
  image_topic:=/camera/image_raw image_transport:=raw model_path:=/path/to/model.pt
```

모델 파일은 repository에 직접 넣지 않고
[models/README.md](../models/README.md)의 정책을 따른다.
