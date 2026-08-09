# 동적 장애물 rosbag 재생 테스트

아래 명령은 ROS 2 workspace가 `~/heven-ad-2026`에 있고,
`yolo26s.pt`가 기본 모델 경로에 준비된 상태를 기준으로 한다.

먼저 터미널 1에서 detector와 visualizer를 실행하고, 노드가 준비된 뒤
터미널 2에서 rosbag을 재생한다.

## 터미널 1: YOLO 추론 및 OpenCV 시각화

```bash
cd ~/heven-ad-2026
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ad_camera_perception dynamic_obstacle_detection.launch.py \
  device:=cuda:0 \
  show_window:=true
```

`dynamic_obstacle_debug` OpenCV 창에 YOLO bbox가 표시된다. 테스트를 끝낼
때는 터미널 1에서 `Ctrl+C`를 눌러 두 노드와 시각화 창을 함께 종료한다.

## 터미널 2: 카메라 rosbag 재생

```bash
source /opt/ros/humble/setup.bash
ros2 bag play \
  ~/ad_camera_perception_bag/rosbag2_2026_07_24-18_57_43
```

반복 재생이 필요하면 마지막 명령에 `--loop`를 추가한다.

```bash
ros2 bag play \
  ~/ad_camera_perception_bag/rosbag2_2026_07_24-18_57_43 \
  --loop
```

다른 녹화본을 확인할 때는 마지막 bag 디렉터리 이름만 바꾸면 된다.

## 문제 확인

노드가 카메라 영상을 받고 있는지 확인하려면 별도 터미널에서 다음 명령을
실행한다.

```bash
source /opt/ros/humble/setup.bash
source ~/heven-ad-2026/install/setup.bash
ros2 topic hz /ad/sensors/camera/front/compressed
```

CUDA 관련 오류가 발생할 때만 터미널 1의 `device:=cuda:0`을
`device:=cpu`로 바꿔 실행한다.
