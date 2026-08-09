# heven-ad-2026

2026 대학생 AI·S/W 모빌리티 경진대회 AI융합자율주행 부문의 ROS 2 스택이다.

- 기준 OS: Ubuntu 22.04
- ROS: ROS 2 Humble
- Python: 3.10
- 기본 workspace: `~/heven_ad_2026_ws`

## Git 설정

commit 추적을 위해 `~/.gitconfig`에 본인의 영문 이름과 GitHub email을 설정한다.

```bash
git config --global user.name "English Name"
git config --global user.email "github-email@example.com"
git config --global --get user.name
git config --global --get user.email
```

동아리방 공용 컴퓨터에서는 commit 전과 `git push` 전에 위 두 값을 다시 확인하고,
다른 사람의 설정이면 본인 정보로 바꾼다.

## 처음 설치

홈 디렉터리에서 아래 블록 전체를 실행한다. 기본 branch는 `dev`다.

private 저장소는 최초 한 번만 `gh`로 로그인한다. 이후 HTTPS clone·fetch·pull에서는 GitHub 비밀번호를 다시 묻지 않는다.

<!-- heven-ad-workspace-bootstrap -->

```bash
cd "$HOME"

HEVEN_AD_EFFECTIVE_WS_PATH="${HEVEN_AD_WS_PATH:-$HOME/heven_ad_2026_ws}"

heven_ad_bootstrap() (
  set -euo pipefail

  if [[ ! -r /etc/os-release ]]; then
    echo "Ubuntu release 정보를 읽을 수 없습니다." >&2
    return 1
  fi
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
    echo "Ubuntu 22.04가 필요합니다: ${PRETTY_NAME:-unknown}" >&2
    return 1
  fi

  readonly HEVEN_AD_WS_PATH="$HEVEN_AD_EFFECTIVE_WS_PATH"
  readonly HEVEN_AD_REPOSITORY_PATH="$HEVEN_AD_WS_PATH/src/heven_ad_2026"
  readonly HEVEN_AD_REF="${HEVEN_AD_REF:-dev}"
  readonly HEVEN_AD_REMOTE="https://github.com/skku-heven/heven-ad-2026.git"

  sudo apt-get update
  sudo apt-get install --no-install-recommends -y \
    ca-certificates \
    curl \
    gh \
    git \
    locales \
    python3 \
    software-properties-common
  sudo locale-gen en_US en_US.UTF-8
  sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  export LANG=en_US.UTF-8
  sudo add-apt-repository -y universe

  if ! apt-cache show ros-humble-ros-base >/dev/null 2>&1; then
    readonly ROS_APT_SOURCE_VERSION="$(
      curl -fsSL \
        https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])'
    )"
    readonly ROS_APT_SOURCE_DEB="$(mktemp --suffix=.deb)"
    curl -fL \
      -o "$ROS_APT_SOURCE_DEB" \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.jammy_all.deb"
    sudo dpkg -i "$ROS_APT_SOURCE_DEB"
    rm -f -- "$ROS_APT_SOURCE_DEB"
  fi

  sudo apt-get update
  sudo apt-get install --no-install-recommends -y \
    git-lfs \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    python3-yaml \
    ros-dev-tools \
    ros-humble-desktop \
    ros-humble-rosbag2-storage-mcap

  git lfs install
  if ! command -v uv >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -LsSf \
      https://releases.astral.sh/github/uv/releases/download/0.11.8/uv-installer.sh \
      | sh
  fi
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv >/dev/null

  mkdir -p "$HEVEN_AD_WS_PATH/src"
  if [[ -e "$HEVEN_AD_REPOSITORY_PATH" &&
    ! -d "$HEVEN_AD_REPOSITORY_PATH/.git" ]]
  then
    echo "기존 경로가 Git 저장소가 아닙니다: $HEVEN_AD_REPOSITORY_PATH" >&2
    return 1
  fi
  if [[ ! -d "$HEVEN_AD_REPOSITORY_PATH/.git" ]]; then
    if ! GIT_TERMINAL_PROMPT=0 \
      git ls-remote "$HEVEN_AD_REMOTE" "refs/heads/$HEVEN_AD_REF" \
      >/dev/null 2>&1
    then
      if ! gh auth status -h github.com >/dev/null 2>&1; then
        echo "Private repository 접근을 위해 최초 GitHub 로그인을 시작합니다."
        gh auth login -h github.com -w
      fi
      gh auth setup-git
      if ! GIT_TERMINAL_PROMPT=0 \
        git ls-remote "$HEVEN_AD_REMOTE" "refs/heads/$HEVEN_AD_REF" \
        >/dev/null 2>&1
      then
        echo "GitHub 인증 후에도 repository에 접근할 수 없습니다." >&2
        echo "조직 접근 권한과 SSO 승인을 확인하세요: $HEVEN_AD_REMOTE" >&2
        return 1
      fi
    fi
    git clone \
      --branch "$HEVEN_AD_REF" \
      --recurse-submodules \
      "$HEVEN_AD_REMOTE" \
      "$HEVEN_AD_REPOSITORY_PATH"
  fi

  "$HEVEN_AD_REPOSITORY_PATH/scripts/bootstrap_workspace.sh"
)

if heven_ad_bootstrap; then
  source /opt/ros/humble/setup.bash
  source "$HEVEN_AD_EFFECTIVE_WS_PATH/install/setup.bash"
  export AD_DATA_DIR="$HEVEN_AD_EFFECTIVE_WS_PATH/src/heven_ad_2026/ad_data"
  echo "HEVEN AD 준비 완료: $HEVEN_AD_EFFECTIVE_WS_PATH"
else
  echo "HEVEN AD bootstrap 실패: 위에서 처음 발생한 오류를 확인하세요." >&2
fi
unset -f heven_ad_bootstrap
unset HEVEN_AD_EFFECTIVE_WS_PATH
```

MORAI simulator와 GPU driver는 별도로 설치한다.

## 빌드

전체 의존성·build·test를 다시 검증한다.

```bash
cd "$HOME/heven_ad_2026_ws/src/heven_ad_2026"
./scripts/bootstrap_workspace.sh
```

수정한 패키지만 다시 빌드한다.

```bash
cd "$HOME/heven_ad_2026_ws"
source /opt/ros/humble/setup.bash
PACKAGE_NAME=ad_localization
colcon build --symlink-install --packages-up-to "$PACKAGE_NAME"
source install/setup.bash
```

## 실행

```bash
cd "$HOME/heven_ad_2026_ws"
source /opt/ros/humble/setup.bash
source install/setup.bash
export AD_DATA_DIR="$PWD/src/heven_ad_2026/ad_data"
ros2 launch ad_bringup bringup.launch.py control_enabled:=false
```

확인 후 주행할 때만 `control_enabled:=true`로 바꾼다. MORAI UDP 설정은
[protocol coverage](docs/morai/protocol-coverage.md)를 참고한다.

## Rosbag

```bash
source /opt/ros/humble/setup.bash
source "$HOME/heven_ad_2026_ws/install/setup.bash"
mkdir -p "$HOME/heven_ad_2026_ws/bags"
ros2 bag record -s mcap \
  -o "$HOME/heven_ad_2026_ws/bags/run_$(date +%Y%m%d_%H%M%S)" \
  --all
```

모르는 내용이나 오류가 있으면 실행한 명령과 오류 전문을 Agent/LLM에게 전달해 질문한다.
