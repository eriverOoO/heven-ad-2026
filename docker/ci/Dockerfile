# syntax=docker/dockerfile:1.7

FROM ros:humble-ros-base-jammy@sha256:7bea3d9aa2483d3ca34c8e30d921b79273b0913bd7dc64bebf51d082b5d107e4

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG APT_REFRESH=manual

LABEL org.opencontainers.image.title="HEVEN AD ROS 2 Humble CI dependencies" \
      org.opencontainers.image.description="Source-free build and test dependencies for heven-ad-2026" \
      org.opencontainers.image.source="https://github.com/skku-heven/heven-ad-2026"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    set -euo pipefail; \
    echo "APT refresh: ${APT_REFRESH}"; \
    apt-get update; \
    apt-get install --no-install-recommends --yes \
      ccache \
      git \
      git-lfs \
      libunwind-dev \
      python3-vcstool \
      ros-dev-tools; \
    git lfs install --system; \
    ccache --max-size=8G; \
    ccache --set-config=compression=true

ARG DEPENDENCY_FINGERPRINT=unknown

# The repository is exposed only to this RUN instruction as a read-only
# BuildKit bind mount. It is never copied into an image layer.
RUN --mount=type=bind,target=/tmp/ws/src/heven_ad_2026,readonly \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    set -euo pipefail; \
    cd /tmp/ws; \
    vcs import src \
      --input src/heven_ad_2026/dependencies.repos \
      --shallow \
      --skip-existing; \
    src/heven_ad_2026/scripts/apply_dependency_patches.sh /tmp/ws/src; \
    set +u; source /opt/ros/humble/setup.bash; set -u; \
    colcon list --base-paths src/heven_ad_2026 --names-only \
      | sort -u > /tmp/heven-packages.txt; \
    test -s /tmp/heven-packages.txt; \
    mapfile -t heven_packages < /tmp/heven-packages.txt; \
    readonly -a runtime_packages=(autoware_multi_object_tracker); \
    colcon list --packages-up-to \
      "${heven_packages[@]}" "${runtime_packages[@]}" \
      --paths-only \
      | sort -u > /tmp/heven-package-paths.txt; \
    test -s /tmp/heven-package-paths.txt; \
    if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then \
      rosdep init; \
    fi; \
    rosdep update --rosdistro humble; \
    mapfile -t heven_package_paths < /tmp/heven-package-paths.txt; \
    rosdep install --from-paths \
      "${heven_package_paths[@]}" \
      --ignore-src \
      --rosdistro humble \
      -y; \
    find /tmp/ws/src -mindepth 1 -maxdepth 1 \
      ! -name heven_ad_2026 -exec rm -rf -- {} +; \
    rm -rf /tmp/ws/log; \
    rm -f /tmp/heven-packages.txt /tmp/heven-package-paths.txt; \
    mkdir -p /opt/heven-ci; \
    printf '%s\n' "${DEPENDENCY_FINGERPRINT}" \
      > /opt/heven-ci/dependency-fingerprint

WORKDIR /__w/heven-ad-2026/heven-ad-2026
