# MORAI runtime notes (2026-07-23)

## 확인 환경

- build: `25.S4.MolitComp03`, Unity `2020.3.6f1`
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, driver `590.48.01`
- log: `~/.config/unity3d/MORAI/Simulator/Player.log` 및 `Player-prev.log`

## 반복 종료 원인

Vulkan 실행에서 native `SIGSEGV`가 세 번 재현됐다. 스택은 모두 Unity의
`vk::ImagePool`/`vk::Texture`와 `GfxDeviceVK` 내부였으며 ROS 프로세스, OOM,
Python 예외 스택은 아니었다. 대표 경로는 다음 두 가지다.

```text
vk::ImagePool::ProcessFrontImage -> vk::Texture::FreeUnusedPoolImagesImmediate
vk::ImagePool::PopFront -> vk::Texture::Create -> GfxDeviceVK::UploadTexture2D
```

MORAI 문의 시 crash 직후의 `Player.log`, `Player-prev.log`, GPU/driver 정보와 함께
제공한다. 런처의 `Running` 표시는 프로세스 종료 뒤에도 잠시 남을 수 있으므로
`pgrep -af Simulator.x86_64`로 실제 프로세스를 확인한다.

## OpenGL 우회 결과

`Simulator.x86_64 -force-glcore`는 OpenGL 4.5로 기동됐지만 이 빌드에서
`Failed to read GPU texture`가 반복됐고 카메라와 LiDAR UDP가 나오지 않았다.
GPS/IMU는 계속 출력됐지만 인지/DWA 검증에는 사용할 수 없으므로 운영 우회책으로
채택하지 않는다. 센서 설정 JSON은 변경하지 않았다.
