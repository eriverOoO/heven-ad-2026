# Legacy localization utilities

이 디렉터리는 현재 기본 localization stack에 포함되지 않지만, 과거 구현과 지도
생성 과정을 재현할 때 참고할 수 있는 코드를 보존한다.

- `heven_control/`: 과거 경로 추종 구현
- `heven_slam/`: 과거 RTAB-Map 기반 LiDAR SLAM 도구
- `tools/`: MORAI Unity map bundle에서 경로를 추출·회전하던 보조 스크립트

이 코드는 `ad_bringup`에서 자동 실행되지 않는다. 특히 `tools/`는 `UnityPy`가 필요한
과거 방식이며, 현재 저장소의 정본 경로와 corridor는 `ad_data/` 및 최상위
`scripts/`의 결정론적 생성 도구를 사용한다.
