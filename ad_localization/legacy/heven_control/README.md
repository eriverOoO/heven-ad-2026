# heven_control

기존 commissioning 작업의 저속 GPS/IMU pure-pursuit 경로 추종기만 임시로
보존한 package다. MORAI UDP 제어 변환과 arm/disarm service는
`ad_morai_bridge`의 제어 경계로 대체되어 이 package에서 제거했다.

`route_follower`는 아직 기존 `/gps/fix`, `/imu/data`, `/vehicle/command` topic
계약을 사용한다. `ad_control` 이관 전까지는 현재 `/ad/...` 대회용 stack에 직접
연결해 실행하지 않는다. 이 package는 후속 제어 package 이관 commit에서 함께
정리할 대상이다.

경로 계산의 순수 함수와 관련 단위 테스트는 중간 이관 단계의 회귀 확인을 위해
남겨 둔다.
