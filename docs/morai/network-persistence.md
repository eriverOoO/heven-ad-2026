# MORAI network port persistence

`25.S4.MolitComp03`의 `2023_Hyundai_Ioniq5` 네트워크 포트 계약은 다음과 같다.
`Destination Port`는 ROS 2 bridge가 bind하는 포트와 같아야 한다.

| MORAI Network Settings 항목 | 내부 publisher | Host Port | Destination Port | bridge profile |
|---|---|---:|---:|---|
| Competition Vehicle Status | `CompetitioninfoPublisher` (`PUBSUB_TYPE=295`) | 1908 | 1909 | competition `competition_status` |
| Ego Vehicle Status | `MoraiInfoPublisher` (`PUBSUB_TYPE=770`) | 1910 | 1911 | development `ego_status` |

## Why a direct JSON edit can revert

실행 중인 Simulator는 네트워크 설정을 메모리에 보관한다. Network Settings의
Connect 및 Save 동작은 그 메모리 값을 다시 수집한 뒤 아래 파일을
`FileMode.Create`로 새로 쓴다.

```text
$MORAI_ROOT/MoraiLauncher_Lin_Data/SaveFile/Network/25.S4.MolitComp03/NetworkInfo_2023_Hyundai_Ioniq5.json
```

따라서 Simulator가 기존 `908 -> 909`, `910 -> 911` 값을 메모리에 가진 상태에서
파일만 고치면 다음 Connect 또는 Save 때 기존 값으로 되돌아간다. 이 동작은
`Assembly-CSharp.dll`의 `DlgNetworkSetting.OnClickEgoNetConnect`,
`DlgNetworkSave.OnClickSaveFile`, `EgoVehicleController.SaveNetworkJSON` 경로에서도
확인했다.

Scenario load도 별도 주의가 필요하다. `SimulatorManager.LoadScenarioFile`은 모든
network를 먼저 disconnect하고 `<scenario-name>_MN.json`을 찾는다. 현재 sample
scenario에는 이 companion file이 없으므로 저장해 둔 NetworkInfo profile을 자동으로
다시 적용하지 않는다. 2026-07-23 live 재현에서는 이 직후 Connect를 누르자 runtime의
기본값이 위 NetworkInfo 파일을 다시 `908/909`, `910/911`로 덮어썼다. 임의로 기존
NetworkInfo를 `_MN.json`으로 복사하는 방법은 이 build에서 scenario load의 null
reference를 유발했으므로 사용하지 않는다.

## Persistent update order

1. Scenario를 먼저 load한다. Scenario load는 기존 network connection을 끊는다.
2. MORAI Network Settings가 **Disconnected**인지 확인한다.
3. Network Settings의 **Load**에서 위 NetworkInfo profile을 불러온다. 이 build에서는
   성공한 Load가 runtime 메모리를 갱신하고 network도 다시 연결한다. 파일
   관리자 기본 선택은 `NPCGhost.json`일 수 있으므로 반드시 `NetworkInfo...json`
   행을 직접 선택한 뒤 Load한다.
4. UI와 파일의 `PUBSUB_TYPE=295`와 `770` 항목이 각각 `1908 -> 1909`,
   `1910 -> 1911`인지 확인한다.
5. 아직 Disconnected라면 값이 맞는 상태에서만 **Connect**한다. 필요할 때만
   **Save**해서 runtime 메모리와 profile을 같은 값으로 만든다.
6. competition bridge가 UDP `1909`, development bridge가 UDP `1911`에 bind했고
   각 ROS 2 status topic의 수신 시각이 갱신되는지 확인한다.

JSON만 직접 수정한 뒤 바로 Connect하지 않는다. 특히 scenario load 후에는 반드시
고친 profile을 Load해야 한다. Connect 자체가 메모리의 오래된 값을 파일에 저장하기
때문이다.
