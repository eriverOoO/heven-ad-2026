# ops/runner — env와 배포 정본

이 문서는 planner와 self-hosted Actions runner의 **정확한 env/deploy reference**다.
스크립트·프롬프트의 versioned source는 `ops/runner/`에 있고, 실행 머신에는 merge된
source를 workspace의 `runner/` 디렉터리로 배포한다. 머신에서 직접 수정하지 않는다.

| 파일 | 실행 위치 | 역할 |
|---|---|---|
| `bin/agent-tick` | planner machine, systemd timer | GitHub context → Codex 2-phase → issue proposal 생성 |
| `bin/codex-pr-review.sh` | self-hosted Actions runner | Ready PR에 advisory Codex review comment 작성 |
| `bin/pr-set-in-review.sh` | self-hosted Actions runner | linked issue Status를 PR draft 상태와 동기화 |
| `bin/render_prompt.sh` | 공통 | allowlist 변수만 prompt에 렌더링 |
| `repo-context.sh` | 공통 | repo·optional Notion URL·open milestones context |
| `prompts/` | 공통 | planner phase 1/2, PR review prompt |
| `systemd/` | planner machine | `agent-tick` user service/timer |

## Env file

`AGENT_ENV_FILE`로 별도 경로를 지정할 수 있다. 지정하지 않으면
`~/.config/heven-agent/heven-agent.env`를 먼저 읽고, 구 경로
`~/.config/heven-common-test-agent/heven-common-test.env`는 fallback으로만 읽는다.
env file은 커밋하지 않는다.

### 공통/Planner

| key | required | meaning/default |
|---|---|---|
| `GH_REPO` | yes | `owner/repo`; Actions에서는 `GITHUB_REPOSITORY`가 우선 |
| `PROJECT_NO` | planner | planner가 읽을 org ProjectV2 number |
| `AGENT_WS` | no | machine workspace, default `${HEVEN_COMMON_TEST_WS:-$HOME/heven_common_test_ws}` |
| `REPO_CHECKOUT` | no | planner default `$AGENT_WS/src/<repo-name>`; PR review는 nonempty `GITHUB_WORKSPACE`가 있으면 항상 그것을 쓰고, Actions 밖에서만 env 값을 사용 |
| `REPO_ROLE` | no | prompt용 repo 역할, default `HEVEN 팀 repo` |
| `DEFAULT_BRANCH` | no | default `dev` |
| `MODE` | no | `test`=hourly, `production`=daily 00:00 KST |
| `PHASE_GAP_SEC` | no | planner phase 간격, default `30` |
| `MAX_OPEN_AGENT_ISSUES` | no | planner backlog guard, default `10` |
| `CANDIDATE_MIN` / `CANDIDATE_MAX` | no | phase 1 후보 범위, default `3` / `5` |
| `MAX_ISSUES` | no | cycle당 issue 생성 상한, default `3` |
| `PRIORITY_ISSUE_FIELD_ID` | recommended | native Priority field id |
| `NOTION_GUIDE_URL` | no | repo-specific Notion hub; bootstrap이 read-back URL을 기록 |

Prompt는 Notion MCP가 있고 repo 목적·대회·hardware·environment·constraint 확인이 필요할
때만 `NOTION_GUIDE_URL`의 repo hub를 안내한다. 공통 workflow·컨벤션은 repo의 README와
AGENTS에 연결된 heven-common 가이드를 따른다. runner runtime과 automation은 Notion URL을
직접 fetch/curl하지 않는다.

### Project status authentication

`project-status.yml`은 `opened`, `ready_for_review`, `converted_to_draft`, `closed`를
구독하고 PR별 concurrency에서 이전 run을 취소한다. live PR이 draft면 `In progress`,
non-draft면 `In review`가 target이다. mutation 후 PR mode를 다시 확인하고 중간에 mode가
바뀌었으면 제한된 fresh pass로 보상한다. 그 뒤의 변경은 다음 PR event가 다시 수렴시킨다.
GitHub PR과 Project 사이의 원자 transaction을 주장하지 않고 event-driven eventual
convergence를 운영 계약으로 삼는다.

`pr-set-in-review.sh`는 live PR의 state와 draft 여부를 반복 확인하고, 그 PR이 닫는 각
open issue의 정확히 하나인 Project item에서 Project와 Status IDs를 runtime에 조회한다.
repo에 연결된 임의의 Project를 고르지 않는다. 여러 closing issue 중 닫힘·`Done`·이미
target인 항목은 그 항목만 no-op하고, 나머지 유효한 항목은 모두 사전 검증한 뒤
갱신한다. live target은 draft PR이면 `In progress`, non-draft PR이면 `In review`다. 일부
mutation 실패 뒤 재실행해도 완료된 항목은 건너뛰고 남은 항목으로 수렴한다.
`PROJECT_ID`, `STATUS_FIELD_ID`, `STATUS_OPT_*`를 env에 복제하지 않는다.

- `project-status.yml` 자동화를 켤 때 repo Actions secret `PROJECT_PAT`은 필수다.
  workflow는 이 secret을 `GH_TOKEN`으로 전달하며, opt-in 상태에서 비어 있으면 구성
  오류를 성공으로 숨기지 않고 실패한다.
- 스크립트는 진입 시 nonempty `GH_TOKEN`을 먼저 보존하고 최우선으로 쓴다. 이후 읽는
  env file의 `PROJECT_PAT`이나 `GH_TOKEN`은 이 incoming token을 덮지 못한다.
- incoming `GH_TOKEN`이 없을 때만 env file의 `PROJECT_PAT`을 사용한다. env file의
  `GH_TOKEN`은 fallback 인증으로 사용하지 않는다.
- incoming `GH_TOKEN`과 env file의 `PROJECT_PAT`이 모두 없으면 runner의 ambient `gh`
  인증을 사용한다.
- token permission: org Project query/mutation에 필요한 `project`와 대상 repo 접근

### Repo Actions settings

| kind | key | value |
|---|---|---|
| repo variable | `ENABLE_RUNNER_AUTOMATION` | `true`일 때 opt-in jobs 실행 |
| repo variable | `ENABLE_DISCORD_NOTIFICATIONS` | `true`일 때 Discord 알림 job 실행 |
| repo variable | `ENABLE_DISCORD_MENTIONS` | `true`일 때 `DISCORD_USER_MAP`으로 실제 mention 생성 |
| repo variable | `RUNNER_SCRIPTS` | deployed `runner/bin` absolute path |
| repo variable | `DISCORD_USER_MAP` | optional GitHub login→Discord ID JSON |
| repo secret | `PROJECT_PAT` | Project Status 동기화 자동화를 사용할 때 필수 |
| repo secret | `DISCORD_WEBHOOK` | optional notification webhook |

`DISCORD_USER_MAP`은 organization inheritance를 전제로 하지 않고 각 private repo의
repository variable로 설정한다.

## Private repository runner access

org owner는 GitHub **Organization Settings → Actions → Runner groups**에서 runner group의
public repository access를 항상 차단한다.
organization plan이 **Selected repositories**를 지원하면 대상 private repo만 허용한다.
Free plan/default group에서 그 범위를 선택할 수 없으면 public 접근 차단을 유지하고, 대상 private repo만
`ENABLE_RUNNER_AUTOMATION=true`로 opt-in한다. 다른 repo는 false 또는 미설정으로 둔다.
`ENABLE_RUNNER_AUTOMATION`은 bundled workflow job만 gate하며 runner access 격리 경계가 아니다.
실제 repository 격리는 runner group의 Selected repositories 설정이 담당한다.

runner의 online/offline 상태는 문서에 고정하지 않는다. 배포 시 GitHub API에서 현재 runner,
public access 차단, selected repository를 확인하고 private scheduling canary가 성공한 뒤에만
대상 repo의 `ENABLE_RUNNER_AUTOMATION=true`를 사용한다.

## Deploy

아래 명령은 repo root에서 실행한다. workspace/host는 machine 환경에 맞게 바꾼다.
`AGENT_WS`나 env file 위치를 기본값과 다르게 쓰면 설치 전에
`systemd/agent-tick.service`의 `ExecStart=`와 `EnvironmentFile=`도 같은 경로로 맞춘다.

```bash
# planner machine
mkdir -p ~/heven_common_test_ws/runner/{bin,prompts} ~/.config/systemd/user
rsync -a ops/runner/bin/     ~/heven_common_test_ws/runner/bin/
rsync -a ops/runner/prompts/ ~/heven_common_test_ws/runner/prompts/
cp ops/runner/repo-context.sh ~/heven_common_test_ws/runner/
cp ops/runner/systemd/* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now agent-tick.timer

# self-hosted Actions runner machine
ssh <runner-host> 'mkdir -p heven_common_test_ws/runner/{bin,prompts}'
scp ops/runner/bin/* <runner-host>:heven_common_test_ws/runner/bin/
scp ops/runner/prompts/* <runner-host>:heven_common_test_ws/runner/prompts/
scp ops/runner/repo-context.sh <runner-host>:heven_common_test_ws/runner/
```

배포 후 source repo와 deployed copy의 script/prompt가 같은지 확인하고, Actions runner와
planner service log에서 실제 실행 경로를 확인한다.
