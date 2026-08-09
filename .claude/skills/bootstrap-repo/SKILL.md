---
name: bootstrap-repo
description: Use when setting up or rerunning bootstrap for a new private repository created from the heven-common template.
---

# bootstrap-repo — 새 private repo 초기 설정

`heven-common`에서 만든 private repo의 Notion hub와 GitHub 설정을 한 흐름으로
초기화한다. public/internal target은 어떤 mutation도 하기 전에 거부한다.

요청 문구:

> 이 repo의 `bootstrap-repo` skill을 사용해서 `skku-heven/<new-repo>`를 부트스트랩해줘.

정본:

- Canonical Notion parent: [repo wiki](https://app.notion.com/p/39c3bf06830080b5a024c7ad91855240)
- Common Notion guide/template: [공통 가이드](https://app.notion.com/p/39c3bf068300807c9e5bcf6b469f94ee)
- Bootstrap/runner operations: [Operations](https://app.notion.com/p/39c3bf068300815fb880e1d20602e662)

canonical parent인 repo wiki 아래에 repo 전용 문서 hub를 준비한다.

## 1. One-shot bootstrap inputs

Collect target repo, REPO_PURPOSE, and all three modes in one request.

- Target repo: owner/repo
- Repository purpose: exact one-line REPO_PURPOSE
- Discord notifications mode: DISCORD_NOTIFICATIONS is enabled or disabled
- Discord mentions mode: DISCORD_MENTIONS is enabled or disabled
- Runner mode: RUNNER_AUTOMATION is enabled or disabled

Explicit opt-outs: ask for every mode; disabled must be selected explicitly and never inferred from silence.
mentions enabled에는 notifications enabled가 필요하다. enabled mode의 조건부 값도 같은
요청에서 식별하고, disabled mode의 조건부 값은 받지 않는다.

### Target provenance preflight

Target REST metadata must prove private visibility, default dev, admin access, and template_repository.full_name skku-heven/heven-common.
Provenance preflight completes before any Notion mutation.

Notion page를 만들거나 재사용하기 전에 repo root에서 다음 read-only preflight를 실행한다.
이 mode는 target과 `gh` 인증만 사용하며 bootstrap input, secret, Notion URL을 요구하지 않는다.

```bash
.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh --preflight-only skku-heven/<new-repo>
```

read-only preflight가 하나라도 다르면 secret을 요청하거나 Notion/GitHub를 mutation하지
않고 종료한다.

### Enabled secret injection

Only DISCORD_WEBHOOK and PROJECT_PAT are plaintext secrets.
Request missing conditional values only through secure environment/secret input, never plaintext chat/docs.
Ask the user to inject only DISCORD_WEBHOOK and PROJECT_PAT through host secure env/secret input.
Verify each enabled secret is present without printing its value.
If host secure input is unavailable, stop before mutation.

- notifications enabled: `DISCORD_WEBHOOK` 필요
- runner enabled: `PROJECT_PAT` 필요
- mentions enabled: non-secret `DISCORD_USER_MAP` 필요
- runner enabled: non-secret `RUNNER_SCRIPTS` 필요

DISCORD_USER_MAP and RUNNER_SCRIPTS are conditional non-secret values; do not persist their values in durable docs.
환경 주입 뒤에는 값이 set/non-empty인지 조건식으로만 확인하고 echo, command argv,
report에 값을 노출하지 않는다.

Local fallback: provide a read -s same-shell wrapper that invokes bootstrap before unsetting secrets.
host에 secure input 기능이 없으면 agent는 실행을 중단하고, 사용자가 non-secret env를 먼저
export한 로컬 terminal의 같은 shell에서 다음 wrapper를 실행하도록 안내한다.

```bash
bootstrap_with_local_secrets() (
  trap 'unset DISCORD_WEBHOOK PROJECT_PAT' EXIT
  local target="$1" rc=0

  if [[ "${DISCORD_NOTIFICATIONS:-}" == enabled ]]; then
    IFS= read -r -s -p 'Discord webhook: ' DISCORD_WEBHOOK || return
    printf '\n' >&2
    export DISCORD_WEBHOOK
  fi
  if [[ "${RUNNER_AUTOMATION:-}" == enabled ]]; then
    IFS= read -r -s -p 'Project PAT: ' PROJECT_PAT || return
    printf '\n' >&2
    export PROJECT_PAT
  fi

  [[ "${DISCORD_NOTIFICATIONS:-}" != enabled || -n "${DISCORD_WEBHOOK:-}" ]] || return 1
  [[ "${RUNNER_AUTOMATION:-}" != enabled || -n "${PROJECT_PAT:-}" ]] || return 1

  .claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh "$target" || rc=$?
  return "$rc"
)
bootstrap_with_local_secrets skku-heven/<new-repo>
unset -f bootstrap_with_local_secrets
```

## 2. Notion hub contract

Notion MCP로 canonical parent와 common guide/template을 fetch한다. target의 repo 이름만
hub title로 사용하고 canonical parent의 direct child만 열거한다.

For the repo hub exact direct-child title match: 0 -> create, 1 -> reuse, and 2+ -> fail.
부분 일치나 workspace 전체 search 결과로 대신하지 않는다.

선택한 hub 바로 아래의 exact title `Project Info`도 0개면 common template에서 만들고,
1개면 덮어쓰지 않고 재사용하며, 2개 이상이면 중단한다.
Create or reuse Project Info from the common template and read it back.

## Repository metadata

hub에는 다음 canonical serialization을 가진 section을 정확히 하나만 둔다.

- GitHub: https://github.com/<owner>/<repo>
- Purpose: <exact REPO_PURPOSE>
- Common guide: https://app.notion.com/p/39c3bf068300807c9e5bcf6b469f94ee
- Project Info: <exact direct-child Project Info URL>

GitHub 값은 target에서 파생하고 Purpose 값은 입력받은 `REPO_PURPOSE`와 정확히 같아야
한다. Project Info에는 앞에서 read back한 direct child URL만 기록한다.

### Reuse and readback

The hub must contain exactly one ## Repository metadata section with exactly one canonical GitHub, Purpose, Common guide, and Project Info bullet.
Repository metadata section count: 0 -> insert this section, 1 -> verify and upsert only missing canonical bullets, and 2+ -> fail.
Preserve all unrelated hub content on reuse.
Upsert only missing canonical metadata bullets.
Fail closed on conflicting or ambiguous existing metadata.
Duplicate canonical metadata labels or values conflicting with expected values fail.

section이 없으면 위 heading과 네 bullet을 삽입한다. 하나면 각 label을 exact match로
parse해 없는 bullet만 추가한다. 같은 label이 여러 번 나오거나 기대값과 다른 값이면
overwrite하지 않고 중단한다. 다른 section, block, child, 사람 문서는 삭제·재배열하지 않는다.

Readback parses only the exact ## Repository metadata section and verifies all four canonical bullets plus exact parent, title, hub page ID, and Project Info page ID before NOTION_REPO_URL is used.
readback은 exact section의 네 값, hub의 canonical parent와 repo title, hub/Project Info의
서로 다른 page ID, Project Info의 exact title과 direct-parent 관계를 모두 확인한다. 모두
통과한 뒤에만 선택한 hub URL을 `NOTION_REPO_URL`, child URL을
`NOTION_PROJECT_INFO_URL`로 사용한다.

Trust boundary: Notion MCP readback is authoritative for the repo hub parent/title and Project Info direct-child relationship.
The shell script does not read Notion; it validates only canonical URL form, distinct page IDs, common-page exclusion, and versioned placement.
따라서 MCP readback을 완료할 수 없으면 direct script 실행도 허용하지 않는다.

Only Notion-dependent work blocks when MCP is unavailable; normal code, CI, and runner work continues.
MCP가 없거나 create/reuse/readback이 실패하면 curl/browser scraping으로 우회하지 않고
이 bootstrap만 GitHub mutation 전에 중단한다.

## 3. Versioned repo hub URL

Only the repo hub URL is versioned in README, AGENTS, and runner context.
The Project Info URL must differ from the repo hub URL and is verified only as its direct child; never version it in those three files.
Write NOTION_REPO_URL into README.md, AGENTS.md, and ops/runner/repo-context.sh before invoking the script.
README 링크에는 검증된 repo hub URL만 기록한다.

세 파일의 `<NOTION_REPO_URL>` placeholder만 exact hub URL로 교체하고 read back한다.
세 값은 hub URL과 같아야 하며 Project Info URL과는 달라야 한다. 공통 URL은 보존한다.

## 4. Task 1 script

non-secret 입력과 enabled 기능의 조건부 값, 검증된 `NOTION_REPO_URL`과
`NOTION_PROJECT_INFO_URL`, 안전하게 주입된 secret이 모두 같은 process environment에
있을 때 repo root에서 실행한다. script는 두 URL이 다르고, 세 versioned field가 hub URL을
가리키며 Project Info URL은 기록되지 않았는지 다시 확인한다.

```bash
.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh skku-heven/<new-repo>
```

script는 입력과 provenance를 다시 fail-closed 검증하고 다음을 멱등 적용한다.

- `main` branch 생성, default `dev`, Wiki off, merge 후 branch 삭제
- squash option 활성화; 기존 merge/rebase 허용 여부는 변경하지 않음
- `agent-proposed`, `agent-stale`, `agent-keep` labels
- enabled 기능의 repo secrets/variables와 Discord/runner mode gate variables
- org Project #14 `[TEMPLATE] heven-common`의 identity와 Project #14 자체의 정확한 5개 Status와 필수 enabled non-Auto-add workflow 검증
- `<repo> Roadmap`이 없으면 복사하고 복사 직후 같은 schema를 read-back한 뒤 연결;
  정확히 하나면 검증 후 재사용

## 5. 완료 전 확인

1. `<repo> Roadmap`의 Auto-add workflow만 GitHub UI에서 설정한다.
   - filter: `is:issue is:open`
   - status: `Backlog`
   - 나머지 Status/workflow는 Project #14에서 복사된다.
2. exact `## Repository metadata` section의 네 canonical bullet과 parent/title/IDs를 다시 확인한다.
3. 세 versioned 파일에는 hub URL만 있고 Project Info URL은 없는지 확인한다.
4. README/CI placeholder를 실제 setup/build/run/test로 교체한다.
5. 의도한 팀원이 private repo와 Project 모두에 접근 가능한지 확인한다.
6. Runner completion verifies the Task 1 script wrote enabled repo secrets/variables without printing values, then deploys the runner.
   runner enabled이면 Task 1 script가 쓴 secret/variable 이름과 존재 여부를 값 출력 없이
   검증한 뒤 `ops/runner/README.md`에 따라 runner를 배포한다. credential을 다시 설정하지 않는다.
   org runner의 public repository access는 항상 차단한다.
   organization plan이 Selected repositories를 지원하면 대상 private repo만 허용한다.
7. 결과를 보고하되 secret과 environment-specific conditional 값은 출력하지 않는다.
