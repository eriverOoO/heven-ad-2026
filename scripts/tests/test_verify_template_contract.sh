#!/usr/bin/env bash

set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
VERIFIER="$ROOT/scripts/verify-template-contract.sh"
CANONICAL_NOTION_URL='https://app.notion.com/p/39c3bf068300807c9e5bcf6b469f94ee'
REPO_WIKI_URL='https://app.notion.com/p/39c3bf06830080b5a024c7ad91855240'
GETTING_STARTED_URL='https://app.notion.com/p/39c3bf06830081fb8e76d5c9a1be6d82'
TEAM_WORKFLOW_URL='https://app.notion.com/p/39c3bf068300810db8adc15fce65450d'
OPERATIONS_URL='https://app.notion.com/p/39c3bf068300815fb880e1d20602e662'
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
failures=0

fail() {
  printf 'not ok - %s\n' "$1" >&2
  failures=$((failures + 1))
}

make_fixture() {
  local name="$1"
  local fixture="$TMP_DIR/$name"

  mkdir -p \
    "$fixture/scripts" \
    "$fixture/.claude/skills/issue-worker/scripts" \
    "$fixture/.claude/skills/bootstrap-repo/scripts" \
    "$fixture/.github/workflows" \
    "$fixture/ops/runner" \
    "$fixture/tools"

  cp "$VERIFIER" "$fixture/scripts/verify-template-contract.sh"
  chmod +x "$fixture/scripts/verify-template-contract.sh"

  cat > "$fixture/README.md" <<EOF
Human guide: $CANONICAL_NOTION_URL
Repo wiki: $REPO_WIKI_URL
Repo-specific Notion hub: <NOTION_REPO_URL>
Getting Started: $GETTING_STARTED_URL
Team Workflow: $TEAM_WORKFLOW_URL
Operations: $OPERATIONS_URL

## 템플릿 구성

이 저장소는 HEVEN 팀의 공통 private template이며, 앞으로 만드는 팀 repository의 출발점이다.
새 repository의 코드 설정은 이 template에서, 전용 문서는 공통 repo wiki 아래에서 시작한다.
Notion은 일반 코드·CI·runner의 실행 의존성이 아니다. 다만 bootstrap은 repository 전용 Notion hub를 준비하고 확인한 뒤 GitHub 설정을 바꾼다.

이 repo의 \`bootstrap-repo\` skill을 사용해서 \`skku-heven/<new-repo>\`를 부트스트랩해줘.

직접 실행하기 전에는 Notion MCP로 repo hub와 그 아래의 \`Project Info\`를 확인해야 한다.
자세한 조건은 \`bootstrap-repo\` skill을 따른다.

cd <new-repo>
.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh skku-heven/<new-repo>

Project #14 \`[TEMPLATE] heven-common\` is copied or reused and linked automatically.
Only the Auto-add workflow remains: Auto-add workflow만 \`is:issue is:open\` → \`Backlog\`.
The squash option is enabled; existing merge/rebase 허용 여부는 변경하지 않는다.
EOF
  cat > "$fixture/AGENTS.md" <<EOF
Agent guide: $CANONICAL_NOTION_URL
Repo-specific Notion hub: <NOTION_REPO_URL>

## 정본

| 위치 | 내용 |
|---|---|
| 공통 Notion 가이드 | 공통 workflow·convention |
| repo별 Notion hub | domain·대회·hardware·환경·제약 |

- domain·대회·hardware·환경·제약에 관련된 작업은 Notion MCP로 repo별 hub를 먼저 읽는다. 공통 가이드만 보고 repo 제약을 추측하지 않는다.

## 작업 허가

\`main\`은 현재 stable/release-state branch이다.
After CI and human review, the maintainer chooses a repository에서 허용된 merge 방식.
merge 뒤 remote 작업 branch는 repository 설정에 따라 GitHub가 자동 삭제한다.
EOF
  cat > "$fixture/.claude/skills/bootstrap-repo/SKILL.md" <<EOF
Private bootstrap contract with safe pre-exported optional values. Operations: $OPERATIONS_URL
Canonical Notion parent: $REPO_WIKI_URL
Common Notion guide/template: $CANONICAL_NOTION_URL

## 1. One-shot bootstrap inputs

Collect target repo, REPO_PURPOSE, and all three modes in one request.
- Target repo: owner/repo
- Repository purpose: exact one-line REPO_PURPOSE
- Discord notifications mode: DISCORD_NOTIFICATIONS is enabled or disabled
- Discord mentions mode: DISCORD_MENTIONS is enabled or disabled
- Runner mode: RUNNER_AUTOMATION is enabled or disabled
Explicit opt-outs: ask for every mode; disabled must be selected explicitly and never inferred from silence.

### Enabled secret injection

Only DISCORD_WEBHOOK and PROJECT_PAT are plaintext secrets.
Request missing conditional values only through secure environment/secret input, never plaintext chat/docs.
Ask the user to inject only DISCORD_WEBHOOK and PROJECT_PAT through host secure env/secret input.
Verify each enabled secret is present without printing its value.
If host secure input is unavailable, stop before mutation.
Local fallback: provide a read -s same-shell wrapper that invokes bootstrap before unsetting secrets.
DISCORD_USER_MAP and RUNNER_SCRIPTS are conditional non-secret values; do not persist their values in durable docs.

\`\`\`bash
bootstrap_with_local_secrets() (
  trap 'unset DISCORD_WEBHOOK PROJECT_PAT' EXIT
)
\`\`\`

### Target provenance preflight

Target REST metadata must prove private visibility, default dev, admin access, and template_repository.full_name skku-heven/heven-common.
Provenance preflight completes before any Notion mutation.

\`\`\`bash
.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh --preflight-only skku-heven/<new-repo>
\`\`\`

## 2. Notion hub contract

For the repo hub exact direct-child title match: 0 -> create, 1 -> reuse, and 2+ -> fail.
Create or reuse Project Info from the common template and read it back.

## Repository metadata

- GitHub: https://github.com/<owner>/<repo>
- Purpose: <exact REPO_PURPOSE>
- Common guide: $CANONICAL_NOTION_URL
- Project Info: <exact direct-child Project Info URL>

### Reuse and readback

The hub must contain exactly one ## Repository metadata section with exactly one canonical GitHub, Purpose, Common guide, and Project Info bullet.
Repository metadata section count: 0 -> insert this section, 1 -> verify and upsert only missing canonical bullets, and 2+ -> fail.
Preserve all unrelated hub content on reuse.
Upsert only missing canonical metadata bullets.
Fail closed on conflicting or ambiguous existing metadata.
Duplicate canonical metadata labels or values conflicting with expected values fail.
Readback parses only the exact ## Repository metadata section and verifies all four canonical bullets plus exact parent, title, hub page ID, and Project Info page ID before NOTION_REPO_URL is used.
Trust boundary: Notion MCP readback is authoritative for the repo hub parent/title and Project Info direct-child relationship.
The shell script does not read Notion; it validates only canonical URL form, distinct page IDs, common-page exclusion, and versioned placement.
Only Notion-dependent work blocks when MCP is unavailable; normal code, CI, and runner work continues.

## 3. Versioned repo hub URL

Only the repo hub URL is versioned in README, AGENTS, and runner context.
The Project Info URL must differ from the repo hub URL and is verified only as its direct child; never version it in those three files.
Write NOTION_REPO_URL into README.md, AGENTS.md, and ops/runner/repo-context.sh before invoking the script.

## 4. Task 1 script

이 repo의 \`bootstrap-repo\` skill을 사용해서 \`skku-heven/<new-repo>\`를 부트스트랩해줘.
The full script requires distinct NOTION_REPO_URL and NOTION_PROJECT_INFO_URL values.

## 5. 완료 전 확인

Runner completion verifies the Task 1 script wrote enabled repo secrets/variables without printing values, then deploys the runner.
EOF
  cat > "$fixture/.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh" <<'EOF'
#!/usr/bin/env bash
# Trust boundary: Notion MCP readback verifies hub parent/title and Project Info direct-child. This shell does not read Notion; it validates only canonical URL form, distinct page IDs, common-page exclusion, and versioned placement.
# Optional secret values must be exported safely before this command runs.
printf 'bootstrap placeholder\n'
EOF
  cat > "$fixture/.claude/skills/issue-worker/SKILL.md" <<EOF
Issue workflow contract. Team Workflow: $TEAM_WORKFLOW_URL
이 repo의 \`issue-worker\` skill을 사용해서 이슈 #42를 잡아줘.
merge 뒤 remote 작업 branch는 repository 설정에 따라 GitHub가 자동 삭제한다.
EOF
  cat > "$fixture/.github/workflows/ci.yml" <<'EOF'
permissions:
  contents: read
jobs:
  build:
    steps:
      - uses: actions/checkout@v4
        with:
          persist-credentials: false
      - name: Install heven-common contract dependencies
        if: ${{ github.repository == 'skku-heven/heven-common' }}
        run: |
          sudo apt-get update
          sudo apt-get install -y ripgrep
      - name: heven-common contract CI
        run: |
          scripts/verify-template-contract.sh
          .claude/skills/bootstrap-repo/tests/test_bootstrap_repo.sh
          .claude/skills/issue-worker/tests/test_claim_issue.sh
          ops/runner/tests/test_env_precedence.sh
          ops/runner/tests/test_pr_set_in_review.sh
          scripts/tests/test_verify_template_contract.sh
EOF
  cat > "$fixture/.github/workflows/notify-discord.yml" <<'EOF'
if: vars.ENABLE_DISCORD_NOTIFICATIONS == 'true'
MENTIONS_ENABLED: vars.ENABLE_DISCORD_MENTIONS
EOF
  cat > "$fixture/ops/runner/README.md" <<'EOF'
Runner public repository access는 항상 차단한다.
Organization plan이 Selected repositories를 지원하면 대상 private repo만 허용한다.
ENABLE_RUNNER_AUTOMATION은 bundled workflow job만 gate하며 runner access 격리 경계가 아니다.
EOF

  cat > "$fixture/ops/runner/repo-context.sh" <<'EOF'
#!/usr/bin/env bash
export NOTION_GUIDE_URL="${NOTION_GUIDE_URL:-<NOTION_REPO_URL>}"
EOF

  cat > "$fixture/.claude/skills/issue-worker/scripts/claim_issue.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
die() {
  printf '%s\n' "$1" >&2
  exit 1
}
status="${1:-Ready}"
ready_error='Status must be Ready before claiming'
[[ "$status" == "Ready" ]] || die "$ready_error"
EOF

  cat > "$fixture/tools/direct-bash" <<'EOF'
#!/bin/bash
printf 'direct bash\n'
EOF

  cat > "$fixture/tools/env-sh" <<'EOF'
#!/usr/bin/env sh
printf 'env sh\n'
EOF

  chmod +x \
    "$fixture/.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh" \
    "$fixture/.claude/skills/issue-worker/scripts/claim_issue.sh" \
    "$fixture/tools/direct-bash" \
    "$fixture/tools/env-sh"

  git -C "$fixture" init -q
  git -C "$fixture" config user.name 'contract-test'
  git -C "$fixture" config user.email 'contract-test@example.invalid'
  git -C "$fixture" add .
  git -C "$fixture" commit -qm 'baseline fixture'
}

run_verifier() {
  local fixture="$1"
  local path_prefix="${2:-}"

  if [[ -n "$path_prefix" ]]; then
    (cd "$fixture" && PATH="$path_prefix:$PATH" scripts/verify-template-contract.sh)
  else
    (cd "$fixture" && scripts/verify-template-contract.sh)
  fi
}

expect_pass() {
  local label="$1" fixture="$2" path_prefix="${3:-}" output
  if ! output=$(run_verifier "$fixture" "$path_prefix" 2>&1); then
    printf '%s\n' "$output" >&2
    fail "$label should pass"
    return
  fi
  if grep -Fq 'No such file or directory' <<< "$output"; then
    printf '%s\n' "$output" >&2
    fail "$label emitted a missing-file warning"
    return
  fi
  printf 'ok - %s\n' "$label"
}

expect_fail() {
  local label="$1" fixture="$2" path_prefix="${3:-}" output
  if output=$(run_verifier "$fixture" "$path_prefix" 2>&1); then
    printf '%s\n' "$output" >&2
    fail "$label should fail"
    return
  fi
  printf 'ok - %s rejected\n' "$label"
}

expect_fail_with_output() {
  local label="$1" fixture="$2" path_prefix="$3" expected="$4" output
  if output=$(run_verifier "$fixture" "$path_prefix" 2>&1); then
    printf '%s\n' "$output" >&2
    fail "$label should fail"
    return
  fi
  if ! grep -Fq "$expected" <<< "$output"; then
    printf '%s\n' "$output" >&2
    fail "$label did not preserve scan error output"
    return
  fi
  printf 'ok - %s rejected with preserved output\n' "$label"
}

test_documented_secret_wrapper() {
  local sandbox="$TMP_DIR/documented-secret-wrapper"
  local extracted="$sandbox/wrapper.sh"
  local call_log="$sandbox/calls.log"
  local output="$sandbox/output.log"
  local old_pwd="$PWD"
  local failures_before="$failures"

  mkdir -p "$sandbox/.claude/skills/bootstrap-repo/scripts"
  if ! awk '
    $0 == "bootstrap_with_local_secrets() {" ||
    $0 == "bootstrap_with_local_secrets() (" { capture = 1 }
    capture { print }
    capture && ($0 == "}" || $0 == ")") { exit }
  ' "$ROOT/.claude/skills/bootstrap-repo/SKILL.md" > "$extracted"; then
    fail 'documented secret wrapper extraction failed'
    return
  fi
  if ! rg --quiet '^bootstrap_with_local_secrets\(\) [({]$' "$extracted"; then
    fail 'documented secret wrapper definition missing'
    return
  fi

  cat > "$sandbox/.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh" <<'EOF'
#!/usr/bin/env bash
set -eu
[[ -n "${DISCORD_WEBHOOK:-}" ]]
[[ -n "${PROJECT_PAT:-}" ]]
printf '%s\n' "$1" >> "$WRAPPER_CALL_LOG"
EOF
  chmod +x "$sandbox/.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh"

  cd "$sandbox"
  # shellcheck source=/dev/null
  source "$extracted"
  export DISCORD_NOTIFICATIONS=enabled
  export RUNNER_AUTOMATION=enabled
  export RUNNER_SCRIPTS="$sandbox/runner/bin"
  export WRAPPER_CALL_LOG="$call_log"

  unset DISCORD_WEBHOOK PROJECT_PAT
  if bootstrap_with_local_secrets owner/repo <<< 'discord-eof' > "$output" 2>&1; then
    fail 'documented secret wrapper accepted second-read EOF'
  fi
  if [[ -v DISCORD_WEBHOOK || -v PROJECT_PAT ]]; then
    fail 'documented secret wrapper leaked a secret after second-read EOF'
  fi
  if [[ -s "$call_log" ]]; then
    fail 'documented secret wrapper invoked bootstrap after second-read EOF'
  fi

  unset DISCORD_WEBHOOK PROJECT_PAT
  if bootstrap_with_local_secrets owner/repo <<< $'discord-empty\n' > "$output" 2>&1; then
    fail 'documented secret wrapper accepted an empty second secret'
  fi
  if [[ -v DISCORD_WEBHOOK || -v PROJECT_PAT ]]; then
    fail 'documented secret wrapper leaked a secret after empty second secret'
  fi
  if [[ -s "$call_log" ]]; then
    fail 'documented secret wrapper invoked bootstrap with an empty second secret'
  fi

  unset DISCORD_WEBHOOK PROJECT_PAT
  if ! bootstrap_with_local_secrets owner/repo <<< $'discord-success\nproject-success' \
      > "$output" 2>&1; then
    fail 'documented secret wrapper rejected valid secrets'
  fi
  if [[ -v DISCORD_WEBHOOK || -v PROJECT_PAT ]]; then
    fail 'documented secret wrapper leaked a secret after success'
  fi
  if [[ "$(wc -l < "$call_log")" -ne 1 ]] ||
      ! grep -Fxq 'owner/repo' "$call_log"; then
    fail 'documented secret wrapper did not invoke bootstrap exactly once on success'
  fi

  unset -f bootstrap_with_local_secrets
  unset DISCORD_NOTIFICATIONS RUNNER_AUTOMATION RUNNER_SCRIPTS WRAPPER_CALL_LOG
  unset DISCORD_WEBHOOK PROJECT_PAT
  cd "$old_pwd"

  if ((failures == failures_before)); then
    printf 'ok - documented secret wrapper cleanup behavior\n'
  fi
}

test_documented_secret_wrapper
if [[ "${TEMPLATE_CONTRACT_TEST_ONLY:-}" == documented-secret-wrapper ]]; then
  ((failures == 0)) || exit 1
  exit 0
fi

fixture="$TMP_DIR/baseline"
make_fixture baseline
expect_pass 'baseline contract' "$fixture"

fixture="$TMP_DIR/deleted-notion-raw-id"
make_fixture deleted-notion-raw-id
printf '%s\n' 'stale Notion page: 39b3''bf0683008007b0dbc58766947040' \
  > "$fixture/tools/stale-notion-id.txt"
git -C "$fixture" add tools/stale-notion-id.txt
expect_fail 'deleted Notion raw page ID remains in tracked content' "$fixture"

fixture="$TMP_DIR/ci-top-level-read-permission-missing"
make_fixture ci-top-level-read-permission-missing
sed -i '/^permissions:$/,/^  contents: read$/d' \
  "$fixture/.github/workflows/ci.yml"
expect_fail 'CI top-level contents read permission missing' "$fixture"

fixture="$TMP_DIR/ci-checkout-persist-credentials-missing"
make_fixture ci-checkout-persist-credentials-missing
sed -i '/persist-credentials: false/d' \
  "$fixture/.github/workflows/ci.yml"
expect_fail 'CI checkout credential persistence disabled missing' "$fixture"

fixture="$TMP_DIR/ci-ripgrep-dependency-missing"
make_fixture ci-ripgrep-dependency-missing
sed -i '/sudo apt-get install -y ripgrep/d' \
  "$fixture/.github/workflows/ci.yml"
expect_fail 'CI ripgrep contract dependency missing' "$fixture"

while IFS='|' read -r name command; do
  fixture="$TMP_DIR/$name"
  make_fixture "$name"
  sed -i "\#${command}#d" "$fixture/.github/workflows/ci.yml"
  expect_fail "$name" "$fixture"
done <<'EOF'
ci-current-contract-missing|scripts/verify-template-contract.sh
ci-bootstrap-contract-missing|.claude/skills/bootstrap-repo/tests/test_bootstrap_repo.sh
ci-issue-worker-contract-missing|.claude/skills/issue-worker/tests/test_claim_issue.sh
ci-runner-env-contract-missing|ops/runner/tests/test_env_precedence.sh
ci-project-status-contract-missing|ops/runner/tests/test_pr_set_in_review.sh
ci-verifier-regressions-missing|scripts/tests/test_verify_template_contract.sh
EOF

fixture="$TMP_DIR/discord-notification-gate-missing"
make_fixture discord-notification-gate-missing
sed -i '/ENABLE_DISCORD_NOTIFICATIONS/d' \
  "$fixture/.github/workflows/notify-discord.yml"
expect_fail 'Discord notification gate missing' "$fixture"

fixture="$TMP_DIR/discord-mention-gate-missing"
make_fixture discord-mention-gate-missing
sed -i '/ENABLE_DISCORD_MENTIONS/d' \
  "$fixture/.github/workflows/notify-discord.yml"
expect_fail 'Discord mention gate missing' "$fixture"

fixture="$TMP_DIR/source-ownership-duplicate-canonical-row"
make_fixture source-ownership-duplicate-canonical-row
sed -i '/| 공통 Notion 가이드 | 공통 workflow·convention |/a | 공통 Notion 가이드 | 공통 workflow·convention |' \
  "$fixture/AGENTS.md"
expect_fail 'duplicate canonical source ownership row' "$fixture"

fixture="$TMP_DIR/source-ownership-uppercase-contradictory-duplicate"
make_fixture source-ownership-uppercase-contradictory-duplicate
sed -i '/| 공통 Notion 가이드 | 공통 workflow·convention |/a | 공통 Notion 가이드 | Deprecated Competition specifications |' \
  "$fixture/AGENTS.md"
expect_fail 'canonical plus uppercase contradictory source ownership row' "$fixture"

fixture="$TMP_DIR/source-ownership-leading-cell-whitespace-duplicate"
make_fixture source-ownership-leading-cell-whitespace-duplicate
sed -i '/| 공통 Notion 가이드 | 공통 workflow·convention |/a |  공통 Notion 가이드 | Deprecated Competition specifications |' \
  "$fixture/AGENTS.md"
expect_fail 'source ownership duplicate with leading cell whitespace' "$fixture"

fixture="$TMP_DIR/source-ownership-trailing-cell-whitespace-duplicate"
make_fixture source-ownership-trailing-cell-whitespace-duplicate
sed -i '/| 공통 Notion 가이드 | 공통 workflow·convention |/a | 공통 Notion 가이드| Deprecated Competition specifications |' \
  "$fixture/AGENTS.md"
expect_fail 'source ownership duplicate without trailing cell whitespace' "$fixture"

fixture="$TMP_DIR/source-ownership-only-contradictory-row"
make_fixture source-ownership-only-contradictory-row
sed -i 's#| 공통 Notion 가이드 | 공통 workflow·convention |#| 공통 Notion 가이드 | Deprecated Competition specifications |#' \
  "$fixture/AGENTS.md"
expect_fail 'only contradictory source ownership row' "$fixture"

fixture="$TMP_DIR/source-ownership-repo-hub-spec-missing"
make_fixture source-ownership-repo-hub-spec-missing
sed -i 's/·대회//' "$fixture/AGENTS.md"
expect_fail 'repo-specific hub competition specification ownership missing' "$fixture"

fixture="$TMP_DIR/source-ownership-glossary-contradiction"
make_fixture source-ownership-glossary-contradiction
cat >> "$fixture/AGENTS.md" <<'EOF'

## Glossary

| 공통 Notion 가이드 | deprecated competition specifications and 대회 명세 |
EOF
expect_pass 'source ownership contradiction outside its section is ignored' "$fixture"

if [[ "${TEMPLATE_CONTRACT_TEST_ONLY:-}" == source-ownership ]]; then
  ((failures == 0)) || exit 1
  exit 0
fi

fixture="$TMP_DIR/readme-bootstrap-prompt-missing"
make_fixture readme-bootstrap-prompt-missing
sed -i '/bootstrap-repo.*skill을 사용해서/d' "$fixture/README.md"
expect_fail 'README bootstrap skill prompt missing' "$fixture"

fixture="$TMP_DIR/bootstrap-skill-prompt-missing"
make_fixture bootstrap-skill-prompt-missing
sed -i '/bootstrap-repo.*skill을 사용해서/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'bootstrap skill prompt missing' "$fixture"

fixture="$TMP_DIR/issue-skill-prompt-missing"
make_fixture issue-skill-prompt-missing
sed -i '/issue-worker.*skill을 사용해서/d' \
  "$fixture/.claude/skills/issue-worker/SKILL.md"
expect_fail 'issue-worker skill prompt missing' "$fixture"

fixture="$TMP_DIR/agents-automatic-branch-deletion-missing"
make_fixture agents-automatic-branch-deletion-missing
sed -i '/merge 뒤 remote 작업 branch는.*GitHub가 자동 삭제/d' \
  "$fixture/AGENTS.md"
expect_fail 'AGENTS automatic branch deletion missing' "$fixture"

fixture="$TMP_DIR/issue-skill-automatic-branch-deletion-missing"
make_fixture issue-skill-automatic-branch-deletion-missing
sed -i '/merge 뒤 remote 작업 branch는.*GitHub가 자동 삭제/d' \
  "$fixture/.claude/skills/issue-worker/SKILL.md"
expect_fail 'issue-worker automatic branch deletion missing' "$fixture"

fixture="$TMP_DIR/current-public-template-wording"
make_fixture current-public-template-wording
printf '현재 public template와 bootstrap 경로를 안내한다.\n' >> "$fixture/README.md"
expect_fail 'current public template wording' "$fixture"

fixture="$TMP_DIR/current-public-markdown-template-wording"
make_fixture current-public-markdown-template-wording
printf '현재 public **template**와 bootstrap 경로를 안내한다.\n' \
  >> "$fixture/README.md"
expect_fail 'current public Markdown template wording' "$fixture"

fixture="$TMP_DIR/manual-remote-branch-deletion"
make_fixture manual-remote-branch-deletion
printf 'gh pr merge 42 --merge --delete-branch\n' \
  >> "$fixture/.claude/skills/issue-worker/SKILL.md"
expect_fail 'manual remote branch deletion' "$fixture"

fixture="$TMP_DIR/standalone-delete-branch-option"
make_fixture standalone-delete-branch-option
printf '원격 작업 branch 정리에는 `--delete-branch` option을 사용한다.\n' \
  >> "$fixture/AGENTS.md"
expect_fail 'standalone delete-branch option' "$fixture"

fixture="$TMP_DIR/multiline-manual-remote-branch-deletion"
make_fixture multiline-manual-remote-branch-deletion
cat >> "$fixture/.claude/skills/issue-worker/SKILL.md" <<'EOF'
gh pr merge 42 --merge \
  --delete-branch
EOF
expect_fail 'multiline manual remote branch deletion' "$fixture"

fixture="$TMP_DIR/natural-language-manual-remote-branch-deletion"
make_fixture natural-language-manual-remote-branch-deletion
printf 'maintainer가 merge 후 remote 작업 branch를 직접 삭제한다.\n' \
  >> "$fixture/AGENTS.md"
expect_fail 'natural-language manual remote branch deletion' "$fixture"

fixture="$TMP_DIR/natural-language-implicit-remote-branch-deletion"
make_fixture natural-language-implicit-remote-branch-deletion
printf 'maintainer가 merge 후 remote 작업 branch를 삭제한다.\n' \
  >> "$fixture/AGENTS.md"
expect_fail 'implicit natural-language remote branch deletion' "$fixture"

fixture="$TMP_DIR/git-push-remote-branch-deletion"
make_fixture git-push-remote-branch-deletion
printf 'git push origin --delete feat/42-example\n' \
  >> "$fixture/.claude/skills/issue-worker/SKILL.md"
expect_fail 'git push remote branch deletion' "$fixture"

fixture="$TMP_DIR/negative-manual-remote-branch-deletion"
make_fixture negative-manual-remote-branch-deletion
printf '별도 remote branch 삭제 명령을 실행할 필요가 없습니다.\n' \
  >> "$fixture/AGENTS.md"
expect_pass 'negative manual remote branch deletion statement' "$fixture"

fixture="$TMP_DIR/negative-direct-manual-remote-branch-deletion"
make_fixture negative-direct-manual-remote-branch-deletion
printf 'maintainer가 merge 후 remote 작업 branch를 직접 삭제할 필요가 없습니다.\n' \
  >> "$fixture/AGENTS.md"
expect_pass 'negative direct manual remote branch deletion statement' "$fixture"

fixture="$TMP_DIR/negative-korean-remote-branch-deletion"
make_fixture negative-korean-remote-branch-deletion
printf 'maintainer는 merge 후 remote 작업 branch를 삭제하지 않는다.\n' \
  >> "$fixture/AGENTS.md"
expect_pass 'negative Korean remote branch deletion statement' "$fixture"

fixture="$TMP_DIR/negative-english-remote-branch-deletion"
make_fixture negative-english-remote-branch-deletion
printf 'The remote work branch should never be manually deleted.\n' \
  >> "$fixture/AGENTS.md"
expect_pass 'negative English remote branch deletion statement' "$fixture"

fixture="$TMP_DIR/agents-main-stable-state-missing"
make_fixture agents-main-stable-state-missing
sed -i '/main.*현재 stable\/release-state branch/d' "$fixture/AGENTS.md"
expect_fail 'AGENTS main stable release-state missing' "$fixture"

fixture="$TMP_DIR/readme-private-template-identity-missing"
make_fixture readme-private-template-identity-missing
sed -i '/^이 저장소는 HEVEN 팀의 공통 private template이며/d' "$fixture/README.md"
expect_fail 'README private-template identity missing' "$fixture"

fixture="$TMP_DIR/wrapped-positive-contracts"
make_fixture wrapped-positive-contracts
sed -i '/bootstrap-repo.*skill을 사용해서/d' "$fixture/README.md"
sed -i '/^cd <new-repo>$/i\
> 이 repo의 `bootstrap-repo` skill을 사용해서\
> `skku-heven/<new-repo>`를 부트스트랩해줘.\
' "$fixture/README.md"
sed -i '/bootstrap-repo.*skill을 사용해서/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
cat >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md" <<'EOF'
> 이 repo의 `bootstrap-repo` skill을 사용해서
> `skku-heven/<new-repo>`를 부트스트랩해줘.
EOF
sed -i '/issue-worker.*skill을 사용해서/d' \
  "$fixture/.claude/skills/issue-worker/SKILL.md"
cat >> "$fixture/.claude/skills/issue-worker/SKILL.md" <<'EOF'
> 이 repo의 `issue-worker` skill을 사용해서
> 이슈 #42를 잡아줘.
EOF
sed -i '/merge 뒤 remote 작업 branch는.*GitHub가 자동 삭제/d' \
  "$fixture/AGENTS.md" "$fixture/.claude/skills/issue-worker/SKILL.md"
cat >> "$fixture/AGENTS.md" <<'EOF'
merge 뒤 remote 작업 branch는 repository 설정에 따라
GitHub가 자동 삭제한다.
EOF
cat >> "$fixture/.claude/skills/issue-worker/SKILL.md" <<'EOF'
merge 뒤 remote 작업 branch는 repository 설정에 따라
GitHub가 자동 삭제한다.
EOF
expect_pass 'line-wrapped positive contracts' "$fixture"

fixture="$TMP_DIR/semantic-versioning-policy"
make_fixture semantic-versioning-policy
printf 'production 배포에는 Semantic Versioning을 적용한다.\n' \
  >> "$fixture/README.md"
expect_fail 'Semantic Versioning policy' "$fixture"

fixture="$TMP_DIR/semver-policy"
make_fixture semver-policy
printf '버전 번호는 SemVer를 따른다.\n' >> "$fixture/AGENTS.md"
expect_fail 'SemVer policy' "$fixture"

fixture="$TMP_DIR/github-release-policy"
make_fixture github-release-policy
printf '배포할 때 GitHub Release를 만든다.\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'GitHub Release policy' "$fixture"

fixture="$TMP_DIR/release-tag-policy"
make_fixture release-tag-policy
printf '각 배포에 release tag를 만든다.\n' \
  >> "$fixture/.claude/skills/issue-worker/SKILL.md"
expect_fail 'release tag policy' "$fixture"

fixture="$TMP_DIR/git-tag-guidance"
make_fixture git-tag-guidance
printf '배포 준비 후 `git tag v1.2.3`을 실행한다.\n' \
  >> "$fixture/ops/runner/README.md"
expect_fail 'git tag guidance' "$fixture"

fixture="$TMP_DIR/markdown-semantic-versioning-policy"
make_fixture markdown-semantic-versioning-policy
printf 'production 배포에는 Semantic **Versioning**을 적용한다.\n' \
  >> "$fixture/README.md"
expect_fail 'Markdown Semantic Versioning policy' "$fixture"

fixture="$TMP_DIR/markdown-github-release-policy"
make_fixture markdown-github-release-policy
printf '배포할 때 GitHub **Release**를 만든다.\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'Markdown GitHub Release policy' "$fixture"

fixture="$TMP_DIR/markdown-release-tag-policy"
make_fixture markdown-release-tag-policy
printf '각 배포에 release **tag**를 만든다.\n' \
  >> "$fixture/.claude/skills/issue-worker/SKILL.md"
expect_fail 'Markdown release tag policy' "$fixture"

fixture="$TMP_DIR/markdown-git-tag-guidance"
make_fixture markdown-git-tag-guidance
printf '배포 준비 후 git **tag** v1.2.3을 실행한다.\n' \
  >> "$fixture/ops/runner/README.md"
expect_fail 'Markdown git tag guidance' "$fixture"

fixture="$TMP_DIR/historical-plan-current-literal-exemption"
make_fixture historical-plan-current-literal-exemption
mkdir -p "$fixture/docs/superpowers/plans"
cat > "$fixture/docs/superpowers/plans/legacy-workflow.md" <<'EOF'
Historical proposal only: public **template**, Semantic Versioning, GitHub Release,
release tag, `git tag v1.2.3`, and git push origin --delete feat/42-example.
EOF
git -C "$fixture" add docs/superpowers/plans/legacy-workflow.md
expect_fail 'tracked docs/superpowers metadata' "$fixture"

fixture="$TMP_DIR/deleted-docs-superpowers-metadata"
make_fixture deleted-docs-superpowers-metadata
mkdir -p "$fixture/docs/superpowers/plans"
printf 'metadata scheduled for deletion\n' \
  > "$fixture/docs/superpowers/plans/legacy-workflow.md"
git -C "$fixture" add docs/superpowers/plans/legacy-workflow.md
git -C "$fixture" commit -qm 'add legacy workflow metadata'
rm "$fixture/docs/superpowers/plans/legacy-workflow.md"
expect_pass 'deleted docs/superpowers metadata is absent from the proposed tree' "$fixture"

fixture="$TMP_DIR/guard-message-only"
make_fixture guard-message-only
sed -i '/status.*Ready.*||/d' \
  "$fixture/.claude/skills/issue-worker/scripts/claim_issue.sh"
grep -Fq 'Status must be Ready before claiming' \
  "$fixture/.claude/skills/issue-worker/scripts/claim_issue.sh" \
  || fail 'message-only fixture lost unrelated Ready error text'
expect_fail 'Ready message without executable guard' "$fixture"

fixture="$TMP_DIR/guard-true-action"
make_fixture guard-true-action
sed -i 's/|| die .*/|| true/' \
  "$fixture/.claude/skills/issue-worker/scripts/claim_issue.sh"
expect_fail 'Ready comparison without failure action' "$fixture"

fixture="$TMP_DIR/missing-scope"
make_fixture missing-scope
rm -rf "$fixture/.github"
expect_fail 'missing scan scope' "$fixture"

fixture="$TMP_DIR/wiki-variable"
make_fixture wiki-variable
printf 'WIKI_CLONE_DIR=/tmp/legacy\n' >> "$fixture/ops/runner/README.md"
expect_fail 'arbitrary WIKI_* variable' "$fixture"

fixture="$TMP_DIR/raw-wiki-url"
make_fixture raw-wiki-url
printf 'https://RAW.GITHUBUSERCONTENT.COM/WIKI/acme/template/Home.md\n' \
  >> "$fixture/README.md"
expect_fail 'mixed-case raw Wiki URL' "$fixture"

fixture="$TMP_DIR/wiki-git-url"
make_fixture wiki-git-url
printf 'https://github.com/acme/template.WiKi.GiT\n' >> "$fixture/README.md"
expect_fail 'mixed-case .wiki.git URL' "$fixture"

fixture="$TMP_DIR/public-runner-spacing"
make_fixture public-runner-spacing
printf 'allows_public_repositories = true\n' >> "$fixture/README.md"
expect_fail 'spaced public-runner setting' "$fixture"

fixture="$TMP_DIR/public-runner-yaml"
make_fixture public-runner-yaml
printf 'allows_public_repositories: true\n' >> "$fixture/README.md"
expect_fail 'YAML public-runner setting' "$fixture"

fixture="$TMP_DIR/public-runner-single-quoted-yaml"
make_fixture public-runner-single-quoted-yaml
printf "'allows_public_repositories': true\n" >> "$fixture/README.md"
expect_fail 'single-quoted YAML public-runner setting' "$fixture"

fixture="$TMP_DIR/public-runner-json"
make_fixture public-runner-json
printf '"allows_public_repositories": true\n' >> "$fixture/README.md"
expect_fail 'JSON public-runner setting' "$fixture"

fixture="$TMP_DIR/github-wiki-root"
make_fixture github-wiki-root
printf '[legacy](https://GitHub.com/acme/template/WiKi)\n' >> "$fixture/README.md"
expect_fail 'GitHub Wiki root without trailing slash' "$fixture"

fixture="$TMP_DIR/mixed-wiki-operations"
make_fixture mixed-wiki-operations
printf 'WiKi OpErAtIoNs\n' >> "$fixture/README.md"
expect_fail 'mixed-case Wiki Operations vocabulary' "$fixture"

fixture="$TMP_DIR/ruleset-api"
make_fixture ruleset-api
printf 'gh api "RePoS/acme/template/RuLeSeTs"\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'mixed-case ruleset API' "$fixture"

fixture="$TMP_DIR/evil-notion"
make_fixture evil-notion
sed -i 's#https://app\.notion\.com#https://evilapp.notion.com#' \
  "$fixture/README.md"
expect_fail 'evil Notion subdomain in README' "$fixture"

fixture="$TMP_DIR/one-doc-missing-notion"
make_fixture one-doc-missing-notion
sed -i "s#${CANONICAL_NOTION_URL}#Notion guide unavailable#" \
  "$fixture/AGENTS.md"
expect_fail 'canonical Notion URL missing from one document' "$fixture"

fixture="$TMP_DIR/readme-missing-repo-wiki-hub"
make_fixture readme-missing-repo-wiki-hub
sed -i "s#${REPO_WIKI_URL}#repo wiki unavailable#" "$fixture/README.md"
expect_fail 'repo wiki hub missing from README' "$fixture"

fixture="$TMP_DIR/bootstrap-skill-canonical-parent-missing"
make_fixture bootstrap-skill-canonical-parent-missing
sed -i "s#${REPO_WIKI_URL}#canonical parent unavailable#" \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'canonical Notion parent missing from bootstrap skill' "$fixture"

fixture="$TMP_DIR/bootstrap-skill-repo-hub-create-reuse-missing"
make_fixture bootstrap-skill-repo-hub-create-reuse-missing
sed -i '/exact direct-child title match/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'repo-hub exact create or reuse contract missing' "$fixture"

fixture="$TMP_DIR/bootstrap-skill-secure-missing-input-request-missing"
make_fixture bootstrap-skill-secure-missing-input-request-missing
sed -i '/secure environment\/secret input/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'secure missing-input request contract missing' "$fixture"

fixture="$TMP_DIR/agents-llm-notion-discovery-missing"
make_fixture agents-llm-notion-discovery-missing
sed -i '/^- domain·대회·hardware·환경·제약에 관련된 작업은 Notion MCP로 repo별 hub를 먼저 읽는다\./d' \
  "$fixture/AGENTS.md"
expect_fail 'LLM Notion discovery rule missing from AGENTS' "$fixture"

fixture="$TMP_DIR/agents-llm-notion-discovery-glossary-decoy"
make_fixture agents-llm-notion-discovery-glossary-decoy
sed -i '/^- domain·대회·hardware·환경·제약에 관련된 작업은 Notion MCP로 repo별 hub를 먼저 읽는다\./d' \
  "$fixture/AGENTS.md"
cat >> "$fixture/AGENTS.md" <<'EOF'

## Glossary

- domain·대회·hardware·환경·제약에 관련된 작업은 Notion MCP로 repo별 hub를 먼저 읽는다. 공통 가이드만 보고 repo 제약을 추측하지 않는다.
EOF
expect_fail 'LLM Notion discovery glossary decoy' "$fixture"

fixture="$TMP_DIR/bootstrap-one-shot-inputs-missing"
make_fixture bootstrap-one-shot-inputs-missing
sed -i '/^Collect target repo, REPO_PURPOSE, and all three modes in one request\.$/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'one-shot bootstrap inputs missing' "$fixture"

fixture="$TMP_DIR/bootstrap-explicit-opt-outs-missing"
make_fixture bootstrap-explicit-opt-outs-missing
sed -i '/^Explicit opt-outs:/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'explicit mode opt-outs missing' "$fixture"

fixture="$TMP_DIR/bootstrap-provenance-heading-after-notion"
make_fixture bootstrap-provenance-heading-after-notion
sed -i '/^### Target provenance preflight$/,/^Provenance preflight completes before any Notion mutation\.$/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
sed -i '/^## Repository metadata$/i\
### Target provenance preflight\
\
Target REST metadata must prove private visibility, default dev, admin access, and template_repository.full_name skku-heven/heven-common.\
Provenance preflight completes before any Notion mutation.\
' "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'provenance heading moved after Notion mutation' "$fixture"

fixture="$TMP_DIR/bootstrap-preflight-command-missing"
make_fixture bootstrap-preflight-command-missing
sed -i '/bootstrap-repo\.sh --preflight-only/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'executable provenance preflight missing' "$fixture"

fixture="$TMP_DIR/bootstrap-preflight-command-after-notion"
make_fixture bootstrap-preflight-command-after-notion
sed -i '/bootstrap-repo\.sh --preflight-only/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
sed -i '/^## Repository metadata$/i\
.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh --preflight-only skku-heven/<new-repo>\
' "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'executable provenance preflight moved after Notion mutation' "$fixture"

fixture="$TMP_DIR/bootstrap-canonical-common-url-missing"
make_fixture bootstrap-canonical-common-url-missing
sed -i "s#${CANONICAL_NOTION_URL}#common guide unavailable#g" \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'canonical common URL missing from bootstrap skill' "$fixture"

while IFS='|' read -r name fragment; do
  fixture="$TMP_DIR/$name"
  make_fixture "$name"
  sed -i "\\|${fragment}|d" "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
  expect_fail "$name" "$fixture"
done <<'EOF'
hub-metadata-github-url-missing|^- GitHub:
hub-metadata-purpose-missing|^- Purpose:
hub-metadata-common-guide-missing|^- Common guide:
hub-metadata-project-info-missing|^- Project Info:
EOF

fixture="$TMP_DIR/hub-metadata-glossary-decoy"
make_fixture hub-metadata-glossary-decoy
sed -i '/^- GitHub:/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
cat >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md" <<'EOF'

## Glossary

- GitHub: https://github.com/<owner>/<repo>
EOF
expect_fail 'hub metadata field only outside exact section' "$fixture"

fixture="$TMP_DIR/hub-metadata-duplicate-heading"
make_fixture hub-metadata-duplicate-heading
cat >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md" <<EOF

## Repository metadata

- GitHub: https://github.com/<owner>/<repo>
- Purpose: <exact REPO_PURPOSE>
- Common guide: $CANONICAL_NOTION_URL
- Project Info: <exact direct-child Project Info URL>
EOF
expect_fail 'duplicate Repository metadata heading' "$fixture"

fixture="$TMP_DIR/hub-metadata-duplicate-field"
make_fixture hub-metadata-duplicate-field
sed -i '/^- GitHub:/a - GitHub: https://github.com/<owner>/<repo>' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'duplicate canonical metadata field' "$fixture"

fixture="$TMP_DIR/hub-metadata-conflicting-field"
make_fixture hub-metadata-conflicting-field
sed -i 's#^- GitHub: https://github.com/<owner>/<repo>$#- GitHub: https://github.com/wrong/repo#' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'conflicting canonical metadata field' "$fixture"

while IFS='|' read -r name fragment; do
  fixture="$TMP_DIR/$name"
  make_fixture "$name"
  sed -i "\\|${fragment}|d" "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
  expect_fail "$name" "$fixture"
done <<'EOF'
hub-metadata-one-section-contract-missing|The hub must contain exactly one ## Repository metadata section
hub-metadata-count-transition-missing|Repository metadata section count: 0 -> insert this section
hub-metadata-duplicate-conflict-rule-missing|Duplicate canonical metadata labels or values conflicting with expected values fail
hub-metadata-exact-readback-missing|Readback parses only the exact ## Repository metadata section
EOF

while IFS='|' read -r name fragment; do
  fixture="$TMP_DIR/$name"
  make_fixture "$name"
  sed -i "\\#${fragment}#d" "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
  expect_fail "$name" "$fixture"
done <<'EOF'
hub-reuse-preservation-missing|Preserve all unrelated hub content on reuse
hub-missing-field-upsert-missing|Upsert only missing canonical metadata bullets
hub-conflict-fail-closed-missing|Fail closed on conflicting or ambiguous existing metadata
EOF

fixture="$TMP_DIR/bootstrap-notion-only-blocking-rule-missing"
make_fixture bootstrap-notion-only-blocking-rule-missing
sed -i '/^Only Notion-dependent work blocks/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'Notion-only blocking rule missing from bootstrap skill' "$fixture"

fixture="$TMP_DIR/bootstrap-notion-mcp-trust-boundary-missing"
make_fixture bootstrap-notion-mcp-trust-boundary-missing
sed -i '/^Trust boundary: Notion MCP readback is authoritative/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'Notion MCP trust boundary missing from bootstrap skill' "$fixture"

fixture="$TMP_DIR/bootstrap-shell-notion-scope-missing"
make_fixture bootstrap-shell-notion-scope-missing
sed -i '/^The shell script does not read Notion;/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'bootstrap shell Notion validation scope missing from skill' "$fixture"

fixture="$TMP_DIR/bootstrap-script-notion-trust-comment-missing"
make_fixture bootstrap-script-notion-trust-comment-missing
sed -i '/^# Trust boundary: Notion MCP readback verifies hub parent\/title and Project Info direct-child\./d' \
  "$fixture/.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh"
expect_fail 'bootstrap script Notion trust boundary comment missing' "$fixture"

fixture="$TMP_DIR/readme-bootstrap-notion-boundary-missing"
make_fixture readme-bootstrap-notion-boundary-missing
sed -i '/^Notion은 일반 코드·CI·runner의 실행 의존성이 아니다\./d' \
  "$fixture/README.md"
expect_fail 'README bootstrap Notion boundary missing' "$fixture"

fixture="$TMP_DIR/readme-direct-bootstrap-notion-boundary-missing"
make_fixture readme-direct-bootstrap-notion-boundary-missing
sed -i '/^직접 실행하기 전에는 Notion MCP로 repo hub와/,/^자세한 조건은 `bootstrap-repo` skill을 따른다\.$/d' \
  "$fixture/README.md"
expect_fail 'README direct bootstrap Notion trust boundary missing' "$fixture"

while IFS='|' read -r name relative_file; do
  fixture="$TMP_DIR/$name"
  make_fixture "$name"
  sed -i 's/<NOTION_REPO_URL>/<MISSING_NOTION_REPO_URL>/' \
    "$fixture/$relative_file"
  expect_fail "$name" "$fixture"
done <<'EOF'
readme-repo-hub-placeholder-missing|README.md
agents-repo-hub-placeholder-missing|AGENTS.md
runner-repo-hub-placeholder-missing|ops/runner/repo-context.sh
EOF

fixture="$TMP_DIR/readme-template-origin-glossary-decoy"
make_fixture readme-template-origin-glossary-decoy
sed -i '/^이 저장소는 HEVEN 팀의 공통 private template이며/d' \
  "$fixture/README.md"
cat >> "$fixture/README.md" <<'EOF'

## Glossary

이 저장소는 HEVEN 팀의 공통 private template이며, 앞으로 만드는 팀 repository의 출발점이다.
EOF
expect_fail 'template identity appears only in glossary decoy' "$fixture"

for fixed_repo in heven-ad-2026 heven-jj-2026; do
  fixture="$TMP_DIR/readme-fixed-$fixed_repo"
  make_fixture "readme-fixed-$fixed_repo"
  printf 'This template is fixed to %s.\n' "$fixed_repo" >> "$fixture/README.md"
  expect_fail "README fixed $fixed_repo wording" "$fixture"
done

while IFS='|' read -r name fragment; do
  fixture="$TMP_DIR/$name"
  make_fixture "$name"
  sed -i "\\#${fragment}#d" "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
  expect_fail "$name" "$fixture"
done <<'EOF'
secure-secret-classification-missing|Only DISCORD_WEBHOOK and PROJECT_PAT are plaintext secrets
secure-host-injection-missing|Ask the user to inject only DISCORD_WEBHOOK and PROJECT_PAT
secure-presence-without-printing-missing|Verify each enabled secret is present without printing its value
secure-unavailable-stop-missing|If host secure input is unavailable, stop before mutation
secure-read-s-fallback-missing|Local fallback: provide a read -s same-shell wrapper
conditional-nonsecret-classification-missing|DISCORD_USER_MAP and RUNNER_SCRIPTS are conditional non-secret values
EOF

fixture="$TMP_DIR/secure-wrapper-subshell-missing"
make_fixture secure-wrapper-subshell-missing
sed -i 's/^bootstrap_with_local_secrets() ($/bootstrap_with_local_secrets() {/' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'secret wrapper subshell scope missing' "$fixture"

fixture="$TMP_DIR/secure-wrapper-exit-cleanup-missing"
make_fixture secure-wrapper-exit-cleanup-missing
sed -i "/trap 'unset DISCORD_WEBHOOK PROJECT_PAT' EXIT/d" \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'secret wrapper EXIT cleanup missing' "$fixture"

fixture="$TMP_DIR/runner-completion-verification-missing"
make_fixture runner-completion-verification-missing
sed -i '/^Runner completion verifies the Task 1 script wrote enabled repo secrets\/variables/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'runner completion verification missing' "$fixture"

fixture="$TMP_DIR/runner-completion-second-mutation-path"
make_fixture runner-completion-second-mutation-path
printf 'runner enabled이면 repo variables/secrets를 설정하고 runner를 배포한다.\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'runner completion second credential mutation path' "$fixture"

fixture="$TMP_DIR/repo-hub-url-ownership-missing"
make_fixture repo-hub-url-ownership-missing
sed -i '/^Only the repo hub URL is versioned/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'repo hub URL ownership contract missing' "$fixture"

fixture="$TMP_DIR/project-info-url-child-only-missing"
make_fixture project-info-url-child-only-missing
sed -i '/^The Project Info URL must differ/d' \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'Project Info child-only URL contract missing' "$fixture"

fixture="$TMP_DIR/readme-project-info-url-conflation"
make_fixture readme-project-info-url-conflation
printf 'Project Info URL is versioned in README, AGENTS, and runner context.\n' \
  >> "$fixture/README.md"
expect_fail 'Project Info URL conflated with repo hub URL' "$fixture"

while IFS='|' read -r name url; do
  fixture="$TMP_DIR/$name"
  make_fixture "$name"
  sed -i "s#${url}#missing direct guide link#" "$fixture/README.md"
  expect_fail "$name" "$fixture"
done <<EOF
missing-getting-started|$GETTING_STARTED_URL
missing-team-workflow|$TEAM_WORKFLOW_URL
missing-operations|$OPERATIONS_URL
EOF

fixture="$TMP_DIR/marker-provenance-contradiction"
make_fixture marker-provenance-contradiction
printf 'template marker 파일도 provenance로 반드시 확인한다.\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'marker provenance contradicts exact GitHub metadata' "$fixture"

fixture="$TMP_DIR/multiline-marker-provenance-contradiction"
make_fixture multiline-marker-provenance-contradiction
cat >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md" <<'EOF'
dev에 template marker가
그대로 있어야 한다. metadata와 marker를 mutation 전에 확인한다.
EOF
expect_fail 'multiline marker provenance contradicts exact GitHub metadata' "$fixture"

fixture="$TMP_DIR/project-manual-division"
make_fixture project-manual-division
sed -i 's/Auto-add workflow만/Auto-add workflow/' "$fixture/README.md"
expect_fail 'Auto-add is not the only manual Project workflow' "$fixture"

fixture="$TMP_DIR/manual-project-contradiction"
make_fixture manual-project-contradiction
printf 'ProjectV2를 수동 생성하고 PR linked to issue workflow도 수동 설정한다.\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'manual Project creation contradicts bootstrap copy' "$fixture"

fixture="$TMP_DIR/non-autoadd-workflow-setup"
make_fixture non-autoadd-workflow-setup
printf 'PR linked to issue workflow를 설정한다.\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'non-Auto-add workflow setup contradicts copied workflows' "$fixture"

fixture="$TMP_DIR/full-name-non-autoadd-workflow-setup"
make_fixture full-name-non-autoadd-workflow-setup
printf 'Pull request linked to issue workflow를 수동 설정한다.\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'full-name non-Auto-add workflow setup contradicts copied workflows' "$fixture"

fixture="$TMP_DIR/full-name-merged-workflow-setup"
make_fixture full-name-merged-workflow-setup
printf 'Pull request merged workflow를 활성화한다.\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'full-name merged workflow setup contradicts copied workflows' "$fixture"

fixture="$TMP_DIR/copied-non-autoadd-workflow-description"
make_fixture copied-non-autoadd-workflow-description
printf 'Project #14에서 복사된 PR linked to issue workflow 설정을 그대로 사용한다.\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_pass 'copied non-Auto-add workflow description' "$fixture"

fixture="$TMP_DIR/copied-full-name-non-autoadd-workflow-description"
make_fixture copied-full-name-non-autoadd-workflow-description
printf 'Project #14에서 복사된 Pull request linked to issue workflow를 그대로 사용한다.\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_pass 'copied full-name non-Auto-add workflow description' "$fixture"

fixture="$TMP_DIR/merge-method-enforcement"
make_fixture merge-method-enforcement
sed -i 's/existing merge\/rebase 허용 여부는 변경하지 않는다/squash merge만 허용한다/' \
  "$fixture/README.md"
expect_fail 'merge methods are forced' "$fixture"

fixture="$TMP_DIR/squash-mandate"
make_fixture squash-mandate
printf 'PR은 dev로 squash merge합니다.\n' >> "$fixture/AGENTS.md"
expect_fail 'squash mandate in agent contract' "$fixture"

fixture="$TMP_DIR/optional-squash-wording"
make_fixture optional-squash-wording
printf 'squash merge is optional; maintainer may choose another allowed method.\n' \
  >> "$fixture/AGENTS.md"
expect_pass 'optional squash wording' "$fixture"

fixture="$TMP_DIR/readme-direct-script-missing"
make_fixture readme-direct-script-missing
sed -i '/bootstrap-repo\.sh skku-heven\/<new-repo>/d' "$fixture/README.md"
expect_fail 'README direct bootstrap script missing' "$fixture"

fixture="$TMP_DIR/readme-cd-missing"
make_fixture readme-cd-missing
sed -i '/cd <new-repo>/d' "$fixture/README.md"
expect_fail 'README cd step missing' "$fixture"

fixture="$TMP_DIR/readme-direct-script-before-skill-prompt"
make_fixture readme-direct-script-before-skill-prompt
sed -i '/bootstrap-repo\.sh skku-heven\/<new-repo>/d' "$fixture/README.md"
sed -i '/bootstrap-repo.*skill을 사용해서/i\\`.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh skku-heven/<new-repo>\\`' \
  "$fixture/README.md"
expect_fail 'README direct script appears before bootstrap skill prompt' "$fixture"

fixture="$TMP_DIR/detailed-lifecycle-in-readme"
make_fixture detailed-lifecycle-in-readme
printf '\n## 작업 lifecycle\nDetailed human workflow.\n' >> "$fixture/README.md"
expect_fail 'detailed lifecycle remains in bootstrap README' "$fixture"

fixture="$TMP_DIR/inline-bootstrap-secret"
make_fixture inline-bootstrap-secret
printf 'PROJECT_PAT=<token> bootstrap-repo.sh owner/repo\n' >> "$fixture/README.md"
expect_fail 'inline bootstrap secret example' "$fixture"

fixture="$TMP_DIR/skill-squash-mandate"
make_fixture skill-squash-mandate
printf '작업 PR은 dev로 squash merge한다.\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'squash mandate in skill documentation' "$fixture"

fixture="$TMP_DIR/skill-inline-secret"
make_fixture skill-inline-secret
printf 'PROJECT_PAT=<token> bootstrap-repo.sh owner/repo\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'inline secret example in skill documentation' "$fixture"

fixture="$TMP_DIR/realistic-inline-pat"
make_fixture realistic-inline-pat
printf 'PROJECT_PAT=ghp_example_token bootstrap-repo.sh owner/repo\n' \
  >> "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'realistic inline PAT assignment' "$fixture"

fixture="$TMP_DIR/realistic-inline-webhook"
make_fixture realistic-inline-webhook
printf 'DISCORD_WEBHOOK=https://discord.example/hooks/test command\n' \
  >> "$fixture/ops/runner/README.md"
expect_fail 'realistic inline webhook assignment' "$fixture"

fixture="$TMP_DIR/bootstrap-script-header-inline-secret"
make_fixture bootstrap-script-header-inline-secret
sed -i '2i# PROJECT_PAT=ghp_example_token bootstrap-repo.sh owner/repo' \
  "$fixture/.claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh"
expect_fail 'inline PAT assignment in bootstrap script header' "$fixture"

fixture="$TMP_DIR/bootstrap-skill-direct-link"
make_fixture bootstrap-skill-direct-link
sed -i "s#${OPERATIONS_URL}#missing operations link#" \
  "$fixture/.claude/skills/bootstrap-repo/SKILL.md"
expect_fail 'bootstrap skill direct Operations link missing' "$fixture"

fixture="$TMP_DIR/issue-skill-direct-link"
make_fixture issue-skill-direct-link
sed -i "s#${TEAM_WORKFLOW_URL}#missing team workflow link#" \
  "$fixture/.claude/skills/issue-worker/SKILL.md"
expect_fail 'issue-worker skill direct Team Workflow link missing' "$fixture"

fixture="$TMP_DIR/runner-public-boundary-missing"
make_fixture runner-public-boundary-missing
sed -i '/public repository access/d' "$fixture/ops/runner/README.md"
expect_fail 'runner public-access boundary missing' "$fixture"

fixture="$TMP_DIR/runner-plan-fallback-missing"
make_fixture runner-plan-fallback-missing
sed -i '/Selected repositories/d' "$fixture/ops/runner/README.md"
expect_fail 'runner selected-repository capability is unconditional' "$fixture"

fixture="$TMP_DIR/runner-job-gate-confusion"
make_fixture runner-job-gate-confusion
sed -i '/bundled workflow job/d' "$fixture/ops/runner/README.md"
expect_fail 'workflow opt-in confused with runner access isolation' "$fixture"

while IFS='|' read -r name suffix; do
  fixture="$TMP_DIR/$name"
  make_fixture "$name"
  sed -i \
    "s|${CANONICAL_NOTION_URL}|${CANONICAL_NOTION_URL}${suffix}|" \
    "$fixture/README.md"
  expect_fail "canonical Notion URL suffix $suffix" "$fixture"
done <<'EOF'
notion-slash|/child
notion-query|?view=1
notion-fragment|#section
notion-dot|.html
notion-percent|%2Fchild
EOF

fixture="$TMP_DIR/tracked-env-sh-error"
make_fixture tracked-env-sh-error
cat > "$fixture/tools/bad-env-sh" <<'EOF'
#!/usr/bin/env sh
if then
EOF
chmod +x "$fixture/tools/bad-env-sh"
git -C "$fixture" add tools/bad-env-sh
expect_fail 'tracked env sh syntax error' "$fixture"

fixture="$TMP_DIR/tracked-direct-bash-error"
make_fixture tracked-direct-bash-error
cat > "$fixture/tools/bad-direct-bash" <<'EOF'
#!/bin/bash
case value in
EOF
chmod +x "$fixture/tools/bad-direct-bash"
git -C "$fixture" add tools/bad-direct-bash
expect_fail 'tracked direct bash syntax error' "$fixture"

fixture="$TMP_DIR/untracked-shell-error"
make_fixture untracked-shell-error
cat > "$fixture/tools/untracked-bad.sh" <<'EOF'
#!/usr/bin/env bash
if then
EOF
expect_pass 'untracked bad shell is ignored' "$fixture"

fixture="$TMP_DIR/untracked-forbidden-content"
make_fixture untracked-forbidden-content
printf 'WIKI_CLONE_DIR=/tmp/untracked-legacy\n' \
  > "$fixture/ops/runner/scratch.md"
expect_pass 'untracked forbidden content is ignored' "$fixture"

fixture="$TMP_DIR/git-producer-error"
make_fixture git-producer-error
mkdir -p "$fixture/fake-bin"
cat > "$fixture/fake-bin/git" <<'EOF'
#!/usr/bin/env bash
exit 73
EOF
chmod +x "$fixture/fake-bin/git"
expect_fail 'git ls-files producer error' "$fixture" "$fixture/fake-bin"

fixture="$TMP_DIR/rg-scan-error"
make_fixture rg-scan-error
mkdir -p "$fixture/fake-bin"
cat > "$fixture/fake-bin/rg" <<'EOF'
#!/usr/bin/env bash
printf 'isolated fake rg scan error\n' >&2
exit 2
EOF
chmod +x "$fixture/fake-bin/rg"
expect_fail_with_output 'isolated rg exit 2' "$fixture" "$fixture/fake-bin" \
  'isolated fake rg scan error'

grep -Fq '| GitHub Issue / Project / PR / Milestone |' "$ROOT/AGENTS.md" \
  || fail 'AGENTS ownership row does not assign Milestone to GitHub'

if ((failures > 0)); then
  printf 'FAILED: %d template contract regression expectation(s)\n' \
    "$failures" >&2
  exit 1
fi

printf 'ok - all template contract verifier regressions\n'
