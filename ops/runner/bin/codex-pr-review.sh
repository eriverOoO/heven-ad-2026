#!/usr/bin/env bash
# codex PR review runner (1-phase). Called from GitHub Actions self-hosted runner.
#
# 리뷰는 advisory(사람이 최종 결정)라 1-phase로 충분하다. codex 한 번 호출로
# 내부 분석 → review comment 를 만들어 PR 에 게시한다. (planner 는 2-phase 유지)
#
# Usage:
#   codex-pr-review.sh <PR_NUMBER>
#
# 설정 (env 파일 또는 Actions 환경):
#   AGENT_WS   runner 워크스페이스 (기본 $HOME/heven_common_test_ws)
#   GH_REPO    owner/repo — Actions에서는 GITHUB_REPOSITORY로 자동

set -Eeuo pipefail

# Standard paths first, then nvm so nvm's node stays in front of /usr/bin.
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

PR_NUMBER="${1:?PR number required}"

# env 파일 먼저 로드 (파생 경로 계산 전에 — AGENT_WS 등이 반영되도록).
# 새 경로 우선, 구 경로(testbed 시절) fallback.
for _f in "${AGENT_ENV_FILE:-}" \
          "$HOME/.config/heven-agent/heven-agent.env" \
          "$HOME/.config/heven-common-test-agent/heven-common-test.env"; do
  if [[ -n "$_f" && -f "$_f" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$_f"
    set +a
    break
  fi
done

: "${AGENT_WS:=${HEVEN_COMMON_TEST_WS:-$HOME/heven_common_test_ws}}"
PROMPTS_DIR="$AGENT_WS/runner/prompts"
CONTEXT_FILE="$AGENT_WS/runner/repo-context.sh"
RENDER_SH="$AGENT_WS/runner/bin/render_prompt.sh"

# repo: Actions 컨텍스트가 정본 (org 공용 runner의 env 파일이 다른 repo를
# 가리켜도 오발사하지 않도록 GITHUB_REPOSITORY 우선). 하드코딩 fallback 없음.
if [[ -n "${GITHUB_REPOSITORY:-}" ]]; then GH_REPO="$GITHUB_REPOSITORY"; fi
: "${GH_REPO:?GH_REPO 필요 — env 파일 또는 GITHUB_REPOSITORY}"

# checkout: Actions workspace가 있으면 env 파일의 stale checkout보다 항상 우선한다.
# Actions 밖에서만 env REPO_CHECKOUT을 쓰고, 없으면 machine workspace로 fallback한다.
if [[ -n "${GITHUB_WORKSPACE:-}" ]]; then
  REPO_CHECKOUT="$GITHUB_WORKSPACE"
else
  : "${REPO_CHECKOUT:=$AGENT_WS/src/${GH_REPO#*/}}"
fi

LOG_DIR="$AGENT_WS/runner/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pr-review.log"

log_msg() {
  printf '[%s] [pr-%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$PR_NUMBER" "$1" \
    | tee -a "$LOG_FILE"
}

log_msg "=== PR review starting (1-phase, repo=$GH_REPO) ==="

TMP_DIR=$(mktemp -d)
REVIEW_FILE="$TMP_DIR/review.md"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

cd "$REPO_CHECKOUT"

# === Single-phase: 분석 → review comment (codex 1회 호출) ===
log_msg "rendering prompt"
PR_NUMBER="$PR_NUMBER" "$RENDER_SH" \
  "$PROMPTS_DIR/pr-review-tpl.md" "$CONTEXT_FILE" > "$TMP_DIR/prompt.md"

log_msg "codex exec starting"
codex exec --sandbox read-only \
  -C "$REPO_CHECKOUT" \
  --output-last-message "$REVIEW_FILE" \
  "$(cat "$TMP_DIR/prompt.md")" >> "$LOG_FILE" 2>&1

if [[ ! -s "$REVIEW_FILE" ]]; then
  log_msg "empty output, abort"
  exit 1
fi
log_msg "review generated"

# === Post comment ===
log_msg "posting review comment to PR"
gh pr comment "$PR_NUMBER" -R "$GH_REPO" --body-file "$REVIEW_FILE" >> "$LOG_FILE" 2>&1
log_msg "review posted"
log_msg "=== done ==="
