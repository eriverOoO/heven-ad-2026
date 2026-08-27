#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

test_actions_checkout_precedence() {
  local case_dir="$TMP_DIR/actions-checkout"
  local case_home="$case_dir/home"
  local agent_ws="$case_dir/agent-ws"
  local actions_checkout="$case_dir/actions-workspace"
  local stale_checkout="$case_dir/stale-env-checkout"
  local env_file="$case_dir/agent.env"
  local codex_checkout_capture="$case_dir/codex-checkout"

  mkdir -p "$case_home/.local/bin" "$agent_ws/runner/bin" \
    "$actions_checkout" "$stale_checkout"

  cat > "$case_home/.local/bin/codex" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
while (($#)); do
  case "$1" in
    -C)
      printf '%s\n' "$2" > "$CODEX_CHECKOUT_CAPTURE"
      shift 2
      ;;
    --output-last-message)
      printf 'isolated test review\n' > "$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
EOF

  cat > "$case_home/.local/bin/gh" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF

  cat > "$agent_ws/runner/bin/render_prompt.sh" <<'EOF'
#!/usr/bin/env bash
printf 'isolated test prompt\n'
EOF

  chmod +x "$case_home/.local/bin/codex" "$case_home/.local/bin/gh" \
    "$agent_ws/runner/bin/render_prompt.sh"

  printf 'AGENT_WS=%s\nGH_REPO=stale/repo\nREPO_CHECKOUT=%s\n' \
    "$agent_ws" "$stale_checkout" > "$env_file"

  HOME="$case_home" \
    PATH="$case_home/.local/bin:/usr/local/bin:/usr/bin:/bin" \
    AGENT_ENV_FILE="$env_file" \
    GITHUB_REPOSITORY="example/actions-repo" \
    GITHUB_WORKSPACE="$actions_checkout" \
    CODEX_CHECKOUT_CAPTURE="$codex_checkout_capture" \
    "$ROOT_DIR/ops/runner/bin/codex-pr-review.sh" 17 \
    > "$case_dir/stdout" 2> "$case_dir/stderr" || {
      cat "$case_dir/stdout" "$case_dir/stderr" >&2
      return 1
    }

  local actual_checkout
  actual_checkout=$(<"$codex_checkout_capture")
  if [[ "$actual_checkout" != "$actions_checkout" ]]; then
    printf 'FAIL: codex -C expected %s, got %s\n' \
      "$actions_checkout" "$actual_checkout" >&2
    return 1
  fi
  printf 'PASS: codex PR review prefers GITHUB_WORKSPACE\n'
}

test_incoming_token_precedence() {
  local case_dir="$TMP_DIR/incoming-token"
  local case_home="$case_dir/home"
  local agent_ws="$case_dir/agent-ws"
  local env_file="$case_dir/agent.env"
  local token_capture="$case_dir/gh-token"

  mkdir -p "$case_home/.local/bin" "$agent_ws"

  cat > "$case_home/.local/bin/gh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail
printf '%s\n' "${GH_TOKEN-__UNSET__}" > "$GH_TOKEN_CAPTURE"
printf 'PR\tCLOSED\tfalse\n'
EOF
  chmod +x "$case_home/.local/bin/gh"

  printf 'AGENT_WS=%s\nGH_REPO=stale/repo\nPROJECT_PAT=env-project-token\nGH_TOKEN=env-file-token\n' \
    "$agent_ws" > "$env_file"

  HOME="$case_home" \
    PATH="$case_home/.local/bin:/usr/local/bin:/usr/bin:/bin" \
    AGENT_ENV_FILE="$env_file" \
    GITHUB_REPOSITORY="example/actions-repo" \
    GH_TOKEN="incoming-workflow-token" \
    GH_TOKEN_CAPTURE="$token_capture" \
    "$ROOT_DIR/ops/runner/bin/pr-set-in-review.sh" 23 \
    > "$case_dir/stdout" 2> "$case_dir/stderr" || {
      cat "$case_dir/stdout" "$case_dir/stderr" >&2
      return 1
    }

  local actual_token
  actual_token=$(<"$token_capture")
  if [[ "$actual_token" != "incoming-workflow-token" ]]; then
    printf 'FAIL: gh expected incoming GH_TOKEN, got %s\n' "$actual_token" >&2
    return 1
  fi
  printf 'PASS: project status preserves incoming GH_TOKEN\n'
}

failures=0
if ! test_actions_checkout_precedence; then
  failures=$((failures + 1))
fi
if ! test_incoming_token_precedence; then
  failures=$((failures + 1))
fi

if ((failures > 0)); then
  printf 'FAILED: %d precedence regression test(s)\n' "$failures" >&2
  exit 1
fi

printf 'PASS: all env precedence regressions\n'
