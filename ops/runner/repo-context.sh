#!/usr/bin/env bash
# Repo variables for prompt template rendering.
# Sourced (not executed).
#
# repo 식별은 하드코딩하지 않는다 — env 파일의 GH_REPO(또는 Actions의
# GITHUB_REPOSITORY)에서 온다. REPO_ROLE만 repo 성격에 맞게 env에서 덮어쓴다.

: "${GH_REPO:=${GITHUB_REPOSITORY:-}}"
: "${GH_REPO:?GH_REPO 필요 — env 파일(heven-agent.env)에 설정하세요}"

export REPO_FULL_NAME="$GH_REPO"
export REPO_ROLE="${REPO_ROLE:-HEVEN 팀 repo}"
export DEFAULT_BRANCH="${DEFAULT_BRANCH:-dev}"
export WORKSPACE_PATH="${AGENT_WS:-$HOME/heven_common_test_ws}"

# Planning tuning
export CANDIDATE_MIN="${CANDIDATE_MIN:-3}"
export CANDIDATE_MAX="${CANDIDATE_MAX:-5}"
export MAX_ISSUES="${MAX_ISSUES:-3}"

# Repo-specific human documentation. Bootstrap replaces the placeholder with the
# exact read-back repo hub URL, never the Project Info child URL. Automation
# fetches the repo hub only through an available MCP.
export NOTION_GUIDE_URL="${NOTION_GUIDE_URL:-https://app.notion.com/p/39c3bf068300801d818ad78812de50f3}"
export OPEN_MILESTONES="[]"

if command -v gh >/dev/null 2>&1; then
  OPEN_MILESTONES=$(gh api "repos/${REPO_FULL_NAME}/milestones?state=open" \
    --jq '[.[] | {number, title, description, due_on}]' 2>/dev/null || echo "[]")
  export OPEN_MILESTONES
else
  export OPEN_MILESTONES="[]"
fi
