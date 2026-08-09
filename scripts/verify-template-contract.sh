#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$ROOT"

failed=0
check_absent() {
  local label="$1" pattern="$2" rc=0
  shift 2

  if (($# == 0)); then
    echo "FAIL: $label scan scope is empty" >&2
    failed=1
    return
  fi

  rg --hidden -n -- "$pattern" "$@" || rc=$?
  case "$rc" in
    0)
      echo "FAIL: $label" >&2
      failed=1
      ;;
    1)
      ;;
    *)
      echo "FAIL: $label scan error (rg exit $rc)" >&2
      failed=1
      ;;
  esac
}

check_present() {
  local label="$1" pattern="$2" rc=0
  shift 2

  if (($# == 0)); then
    echo "FAIL: $label scan scope is empty" >&2
    failed=1
    return
  fi

  rg --hidden -n -- "$pattern" "$@" >/dev/null || rc=$?
  case "$rc" in
    0)
      ;;
    1)
      echo "FAIL: $label" >&2
      failed=1
      ;;
    *)
      echo "FAIL: $label scan error (rg exit $rc)" >&2
      failed=1
      ;;
  esac
}

check_ci_top_level_contents_read() {
  local label="$1" file
  shift

  if (($# == 0)); then
    echo "FAIL: $label scan scope is empty" >&2
    failed=1
    return
  fi

  for file in "$@"; do
    if ! awk '
      /^permissions:[[:space:]]*(#.*)?$/ {
        permissions_blocks++
        in_permissions = 1
        next
      }
      /^[^[:space:]#]/ { in_permissions = 0 }
      in_permissions && /^  contents:[[:space:]]*read[[:space:]]*(#.*)?$/ {
        contents_read++
      }
      END {
        exit !(permissions_blocks == 1 && contents_read == 1)
      }
    ' "$file"; then
      echo "FAIL: $label ($file)" >&2
      failed=1
    fi
  done
}

check_ci_checkout_credentials_disabled() {
  local label="$1" file
  shift

  if (($# == 0)); then
    echo "FAIL: $label scan scope is empty" >&2
    failed=1
    return
  fi

  for file in "$@"; do
    if ! awk '
      function leading_spaces(text) {
        match(text, /[^ ]/)
        return RSTART ? RSTART - 1 : length(text)
      }
      {
        line = $0
        indent = leading_spaces(line)

        if (line ~ /^[ ]*-[ ]+uses:[ ]+actions\/checkout@v4[[:space:]]*(#.*)?$/) {
          checkout_count++
          active_checkout = 1
          step_indent = indent
          in_with = 0
          next
        }

        if (!active_checkout) {
          next
        }
        if (line ~ /^[ ]*-[ ]+/ && indent == step_indent) {
          active_checkout = 0
          in_with = 0
          next
        }
        if (line !~ /^[[:space:]]*(#.*)?$/ && indent <= step_indent) {
          active_checkout = 0
          in_with = 0
          next
        }
        if (line ~ /^[ ]*with:[[:space:]]*(#.*)?$/ && indent == step_indent + 2) {
          in_with = 1
          next
        }
        if (in_with && line !~ /^[[:space:]]*(#.*)?$/ && indent == step_indent + 2) {
          in_with = 0
        }
        if (in_with && indent == step_indent + 4 &&
            line ~ /^[ ]*persist-credentials:[[:space:]]*false[[:space:]]*(#.*)?$/) {
          secured_checkout_count++
          in_with = 0
        }
      }
      END {
        exit !(checkout_count > 0 && secured_checkout_count == checkout_count)
      }
    ' "$file"; then
      echo "FAIL: $label ($file)" >&2
      failed=1
    fi
  done
}

check_ci_ripgrep_dependency() {
  local label="$1" file
  shift

  if (($# == 0)); then
    echo "FAIL: $label scan scope is empty" >&2
    failed=1
    return
  fi

  for file in "$@"; do
    if ! awk '
      /^      - name: Install heven-common contract dependencies[[:space:]]*(#.*)?$/ {
        install_steps++
        in_install_step = 1
        next
      }
      in_install_step && /^      - / {
        in_install_step = 0
      }
      in_install_step && /^        if: \$\{\{ github\.repository == '\''skku-heven\/heven-common'\'' \}\}[[:space:]]*(#.*)?$/ {
        repo_gate++
      }
      in_install_step && /^          sudo apt-get update[[:space:]]*(#.*)?$/ {
        apt_update++
      }
      in_install_step && /^          sudo apt-get install -y ripgrep[[:space:]]*(#.*)?$/ {
        ripgrep_install++
      }
      END {
        exit !(install_steps == 1 && repo_gate == 1 && apt_update == 1 && ripgrep_install == 1)
      }
    ' "$file"; then
      echo "FAIL: $label ($file)" >&2
      failed=1
    fi
  done
}

check_present_normalized() {
  local label="$1" pattern="$2" file normalized rc
  shift 2

  if (($# == 0)); then
    echo "FAIL: $label scan scope is empty" >&2
    failed=1
    return
  fi

  for file in "$@"; do
    if ! normalized=$(
      sed -E 's/^[[:space:]]*(>[[:space:]]*)+//' "$file" |
        tr '\n\r\t' '   '
    ); then
      echo "FAIL: $label normalization error ($file)" >&2
      failed=1
      continue
    fi
    rc=0
    rg --quiet -- "$pattern" <<< "$normalized" || rc=$?
    case "$rc" in
      0)
        ;;
      1)
        echo "FAIL: $label ($file)" >&2
        failed=1
        ;;
      *)
        echo "FAIL: $label scan error (rg exit $rc, $file)" >&2
        failed=1
        ;;
    esac
  done
}

extract_markdown_section_once() {
  local heading="$1" file="$2" heading_count

  heading_count=$(awk -v target="$heading" '
    $0 == target { count++ }
    END { print count + 0 }
  ' "$file") || return 2
  [[ "$heading_count" == 1 ]] || return 3

  awk -v target="$heading" '
    function heading_level(text) {
      match(text, /^#+/)
      return RLENGTH
    }
    $0 == target {
      active = 1
      base_level = heading_level($0)
      print
      next
    }
    active && /^#+[[:space:]]/ {
      if (heading_level($0) <= base_level) {
        exit
      }
    }
    active { print }
  ' "$file"
}

check_markdown_section_literal() {
  local label="$1" heading="$2" literal="$3" file section rc
  shift 3

  if (($# == 0)); then
    echo "FAIL: $label scan scope is empty" >&2
    failed=1
    return
  fi

  for file in "$@"; do
    rc=0
    section=$(extract_markdown_section_once "$heading" "$file") || rc=$?
    if ((rc != 0)); then
      if ((rc == 3)); then
        echo "FAIL: $label (expected exactly one section $heading in $file)" >&2
      else
        echo "FAIL: $label section scan error (exit $rc, $file)" >&2
      fi
      failed=1
      continue
    fi

    rc=0
    rg --fixed-strings --quiet -- "$literal" <<< "$section" || rc=$?
    case "$rc" in
      0)
        ;;
      1)
        echo "FAIL: $label ($file)" >&2
        failed=1
        ;;
      *)
        echo "FAIL: $label scan error (rg exit $rc, $file)" >&2
        failed=1
        ;;
    esac
  done
}

check_markdown_section_canonical_bullet() {
  local label="$1" heading="$2" field="$3" expected="$4"
  local file section rc label_count exact_count
  shift 4

  if (($# == 0)); then
    echo "FAIL: $label scan scope is empty" >&2
    failed=1
    return
  fi

  for file in "$@"; do
    rc=0
    section=$(extract_markdown_section_once "$heading" "$file") || rc=$?
    if ((rc != 0)); then
      if ((rc == 3)); then
        echo "FAIL: $label (expected exactly one section $heading in $file)" >&2
      else
        echo "FAIL: $label section scan error (exit $rc, $file)" >&2
      fi
      failed=1
      continue
    fi

    label_count=$(awk -v prefix="- $field:" '
      index($0, prefix) == 1 { count++ }
      END { print count + 0 }
    ' <<< "$section")
    exact_count=$(awk -v expected="$expected" '
      $0 == expected { count++ }
      END { print count + 0 }
    ' <<< "$section")
    if [[ "$label_count" != 1 || "$exact_count" != 1 ]]; then
      echo "FAIL: $label ($file)" >&2
      failed=1
    fi
  done
}

check_markdown_section_canonical_row() {
  local label="$1" heading="$2" source_label="$3" expected="$4"
  local file section rc label_count exact_count
  shift 4

  if (($# == 0)); then
    echo "FAIL: $label scan scope is empty" >&2
    failed=1
    return
  fi

  for file in "$@"; do
    rc=0
    section=$(extract_markdown_section_once "$heading" "$file") || rc=$?
    if ((rc != 0)); then
      if ((rc == 3)); then
        echo "FAIL: $label (expected exactly one section $heading in $file)" >&2
      else
        echo "FAIL: $label section scan error (exit $rc, $file)" >&2
      fi
      failed=1
      continue
    fi

    label_count=$(awk -v source_label="$source_label" '
      {
        cell_count = split($0, cells, "|")
        if (cell_count < 3) {
          next
        }
        prefix = cells[1]
        first_cell = cells[2]
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", prefix)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", first_cell)
        if (prefix == "" && first_cell == source_label) {
          count++
        }
      }
      END { print count + 0 }
    ' <<< "$section")
    exact_count=$(awk -v expected="$expected" '
      $0 == expected { count++ }
      END { print count + 0 }
    ' <<< "$section")
    if [[ "$label_count" != 1 || "$exact_count" != 1 ]]; then
      echo "FAIL: $label ($file)" >&2
      failed=1
    fi
  done
}

check_absent_normalized() {
  local label="$1" pattern="$2" file normalized rc
  shift 2

  if (($# == 0)); then
    echo "FAIL: $label scan scope is empty" >&2
    failed=1
    return
  fi

  for file in "$@"; do
    if ! normalized=$(tr '\n' ' ' < "$file"); then
      echo "FAIL: $label normalization error ($file)" >&2
      failed=1
      continue
    fi
    rc=0
    rg --quiet -- "$pattern" <<< "$normalized" || rc=$?
    case "$rc" in
      0)
        echo "FAIL: $label ($file)" >&2
        failed=1
        ;;
      1)
        ;;
      *)
        echo "FAIL: $label scan error (rg exit $rc, $file)" >&2
        failed=1
        ;;
    esac
  done
}

check_manual_remote_branch_deletion() {
  local label="$1" file normalized statements statement rc
  local deletion_pattern automatic_pattern negative_pattern
  shift

  if (($# == 0)); then
    echo "FAIL: $label scan scope is empty" >&2
    failed=1
    return
  fi

  deletion_pattern='(?i:(remote|원격)[^.!?。]{0,120}(branch|브랜치)[^.!?。]{0,100}(삭제|delete))'
  automatic_pattern='(?i:자동|automatically)'
  negative_pattern='(?i:(삭제|delete)[^.!?。]{0,48}(필요(가)?[[:space:]]*(없|않)|하지[[:space:]]*(않|말)|불필요|금지|no[[:space:]]+need|not[[:space:]]+need)|(never|must[[:space:]]+not|should[[:space:]]+not|do[[:space:]]+not|don.t|cannot|can.t)[^.!?。]{0,80}(삭제|delete))'

  for file in "$@"; do
    if ! normalized=$(tr '\n\r\t' '   ' < "$file"); then
      echo "FAIL: $label normalization error ($file)" >&2
      failed=1
      continue
    fi
    statements="$normalized"
    statements="${statements//./$'\n'}"
    statements="${statements//!/$'\n'}"
    statements="${statements//\?/$'\n'}"
    statements="${statements//。/$'\n'}"

    while IFS= read -r statement || [[ -n "$statement" ]]; do
      rc=0
      rg --quiet -- "$deletion_pattern" <<< "$statement" || rc=$?
      case "$rc" in
        1) continue ;;
        0) ;;
        *)
          echo "FAIL: $label scan error (rg exit $rc, $file)" >&2
          failed=1
          continue
          ;;
      esac

      rc=0
      rg --quiet -- "$automatic_pattern" <<< "$statement" || rc=$?
      case "$rc" in
        0) continue ;;
        1) ;;
        *)
          echo "FAIL: $label scan error (rg exit $rc, $file)" >&2
          failed=1
          continue
          ;;
      esac

      rc=0
      rg --quiet -- "$negative_pattern" <<< "$statement" || rc=$?
      case "$rc" in
        0) continue ;;
        1) ;;
        *)
          echo "FAIL: $label scan error (rg exit $rc, $file)" >&2
          failed=1
          continue
          ;;
      esac

      echo "FAIL: $label ($file: ${statement# })" >&2
      failed=1
    done <<< "$statements"
  done
}

check_non_autoadd_workflow_setup() {
  local label="$1" file normalized statements statement rc
  local workflow_pattern setup_pattern copied_pattern negative_pattern
  shift

  if (($# == 0)); then
    echo "FAIL: $label scan scope is empty" >&2
    failed=1
    return
  fi

  workflow_pattern='(?i:(PR|Pull[[:space:]]+request)[[:space:]]+linked[[:space:]]+to[[:space:]]+issue|(PR|Pull[[:space:]]+request)[[:space:]]+merged|Item[[:space:]]+closed)'
  setup_pattern='(?i:설정|구성|활성화|configure|configuration|enable|create|만들)'
  copied_pattern='(?i:복사|copied|copy[[:space:]]+from|template[^.!?。]{0,120}(자동|copy))'
  negative_pattern='(?i:(설정|구성|활성화)[^.!?。]{0,24}(않|없|불필요)|do[[:space:]]+not[[:space:]]+(configure|enable)|no[[:space:]]+need[[:space:]]+to[[:space:]]+(configure|enable))'

  for file in "$@"; do
    if ! normalized=$(tr '\n' ' ' < "$file"); then
      echo "FAIL: $label normalization error ($file)" >&2
      failed=1
      continue
    fi
    statements="$normalized"
    statements="${statements//./$'\n'}"
    statements="${statements//!/$'\n'}"
    statements="${statements//\?/$'\n'}"
    statements="${statements//。/$'\n'}"

    while IFS= read -r statement || [[ -n "$statement" ]]; do
      rc=0
      rg --quiet -- "$workflow_pattern" <<< "$statement" || rc=$?
      case "$rc" in
        1) continue ;;
        0) ;;
        *)
          echo "FAIL: $label scan error (rg exit $rc, $file)" >&2
          failed=1
          continue
          ;;
      esac

      rc=0
      rg --quiet -- "$setup_pattern" <<< "$statement" || rc=$?
      case "$rc" in
        1) continue ;;
        0) ;;
        *)
          echo "FAIL: $label scan error (rg exit $rc, $file)" >&2
          failed=1
          continue
          ;;
      esac

      rc=0
      rg --quiet -- "$copied_pattern" <<< "$statement" || rc=$?
      case "$rc" in
        0) continue ;;
        1) ;;
        *)
          echo "FAIL: $label scan error (rg exit $rc, $file)" >&2
          failed=1
          continue
          ;;
      esac

      rc=0
      rg --quiet -- "$negative_pattern" <<< "$statement" || rc=$?
      case "$rc" in
        0) continue ;;
        1) ;;
        *)
          echo "FAIL: $label scan error (rg exit $rc, $file)" >&2
          failed=1
          continue
          ;;
      esac

      echo "FAIL: $label ($file: ${statement# })" >&2
      failed=1
    done <<< "$statements"
  done
}

check_ordered_patterns() {
  local label="$1" file="$2"
  shift 2
  local previous=0 pattern match line rc

  for pattern in "$@"; do
    rc=0
    match=$(rg --no-heading -n -m 1 -- "$pattern" "$file") || rc=$?
    if ((rc != 0)); then
      if ((rc == 1)); then
        echo "FAIL: $label (missing ordered step: $pattern)" >&2
      else
        echo "FAIL: $label scan error (rg exit $rc)" >&2
      fi
      failed=1
      return
    fi
    line="${match%%:*}"
    if [[ ! "$line" =~ ^[0-9]+$ ]] || ((line <= previous)); then
      echo "FAIL: $label" >&2
      failed=1
      return
    fi
    previous="$line"
  done
}

for required_file in README.md AGENTS.md; do
  if [[ ! -f "$required_file" ]]; then
    echo "FAIL: required root missing: $required_file" >&2
    failed=1
  fi
done
for required_dir in .claude .github ops/runner; do
  if [[ ! -d "$required_dir" ]]; then
    echo "FAIL: required root missing: $required_dir" >&2
    failed=1
  fi
done

tracked_list=$(mktemp)
tracked_inventory_ok=0
tracked_files=()
if git ls-files -z > "$tracked_list"; then
  while IFS= read -r -d '' path; do
    tracked_files+=("$path")
  done < "$tracked_list"
  tracked_inventory_ok=1
else
  rc=$?
  echo "FAIL: tracked file discovery (git ls-files exit $rc)" >&2
  failed=1
fi
rm -f "$tracked_list"

if ((tracked_inventory_ok)); then
  tracked_content_scope=()
  content_scope=()
  ruleset_scope=()
  public_runner_scope=()
  claim_scope=()
  readme_scope=()
  agents_scope=()
  human_docs_scope=()
  ops_readme_scope=()
  runner_context_scope=()
  bootstrap_skill_scope=()
  bootstrap_script_scope=()
  issue_skill_scope=()
  ci_workflow_scope=()
  discord_workflow_scope=()

  for path in "${tracked_files[@]}"; do
    if [[ -e "$path" || -L "$path" ]]; then
      tracked_content_scope+=("$path")
    fi
    case "$path" in
      docs/superpowers|docs/superpowers/*)
        if [[ -e "$path" || -L "$path" ]]; then
          echo "FAIL: tracked docs/superpowers metadata: $path" >&2
          failed=1
        fi
        ;;
    esac
    case "$path" in
      README.md|AGENTS.md|.claude/*|.github/*|ops/runner/*)
        content_scope+=("$path")
        ;;
    esac
    case "$path" in
      README.md|.claude/skills/bootstrap-repo/*)
        ruleset_scope+=("$path")
        ;;
    esac
    case "$path" in
      README.md|.claude/*|ops/runner/*)
        public_runner_scope+=("$path")
        ;;
    esac
    case "$path" in
      .claude/skills/issue-worker/scripts/claim_issue.sh)
        claim_scope+=("$path")
        ;;
      README.md)
        readme_scope+=("$path")
        ;;
      AGENTS.md)
        agents_scope+=("$path")
        ;;
    esac
    case "$path" in
      README.md|AGENTS.md|.claude/skills/*/SKILL.md|ops/runner/README.md)
        human_docs_scope+=("$path")
        ;;
    esac
    case "$path" in
      ops/runner/README.md)
        ops_readme_scope+=("$path")
        ;;
      ops/runner/repo-context.sh)
        runner_context_scope+=("$path")
        ;;
    esac
    case "$path" in
      .claude/skills/bootstrap-repo/SKILL.md)
        bootstrap_skill_scope+=("$path")
        ;;
      .claude/skills/bootstrap-repo/scripts/bootstrap-repo.sh)
        bootstrap_script_scope+=("$path")
        ;;
      .claude/skills/issue-worker/SKILL.md)
        issue_skill_scope+=("$path")
        ;;
      .github/workflows/ci.yml)
        ci_workflow_scope+=("$path")
        ;;
      .github/workflows/notify-discord.yml)
        discord_workflow_scope+=("$path")
        ;;
    esac
  done

  check_absent 'Wiki git/runtime dependency' \
    '(?i:\bwiki_[a-z0-9_]*\b|raw\.githubusercontent\.com/wiki\b|\.wiki\.git\b|\btemplate_src\b)' \
    "${content_scope[@]}"
  check_absent 'ruleset API/bootstrap dependency' \
    '(?i:\brepos/[^[:space:]]+/rulesets\b|\bprotect-(dev|main)\b)' \
    "${ruleset_scope[@]}"
  check_absent 'public runner setup in private profile' \
    "(?i:[\"']?\ballows_public_repositories\b[\"']?[[:space:]]*[=:][[:space:]]*true\b)" \
    "${public_runner_scope[@]}"
  check_absent 'GitHub Wiki relative link' \
    '(?i:(\.\./)+wiki(?:/|[^a-z0-9._~-]|$)|github\.com/[^/[:space:]]+/[^/[:space:]]+/wiki(?:/|[^a-z0-9._~-]|$))' \
    "${content_scope[@]}"
  check_absent 'stale human Wiki vocabulary' \
    '(?i:관련[[:space:]]*파일[[:space:]]*/[[:space:]]*위키|위키[[:space:]]+operations|wiki[[:space:]]+operations|위키[[:space:]]+\[)' \
    "${content_scope[@]}"
  check_absent 'deleted Notion page ID' \
    '39[b]3bf068300' \
    "${tracked_content_scope[@]}"

  check_present 'claim script lacks executable Ready guard with failure action' \
    '^[[:space:]]*\[\[[[:space:]]+"\$status"[[:space:]]+==[[:space:]]+"Ready"[[:space:]]+\]\][[:space:]]+\|\|[[:space:]]+die([[:space:]]|$)' \
    "${claim_scope[@]}"

  canonical_notion_pattern='https://app\.notion\.com/p/39c3bf068300807c9e5bcf6b469f94ee([[:space:]]|\)|$)'
  check_present 'repo wiki common guide missing from README.md' \
    "$canonical_notion_pattern" "${readme_scope[@]}"
  check_present 'repo wiki common guide missing from AGENTS.md' \
    "$canonical_notion_pattern" "${agents_scope[@]}"

  repo_wiki_pattern='https://app\.notion\.com/p/39c3bf06830080b5a024c7ad91855240([[:space:]]|\)|$)'
  check_present 'repo wiki hub missing from README.md' \
    "$repo_wiki_pattern" "${readme_scope[@]}"
  check_present 'canonical Notion parent missing from bootstrap skill' \
    "$repo_wiki_pattern" "${bootstrap_skill_scope[@]}"
  check_present 'canonical common URL missing from bootstrap skill' \
    "$canonical_notion_pattern" "${bootstrap_skill_scope[@]}"

  one_shot_heading='## 1. One-shot bootstrap inputs'
  check_markdown_section_literal 'one-shot bootstrap collection missing' \
    "$one_shot_heading" \
    'Collect target repo, REPO_PURPOSE, and all three modes in one request.' \
    "${bootstrap_skill_scope[@]}"
  for required_input in \
      'Target repo: owner/repo' \
      'Repository purpose: exact one-line REPO_PURPOSE' \
      'DISCORD_NOTIFICATIONS is enabled or disabled' \
      'DISCORD_MENTIONS is enabled or disabled' \
      'RUNNER_AUTOMATION is enabled or disabled' \
      'Explicit opt-outs: ask for every mode; disabled must be selected explicitly and never inferred from silence.'; do
    check_markdown_section_literal "one-shot input contract missing: $required_input" \
      "$one_shot_heading" "$required_input" "${bootstrap_skill_scope[@]}"
  done

  secret_heading='### Enabled secret injection'
  for secret_contract in \
      'Only DISCORD_WEBHOOK and PROJECT_PAT are plaintext secrets.' \
      'Request missing conditional values only through secure environment/secret input, never plaintext chat/docs.' \
      'Ask the user to inject only DISCORD_WEBHOOK and PROJECT_PAT through host secure env/secret input.' \
      'Verify each enabled secret is present without printing its value.' \
      'If host secure input is unavailable, stop before mutation.' \
      'Local fallback: provide a read -s same-shell wrapper that invokes bootstrap before unsetting secrets.' \
      'DISCORD_USER_MAP and RUNNER_SCRIPTS are conditional non-secret values; do not persist their values in durable docs.' \
      'bootstrap_with_local_secrets() (' \
      "trap 'unset DISCORD_WEBHOOK PROJECT_PAT' EXIT"; do
    check_markdown_section_literal "secure input contract missing: $secret_contract" \
      "$secret_heading" "$secret_contract" "${bootstrap_skill_scope[@]}"
  done

  provenance_heading='### Target provenance preflight'
  for provenance_contract in \
      'Target REST metadata must prove private visibility, default dev, admin access, and template_repository.full_name skku-heven/heven-common.' \
      'Provenance preflight completes before any Notion mutation.'; do
    check_markdown_section_literal "provenance contract missing: $provenance_contract" \
      "$provenance_heading" "$provenance_contract" "${bootstrap_skill_scope[@]}"
  done
  check_ordered_patterns 'provenance, Notion metadata, and URL use order is invalid' \
    .claude/skills/bootstrap-repo/SKILL.md \
    '^### Target provenance preflight$' \
    'bootstrap-repo\.sh[[:space:]]+--preflight-only[[:space:]]+skku-heven/<new-repo>' \
    '^## 2\. Notion hub contract$' \
    '^## Repository metadata$' \
    '^## 3\. Versioned repo hub URL$'

  notion_heading='## 2. Notion hub contract'
  for notion_contract in \
      'For the repo hub exact direct-child title match: 0 -> create, 1 -> reuse, and 2+ -> fail.' \
      'Create or reuse Project Info from the common template and read it back.'; do
    check_markdown_section_literal "Notion hub contract missing: $notion_contract" \
      "$notion_heading" "$notion_contract" "${bootstrap_skill_scope[@]}"
  done

  hub_metadata_heading='## Repository metadata'
  check_markdown_section_canonical_bullet 'canonical GitHub metadata bullet missing, duplicated, or conflicting' \
    "$hub_metadata_heading" 'GitHub' \
    '- GitHub: https://github.com/<owner>/<repo>' \
    "${bootstrap_skill_scope[@]}"
  check_markdown_section_canonical_bullet 'canonical Purpose metadata bullet missing, duplicated, or conflicting' \
    "$hub_metadata_heading" 'Purpose' \
    '- Purpose: <exact REPO_PURPOSE>' \
    "${bootstrap_skill_scope[@]}"
  check_markdown_section_canonical_bullet 'canonical Common guide metadata bullet missing, duplicated, or conflicting' \
    "$hub_metadata_heading" 'Common guide' \
    '- Common guide: https://app.notion.com/p/39c3bf068300807c9e5bcf6b469f94ee' \
    "${bootstrap_skill_scope[@]}"
  check_markdown_section_canonical_bullet 'canonical Project Info metadata bullet missing, duplicated, or conflicting' \
    "$hub_metadata_heading" 'Project Info' \
    '- Project Info: <exact direct-child Project Info URL>' \
    "${bootstrap_skill_scope[@]}"

  reuse_heading='### Reuse and readback'
  for reuse_contract in \
      'The hub must contain exactly one ## Repository metadata section with exactly one canonical GitHub, Purpose, Common guide, and Project Info bullet.' \
      'Repository metadata section count: 0 -> insert this section, 1 -> verify and upsert only missing canonical bullets, and 2+ -> fail.' \
      'Preserve all unrelated hub content on reuse.' \
      'Upsert only missing canonical metadata bullets.' \
      'Fail closed on conflicting or ambiguous existing metadata.' \
      'Duplicate canonical metadata labels or values conflicting with expected values fail.' \
      'Readback parses only the exact ## Repository metadata section and verifies all four canonical bullets plus exact parent, title, hub page ID, and Project Info page ID before NOTION_REPO_URL is used.' \
      'Trust boundary: Notion MCP readback is authoritative for the repo hub parent/title and Project Info direct-child relationship.' \
      'The shell script does not read Notion; it validates only canonical URL form, distinct page IDs, common-page exclusion, and versioned placement.' \
      'Only Notion-dependent work blocks when MCP is unavailable; normal code, CI, and runner work continues.'; do
    check_markdown_section_literal "hub reuse/readback contract missing: $reuse_contract" \
      "$reuse_heading" "$reuse_contract" "${bootstrap_skill_scope[@]}"
  done

  versioned_url_heading='## 3. Versioned repo hub URL'
  for url_contract in \
      'Only the repo hub URL is versioned in README, AGENTS, and runner context.' \
      'The Project Info URL must differ from the repo hub URL and is verified only as its direct child; never version it in those three files.' \
      'Write NOTION_REPO_URL into README.md, AGENTS.md, and ops/runner/repo-context.sh before invoking the script.'; do
    check_markdown_section_literal "versioned repo hub URL contract missing: $url_contract" \
      "$versioned_url_heading" "$url_contract" "${bootstrap_skill_scope[@]}"
  done

  check_present 'README repo hub placeholder missing' \
    '<NOTION_REPO_URL>' "${readme_scope[@]}"
  check_present 'AGENTS repo hub placeholder missing' \
    '<NOTION_REPO_URL>' "${agents_scope[@]}"
  check_present 'runner context repo hub placeholder missing' \
    '<NOTION_REPO_URL>' "${runner_context_scope[@]}"

  template_identity_heading='## 템플릿 구성'
  check_markdown_section_literal 'future private template identity missing from README' \
    "$template_identity_heading" \
    '이 저장소는 HEVEN 팀의 공통 private template이며, 앞으로 만드는 팀 repository의 출발점이다.' \
    "${readme_scope[@]}"
  check_markdown_section_literal 'future repository Notion origin missing from README' \
    "$template_identity_heading" \
    '새 repository의 코드 설정은 이 template에서, 전용 문서는 공통 repo wiki 아래에서 시작한다.' \
    "${readme_scope[@]}"
  check_markdown_section_literal 'README Notion bootstrap boundary missing' \
    "$template_identity_heading" \
    'Notion은 일반 코드·CI·runner의 실행 의존성이 아니다. 다만 bootstrap은 repository 전용 Notion hub를 준비하고 확인한 뒤 GitHub 설정을 바꾼다.' \
    "${readme_scope[@]}"
  check_present_normalized 'README direct bootstrap Notion trust boundary missing' \
    '직접 실행하기 전에는 Notion MCP로 repo hub와 그 아래의 `Project Info`를 확인해야 한다\.[[:space:]]+자세한 조건은 `bootstrap-repo` skill을 따른다\.' \
    "${readme_scope[@]}"
  check_absent 'fixed 2026 project wording retained in README' \
    '(?i:heven-(ad|jj)-2026)' "${readme_scope[@]}"
  check_absent 'Project Info URL conflated with versioned repo hub URL' \
    'Project Info URL is versioned in README, AGENTS, and runner context\.' \
    "${readme_scope[@]}"

  source_ownership_heading='## 정본'
  common_guide_ownership_literal='| 공통 Notion 가이드 | 공통 workflow·convention |'
  repo_hub_ownership_literal='| repo별 Notion hub | domain·대회·hardware·환경·제약 |'
  check_markdown_section_canonical_row 'common Notion guide source ownership is missing, duplicated, or conflicting' \
    "$source_ownership_heading" '공통 Notion 가이드' "$common_guide_ownership_literal" \
    "${agents_scope[@]}"
  check_markdown_section_canonical_row 'repo-specific Notion hub source ownership is missing, duplicated, or conflicting' \
    "$source_ownership_heading" 'repo별 Notion hub' "$repo_hub_ownership_literal" \
    "${agents_scope[@]}"

  llm_notion_discovery_literal='- domain·대회·hardware·환경·제약에 관련된 작업은 Notion MCP로 repo별 hub를 먼저 읽는다. 공통 가이드만 보고 repo 제약을 추측하지 않는다.'
  check_markdown_section_literal 'LLM Notion discovery rule missing from AGENTS.md source ownership' \
    "$source_ownership_heading" "$llm_notion_discovery_literal" \
    "${agents_scope[@]}"

  completion_heading='## 5. 완료 전 확인'
  check_markdown_section_literal 'runner completion must verify Task 1 credential writes before deploy' \
    "$completion_heading" \
    'Runner completion verifies the Task 1 script wrote enabled repo secrets/variables without printing values, then deploys the runner.' \
    "${bootstrap_skill_scope[@]}"
  check_absent 'runner completion must not create a second credential mutation path' \
    'runner enabled이면 repo variables/secrets를 설정' \
    "${bootstrap_skill_scope[@]}"

  getting_started_pattern='https://app\.notion\.com/p/39c3bf06830081fb8e76d5c9a1be6d82([[:space:]]|\)|$)'
  team_workflow_pattern='https://app\.notion\.com/p/39c3bf068300810db8adc15fce65450d([[:space:]]|\)|$)'
  operations_pattern='https://app\.notion\.com/p/39c3bf068300815fb880e1d20602e662([[:space:]]|\)|$)'
  check_present 'direct Getting Started link missing from README.md' \
    "$getting_started_pattern" "${readme_scope[@]}"
  check_present 'direct Team Workflow link missing from README.md' \
    "$team_workflow_pattern" "${readme_scope[@]}"
  check_present 'direct Operations link missing from README.md' \
    "$operations_pattern" "${readme_scope[@]}"
  check_present 'direct Operations link missing from bootstrap skill' \
    "$operations_pattern" "${bootstrap_skill_scope[@]}"
  check_present 'direct Team Workflow link missing from issue-worker skill' \
    "$team_workflow_pattern" "${issue_skill_scope[@]}"

  bootstrap_prompt_pattern='이 repo의[[:space:]]+`bootstrap-repo`[[:space:]]+skill을[[:space:]]+사용해서[[:space:]]+`skku-heven/<new-repo>`를[[:space:]]+부트스트랩해줘\.'
  issue_prompt_pattern='이 repo의[[:space:]]+`issue-worker`[[:space:]]+skill을[[:space:]]+사용해서[[:space:]]+이슈[[:space:]]+#42를[[:space:]]+잡아줘\.'
  automatic_branch_deletion_pattern='merge 뒤 remote 작업 branch는 repository 설정에 따라 GitHub가 자동 삭제한다\.'

  check_present_normalized 'bootstrap skill prompt missing from README.md' \
    "$bootstrap_prompt_pattern" "${readme_scope[@]}"
  check_present_normalized 'bootstrap skill prompt missing from bootstrap skill' \
    "$bootstrap_prompt_pattern" "${bootstrap_skill_scope[@]}"
  check_present_normalized 'issue-worker skill prompt missing from issue-worker skill' \
    "$issue_prompt_pattern" "${issue_skill_scope[@]}"
  check_present_normalized 'automatic branch deletion missing from AGENTS.md' \
    "$automatic_branch_deletion_pattern" "${agents_scope[@]}"
  check_present_normalized 'automatic branch deletion missing from issue-worker skill' \
    "$automatic_branch_deletion_pattern" "${issue_skill_scope[@]}"

  check_present_normalized 'README private-template identity missing' \
    '(?i:\bprivate(?:[[:space:]]|[*_`~-]){1,24}template)' \
    "${readme_scope[@]}"
  check_present_normalized 'AGENTS main stable release-state missing' \
    '`?main`?[^.!?。]{0,80}현재[[:space:]]+stable/release-state[[:space:]]+branch' \
    "${agents_scope[@]}"

  check_present 'Project #14 template copy contract missing from README.md' \
    'Project #14.*\[TEMPLATE\][[:space:]]+heven-common' "${readme_scope[@]}"
  check_present 'Auto-add-only manual contract missing from README.md' \
    'Auto-add workflow만' "${readme_scope[@]}"
  check_present 'Auto-add filter missing from README.md' \
    'is:issue[[:space:]]+is:open.*Backlog' "${readme_scope[@]}"
  check_present 'merge method preservation missing from README.md' \
    'merge/rebase 허용 여부는 변경하지 않는다' "${readme_scope[@]}"
  check_present 'bootstrap Project Info URL runtime contract missing' \
    'NOTION_PROJECT_INFO_URL' "${bootstrap_skill_scope[@]}" "${bootstrap_script_scope[@]}"
  for contract_command in \
      'scripts/verify-template-contract\.sh' \
      '.claude/skills/bootstrap-repo/tests/test_bootstrap_repo\.sh' \
      '.claude/skills/issue-worker/tests/test_claim_issue\.sh' \
      'ops/runner/tests/test_env_precedence\.sh' \
      'ops/runner/tests/test_pr_set_in_review\.sh' \
      'scripts/tests/test_verify_template_contract\.sh'; do
    check_present "heven-common CI contract command missing: $contract_command" \
      "$contract_command" "${ci_workflow_scope[@]}"
  done
  check_ci_top_level_contents_read \
    'heven-common CI must declare top-level contents read permission' \
    "${ci_workflow_scope[@]}"
  check_ci_checkout_credentials_disabled \
    'heven-common CI checkout must disable persisted credentials' \
    "${ci_workflow_scope[@]}"
  check_ci_ripgrep_dependency \
    'heven-common CI must install the ripgrep contract dependency' \
    "${ci_workflow_scope[@]}"
  check_present 'Discord notification mode gate missing from workflow' \
    "ENABLE_DISCORD_NOTIFICATIONS[[:space:]]*==[[:space:]]*'true'" \
    "${discord_workflow_scope[@]}"
  check_present 'Discord mention mode gate missing from workflow' \
    'ENABLE_DISCORD_MENTIONS' "${discord_workflow_scope[@]}"
  check_present 'maintainer merge-method choice missing from AGENTS.md' \
    'repository에서 허용된 merge 방식' "${agents_scope[@]}"
  check_absent_normalized 'marker files used as provenance in human contracts' \
    '(?i:(template[[:space:]-]*marker|marker[[:space:]]*(파일|file))[^.!?。]{0,200}(provenance|검증|확인|필수|must|required|있어야))' \
    "${human_docs_scope[@]}"
  check_absent 'manual Project creation retained in human contracts' \
    '(?i:Project(V2|[[:space:]]*보드)?(를|을)?[[:space:]]*(직접|수동)?[[:space:]]*(만들|생성|create))' \
    "${human_docs_scope[@]}"
  check_non_autoadd_workflow_setup \
    'manual non-Auto-add Project workflow retained' \
    "${human_docs_scope[@]}"
  check_absent 'squash-only mandate in human contracts' \
    '(?i:--squash|squash[-[:space:]]*only|squash[[:space:]]+merge(만|한다|합니다|해야|필수|강제)|squash([[:space:]]+merge)?[[:space:]]*(만|필수|강제))' \
    "${human_docs_scope[@]}"
  check_absent 'human lifecycle detail retained in bootstrap README' \
    '^##[[:space:]]+(작업 lifecycle|셋업 검증)([[:space:]]|$)' \
    "${readme_scope[@]}"
  check_absent 'current public-template wording retained in human contracts' \
    '(?i:\bpublic(?:[[:space:]]|[*_`~-]){1,24}template)' \
    "${human_docs_scope[@]}"
  check_absent 'manual remote branch deletion retained in human contracts' \
    '(?i:--delete-branch)' \
    "${human_docs_scope[@]}"
  check_absent 'git push remote branch deletion retained in human contracts' \
    '(?i:\bgit[[:space:]]+push[[:space:]]+origin[^[:cntrl:]]*--delete([=[:space:]]|$))' \
    "${human_docs_scope[@]}"
  check_manual_remote_branch_deletion \
    'natural-language manual remote branch deletion retained in human contracts' \
    "${human_docs_scope[@]}"
  check_absent 'release/tag policy retained in human contracts' \
    '(?i:\bsemantic(?:[[:space:]]|[*_`~-]){1,24}versioning([^A-Za-z0-9_]|$)|\bsemver([^A-Za-z0-9_]|$)|\bgithub(?:[[:space:]]|[*_`~-]){1,24}releases?([^A-Za-z0-9_]|$)|\brelease(?:[[:space:]]|[*_`~-]){1,24}tags?([^A-Za-z0-9_]|$)|\bgit(?:[[:space:]]|[*_`~-]){1,24}tag([^A-Za-z0-9_]|$))' \
    "${human_docs_scope[@]}"
  check_absent 'inline bootstrap secret example in human docs' \
    '(?i:(PROJECT_PAT|DISCORD_WEBHOOK)[[:space:]]*=)' \
    "${human_docs_scope[@]}"
  check_absent 'inline bootstrap secret example in bootstrap script comments' \
    '^[[:space:]]*#[^[:cntrl:]]*(PROJECT_PAT|DISCORD_WEBHOOK)[[:space:]]*=' \
    "${bootstrap_script_scope[@]}"
  check_present 'bootstrap script Notion trust boundary comment missing' \
    '^# Trust boundary: Notion MCP readback verifies hub parent/title and Project Info direct-child\. This shell does not read Notion; it validates only canonical URL form, distinct page IDs, common-page exclusion, and versioned placement\.$' \
    "${bootstrap_script_scope[@]}"

  check_present 'runner public repository boundary missing from operations docs' \
    'public repository access.*항상.*차단' "${ops_readme_scope[@]}"
  check_present 'runner selected-repository plan fallback missing from operations docs' \
    'Selected repositories.*지원하면' "${ops_readme_scope[@]}"
  check_present 'workflow opt-in confused with runner access isolation' \
    'ENABLE_RUNNER_AUTOMATION.*bundled workflow job.*runner access.*아니' \
    "${ops_readme_scope[@]}"

  check_ordered_patterns 'README bootstrap guidance must order skill prompt, cd, and direct script' README.md \
    '이 repo의[[:space:]]+`bootstrap-repo`[[:space:]]+skill을[[:space:]]+사용해서' \
    '^[[:space:]]*cd[[:space:]]+<new-repo>[[:space:]]*$' \
    '^[[:space:]]*\.claude/skills/bootstrap-repo/scripts/bootstrap-repo\.sh[[:space:]]+skku-heven/<new-repo>[[:space:]]*$'

  direct_shell_re='^#![[:space:]]*/([^[:space:]]*/)*(bash|sh)([[:space:]]|$)'
  env_shell_re='^#![[:space:]]*/([^[:space:]]*/)*env[[:space:]]+(-S[[:space:]]+)?(bash|sh)([[:space:]]|$)'

  for script in "${tracked_files[@]}"; do
    [[ -e "$script" || -L "$script" ]] || continue
    is_shell=0
    case "$script" in
      *.sh)
        is_shell=1
        ;;
      *)
        first_line=''
        IFS= read -r first_line < "$script" || true
        if [[ "$first_line" =~ $direct_shell_re || "$first_line" =~ $env_shell_re ]]; then
          is_shell=1
        fi
        ;;
    esac

    if ((is_shell)) && ! bash -n "$script"; then
      echo "FAIL: tracked shell syntax: $script" >&2
      failed=1
    fi
  done
fi

(( failed == 0 )) || exit 1
echo 'template contract: ok'
