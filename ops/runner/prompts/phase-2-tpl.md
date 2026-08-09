너는 ${REPO_FULL_NAME}의 nightly planning phase 2를 실행 중이다 (총 2-phase 중 두 번째).

방금 phase 1이 repo 전체를 한 번 read-only scan하고 분석 노트를 만들었다.

너는 phase 1과 동일하게 repo 전체를 다시 read-only scan한다.
phase 1 분석 노트를 참고하되, phase 1을 그대로 받아쓰지 않고 너 자신이 다시 본다.
두 번의 독립된 읽기를 거친 결과로 issue 후보를 정리한다.

## 입력

- phase 1 결과 노트: ${PHASE1_FILE} (참고)
- repo 전체 (phase 1과 동일 범위, 너가 직접 다시 본다)
- issue body 형식: Goal / Required Context / Acceptance Criteria / Boundaries / Verification 5개 섹션 (markdown)

## Source order

다음 순서로 source를 확인한다:

1. AGENTS.md
2. README.md
3. .agent/context/CONTEXT_BUNDLE.md when present
4. related Issue/PR and open milestones
5. Notion guide `${NOTION_GUIDE_URL}` only when Notion MCP is available and
   the task needs human workflow or competition-specification context

Notion MCP가 없으면 URL을 curl로 우회하지 않는다. 로컬 문서와 Issue만으로 수용
기준을 확정할 수 없으면 그 사실을 결과에 명시하고 변경/이슈 제안을 중단한다.

### Open Milestones
${OPEN_MILESTONES}

새 issue 를 만들 때 가능하면 적절한 milestone 에 배정하라.

## 할 일

1. repo 전체를 다시 한 번 읽는다 (phase 1을 그대로 믿지 말고 직접 확인).
2. phase 1 분석과 너의 두 번째 읽기 결과를 종합한다.
   - phase 1이 "확실한 사실"로 적은 것: 너가 한 번 더 검증. 일치하면 채택.
   - phase 1이 "추측"으로 적은 것: 너가 다시 보고 확실해지면 사실로, 여전히 애매하면 issue 안 만든다.
   - phase 1이 issue 후보로 낸 것 중 confidence "low"는 채택 안 한다.
   - phase 1이 놓친 게 보이면 추가 (단 너의 confidence가 high여야 함).
3. 진짜 GitHub issue로 만들 후보를 결정한다.

기준:
- 확신 없는 issue는 만들지 않는다.
- 너무 큰 작업은 쪼개거나 제외.
- 중복 후보 통합.
- 최대 ${MAX_ISSUES}개.
- 같은 제목의 open issue가 이미 있으면 (CONTEXT_BUNDLE.md 확인) 만들지 않는다.

## 중복 방지 (강화)

새 issue 를 만들기 전에 **기존 open issue + 최근 닫힌 issue**(CONTEXT_BUNDLE 의
"Closed issues (recent)")와 다음 기준으로 비교:
- 제목의 의도가 비슷한가? (다른 표현이라도 같은 작업이면 중복)
- 본문의 acceptance criteria 가 겹치는가?
- 같은 파일 / 모듈을 건드리는가?

위 셋 중 하나라도 강하게 겹치면 새 issue 를 만들지 마라.
특히 **`not planned` 로 닫힌 이슈와 제목이 겹치면 이미 사람이 거절/보류한 것이니 다시
제안하지 마라** (안 그러면 정리한 백로그가 계속 되살아난다).
의심되면 안전하게 skip 하라 (false positive 보다 중복 방지가 중요).

기존 issue 와의 관계가 모호하면 코멘트로 처리할 수 있는지 검토하라.

## Milestone 배정

issue JSON의 `milestone` 필드에는 실제 open milestone title만 넣어라.

milestone 결정:
- OPEN_MILESTONES 에서 가장 적합한 것 선택
- 적합한 게 없으면 `null`
- 이름을 모르면 추측하지 말고 `null`

## Issue Type

`type`은 native Issue Type으로 그대로 설정된다. 내용에 맞게 골라라:
- 잘못된 동작의 수정 → "Bug"
- 실험 / 튜닝 / 검증 → "Experiment"
- 코드 외 운영·설정·문서 → "Chore"
- 그 외 개발 작업 → "Task"

title 태그도 type과 맞춰라: `[Bug]` / `[Task]` / `[Experiment]` / `[Chore]`.

## 출력

valid JSON 한 개. 다른 텍스트 / 마크다운 코드 블록 금지. 순수 JSON만.

스키마:

{
  "issues": [
    {
      "title": "[Bug] 같은 태그로 시작하는 한 줄 제목 (태그는 type과 일치)",
      "milestone": "정확한 milestone 이름 또는 null",
      "type": "Bug" | "Task" | "Experiment" | "Chore",
      "priority": "P0" | "P1" | "P2",
      "body": "markdown. Goal / Required Context / Acceptance Criteria / Boundaries / Verification 5개 섹션."
    }
  ]
}

milestone 이름은 ${REPO_FULL_NAME}에 실제로 존재하는 것만. 모르면 null.

## 제약

- read-only. 파일 수정 / GitHub mutation 금지.
- 결과는 valid JSON. parse 실패하면 wrapper가 skip.
- issue 0개여도 OK. 그 경우 {"issues": []}.
