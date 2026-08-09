너는 ${REPO_FULL_NAME}의 nightly planning phase 1을 실행 중이다 (총 2-phase 중 첫 번째).

이 phase는 read-only scan이다. 파일 수정 / issue 생성 / GitHub mutation 모두 금지.
출력은 markdown 노트로 저장되고, 곧이어 phase 2가 그 노트를 읽는다.

## Repository

- name: ${REPO_FULL_NAME}
- role: ${REPO_ROLE}
- default branch: ${DEFAULT_BRANCH}
- workspace: ${WORKSPACE_PATH}

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

repo 전체를 쫙 읽고, 다음에 만들 만한 issue 후보를 생각해라.

## 읽을 것

repo 안 모든 파일을 자유롭게 보되 위 source order를 우선한다. 최근 commit, 열린
Issue/PR, 영향 받는 파일은 직접 확인한다.

## 작성 원칙 (메타인지)

이 노트는 phase 2의 기반 자료가 된다. phase 2가 너의 보고를 신뢰할 수 있어야 한다.

- 자세히 써라. 분석 근거 (어떤 파일의 어떤 부분 보고 결론 냈는지) 를 같이 적어라.
- 확실한 것만 단정으로 적어라. 직접 파일에서 확인한 사실, repo에 명시적으로 적힌 정보.
- 추측은 추측이라고 표기해라. "~인 것 같다", "~로 보인다", "확인 못 함" 같은 hedging 사용.
- 모르는 건 모른다고 적어라. 추측으로 채우지 말 것.
- 확신도가 다르면 별도 섹션으로 분리해라.

## 출력

지금 repo 상태에 대한 자세한 분석 노트. markdown.

다음을 포함해라:

### 1. 현재 상태 (확실한 사실 위주)
1-2 단락. 직접 확인한 것만.

### 2. 확실한 문제나 누락
파일 / commit / issue 근거 명시. 추측 섞지 말 것.

### 3. 추측 / 의심 / 확인 필요한 부분
"~로 보인다" / "확인 필요" 형태. 왜 추측인지도 같이.

### 4. 다음에 만들 만한 issue 후보 ${CANDIDATE_MIN}-${CANDIDATE_MAX}개
각 후보:
- 한 줄 제목
- 한 단락 설명 (확실한 근거가 있으면 그것, 추측이면 "추측 기반"임을 명시)
- 이 후보를 issue로 만드는 게 적절한지에 대한 자기 평가 (확신도 high / medium / low)

## 제약

- read-only.
- 추측을 사실처럼 적지 말 것. 확신 없이 issue 후보 양산하지 말 것.
- 보고할 게 없는 섹션은 "(해당 사항 없음)" 명시.
