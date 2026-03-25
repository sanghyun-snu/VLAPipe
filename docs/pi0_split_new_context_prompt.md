# Prompt Template (Use with `pi0_split_new_context.md`)

Use this prompt in a new chat:

```text
다음 문서를 먼저 읽고 그 내용을 작업 기준 컨텍스트로 사용해줘:
/workspace/openpi/docs/pi0_split_new_context.md

요구사항:
1) 문서의 현재 상태를 5줄 이내로 요약
2) 남은 리스크를 정확도/성능/운영 관점으로 나눠 제시
3) 바로 실행 가능한 검증 커맨드(prefix/suffix/parity) 제시
4) 필요하면 최소 변경 패치부터 적용하고, 변경 파일 목록/의도/검증 결과를 보고

중요:
- 기존 구조(layer-wise split)는 유지
- 가급적 작은 패치 단위로 진행
- sidecar/direct 모드 호환성 깨지지 않게
```

Optional stricter variant:

```text
`/workspace/openpi/docs/pi0_split_new_context.md` 를 단일 source of truth로 간주해.
문서와 충돌하는 가정이 있으면 먼저 문서 기준으로 정리하고 진행해.
작업 전후로 parity/latency 관점의 성공 기준을 명시해줘.
```
