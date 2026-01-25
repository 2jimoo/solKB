from __future__ import annotations
import json
from typing import Any, Dict


def _build_parent_map(events: List[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    parent: Dict[str, Optional[str]] = {}
    for e in events:
        if e.get("event") == "node_created":
            node = e.get("node") or {}
            tid = node.get("task_id")
            pid = node.get("parent_id")
            if tid:
                parent[tid] = pid
    return parent


def _collect_subtree_ids(events: List[Dict[str, Any]], root_id: str) -> Set[str]:
    # child adjacency
    children: Dict[str, List[str]] = {}
    for e in events:
        if e.get("event") == "node_created":
            node = e.get("node") or {}
            tid = node.get("task_id")
            pid = node.get("parent_id")
            if tid:
                children.setdefault(tid, [])
            if tid and pid:
                children.setdefault(pid, []).append(tid)

    # dfs
    stack = [root_id]
    out: Set[str] = set()
    while stack:
        cur = stack.pop()
        if cur in out:
            continue
        out.add(cur)
        stack.extend(children.get(cur, []))
    return out


def summarize_history_for_planning(
    events: List[Dict[str, Any]],
    node_id: str,
    limit_events: int = 400,  # 너무 길어지는 것 방지
    max_items: int = 30,  # 요약 bullet 수 제한
) -> str:
    subtree_ids = _collect_subtree_ids(events, node_id)
    if not subtree_ids:
        return None

    # 최근 이벤트만 보되, subtree 관련만 필터
    recent = [e for e in events[-limit_events:] if e.get("task_id") in subtree_ids]

    if not recent:
        return None

    bullets: List[str] = []

    # 1) 최근 분해 결과 (decomposed 이벤트를 선호)
    for e in reversed(recent):
        if e.get("event") == "decomposed":
            subs = e.get("subtasks")
            if isinstance(subs, list) and subs:
                bullets.append(
                    f"- Previous decomposition for {e.get('task_id')}: {subs[:5]}"
                )
                break  # 가장 최근 1개만

    # 2) 자식 노드들의 SLM 답 + 검증 실패 이유
    for e in recent:
        ev = e.get("event")
        tid = e.get("task_id")
        if ev == "slm_done":
            ans = str((e.get("slm_res") or {}).get("answer", "")).strip()
            if ans:
                bullets.append(f"- [{tid}] SLM answer: {ans[:300]}")
        elif ev in ("verified", "node_final_verified"):
            verdict = e.get("verdict")
            if verdict and verdict != "correct":
                reason = str(e.get("reason", "")).strip()
                evidence = str(e.get("evidence", "")).strip()
                line = f"- [{tid}] Verify failed: {verdict}"
                if reason:
                    line += f"; reason={reason[:250]}"
                if evidence:
                    line += f"; evidence={evidence[:250]}"
                bullets.append(line)
        elif ev == "slm_tool_result":
            out = e.get("output")
            # 도구 타임아웃/에러 반복 패턴만 잡기
            if isinstance(out, dict) and out.get("error"):
                bullets.append(f"- [{tid}] Tool error: {str(out.get('error'))[:250]}")

        if len(bullets) >= max_items:
            break

    return "\n".join(bullets).strip()


class JsonlKB:
    """Append-only JSONL KB/logger."""

    def __init__(self, path: str):
        self.path = path

    def append(self, event: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def read_events(self, limit: int | None = None) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        events.append(obj)
                except json.JSONDecodeError:
                    continue
        return events[-limit:] if limit else events

    def read_events_by_type(
        self,
        event_type: str,
        limit: int | None = None,
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and obj.get("event") == event_type:
                        events.append(obj)
                except json.JSONDecodeError:
                    continue
        return events[-limit:] if limit else events

    def planner_context(self, node_id: str, limit_events: int = 400) -> str:
        events = self.read_events()
        return summarize_history_for_planning(
            events, node_id=node_id, limit_events=limit_events
        )

    def collect_sibling_raw_logs(
        self,
        parent_task_id: str,
        limit_events: int = 500,
        max_items: int = 40,
        max_chars: int = 6000,
    ) -> Dict[str, Any]:
        """
        같은 parent 아래에서 이미 수행된 sibling들의 실행 기록만 뽑아 LLM 요약 입력(raw)으로 만든다.
        related 없으면 {}.
        """
        events = self.read_events()
        recent = events[-limit_events:]

        sibling_ids: List[str] = []
        sibling_q: Dict[str, str] = {}

        for e in recent:
            if e.get("event") == "node_created":
                node = e.get("node") or {}
                if node.get("parent_id") == parent_task_id:
                    tid = node.get("task_id")
                    if tid:
                        sibling_ids.append(tid)
                        sibling_q[tid] = (node.get("question") or "")[:300]

        if not sibling_ids:
            return {}

        picked: List[Dict[str, Any]] = []
        for e in recent:
            tid = e.get("task_id")
            if tid not in sibling_ids:
                continue

            ev = e.get("event")
            if ev in ("slm_done", "verified", "node_final_verified", "decomposed"):
                picked.append(e)
            elif ev == "slm_tool_result":
                out = e.get("output")
                if isinstance(out, dict) and out.get("error"):
                    picked.append(
                        {
                            "event": "tool_error",
                            "task_id": tid,
                            "error": str(out["error"])[:500],
                        }
                    )

            if len(picked) >= max_items:
                break

        payload = {
            "parent_task_id": parent_task_id,
            "siblings": [
                {"task_id": tid, "question": sibling_q.get(tid, "")}
                for tid in sibling_ids
            ],
            "events": picked,
        }

        s = json.dumps(payload, ensure_ascii=False)
        if len(s) > max_chars:
            payload["events"] = payload["events"][: max(8, max_items // 2)]
        return payload
