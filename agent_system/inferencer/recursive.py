from __future__ import annotations

import json
import time
from typing import TypedDict, List, Optional, Dict, Any, Callable, Literal, Tuple

SearchMode = Literal["hybrid", "text", "semantic"]


class RecursiveInferencer:
    """
    root_task(str) -> AKB 검색 -> SLM으로 subtasks 생성 -> 순차 실행
      - subtask 실패 시: 실패 컨텍스트로 AKB 재검색 -> SLM replanning -> 재귀적으로 해결
    """

    def __init__(
        self,
        akb_client,  # AKBClient
        slm_runner,  # SLMRunnerHF
        *,
        mode: SearchMode = "hybrid",
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None,
        max_subtasks: int = 5,
        max_depth: int = 3,
        max_retries_per_depth: int = 2,
        backoff_sec: float = 0.5,
    ):
        self.akbclient = akb_client
        self.slm = slm_runner

        self.mode = mode
        self.top_k = top_k
        self.weights = weights or {"text": 0.5, "semantic": 0.5}

        self.max_subtasks = max_subtasks
        self.max_depth = max_depth
        self.max_retries_per_depth = max_retries_per_depth
        self.backoff_sec = backoff_sec

    # ---------- AKB ----------
    def fetch_refs(
        self,
        query: str,
        *,
        mode: Optional[SearchMode] = None,
        top_k: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[TaskResult]:
        mode = mode or self.mode
        top_k = top_k or self.top_k
        weights = weights or self.weights

        if mode == "hybrid":
            return self.akbclient.hybrid_search(
                query=query, top_k=top_k, weights=weights
            )
        if mode == "text":
            return self.akbclient.text_search(query=query, top_k=top_k)
        if mode == "semantic":
            return self.akbclient.semantic_search(query=query, top_k=top_k)
        raise ValueError(f"Unknown search mode: {mode}")

    # ---------- Planning ----------
    def _normalize_reference_cards(
        self,
        refs: List[TaskResult],
        *,
        max_docs: int = 5,
        max_subtasks_per_doc: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        SLM이 이해하기 쉬운 형태로 레퍼런스 subtasks/rationale만 카드화.
        (actions는 참고로 넣어도 되지만, rationale 기반 생성이 목적이라 기본은 제외/축약)
        """
        cards: List[Dict[str, Any]] = []
        for i, r in enumerate(refs[:max_docs]):
            st_list = r.get("subtasks") or []
            if not isinstance(st_list, list) or not st_list:
                continue

            cards.append(
                {
                    "doc_id": r.get("task_id", f"doc_{i}"),
                    "task": r.get("task", ""),
                    "subtasks": [
                        {
                            "subgoal": st.get("subgoal", ""),
                            "rationale": st.get("rationale", ""),
                            # "actions": st.get("actions", [])  # 필요하면 주석 해제
                        }
                        for st in st_list[:max_subtasks_per_doc]
                        if isinstance(st, dict) and st.get("subgoal")
                    ],
                }
            )
        return cards

    def _build_planning_messages(
        self,
        *,
        root_task: str,
        reference_cards: List[Dict[str, Any]],
        max_subtasks: int,
    ) -> List[Dict[str, str]]:
        system = (
            "You are a task planning agent.\n"
            "You MUST output valid JSON only.\n"
            "Goal: Generate subtasks for the ROOT TASK.\n\n"
            "Hard rules:\n"
            "- Use reference subtasks & rationales as evidence/inspiration.\n"
            "- Do NOT copy verbatim; adapt and restructure for the new task.\n"
            "- Output must follow the schema exactly.\n"
            "- Subtasks must be logically ordered and executable.\n"
        )

        user = (
            f"ROOT TASK:\n{root_task}\n\n"
            f"REFERENCE (subtasks & rationales):\n{json.dumps(reference_cards, ensure_ascii=False, indent=2)}\n\n"
            f"Generate at most {max_subtasks} subtasks.\n"
            "Return JSON ONLY with shape:\n"
            "{\n"
            '  "subtasks": [\n'
            '    {"subgoal":"...", "rationale":"...", "actions":["...", "..."]}\n'
            "  ]\n"
            "}\n"
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def plan_subtasks(
        self,
        task: str,
        refs: List[TaskResult],
        *,
        max_subtasks: Optional[int] = None,
    ) -> List[Subtask]:
        max_subtasks = max_subtasks or self.max_subtasks
        cards = self._normalize_reference_cards(refs)

        messages = self._build_planning_messages(
            root_task=task,
            reference_cards=cards,
            max_subtasks=max_subtasks,
        )
        raw = self.slm._generate(messages)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"SLM returned invalid JSON:\n{raw}") from e

        st_list = parsed.get("subtasks", [])
        if not isinstance(st_list, list):
            raise ValueError(f"Invalid 'subtasks' type: {type(st_list)}")

        out: List[Subtask] = []
        for st in st_list[:max_subtasks]:
            if not isinstance(st, dict):
                continue
            if not all(k in st for k in ("subgoal", "rationale", "actions")):
                continue
            actions = st.get("actions")
            if not isinstance(actions, list):
                continue

            out.append(
                {
                    "subgoal": str(st["subgoal"]),
                    "rationale": str(st["rationale"]),
                    "actions": [str(a) for a in actions],
                }
            )
        return out

    def run_subtask_once(
        self,
        subtask: Subtask,
        action_executor: Callable[[str], Any],
    ) -> ExecutionResult:
        outputs: List[Any] = []
        try:
            for act in subtask["actions"]:
                outputs.append(action_executor(act))
            return {"ok": True, "output": outputs}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": outputs}

    def solve_subtask_recursive(
        self,
        *,
        root_task: str,
        subtask: Subtask,
        action_executor: Callable[[str], Any],
        depth: int = 0,
        max_depth: Optional[int] = None,
        max_retries_per_depth: Optional[int] = None,
    ) -> ExecutionResult:
        max_depth = self.max_depth if max_depth is None else max_depth
        max_retries_per_depth = (
            self.max_retries_per_depth
            if max_retries_per_depth is None
            else max_retries_per_depth
        )

        # 1) 동일 계획으로 재시도
        last_err = ""
        for attempt in range(1, max_retries_per_depth + 1):
            r = self.run_subtask_once(subtask, action_executor)
            r["attempts"] = attempt
            r["depth"] = depth
            if r.get("ok"):
                return r
            last_err = r.get("error", "unknown error")
            time.sleep(self.backoff_sec * attempt)

        # 2) 깊이 제한
        if depth >= max_depth:
            return {
                "ok": False,
                "error": f"Max depth reached. Last error: {last_err}",
                "attempts": max_retries_per_depth,
                "depth": depth,
            }

        # 3) 실패 컨텍스트로 AKB 재조회 -> replanning -> 재귀 해결
        failure_query = (
            f"ROOT_TASK: {root_task}\n"
            f"FAILED_SUBTASK: {subtask['subgoal']}\n"
            f"RATIONALE: {subtask.get('rationale','')}\n"
            f"ACTIONS: {json.dumps(subtask.get('actions',[]), ensure_ascii=False)}\n"
            f"ERROR: {last_err}\n"
            "REQUEST: Create an improved, more concrete and smaller-grained subtask plan to resolve this failure."
        )

        refs = self.fetch_refs(failure_query)
        new_subtasks = self.plan_subtasks(failure_query, refs, max_subtasks=3)

        aggregated: List[Any] = []
        for st in new_subtasks:
            child = self.solve_subtask_recursive(
                root_task=root_task,
                subtask=st,
                action_executor=action_executor,
                depth=depth + 1,
                max_depth=max_depth,
                max_retries_per_depth=max_retries_per_depth,
            )
            if not child.get("ok"):
                return child
            aggregated.append(child.get("output"))

        return {
            "ok": True,
            "output": aggregated,
            "attempts": max_retries_per_depth,
            "depth": depth,
        }

    def run(
        self,
        task: str,
        action_executor: Callable[[str], Any],
        *,
        stop_on_first_failure: bool = True,
    ) -> Dict[str, Any]:
        refs = self.fetch_refs(task)
        subtasks = self.plan_subtasks(task, refs, max_subtasks=self.max_subtasks)

        results: List[Dict[str, Any]] = []
        for st in subtasks:
            r = self.solve_subtask_recursive(
                root_task=task,
                subtask=st,
                action_executor=action_executor,
                depth=0,
            )
            results.append({"subtask": st, "result": r})
            if stop_on_first_failure and not r.get("ok"):
                return {
                    "ok": False,
                    "task": task,
                    "planned_subtasks": subtasks,
                    "results": results,
                }

        return {
            "ok": True,
            "task": task,
            "planned_subtasks": subtasks,
            "results": results,
        }
