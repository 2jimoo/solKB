from __future__ import annotations

import json
import time
from typing import TypedDict, List, Optional, Dict, Any, Callable, Literal, Tuple
from agent_system.models import Subtask, TaskResult
from agent_system.reconstructor import SubtaskRewriter

import logging

SearchMode = Literal["hybrid", "text", "semantic"]

logger = logging.getLogger(__name__)


class RecursiveLLMInferencer:

    def __init__(
        self,
        akb_client,  # AKBClient
        llm_runner,  # LLMRunnerWithTools
        *,
        mode: SearchMode = "hybrid",
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None,
        max_subtasks: int = 5,
        max_depth: int = 3,
        max_retries_per_depth: int = 2,
        backoff_sec: float = 0.5,
        subtask_rewriter: Optional[SubtaskRewriter] = None,
        rewrite_before_recursive: bool = True,
        rewrite_retry_budget: int = 1,  # 재기술 후 추가로 몇 번 재시도할지
        rewrite_guidance: Optional[str] = None,
    ):
        self.akbclient = akb_client
        self.llm = llm_runner

        self.mode = mode
        self.top_k = top_k
        self.weights = weights or {"text": 0.5, "semantic": 0.5}

        self.max_subtasks = max_subtasks
        self.max_depth = max_depth
        self.max_retries_per_depth = max_retries_per_depth
        self.backoff_sec = backoff_sec

        self.subtask_rewriter = subtask_rewriter
        self.rewrite_before_recursive = rewrite_before_recursive
        self.rewrite_retry_budget = rewrite_retry_budget
        self.rewrite_guidance = rewrite_guidance

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
        cards: List[Dict[str, Any]] = []
        for i, r in enumerate(refs[:max_docs]):
            st_list = r.get("subtasks") or []
            if not isinstance(st_list, list) or not st_list:
                continue

            cards.append(
                {
                    "task": r.get("task", ""),
                    "subtasks": [
                        {
                            "subgoal": st.get("subgoal", ""),
                            "rationale": st.get("rationale", ""),
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
            '    {"subgoal":"...", "rationale":"..."}\n'
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
        raw = self.llm.run(messages, None)

        try:
            raw = raw.output[0].content[0].text
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                raw = raw.rsplit("```", 1)[0].strip()
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON:\n{raw}") from e

        st_list = parsed.get("subtasks", [])
        if not isinstance(st_list, list):
            raise ValueError(f"Invalid 'subtasks' type: {type(st_list)}")

        out: List[Subtask] = []
        for st in st_list[:max_subtasks]:
            if not isinstance(st, dict):
                continue
            if not all(k in st for k in ("subgoal", "rationale")):
                continue

            out.append(
                {
                    "subgoal": str(st["subgoal"]),
                    "rationale": str(st["rationale"]),
                }
            )
        return out

    # ---------- Difficulty ----------
    def _doc_difficulty(self, ref: "TaskResult") -> int:
        """
        문서 난이도 = 해당 문서(TaskResult)의 subtasks 개수
        """
        st_list = ref.get("subtasks") or []
        return len(st_list) if isinstance(st_list, list) else 0

    def estimate_difficulty(
        self,
        query: str,
        *,
        mode: Optional["SearchMode"] = None,
        top_k: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
        min_docs: int = 1,
    ) -> Tuple[float, List["TaskResult"]]:
        """
        task와 유사한 문서 조회 후,
        task 난이도 = (유사 문서 난이도 평균)
        """
        refs = self.fetch_refs(query, mode=mode, top_k=top_k, weights=weights)

        diffs: List[int] = []
        for r in refs:
            d = self._doc_difficulty(r)
            diffs.append(d)

        # 문서가 너무 없거나 전부 0이면 fallback: 0.0
        if len(diffs) < min_docs:
            return 0.0, refs

        avg = sum(diffs) / max(len(diffs), 1)
        return float(avg), refs

    # ---------- Execution (single) ----------
    def run_query_once(
        self,
        query: str,
        tools: ToolRegistry,
    ) -> "ExecutionResult":
        """
        서브태스크로 쪼개지 않고 query 자체를 바로 푸는 실행 경로
        """
        try:
            outputs = self.llm.run(
                question=query,
                tools=tools,
            )
            return {"ok": True, "output": outputs}
        except Exception as e:
            return {"ok": False, "error": str(e), "output": []}

    # ---------- Adaptive recursive solver ----------
    def run(
        self,
        *,
        root_task: str,
        task: str,
        tools: ToolRegistry,
        difficulty_threshold: float,
        depth: int = 0,
        max_depth: Optional[int] = None,
        stop_on_first_failure: bool = True,
    ) -> Dict[str, Any]:
        logger.info(
            f"====================================== DEPTH {depth} ======================================\n"
            f"Root Task: {root_task}\n"
            f"Current Task: {task}\n"
        )
        max_depth = self.max_depth if max_depth is None else max_depth
        if depth > max_depth:
            return {
                "ok": False,
                "task": task,
                "depth": depth,
                "error": f"Max depth exceeded (>{max_depth})",
            }

        # 1) depth != 0이면 현재 태스크 재기술 시도 (옵션)
        task_to_solve = task
        rewriter_meta = None
        if depth != 0 and self.subtask_rewriter and self.rewrite_before_recursive:
            try:
                rewrite_res = self.subtask_rewriter.rewrite(
                    original_task=root_task,
                    current_subtask={"subgoal": task, "rationale": ""},
                    failure_logs=[],
                    guidance=self.rewrite_guidance,
                )
                logger.info(
                    f"Rewrite\n[BEFORE] {task}\n[AFTER] {rewrite_res['rewritten_subtask']}"
                )
            except Exception as e:
                return {
                    "ok": False,
                    "task": task,
                    "depth": depth,
                    "error": f"rewrite error: {e}",
                }

            if rewrite_res.get("status") == "ABORT":
                return {
                    "ok": False,
                    "task": task,
                    "depth": depth,
                    "error": f"rewrite aborted: {rewrite_res.get('reason')}",
                }
            # OK이면 교체
            task_to_solve = rewrite_res.get("rewritten_subtask") or task
            rewriter_meta = rewrite_res

        # 2) 난이도 측정
        difficulty, refs = self.estimate_difficulty(task_to_solve)
        logger.info(f"Estimated Difficulty: {difficulty}")
        # logger.info(f"Refs: {json.dumps(refs, ensure_ascii=False, indent=2)}")

        # 3) threshold 이하이면 direct 시도
        if difficulty <= difficulty_threshold:
            direct = self.run_query_once(task_to_solve, tools)
            direct.update(
                {
                    "task": task_to_solve,
                    "root_task": root_task,
                    "depth": depth,
                    "estimated_difficulty": difficulty,
                    "strategy": "direct",
                    "rewriter": rewriter_meta,
                }
            )
            logger.info(f"Try to solve directly. result:\n{direct}")
            if direct.get("ok"):
                return direct
            # direct 실패면 분해로 fallback (아래 분해 로직으로 내려감)

        # 4) 분해 시도 (난이도 초과거나 direct 실패)
        planned = self.plan_subtasks(
            task_to_solve, refs, max_subtasks=self.max_subtasks
        )
        planned_str = "\n".join([p["subgoal"] for p in planned])
        logger.info(f"Decomposed to...\n{planned_str}")

        results: List[Dict[str, Any]] = []
        for st in planned:
            subgoal = st["subgoal"]

            # 재귀 호출: 서브태스크에 대해서 동일 규칙 적용
            logger.info(f"Start to recurse: {subgoal}")
            child = self.run(
                root_task=root_task,
                task=subgoal,
                tools=tools,
                difficulty_threshold=difficulty_threshold,
                depth=depth + 1,
                max_depth=max_depth,
                stop_on_first_failure=stop_on_first_failure,
            )
            logger.info(f"Recursioin ended {child}")

            results.append({"subtask": st, "result": child})
            if stop_on_first_failure and not child.get("ok"):
                logger.info(f"Recursioin Failed.")
                return {
                    "ok": False,
                    "task": task_to_solve,
                    "root_task": root_task,
                    "depth": depth,
                    "estimated_difficulty": difficulty,
                    "strategy": "decompose",
                    "planned_subtasks": planned,
                    "results": results,
                }

        logger.info(f"Recursioin Succeeded.")
        return {
            "ok": True,
            "task": task_to_solve,
            "root_task": root_task,
            "depth": depth,
            "estimated_difficulty": difficulty,
            "strategy": "decompose",
            "planned_subtasks": planned,
            "results": results,
            "rewriter": rewriter_meta,
        }
