from __future__ import annotations
import uuid
from typing import Dict, List, Optional
from collections import defaultdict

from agent_system.kb import JsonlKB
from agent_system.models import TaskNode
from agent_system.tools.registry import ToolRegistry
from agent_system.llm.supervisor_v2 import LLMSupervisorV2
from agent_system.slm.runner_hf import SLMRunnerHF
from agent_system.models import SubtaskSpec, SolveReference, Verification

import logging

logger = logging.getLogger(__name__)


class RecursiveSolverV2:
    """
    컨텍스트: root_question X
    - SLM 입력에 넣는 것: parent_question + current_subtask + solved_siblings(Q/A)
    - Decompose에도 sibling_solved 전달 (품질 향상)
    - _solve_node returns max_derived_depth (solvability), 실패는 -1

    KB events(필수만):
      - plan
      - attempt
      - fail
      - record (all subtasks solved at that node)
    """

    def __init__(
        self,
        kb: JsonlKB,
        tools: ToolRegistry,
        supervisor: LLMSupervisorV2,
        slm: SLMRunnerHF,
        max_depth: int = 6,
        max_tool_turns: int = 6,
        max_slm_attempts: int = 10,
        require_all_subtasks: bool = True,
        keep_failed_history: int = 30,
        keep_sibling_ctx: int = 8,
    ):
        self.kb = kb
        self.tools = tools
        self.supervisor = supervisor
        self.slm = slm

        self.max_depth = max_depth
        self.max_tool_turns = max_tool_turns
        self.max_slm_attempts = max_slm_attempts
        self.require_all_subtasks = require_all_subtasks
        self.keep_failed_history = keep_failed_history
        self.keep_sibling_ctx = keep_sibling_ctx

        self._node_counter = 0

    def _new_node_id(self) -> str:
        self._node_counter += 1
        return str(self._node_counter)

    def _build_exec_question(
        self,
        parent_question: str,
        subtask: SubtaskSpec,
        sibling_solved: List[Dict[str, str]],
    ) -> str:
        current_subtask = subtask.subtask
        tail = sibling_solved[-self.keep_sibling_ctx :] if sibling_solved else []
        if not tail:
            return (
                f"Parent question:\n{parent_question}\n\n"
                f"Current subtask:\n{current_subtask}\n\n"
                f"Used Tool:{subtask.expected_tool}\nUsed Tool Parameter:{subtask.expected_tool_params}\n"
                "Answer ONLY the current subtask."
            )

        lines: List[str] = []
        for i, it in enumerate(tail, 1):
            q = (it.get("subtask") or "").strip()
            a = (it.get("answer") or "").strip()
            if q and a:
                lines.append(f"{i}. Q: {q}\n   A: {a}")

        ctx_block = "\n".join(lines).strip()
        if not ctx_block:
            return (
                f"Parent question:\n{parent_question}\n\n"
                f"Current subtask:\n{current_subtask}\n\n"
                f"Used Tool:{subtask.expected_tool}\nUsed Tool Parameter:{subtask.expected_tool_params}\n"
                "Answer ONLY the current subtask."
            )

        return (
            f"Parent question:\n{parent_question}\n\n"
            f"Current subtask:\n{current_subtask}\n\n"
            f"Used Tool:{subtask.expected_tool}\nUsed Tool Parameter:{subtask.expected_tool_params}\n"
            "Solved sibling subtasks (treat as facts; do NOT redo unless necessary):\n"
            f"{ctx_block}\n\n"
            "Answer ONLY the current subtask."
        )

    def run(
        self,
        root_question: str,
        reference_planning: Optional[str] = None,
        actual_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        ref = SolveReference(reference_planning=reference_planning, failed_history=[])
        solvability, solved_records, _ = self._solve_node(
            root_question=root_question,
            parent_question=root_question,
            reference=ref,
            expected_answer=actual_answer,
            actual_answer=actual_answer,
            depth=0,
            sibling_solved=[],  # root-level siblings context starts empty
        )
        return {
            "solved": solvability != -1,
            "solved_records": solved_records,
            "solvability": solvability,
            "failed_history": ref.failed_history,
        }

    def _solve_node(
        self,
        root_question: str,
        parent_question: str,
        reference: SolveReference,
        expected_answer: Optional[str],
        depth: int,
        sibling_solved: List[Dict[str, str]],
        actual_answer=None,
    ) -> (int, List[Any], bool):
        """
        Returns:
          -1 : failed
          >=0: solved -> max derived depth (solvability)
        """
        if depth >= self.max_depth:
            return -1, None, True

        logger.info(
            f"====================================== DEPTH {depth} ======================================"
        )
        node_id = self._new_node_id()

        # 1) Plan (decompose uses failed_history + sibling_solved)
        subtasks = self.supervisor.decompose(
            question=parent_question,
            tools=self.tools,
            kb=self.kb,
            node_id=node_id,
            reference=reference,
            actual_answer=actual_answer,
            depth=depth,
            is_root=(depth == 0),
            sibling_solved=sibling_solved,
        )
        if subtasks is None:
            logger.info(f"[{depth}] Superviser Couldn't Solve.")
            return -1, None, True

        self.kb.append(
            {
                "event": "plan",
                "node_id": node_id,
                "depth": depth,
                "question": parent_question,
                "subtasks": [
                    {"subtask": s.subtask, "expected_answer": s.expected_answer or ""}
                    for s in subtasks
                ],
            }
        )
        subtasks_str = "\n".join(
            [
                f"SubTask: {s.subtask}\nExpected Answer: {s.expected_answer}\n"
                for s in subtasks
            ]
        )
        logger.info(f"[{depth}] Plan:\n{subtasks_str}")

        solved_records: List[Dict[str, Any]] = []
        child_solvabilities: List[int] = []

        # local_siblings accumulates only within this node (this is what you want)
        local_siblings: List[Dict[str, str]] = list(
            sibling_solved[-self.keep_sibling_ctx :]
        )

        # 2) Execute each subtask
        for sidx, s in enumerate(subtasks):
            attempt_id = self._new_node_id()

            exec_q = self._build_exec_question(
                parent_question=parent_question,
                subtask=s,
                sibling_solved=local_siblings,
            )

            slm_attempts = 0
            slm_attempt_histories = []
            while slm_attempts < self.max_slm_attempts:
                slm_attempts += 1
                slm_res = self.slm.solve_with_tools(
                    exec_q,
                    self.tools,
                    kb=self.kb,
                    node_id=attempt_id,
                    max_tool_turns=self.max_tool_turns,
                    slm_attempt_histories=slm_attempt_histories,
                )
                proposed = (str(slm_res.get("answer") or "")).strip()
                logger.info(
                    f"[{depth}-{sidx}({slm_attempts})] SLM solve\n exec_q:\n{exec_q}\n\nProposed:{proposed}\nExpected:{s.expected_answer}"
                )

                v_st = self.supervisor.verify_semantic(
                    question=s.subtask,
                    proposed_answer=proposed,
                    actual_answer=s.expected_answer,
                    tools=self.tools,
                    kb=self.kb,
                    node_id=attempt_id,
                )
                logger.info(
                    f"[{depth}-{sidx}({slm_attempts})] LLM Verify SubTask\n verdict:{v_st.verdict}\n reason:{v_st.reason}\n evidence:{v_st.evidence}"
                )

                # Early stop
                v_final = self.supervisor.verify_semantic(
                    question=root_question,
                    proposed_answer=proposed,
                    actual_answer=actual_answer,
                    tools=self.tools,
                    kb=self.kb,
                    node_id=attempt_id,
                )
                logger.info(
                    f"[{depth}-{sidx}({slm_attempts})] LLM Verify Original Task\n verdict:{v_final.verdict}\n reason:{v_final.reason}\n evidence:{v_final.evidence}"
                )
                if v_st.verdict == "correct" or v_final.verdict == "correct":
                    break
                slm_attempt_histories.append(
                    {
                        "proposed": proposed,
                        "verdict": v_st.verdict,
                        "reason": v_st.reason,
                        "evidence": v_st.evidence,
                    }
                )

            self.kb.append(
                {
                    "event": "attempt",
                    "node_id": attempt_id,
                    "parent_node_id": node_id,
                    "depth": depth + 1,
                    "parent_question": parent_question,
                    "subtask": s.subtask,
                    "expected_answer": s.expected_answer or "",
                    "proposed_answer": proposed,
                    "verdict": v_st.verdict,
                    "reason": v_st.reason,
                    "evidence": v_st.evidence,
                }
            )

            if v_st.verdict == "correct":
                logger.info(f"[{depth}-{sidx}] SLM Corrected Partially.")
                local_siblings.append({"subtask": s.subtask, "answer": proposed})
                local_siblings = local_siblings[-self.keep_sibling_ctx :]

                solved_records.append(
                    {
                        "subtask": s.subtask,
                        "answer": s.expected_answer,
                        "depth": depth + 1,
                    }
                )
                child_solvabilities.append(depth + 1)
                continue
            # Early stop
            if v_final.verdict == "correct":
                logger.info(f"[{depth}-{sidx}] SLM Corrected Ultaimately.")
                max_derived_depth = max([depth] + child_solvabilities)
                solved_records.append(
                    {
                        "subtask": s.subtask,
                        "answer": s.expected_answer,
                        "depth": depth + 1,
                    }
                )
                return max_derived_depth, solved_records, True

            # 3) fail + recurse
            reference.failed_history.append(
                {"task": s.subtask, "reason": v_st.reason or v_st.verdict}
            )
            if len(reference.failed_history) > self.keep_failed_history:
                reference.failed_history = reference.failed_history[
                    -self.keep_failed_history :
                ]

            self.kb.append(
                {
                    "event": "recurse",
                    "node_id": attempt_id,
                    "parent_node_id": node_id,
                    "depth": depth + 1,
                    "subtask": s.subtask,
                    "reason": v_st.reason or v_st.verdict,
                }
            )

            child_solvability, _, is_finished = self._solve_node(
                root_question=root_question,
                parent_question=s.subtask,
                reference=reference,
                expected_answer=s.expected_answer,
                depth=depth + 1,
                # ✅ 재귀로 들어갈 때도 “현재까지 solved된 형제 컨텍스트”를 전달하면 안정적
                sibling_solved=local_siblings,
            )
            logger.info(
                f"[{depth}-{sidx}] SLM failed and recursed({child_solvability})."
            )

            # Early stopped
            if is_finished:
                solved_records.append(
                    {
                        "subtask": s.subtask,
                        "answer": s.expected_answer,
                        "depth": depth + 1,
                    }
                )
                child_solvabilities.append(child_solvability)
                max_derived_depth = max([depth] + child_solvabilities)
                return max_derived_depth, solved_records, True

            if child_solvability != -1:
                local_siblings.append(
                    {"subtask": s.subtask, "answer": s.expected_answer}
                )
                local_siblings = local_siblings[-self.keep_sibling_ctx :]

                solved_records.append(
                    {
                        "subtask": s.subtask,
                        "answer": s.expected_answer,
                        "depth": depth + 1,
                    }
                )
                child_solvabilities.append(child_solvability)
                continue

            if self.require_all_subtasks:
                return -1, None, True

        # 4) Decide success and compute solvability
        if self.require_all_subtasks:
            ok = len(subtasks) > 0 and len(solved_records) == len(subtasks)
        else:
            ok = len(solved_records) > 0

        if not ok:
            return -1

        max_derived_depth = max([depth] + child_solvabilities)

        # 5) record (only when all subtasks solved at this node)
        self.kb.append(
            {
                "event": "done",
                "node_id": node_id,
                "depth": depth,
                "question": parent_question,
                "records": solved_records,
                "solvability": max_derived_depth,
            }
        )

        return max_derived_depth, solved_records, True
