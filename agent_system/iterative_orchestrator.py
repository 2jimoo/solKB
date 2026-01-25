from __future__ import annotations
import uuid
from typing import Dict, List, Optional
from collections import defaultdict

from agent_system.kb import JsonlKB
from agent_system.models import TaskNode
from agent_system.tools.registry import ToolRegistry
from agent_system.llm.supervisor import LLMSupervisor
from agent_system.slm.runner_hf import SLMRunnerHF

import logging

logger = logging.getLogger(__name__)


class IterativeSolver:
    """
    - Queue 누적 X
    - 항상 "현재 상태" -> LLM이 '새 계획(plan)'을 다시 수립
    - SLM은 그 plan의 step을 순서대로 실행
    - 실패하면: 남은 plan 버리고, 실패 로그 포함해 LLM이 새 plan 수립
    - successes(검증 통과한 SLM 결과)만 누적
    """

    def __init__(
        self,
        kb: JsonlKB,
        tools: ToolRegistry,
        supervisor: LLMSupervisor,
        slm: SLMRunnerHF,
        max_depth: int = 6,
        max_iter: int = 80,
        max_tool_turns: int = 6,
        max_replans: int = 12,
        early_stop: bool = True,  # successes로 루트 답 합성 후 맞으면 즉시 종료
    ):
        self.kb = kb
        self.tools = tools
        self.supervisor = supervisor
        self.slm = slm
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.max_tool_turns = max_tool_turns
        self.max_replans = max_replans
        self.early_stop = early_stop

        self.id_counters: Dict[str, int] = defaultdict(int)

    def _new_id(self, parent_id: Optional[str]) -> str:
        if parent_id is None:
            self.id_counters["ROOT"] += 1
            return str(self.id_counters["ROOT"])
        self.id_counters[parent_id] += 1
        return f"{parent_id}.{self.id_counters[parent_id]}"

    def _make_state_prompt(
        self,
        root_question: str,
        successes: List[Dict[str, str]],
    ) -> str:
        """
        LLM이 '현재 상태'를 보고 새 계획을 만들 수 있게 하는 상태 요약 프롬프트.
        """
        solved_lines = []
        for i, s in enumerate(successes, 1):
            solved_lines.append(f"{i}. {s['subtask']} -> {s['answer']}")

        solved_block = "\n".join(solved_lines) if solved_lines else "(none)"

        histories = self.kb.read_events_by_type(event_type="slm_done")
        return (
            f"[Root question]\n{root_question}\n\n"
            f"[Solved facts/steps so far (verified correct)]\n{solved_block}\n\n"
            f"[Planning/execution history]\n{histories}\n\n"
            "Now propose a NEW plan: a short ordered list of subtasks that "
            "moves closer to answering the root question, using available tools. "
            "Do NOT include already-solved steps unless strictly necessary. "
            "Make steps specific and verifiable."
        )

    def run(
        self,
        root_question: str,
        reference_planning: str,
        reference_answer: Optional[str] = None,
    ) -> Dict[str, object]:
        root_id = self._new_id(None)
        logger.info(
            f"============================================ START ROOT {root_id} ============================================"
        )
        logger.info(f"[ROOT] question: {root_question}")
        self.kb.append(
            {"event": "root_created", "root_id": root_id, "question": root_question}
        )

        # 성공한 step만 누적 (요청사항)
        successes: List[Dict[str, str]] = []

        # 0) 초기 계획(레퍼런스 플래닝 기반)
        logger.info("[ROOT] initial planning via root_decompose ...")
        plan = self.supervisor.root_decompose(
            reference_planning, self.tools, self.kb, root_id
        )
        logger.info(f"[ROOT] initial plan steps={len(plan)}")
        logger.info(f"[ROOT] plan:\n{plan}")
        self.kb.append(
            {"event": "plan_created", "root_id": root_id, "plan": plan, "replan_idx": 0}
        )

        replans_used = 0
        it = 0

        # 1) while loop
        while it < self.max_iter:
            it += 1

            if not plan:
                # 계획이 비었으면: 현재 상태 기반으로 재계획
                if replans_used >= self.max_replans:
                    logger.info(
                        f"[ROOT] stop: max_replans reached ({replans_used}/{self.max_replans})"
                    )
                    break

                replans_used += 1
                state_prompt = self._make_state_prompt(root_question, successes)
                logger.info(f"[ROOT] replan #{replans_used} (plan was empty) ...")
                plan = self.supervisor.decompose(
                    root_question,
                    state_prompt,
                    depth=0,
                    tools=self.tools,
                    kb=self.kb,
                    node_id=root_id,
                )
                logger.info(f"[ROOT] new plan steps={len(plan)}")
                logger.info(f"[ROOT] new plan:\n{plan}")
                self.kb.append(
                    {
                        "event": "replanned",
                        "root_id": root_id,
                        "replan_idx": replans_used,
                        "plan": plan,
                        "state_prompt": state_prompt,
                    }
                )
                continue

            # 2) 현재 plan의 다음 step 하나만 수행 (Queue처럼 쌓지 않음)
            step = plan.pop(0)
            step_id = self._new_id(root_id)

            logger.info(
                f"============================================ START {step_id} (iter={it}) ============================================"
            )
            logger.info(f"[{step_id}] step: {step}")
            self.kb.append(
                {
                    "event": "step_started",
                    "task_id": step_id,
                    "step": step,
                    "iter": it,
                    "replan_idx": replans_used,
                }
            )

            # depth guard (여기서는 plan-step을 depth=1로 취급하거나, 재계획 반복 횟수로 대체)
            # 필요하면 step별 depth를 별도로 계산 가능.
            if replans_used >= self.max_depth:
                logger.info(
                    f"[{step_id}] dropped: exceeded max_depth-by-replan (replans_used={replans_used})"
                )
                self.kb.append({"event": "dropped_depth", "task_id": step_id})
                logger.info(
                    f"============================================ END {step_id} ============================================"
                )
                break

            exec_question = step
            if successes:
                solved_lines = "\n".join(
                    [f"- {s['subtask']} -> {s['answer']}" for s in successes[-8:]]
                )
                exec_question = (
                    f"{step}\n\n"
                    f"[Already solved (verified)]\n{solved_lines}\n\n"
                    "Use the solved info above. Do not redo those unless necessary."
                )

            logger.info(
                f"[{step_id}] SLM solve_with_tools (max_tool_turns={self.max_tool_turns}) ..."
            )
            slm_res = self.slm.solve_with_tools(
                exec_question,
                self.tools,
                kb=self.kb,
                node_id=step_id,
                max_tool_turns=self.max_tool_turns,
            )
            slm_answer = (str(slm_res.get("answer")) or "").strip()
            logger.info(f"[{step_id}] SLM answer:\n{slm_answer}")
            self.kb.append(
                {
                    "event": "slm_done",
                    "task_id": step_id,
                    "exec_question": exec_question,
                    "slm_res": slm_res,
                }
            )
            logger.info(
                f"[{step_id}] early_stop check: synthesize root answer from successes ..."
            )
            candidate = self.supervisor.synthesize(
                root_question,
                slm_answer,
                self.tools,
                self.kb,
                root_id,
            )
            logger.info(f"[{step_id}] early_stop candidate: {candidate}")
            v_root = self.supervisor.verify(
                root_question,
                candidate,
                self.tools,
                self.kb,
                root_id,
                reference_answer=reference_answer,
            )
            logger.info(
                f"[ROOT] early_stop verdict={v_root.get('verdict')} reason={v_root.get('reason')}"
            )
            self.kb.append(
                {"event": "early_stop_checked", "root_id": root_id, **v_root}
            )
            if v_root.get("verdict") == "final_correct":
                logger.info("[ROOT] ✅ early_stop success")
                final_answer = candidate
                self.kb.append(
                    {
                        "event": "root_final",
                        "root_id": root_id,
                        "final_answer": final_answer,
                    }
                )
                logger.info(
                    f"============================================ END ROOT {root_id} ============================================"
                )
                return {
                    "root_id": root_id,
                    "status": "solved",
                    "iters_used": it,
                    "replans_used": replans_used,
                    "successful_plan": successes,
                    "final_answer": final_answer,
                }

            logger.info(f"[{step_id}] LLM verify step ...")
            v = self.supervisor.verify(
                exec_question,
                slm_answer,
                self.tools,
                self.kb,
                step_id,
                reference_answer=None,
            )
            logger.info(
                f"[{step_id}] verify verdict={v.get('verdict')} reason={v.get('reason')}"
            )
            self.kb.append({"event": "verified", "task_id": step_id, **v})

            if v.get("verdict") == "partial_correct":
                successes.append({"subtask": step, "answer": slm_answer})
                self.kb.append({"event": "step_solved", "task_id": step_id})
                logger.info(
                    f"[{step_id}] ✅ step solved({v_root.get('verdict')}); successes={len(successes)}"
                )
                logger.info(
                    f"============================================ END {step_id} ============================================"
                )
                continue
            if v_root.get("verdict") == "final_correct":
                logger.info(
                    f"[{step_id}] ✅ step solved({v_root.get('verdict')}); successes={len(successes)}"
                )
                final_answer = slm_answer
                self.kb.append(
                    {
                        "event": "root_final",
                        "root_id": root_id,
                        "final_answer": final_answer,
                    }
                )
                logger.info(
                    f"============================================ END ROOT {root_id} ============================================"
                )
                return {
                    "root_id": root_id,
                    "status": "solved",
                    "iters_used": it,
                    "replans_used": replans_used,
                    "successful_plan": successes,
                    "final_answer": final_answer,
                }

            # 실패: 남은 plan 버리고 재계획
            last_failure = {
                "subtask": step,
                "slm_answer": slm_answer,
                "verifier_reason": v.get("reason"),
                "verifier_evidence": v.get("evidence"),
            }
            self.kb.append({"event": "step_failed", "task_id": step_id, **last_failure})
            logger.info(
                f"[{step_id}] ❌ step failed -> discard remaining plan ({len(plan)} steps) and replan"
            )

            plan = []  # ✅ 남은 계획 버림 (Queue 누적 X)

            logger.info(
                f"============================================ END {step_id} ============================================"
            )

        # 루프 종료 후: successes 기반으로 최종 답(옵션) 만들기
        logger.info(
            f"============================================ FINISH ROOT {root_id} (iters_used={it}, replans_used={replans_used}) ============================================"
        )
        logger.info(
            f"[ROOT] successful_plan_len={len(successes)} plan_left={len(plan)}"
        )

        final_answer = None
        if successes:
            logger.info("[ROOT] synthesize final answer from successful_plan ...")
            final_answer = self.supervisor.synthesize(
                root_question,
                successes,
                self.tools,
                self.kb,
                root_id,
            )
            logger.info(f"[ROOT] final_answer:\n{final_answer}")
            self.kb.append(
                {
                    "event": "root_synthesized",
                    "root_id": root_id,
                    "final_answer": final_answer,
                }
            )

        status = "solved" if final_answer else "failed"
        logger.info(
            f"============================================ END ROOT {root_id} ============================================"
        )

        return {
            "root_id": root_id,
            "status": status,
            "iters_used": it,
            "replans_used": replans_used,
            "successful_plan": successes,  # ✅ 누적되는 건 이것뿐
            "final_answer_optional": final_answer,
            "last_failure_optional": last_failure,
        }
