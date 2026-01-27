from __future__ import annotations
import uuid
from typing import Dict, List, Optional
from collections import defaultdict

from agent_system.kb import JsonlKB
from agent_system.models import TaskNode
from agent_system.tools.registry import ToolRegistry
from agent_system.llm import LLMSupervisor
from agent_system.slm.runner_hf import SLMRunnerHF

import logging

logger = logging.getLogger(__name__)


class RecursiveSolver:
    def __init__(
        self,
        kb: JsonlKB,
        tools: ToolRegistry,
        supervisor: LLMSupervisor,
        slm: SLMRunnerHF,
        max_depth: int = 6,
        require_all_subtasks: bool = False,
    ):
        self.kb = kb
        self.tools = tools
        self.supervisor = supervisor
        self.slm = slm
        self.max_depth = max_depth
        self.require_all_subtasks = require_all_subtasks
        self.nodes: Dict[str, TaskNode] = {}
        self.id_counters: Dict[str, int] = defaultdict(int)

    def _new_node(
        self, question: str, depth: int, parent_id: Optional[str]
    ) -> TaskNode:
        # 계층적 ID 생성:
        # - 루트: "1", "2", ...
        # - 자식: "<parent>.<n>" (예: "1.1", "1.2")
        if parent_id is None:
            # 루트 노드
            self.id_counters["ROOT"] += 1
            node_id = str(self.id_counters["ROOT"])
        else:
            # 부모의 카운터 증가 후 접두사로 사용
            self.id_counters[parent_id] += 1
            node_id = f"{parent_id}.{self.id_counters[parent_id]}"

        node = TaskNode(
            task_id=node_id,
            question=question,
            depth=depth,
            parent_id=parent_id,
        )
        self.nodes[node.task_id] = node
        self.kb.append({"event": "node_created", "node": node.__dict__})
        return node

    def _update(self, node: TaskNode, **updates) -> None:
        for k, v in updates.items():
            setattr(node, k, v)
        self.kb.append(
            {"event": "node_updated", "task_id": node.task_id, "updates": updates}
        )

    def _recompute_max_depth(self, node_id: str) -> int:
        node = self.nodes[node_id]
        if not node.children_ids:
            node.max_derived_depth = node.depth
            return node.max_derived_depth
        child_max = max(self._recompute_max_depth(cid) for cid in node.children_ids)
        node.max_derived_depth = child_max
        return child_max

    def run(
        self,
        root_question: str,
        reference_planning: str,
        reference_answer: Optional[str] = None,
    ) -> Dict[str, object]:
        root = self._new_node(root_question, depth=0, parent_id=None)
        self._solve_node(
            root_question,
            root,
            reference_planning=reference_planning,
            reference_answer=reference_answer,
        )

        self._recompute_max_depth(root.task_id)
        self.kb.append(
            {"event": "root_final", "root": self.nodes[root.task_id].__dict__}
        )

        return {
            "root_id": root.task_id,
            "status": self.nodes[root.task_id].status,
            "final_answer": self.nodes[root.task_id].final_answer,
            "root_max_derived_depth": self.nodes[root.task_id].max_derived_depth,
            "root_solvable": self.nodes[root.task_id].solvable,
        }

    def _solve_node(
        self,
        root_question: str,
        node: TaskNode,
        reference_planning: str,
        reference_answer: Optional[str],
    ) -> None:
        logger.info(
            f"============================================ START {node.task_id} ============================================"
        )
        if node.depth >= self.max_depth:
            self._update(node, status="failed")
            return

        # 1) Decompose
        if node.depth == 0:
            subtasks = self.supervisor.root_decompose(
                reference_planning,
                self.tools,
                self.kb,
                node.task_id,
            )
        else:
            subtasks = self.supervisor.decompose(
                root_question,
                node.question,
                node.depth,
                self.tools,
                self.kb,
                node.task_id,
            )
        logger.info(f"LLM tried to decompose:\n {subtasks}")
        self._update(node, status="expanded")
        self.kb.append(
            {"event": "decomposed", "task_id": node.task_id, "subtasks": subtasks}
        )

        solved_subtasks: List[Dict[str, str]] = []
        solved_count = 0

        # 2) Solve each subtask
        for st_idx, st in enumerate(subtasks):
            logger.info(f"{node.depth}-{st_idx} Task: {st} ")
            child = self._new_node(st, depth=node.depth + 1, parent_id=node.task_id)
            node.children_ids.append(child.task_id)
            self._update(node, children_ids=node.children_ids)

            # 2a) Summary for SLM
            sibling_ctx = self.supervisor.summarize_sibling_context_llm(
                root_question=root_question,
                current_subtask=child.question,
                tools=self.tools,
                kb=self.kb,
                parent_task_id=node.task_id,
                node_id_for_trace=child.task_id,
            )
            if sibling_ctx:
                self.kb.append(
                    {
                        "event": "sibling_ctx_summary",
                        "task_id": child.task_id,
                        "parent_task_id": node.task_id,
                        "summary": sibling_ctx,
                    }
                )
            exec_question = child.question
            if sibling_ctx:
                logger.info(f"LLM summerized history:\n{sibling_ctx}")
                exec_question = (
                    f"{child.question}\n\n"
                    f"[Sibling context]\n{sibling_ctx}\n\n"
                    "Use the sibling context above. Do not redo solved facts unless necessary."
                )

            # 2b) SLM solve with tools
            slm_res = self.slm.solve_with_tools(
                exec_question,
                self.tools,
                kb=self.kb,
                node_id=child.task_id,
                max_tool_turns=6,
            )
            slm_answer = (str(slm_res.get("answer")) or "").strip()
            logger.info(f"SLM tried to solve:\n {slm_answer}")
            self._update(child, slm_answer=slm_answer)
            self.kb.append(
                {"event": "slm_done", "task_id": child.task_id, "slm_res": slm_res}
            )

            # 2b-1) LLM try to stop early
            temp_solved_subtasks = solved_subtasks + [
                {"subtask": child.question, "answer": slm_answer}
            ]
            early_final_answer = self.supervisor.synthesize(
                root_question,
                temp_solved_subtasks,
                self.tools,
                self.kb,
                node.task_id,
            )
            v_final = self.supervisor.verify(
                root_question,
                early_final_answer,
                self.tools,
                self.kb,
                node.task_id,
                reference_answer=reference_answer,
            )
            logger.info(
                f"LLM tried to stop: {early_final_answer}, {v_final['verdict']}\n-> {v_final['reason']}"
            )
            if v_final["verdict"] == "correct":
                self._update(
                    node,
                    status="solved",
                    solvable=1,
                    final_answer=early_final_answer,
                    llm_verdict=v_final["verdict"],
                    llm_reason=v_final["reason"],
                    llm_evidence=v_final["evidence"],
                )
                logger.info(
                    f"============================================ END {node.task_id} ============================================"
                )
                return

            # 2b-2) LLM verify with tools
            v = self.supervisor.verify(
                exec_question,
                slm_answer,
                self.tools,
                self.kb,
                child.task_id,
                reference_answer=None,
            )
            logger.info(
                f"LLM tried to verify subtask: {v['verdict']}\n-> {v['reason']}"
            )
            self._update(
                child,
                llm_verdict=v["verdict"],
                llm_reason=v["reason"],
                llm_evidence=v["evidence"],
            )
            self.kb.append({"event": "verified", "task_id": child.task_id, **v})

            if v["verdict"] == "correct":
                self._update(
                    child, status="solved", solvable=1, final_answer=slm_answer
                )
                solved_subtasks.append(
                    {"subtask": child.question, "answer": slm_answer}
                )
                solved_count += 1
                logger.info(f"{node.depth}-{st_idx} Task corrected directly. ")
                continue

            # 2c) Recurse if not solved
            if child.depth < self.max_depth - 1:
                self._solve_node(
                    root_question,
                    child,
                    reference_planning=None,
                    reference_answer=reference_answer,
                )
                if self.nodes[child.task_id].status == "solved":
                    bubbled = (
                        self.nodes[child.task_id].final_answer
                        or self.nodes[child.task_id].slm_answer
                        or ""
                    ).strip()
                    solved_subtasks.append(
                        {"subtask": child.question, "answer": bubbled}
                    )
                    solved_count += 1
                    self._update(node, solvable=1)
                    logger.info(f"{node.depth}-{st_idx} Task corrected recursively. ")
                    continue

            self._update(child, status="failed")
            logger.info(f"{node.depth}-{st_idx} Task failed. ")
            break

        # 3) Synthesize final answer for this node
        final = self.supervisor.synthesize(
            node.question, solved_subtasks, self.tools, self.kb, node.task_id
        )
        self._update(node, final_answer=final)

        # 4) Verify final (apply reference only at root if provided)
        v_final = self.supervisor.verify(
            node.question,
            final,
            self.tools,
            self.kb,
            node.task_id,
            reference_answer=(reference_answer if node.depth == 0 else None),
        )
        logger.info(
            f"LLM tried to verify task: {v_final['verdict']}\n-> {v_final['reason']}"
        )
        self._update(
            node,
            llm_verdict=v_final["verdict"],
            llm_reason=v_final["reason"],
            llm_evidence=v_final["evidence"],
        )
        self.kb.append(
            {"event": "node_final_verified", "task_id": node.task_id, **v_final}
        )

        # 5) Decide solved/failed
        if node.depth == 0 and reference_answer is not None:
            if v_final["verdict"] == "correct":
                self._update(node, status="solved", solvable=1)
            else:
                self._update(node, status="failed")
            logger.info(
                f"============================================ END {node.task_id} ============================================"
            )
            return

        if self.require_all_subtasks:
            ok = len(subtasks) > 0 and solved_count == len(subtasks)
        else:
            ok = solved_count > 0

        self._update(node, status="solved" if ok else "failed")
        if ok:
            self._update(node, solvable=1)

        logger.info(
            f"============================================ END {node.task_id} ============================================"
        )
