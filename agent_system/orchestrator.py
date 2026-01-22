from __future__ import annotations
import uuid
from typing import Dict, List, Optional

from agent_system.kb import JsonlKB
from agent_system.models import TaskNode
from agent_system.tools.registry import ToolRegistry
from agent_system.llm.supervisor import LLMSupervisor
from agent_system.slm.runner_hf import SLMRunnerHF

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

    def _new_node(self, question: str, depth: int, parent_id: Optional[str]) -> TaskNode:
        node = TaskNode(task_id=str(uuid.uuid4()), question=question, depth=depth, parent_id=parent_id)
        self.nodes[node.task_id] = node
        self.kb.append({"event": "node_created", "node": node.__dict__})
        return node

    def _update(self, node: TaskNode, **updates) -> None:
        for k, v in updates.items():
            setattr(node, k, v)
        self.kb.append({"event": "node_updated", "task_id": node.task_id, "updates": updates})

    def _recompute_max_depth(self, node_id: str) -> int:
        node = self.nodes[node_id]
        if not node.children_ids:
            node.max_derived_depth = node.depth
            return node.max_derived_depth
        child_max = max(self._recompute_max_depth(cid) for cid in node.children_ids)
        node.max_derived_depth = child_max
        return child_max

    def run(self, root_question: str, reference_answer: Optional[str] = None) -> Dict[str, object]:
        root = self._new_node(root_question, depth=0, parent_id=None)
        self._solve_node(root, reference_answer=reference_answer)

        self._recompute_max_depth(root.task_id)
        self.kb.append({"event": "root_final", "root": self.nodes[root.task_id].__dict__})

        return {
            "root_id": root.task_id,
            "status": self.nodes[root.task_id].status,
            "final_answer": self.nodes[root.task_id].final_answer,
            "root_max_derived_depth": self.nodes[root.task_id].max_derived_depth,
            "root_solvable": self.nodes[root.task_id].solvable,
        }

    def _solve_node(self, node: TaskNode, reference_answer: Optional[str]) -> None:
        if node.depth >= self.max_depth:
            self._update(node, status="failed")
            return

        # 1) Decompose
        subtasks = self.supervisor.decompose(node.question, node.depth, self.tools, self.kb, node.task_id)
        self._update(node, status="expanded")
        self.kb.append({"event": "decomposed", "task_id": node.task_id, "subtasks": subtasks})

        solved_subtasks: List[Dict[str, str]] = []
        solved_count = 0

        # 2) Solve each subtask
        for st in subtasks:
            child = self._new_node(st, depth=node.depth + 1, parent_id=node.task_id)
            node.children_ids.append(child.task_id)
            self._update(node, children_ids=node.children_ids)

            # 2a) SLM solve with tools
            slm_res = self.slm.solve_with_tools(child.question, self.tools, kb=self.kb, node_id=child.task_id, max_tool_turns=6)
            slm_answer = (slm_res.get("answer") or "").strip()
            self._update(child, slm_answer=slm_answer)
            self.kb.append({"event": "slm_done", "task_id": child.task_id, "slm_res": slm_res})

            # 2b) LLM verify with tools
            v = self.supervisor.verify(child.question, slm_answer, self.tools, self.kb, child.task_id, reference_answer=None)
            self._update(child, llm_verdict=v["verdict"], llm_reason=v["reason"], llm_evidence=v["evidence"])
            self.kb.append({"event": "verified", "task_id": child.task_id, **v})

            if v["verdict"] == "correct":
                self._update(child, status="solved", solvable=1, final_answer=slm_answer)
                solved_subtasks.append({"subtask": child.question, "answer": slm_answer})
                solved_count += 1
                continue

            # 2c) Recurse if not solved
            if child.depth < self.max_depth - 1:
                self._solve_node(child, reference_answer=reference_answer)
                if self.nodes[child.task_id].status == "solved":
                    bubbled = (self.nodes[child.task_id].final_answer or self.nodes[child.task_id].slm_answer or "").strip()
                    solved_subtasks.append({"subtask": child.question, "answer": bubbled})
                    solved_count += 1
                    self._update(node, solvable=1)
                    continue

            self._update(child, status="failed")

        # 3) Synthesize final answer for this node
        final = self.supervisor.synthesize(node.question, solved_subtasks, self.tools, self.kb, node.task_id)
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
        self._update(node, llm_verdict=v_final["verdict"], llm_reason=v_final["reason"], llm_evidence=v_final["evidence"])
        self.kb.append({"event": "node_final_verified", "task_id": node.task_id, **v_final})

        # 5) Decide solved/failed
        if node.depth == 0 and reference_answer is not None:
            if v_final["verdict"] == "correct":
                self._update(node, status="solved", solvable=1)
            else:
                self._update(node, status="failed")
            return

        if self.require_all_subtasks:
            ok = (len(subtasks) > 0 and solved_count == len(subtasks))
        else:
            ok = (solved_count > 0)

        self._update(node, status="solved" if ok else "failed")
        if ok:
            self._update(node, solvable=1)
