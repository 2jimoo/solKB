from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, TypedDict


@dataclass
class TaskNode:
    task_id: str
    question: str
    depth: int
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # artifacts
    final_answer: Optional[str] = None
    slm_answer: Optional[str] = None
    expected_answer: Optional[str] = None
    llm_verdict: Optional[str] = None
    llm_reason: Optional[str] = None
    llm_evidence: List[str] = field(default_factory=list)

    status: str = "open"  # open|expanded|solved|failed
    solvable: int = 0  # solved via SLM (direct or subtree)
    max_derived_depth: int = 0  # subtree max depth (Solvability metric)


@dataclass
class SubtaskSpec:
    subtask: str
    rationale: str
    expected_answer: Optional[str] = None
    expected_tool: Optional[str] = None
    expected_tool_params: Optional[str] = None


@dataclass
class Verification:
    verdict: str  # "correct" | "incorrect" | "insufficient"
    reason: str
    evidence: List[str]


@dataclass
class SolveReference:
    reference_planning: Optional[str] = None
    failed_history: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.failed_history is None:
            self.failed_history = []


@dataclass
class ToolContribution:
    tool_name: Optional[str]
    tool_args: Optional[str]


class Subtask(TypedDict):
    subgoal: str
    rationale: str
    actions: List[str]


class TaskResult(TypedDict, total=False):
    task_id: str
    task: str
    subtasks: List[Subtask]
    total_score: float


class ExecutionResult(TypedDict, total=False):
    ok: bool
    output: Any
    error: str
    attempts: int
    depth: int
