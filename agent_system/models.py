from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


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
    llm_verdict: Optional[str] = None
    llm_reason: Optional[str] = None
    llm_evidence: List[str] = field(default_factory=list)

    status: str = "open"  # open|expanded|solved|failed
    solvable: int = 0  # solved via SLM (direct or subtree)
    max_derived_depth: int = 0  # subtree max depth (Solvability metric)
