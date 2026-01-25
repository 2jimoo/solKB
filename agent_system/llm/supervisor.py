from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from agent_system.kb import JsonlKB
from agent_system.tools.registry import ToolRegistry
from agent_system.llm.runner_openai import LLMRunnerWithTools

import logging

logger = logging.getLogger(__name__)


class LLMSupervisor:
    def __init__(self, runner: LLMRunnerWithTools):
        self.runner = runner

    def decompose(
        self, question: str, depth: int, tools: ToolRegistry, kb: JsonlKB, node_id: str
    ) -> List[str]:
        schema = {
            "type": "json_schema",
            "name": "decomposition",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "subtasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 5,
                    }
                },
                "required": ["subtasks"],
                "additionalProperties": False,
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "Decompose into smaller subtasks that a small model can solve. "
                    "If facts are needed, include fact-finding subtasks (identify entity, read sources, extract dates)."
                ),
            },
            {
                "role": "user",
                "content": f"Depth={depth}\nTask:\n{question}\nReturn 1-5 subtasks.",
            },
        ]
        resp = self.runner.run(
            messages,
            tools,
            kb=kb,
            node_id=node_id,
            label="llm_decompose",
            schema=schema,
        )
        logger.info(f"supervisor decomposed results: {resp.output_text}")
        data = json.loads(resp.output_text)
        return [s.strip() for s in data["subtasks"] if s.strip()]

    def verify(
        self,
        question: str,
        proposed_answer: str,
        tools: ToolRegistry,
        kb: JsonlKB,
        node_id: str,
        reference_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        schema = {
            "type": "json_schema",
            "name": "verification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["correct", "incorrect", "insufficient"],
                    },
                    "reason": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 6,
                    },
                },
                "required": ["verdict", "reason", "evidence"],
                "additionalProperties": False,
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict verifier. You MAY call tools (serpapi_search, jina_read_url, calc) to check facts. "
                    "Return verdict + reason + 1-6 short evidence bullets (URLs or extracted facts)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "proposed_answer": proposed_answer,
                        "reference_answer": reference_answer,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        resp = self.runner.run(
            messages, tools, kb=kb, node_id=node_id, label="llm_verify", schema=schema
        )
        logger.info(f"supervisor verifying results: {resp.output_text}")
        return json.loads(resp.output_text)

    def synthesize(
        self,
        question: str,
        solved_subtasks: List[Dict[str, str]],
        tools: ToolRegistry,
        kb: JsonlKB,
        node_id: str,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Synthesize a final answer from solved subtasks. Call tools if you need to double-check. Be concise."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "solved_subtasks": solved_subtasks},
                    ensure_ascii=False,
                ),
            },
        ]
        resp = self.runner.run(
            messages, tools, kb=kb, node_id=node_id, label="llm_synthesize", schema=None
        )
        logger.info(f"supervisor synthesized results: {resp.output_text}")
        return (resp.output_text or "").strip()
