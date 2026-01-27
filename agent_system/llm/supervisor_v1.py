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

    def root_decompose(
        self,
        reference_planning: str,
        tools: ToolRegistry,
        kb: JsonlKB,
        node_id: str,
    ):
        steps_schema = {
            "type": "json_schema",
            "name": "decomposition_steps",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 5,
                    }
                },
                "required": ["steps"],
                "additionalProperties": False,
            },
        }

        messages = [
            {
                "role": "system",
                "content": """
                    You are given a reference planning produced by a strong reasoning agent.
                    Your task is to generate a simplified and executable plan for a smaller task agent
                    that directly solves the root question.

                    Instructions:
                    - Return the output as a JSON array of 1 to 5 strings.
                    - Each string represents one planning step.
                    - Each step must be a complete sentence.
                    - Each step must be actionable (i.e., something an agent can directly execute).
                    - Do NOT include sub-bullets or nested steps.
                    - Do NOT restate the reference planning verbatim.
                    - Remove meta-reasoning, delegation details, and internal tool discussions unless strictly necessary.
                    - Preserve only the essential reasoning needed to solve the root question.
                    - Focus on WHAT to do, not HOW to reason internally.
                    - If multiple steps can be merged without losing clarity, merge them.

                    The final plan should:
                    - Use only information and constraints implied by the reference planning.
                    - Be sufficient for a smaller agent to reach the final answer.
                    - End with producing the final answer required by the root question.

                    Return ONLY the JSON array, and nothing else.
                """,
            },
            {
                "role": "user",
                "content": (
                    "Reference planning:\n\n" + reference_planning + "\n\n"
                    'Return JSON: {"steps": [...]} only.'
                ),
            },
        ]

        resp = self.runner.run(
            messages,
            tools,
            kb=kb,
            node_id=node_id,
            label="llm_decompose_from_reference_steps",
            schema=steps_schema,
        )
        logger.debug(
            f"supervisor decomposed (from reference -> steps) results:\n{resp.output_text}"
        )

        data = json.loads(resp.output_text)
        steps = [
            s.strip() for s in data.get("steps", []) if isinstance(s, str) and s.strip()
        ]
        return steps

    def decompose(
        self,
        root_question: str,
        question: str,
        depth: int,
        tools: ToolRegistry,
        kb: JsonlKB,
        node_id: str,
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
        ctx = kb.planner_context(node_id)

        messages = [
            {
                "role": "system",
                "content": (
                    "Make a plan to solve a task.\n"
                    "Use the planning/execution history to avoid repeating mistakes.\n"
                    "- If verification failed before, add subtasks that directly address the failure reason.\n"
                    "- Prefer evidence-gathering subtasks when prior attempts lacked proof.\n"
                    "- Do NOT repeat the same incorrect approach.\n"
                    "- If need, You can try again same parent subtask."
                    "Return 1-5 subtasks."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Depth={depth}\nOriginalTask:{root_question}\nCurrentTask:\n{question}\n"
                    + (f"\n\nPlanning/execution history:\n{ctx}\n" if ctx else "\n")
                    + "\nReturn 1-5 subtasks."
                ),
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
        logger.debug(f"supervisor decomposed results:\n{resp.output_text}")
        data = json.loads(resp.output_text)
        return [s.strip() for s in data["subtasks"] if s.strip()]

    def verify_final(
        self,
        root_question: str,
        proposed_answer: str,
        tools,
        kb,
        node_id: str,
        reference_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        # FINAL 전용 스키마
        schema = {
            "type": "json_schema",
            "name": "verification_final",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["final_correct", "incorrect", "insufficient"],
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

        # FINAL 전용 프롬프트/메시지
        messages = [
            {
                "role": "system",
                "content": """
                    You are a STRICT VERIFIER for another model's output.
                    Your role is ONLY to judge whether the model's answer is semantically equivalent to the actual FINAL answer.

                    You MAY call tools (serpapi_search, jina_read_url, calc) ONLY to CHECK whether the given answer is correct.

                    You MUST NOT solve the task yourself.
                    You MUST NOT provide the correct answer, partial answers, or hints toward the solution.
                    You MUST NOT suggest what should be done next.

                    Decision Criteria:
                    - final_correct: The model’s answer is semantically equivalent to the actual final answer (complete and correct).
                    - incorrect: The model’s answer is not equivalent to the actual final answer.
                    - insufficient: Not enough information to verify correctness.
                """.strip(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "root_question": root_question,
                        "final_model_answer": proposed_answer,
                        "actual_root_answer": reference_answer,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        resp = self.runner.run(
            messages,
            tools,
            kb=kb,
            node_id=node_id,
            label="llm_verify_final",
            schema=schema,
        )
        logger.debug(f"verify_final results:\n{resp.output_text}")
        return json.loads(resp.output_text)

    def verify_intermediate(
        self,
        root_question: str,
        subtask_question: str,
        proposed_answer: str,
        tools,
        kb,
        node_id: str,
        reference_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        # INTERMEDIATE 전용 스키마
        schema = {
            "type": "json_schema",
            "name": "verification_intermediate",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "final_correct",
                            "partial_correct",
                            "incorrect",
                            "insufficient",
                        ],
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

        # INTERMEDIATE 전용 프롬프트/메시지
        messages = [
            {
                "role": "system",
                "content": """
                    You are a STRICT VERIFIER for another model's output.
                    Your role is ONLY to judge whether the model's answer correctly solves the INTERMEDIATE subtask question.
                    Judge correctness for the subtask (not the overall root), using any provided reference if available.

                    You MAY call tools (serpapi_search, jina_read_url, calc) ONLY to CHECK whether the given answer is correct.

                    You MUST NOT solve the task yourself.
                    You MUST NOT provide the correct answer, partial answers, or hints toward the solution.
                    You MUST NOT suggest what should be done next.

                    Decision Criteria:
                    - final_correct: The intermediate answer is correct for the subtask (treat as "correct").
                    - partial_correct: The answer contains some correct intermediate results but is not fully correct.
                    - incorrect: The answer does not correctly solve the subtask.
                    - insufficient: Not enough information to verify correctness.
                """.strip(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "root_question": root_question,
                        "subtask_question": subtask_question,
                        "subtask_model_answer": proposed_answer,
                        "actual_root_answer": reference_answer,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        resp = self.runner.run(
            messages,
            tools,
            kb=kb,
            node_id=node_id,
            label="llm_verify_intermediate",
            schema=schema,
        )
        logger.debug(f"verify_intermediate results:\n{resp.output_text}")
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
                    "Synthesize a final answer from solved subtasks."
                    "Do NOT introduce new facts, assumptions, reasoning or tool calls."
                    "Do NOT include actual answer."
                    "Return only the answer to the question, such as the year, numerical value, or name."
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
        logger.debug(f"supervisor synthesized results:\n{resp.output_text}")
        return (resp.output_text or "").strip()

    def summarize_sibling_context_llm(
        self,
        root_question: str,
        current_subtask: str,
        tools: "ToolRegistry",
        kb: JsonlKB,
        parent_task_id: str,
        node_id_for_trace: str,
    ) -> str:
        raw = kb.collect_sibling_raw_logs(parent_task_id=parent_task_id)
        if not raw:
            return None

        schema = {
            "type": "json_schema",
            "name": "sibling_context_summary",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"context": {"type": "string"}},
                "required": ["context"],
                "additionalProperties": False,
            },
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You summarize prior sibling execution logs for a downstream small model.\n"
                    "Goal: extract ONLY the minimal facts, partial results, and failure-avoidance notes needed to solve CURRENT_SUBTASK.\n"
                    "Rules:\n"
                    "- Be short (<= 10 bullets or <= 900 chars).\n"
                    "- Prefer verified-correct facts and concrete entities/dates/URLs if present.\n"
                    "- If prior attempts failed, include 1-3 'avoid this mistake' notes.\n"
                    "- Do NOT include irrelevant history. Do NOT restate the whole plan.\n"
                    "Output JSON: {context: string}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "ROOT_QUESTION": root_question,
                        "CURRENT_SUBTASK": current_subtask,
                        "SIBLING_RAW_LOGS": raw,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        resp = self.runner.run(
            messages,
            tools,
            kb=kb,
            node_id=node_id_for_trace,
            label="llm_sibling_ctx_summary",
            schema=schema,
        )
        data = json.loads(resp.output_text or "{}")
        logger.debug(f"supervisor summarize previous result:\n{data}")
        return (data.get("context") or "").strip()
