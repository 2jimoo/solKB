from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, TypedDict, Union, Literal
from agent_system.models import SLMFailureRecord, RewriteResult

import logging

logger = logging.getLogger(__name__)


def _safe_json_loads(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response text")
    if isinstance(text, str):
        return {
            "status": "OK",
            "rewritten_subtask": text,
            "reason": "Model returned plain text; treated as rewritten_subtask",
        }
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(text[l : r + 1])
        raise


def slm_history_to_failure_logs(
    history: List[Dict[str, Any]],
) -> List[SLMFailureRecord]:
    out: List[SLMFailureRecord] = []
    for h in history or []:
        if not isinstance(h, dict):
            continue
        rec: SLMFailureRecord = {
            "attempt": int(h.get("attempt", 0)) if h.get("attempt") is not None else 0,
            "turn": int(h.get("turn", 0)) if h.get("turn") is not None else 0,
            "tool_name": h.get("tool_name"),
            "arguments": h.get("arguments"),
            "output": h.get("output"),
        }
        if isinstance(h.get("output"), dict) and "error" in h["output"]:
            rec["error"] = str(h["output"].get("error"))
        out.append(rec)
    return out


class SubtaskRewriter:
    """
    모델(LLMRunnerWithTools or SLMRunnerHF)을 교체 가능하게 만든 Subtask 변환기.
    - 출력은 "변환된 서브태스크 문자열"만(actions/subtask 분해 X)
    """

    def __init__(
        self,
        model: SubtaskRewriteModel,
        *,
        schema: Optional[Dict[str, Any]] = None,
        max_steps: int = 6,
        cooldown_sec: float = 0.15,
        max_failure_logs: int = 30,
    ):
        self.model = model
        self.schema = schema or self.default_schema()
        self.max_steps = max_steps
        self.cooldown_sec = cooldown_sec
        self.max_failure_logs = max_failure_logs

    @staticmethod
    def default_schema() -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "name": "subtask_rewrite",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": ["OK", "ABORT"]},
                    "rewritten_subtask": {
                        "anyOf": [{"type": "null"}, {"type": "string"}]
                    },
                    "reason": {"type": "string"},
                },
                "required": ["status", "rewritten_subtask", "reason"],
            },
            "strict": True,
        }

    def _build_messages(
        self,
        *,
        original_task: str,
        current_subtask: Subtask,
        failure_logs: List[SLMFailureRecord],
        guidance: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        system = (
            "You are a task wording optimizer.\n"
            "Rewrite CURRENT_SUBTASK into a more solvable, specific subtask statement.\n"
            "Return JSON only.\n\n"
            "Hard constraints:\n"
            "- Do NOT decompose into multiple subtasks.\n"
            "- Do NOT output actions/steps/checklists.\n"
            "- Output ONLY one rewritten subtask string.\n"
            "- Keep it 1~2 sentences.\n\n"
            "Use FAILURE_LOGS to adjust scope/constraints/prerequisites.\n"
            "If it is impossible due to permissions/dependencies, return ABORT with reason.\n"
        )
        if guidance:
            system += f"\nAdditional guidance:\n{guidance}\n"

        payload = {
            # "ORIGINAL_TASK": original_task,
            "CURRENT_SUBTASK": {
                "subgoal": current_subtask.get("subgoal", ""),
                "rationale": current_subtask.get("rationale", ""),
            },
            "FAILURE_LOGS": failure_logs[-self.max_failure_logs :],
        }

        user = (
            "Rewrite CURRENT_SUBTASK.\n"
            "Output JSON only with shape:\n"
            '{ "status": "OK"|"ABORT", "rewritten_subtask": "<string|null>", "reason": "<string>" }\n\n'
            f"INPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _validate(self, obj: Any) -> RewriteResult:
        if not isinstance(obj, dict):
            raise ValueError("not an object")
        status = obj.get("status")
        reason = obj.get("reason")
        rewritten = obj.get("rewritten_subtask")
        if status not in ("OK", "ABORT"):
            raise ValueError(f"bad status: {status}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("missing reason")
        if status == "OK":
            if not isinstance(rewritten, str) or not rewritten.strip():
                raise ValueError("OK requires rewritten_subtask string")
            # 과도하게 길면 잘라서 방어
            if len(rewritten) > 400:
                rewritten = rewritten[:400].rstrip()
            return {
                "status": "OK",
                "rewritten_subtask": rewritten.strip(),
                "reason": reason.strip(),
            }
        return {"status": "ABORT", "rewritten_subtask": None, "reason": reason.strip()}

    def rewrite(
        self,
        *,
        original_task: str,
        current_subtask: Subtask,
        failure_logs: List[SLMFailureRecord],
        guidance: Optional[str] = None,
        tools: Optional[Any] = None,
        kb: Optional[Any] = None,
        node_id: Optional[str] = None,
        label: str = "subtask_rewrite",
    ) -> RewriteResult:
        messages = self._build_messages(
            original_task=original_task,
            current_subtask=current_subtask,
            failure_logs=failure_logs,
            guidance=guidance,
        )

        raw = self.model.rewrite_subtask(
            messages=messages,
            tools=tools,
            schema=self.schema,
            max_steps=self.max_steps,
            label=label,
            kb=kb,
            node_id=node_id,
        )
        obj = _safe_json_loads(raw)
        res = self._validate(obj)

        if self.cooldown_sec:
            time.sleep(self.cooldown_sec)
        return res
