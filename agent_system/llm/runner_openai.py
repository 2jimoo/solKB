from __future__ import annotations
import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from agent_system.kb import JsonlKB
from agent_system.tools.registry import ToolRegistry

class LLMRunnerWithTools:
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = OpenAI()
        self.model = model

    def run(
        self,
        messages: List[Dict[str, str]],
        tools: ToolRegistry,
        kb: Optional[JsonlKB] = None,
        node_id: Optional[str] = None,
        label: str = "llm_run",
        response_format: Optional[Dict[str, Any]] = None,
        max_steps: int = 8,
    ):
        resp = self.client.responses.create(
            model=self.model,
            input=messages,
            tools=tools.openai_tools,
            response_format=response_format,
        )

        if kb and node_id:
            kb.append({"event": label, "task_id": node_id, "stage": "start", "response_id": resp.id})

        for step in range(max_steps):
            tool_calls = []
            for item in (getattr(resp, "output", None) or []):
                t = getattr(item, "type", None) or item.get("type")
                if t == "tool_call":
                    tool_calls.append({
                        "id": getattr(item, "id", None) or item.get("id"),
                        "name": getattr(item, "name", None) or item.get("name"),
                        "arguments": getattr(item, "arguments", None) or item.get("arguments"),
                    })

            if not tool_calls:
                if kb and node_id:
                    kb.append({"event": label, "task_id": node_id, "stage": "done", "output_text": getattr(resp, "output_text", "")})
                return resp

            tool_results = []
            for tc in tool_calls:
                raw_args = tc["arguments"]
                if isinstance(raw_args, str):
                    args = json.loads(raw_args) if raw_args.strip() else {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}

                if kb and node_id:
                    kb.append({"event": "llm_tool_call", "task_id": node_id, "tool_name": tc["name"], "arguments": args})

                try:
                    out = tools.call(tc["name"], args)
                except Exception as e:
                    out = {"error": str(e), "tool": tc["name"], "args": args}

                if kb and node_id:
                    kb.append({"event": "llm_tool_result", "task_id": node_id, "tool_name": tc["name"], "output": out})

                tool_results.append({
                    "type": "tool_result",
                    "tool_call_id": tc["id"],
                    "output": json.dumps(out, ensure_ascii=False),
                })

            resp = self.client.responses.create(
                model=self.model,
                previous_response_id=resp.id,
                input=tool_results,
                tools=tools.openai_tools,
                response_format=response_format,
            )
            time.sleep(0.05)

        if kb and node_id:
            kb.append({"event": label, "task_id": node_id, "stage": "max_steps_reached", "output_text": getattr(resp, "output_text", "")})
        return resp
