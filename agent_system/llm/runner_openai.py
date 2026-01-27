from __future__ import annotations

from agent_system.kb import JsonlKB
from agent_system.tools.registry import ToolRegistry
import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

import logging

logger = logging.getLogger(__name__)


class LLMRunnerWithTools:
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model

    def run(
        self,
        messages: List[Dict[str, str]],
        tools,
        kb: Optional[Any] = None,
        node_id: Optional[str] = None,
        label: str = "llm_run",
        schema: Optional[Dict[str, Any]] = None,
        max_steps: int = 8,
    ):
        executed_tool_calls: List[Dict[str, Any]] = []
        # --- first call ---
        kwargs = dict(
            model=self.model,
            input=messages,
            tools=tools.openai_tools,
            text={"format": schema} if schema else None,
        )
        resp = self.client.responses.create(**kwargs)
        logger.debug(f"LLM tried to run.\n-> resp: {resp}")
        if kb and node_id:
            kb.append(
                {
                    "event": label,
                    "task_id": node_id,
                    "stage": "start",
                    "response_id": resp.id,
                }
            )

        # --- tool loop ---
        for step in range(max_steps):
            tool_calls = []

            output_items = getattr(resp, "output", None) or []
            for item in output_items:
                # item can be a pydantic-ish object or a dict
                if isinstance(item, dict):
                    t = item.get("type")
                    name = item.get("name")
                    arguments = item.get("arguments")
                    item_id = item.get("id")
                    call_id = item.get("call_id")
                else:
                    t = getattr(item, "type", None)
                    name = getattr(item, "name", None)
                    arguments = getattr(item, "arguments", None)
                    item_id = getattr(item, "id", None)
                    call_id = getattr(item, "call_id", None)

                # Responses API commonly uses "function_call"
                if t in ("tool_call", "function_call"):
                    tool_calls.append(
                        {
                            "id": item_id,
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments,
                        }
                    )

            # no tool calls => done
            if not tool_calls:
                try:
                    setattr(resp, "_executed_tool_calls", executed_tool_calls)
                except Exception:
                    pass

                if kb and node_id:
                    kb.append(
                        {
                            "event": label,
                            "task_id": node_id,
                            "stage": "done",
                            "output_text": getattr(resp, "output_text", "") or "",
                        }
                    )
                return resp

            # execute tool calls and send tool_results
            tool_results = []
            for tc in tool_calls:
                raw_args = tc.get("arguments")

                # arguments can be a JSON string or dict
                if isinstance(raw_args, str):
                    raw_args = raw_args.strip()
                    args = json.loads(raw_args) if raw_args else {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}

                if kb and node_id:
                    kb.append(
                        {
                            "event": "llm_tool_call",
                            "task_id": node_id,
                            "tool_name": tc.get("name"),
                            "arguments": args,
                        }
                    )

                try:
                    out = tools.call(tc["name"], args)
                except Exception as e:
                    out = {"error": str(e), "tool": tc.get("name"), "args": args}
                logger.debug(
                    f"LLM tried to call tool ({tc['name']}) with parameters ({args}).\n-> res: {out}"
                )

                if kb and node_id:
                    kb.append(
                        {
                            "event": "llm_tool_result",
                            "task_id": node_id,
                            "tool_name": tc.get("name"),
                            "output": out,
                        }
                    )

                # IMPORTANT: tool_call_id should match call_id if present (your log shows call_id='call_...')
                executed_tool_calls.append(
                    {
                        "name": tc.get("name"),
                        "arguments": args,
                    }
                )
                tool_results.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.get("call_id") or tc.get("id"),
                        "output": json.dumps(out, ensure_ascii=False),
                    }
                )
            # follow-up call (continue the same response thread)
            kwargs2 = dict(
                model=self.model,
                previous_response_id=resp.id,
                input=tool_results,
                tools=tools.openai_tools,
                text={"format": schema} if schema else None,
            )
            resp = self.client.responses.create(**kwargs2)
            logger.debug(f"LLM tried to run with tool results.\n-> resp: {resp}")
            time.sleep(0.5)

        # if max steps reached
        try:
            setattr(resp, "_executed_tool_calls", executed_tool_calls)
        except Exception:
            pass

        if kb and node_id:
            kb.append(
                {
                    "event": label,
                    "task_id": node_id,
                    "stage": "max_steps_reached",
                    "output_text": getattr(resp, "output_text", "") or "",
                }
            )
        logger.debug(f"LLM tried to make final answer.\n-> resp: {resp}")
        return resp
