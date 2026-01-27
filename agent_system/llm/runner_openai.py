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

        # --- tool loop ---
        for step in range(max_steps):
            tool_calls = []

            output_items = getattr(resp, "output", None) or []
            for item in output_items:
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

            # If there are no tool calls, assistant has given an answer -> return it.
            if not tool_calls:
                setattr(resp, "_executed_tool_calls", executed_tool_calls)
                return resp

            # --- At this point we have tool calls to execute(1회당 여러개 호출 가능 모두 리스트에 넣어줘야함, 각각 call_id 가짐) ---
            tool_results = []
            for tc in tool_calls:
                raw_args = tc.get("arguments")
                if isinstance(raw_args, str):
                    raw_args = raw_args.strip()
                    args = json.loads(raw_args) if raw_args else {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}

                try:
                    out = tools.call(tc["name"], args)
                except Exception as e:
                    out = {"error": str(e), "tool": tc.get("name"), "args": args}
                logger.debug(
                    f"LLM tried {step}-th tool call ({tc['name']}) with parameters ({args}).\n-> res: {out}"
                )
                executed_tool_calls.append(
                    {
                        "name": tc.get("name"),
                        "arguments": args,
                        "call_id": tc.get("call_id") or tc.get("id"),
                        "output": json.dumps(out, ensure_ascii=False),
                    }
                )
                tool_results.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.get("call_id") or tc.get("id"),
                        "output": json.dumps(out, ensure_ascii=False),
                    }
                )
            # follow-up call (continue the same response thread) to have LLM synthesize an answer
            kwargs2 = dict(
                model=self.model,
                previous_response_id=resp.id,
                # include original messages plus tool outputs; ask LLM explicitly for the answer
                input=messages
                + tool_results
                + [
                    {
                        "role": "user",
                        "content": "\nBased on the above tool results and prior conversation, produce a clear answer now. If you need, you can call tools again.",
                    }
                ],
                tools=tools.openai_tools,
                text={"format": schema} if schema else None,
            )
            resp = self.client.responses.create(**kwargs2)
            logger.debug(f"LLM tried to solve with tool results.\n-> resp: {resp}")

            # If LLM returned final content without requesting further tool calls -> return it immediately.
            # (This will be caught by the top-of-loop check on next iteration, but we can short-circuit here.)
            next_output_items = getattr(resp, "output", None) or []
            has_tool_call = any(
                (
                    (
                        isinstance(it, dict)
                        and it.get("type") in ("tool_call", "function_call")
                    )
                    or (
                        not isinstance(it, dict)
                        and getattr(it, "type", None) in ("tool_call", "function_call")
                    )
                )
                for it in next_output_items
            )

            if not has_tool_call:
                setattr(resp, "_executed_tool_calls", executed_tool_calls)
                return resp
            time.sleep(0.5)

        # --- max steps reached: force a final synthesis from the model ---
        tool_trace_text = json.dumps(executed_tool_calls, ensure_ascii=False, indent=2)
        final_prompt = (
            "Using the conversation history and all tool results provided so far, produce a concise final answer now.\n"
            f"TOOL_TRACE:\n{tool_trace_text}"
        )
        kwargs_final = dict(
            model=self.model,
            input=messages + [{"role": "user", "content": final_prompt}],
            tool_choice="none",
            text={"format": schema} if schema else None,
        )
        final_resp = self.client.responses.create(**kwargs_final)
        setattr(final_resp, "_executed_tool_calls", executed_tool_calls)
        logger.debug(
            f"LLM produced final answer after max steps.\n-> resp: {final_resp}"
        )
        return final_resp
