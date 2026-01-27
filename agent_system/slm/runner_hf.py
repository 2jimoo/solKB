from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from agent_system.kb import JsonlKB
from agent_system.tools.registry import ToolRegistry

import logging

logger = logging.getLogger(__name__)


class SLMRunnerHF:
    """
    HF local Qwen3 runner that 'calls tools' via JSON-only action protocol.

    Required output (JSON ONLY):
      {"action":"TOOL_CALL","tool_name":"...","arguments":{...}}
      {"action":"FINAL","answer":"..."}
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
        device_map: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 5000,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        self.model.eval()

        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.inference_mode()
    def _generate(self, messages: List[Dict[str, str]]) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = out[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def solve_with_tools(
        self,
        question: str,
        tools: ToolRegistry,
        kb: Optional[JsonlKB] = None,
        node_id: Optional[str] = None,
        max_tool_turns: int = 10,
        slm_attempt_histories=None,
    ) -> Dict[str, Any]:
        system = (
            "You are a problem solver. You MUST output valid JSON ONLY.\n"
            "Choose exactly one:\n"
            'A) {"action":"TOOL_CALL","tool_name":"<name>","arguments":{...}}\n'
            'B) {"action":"FINAL","answer":"<final answer>"}\n'
            "If web facts are needed, call serpapi_search then jina_read_url on the best links."
            f"[Attempt History]{slm_attempt_histories}"
        )

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nAvailable tools:\n{tools.list_names()}",
            },
        ]

        trace: List[Dict[str, Any]] = []
        for turn in range(max_tool_turns):
            raw = self._generate(messages)
            logger.debug(f"raw respose:\n{raw}")
            trace.append({"turn": turn, "slm_raw": raw})

            if kb and node_id:
                kb.append(
                    {"event": "slm_raw", "task_id": node_id, "turn": turn, "raw": raw}
                )

            try:
                action = json.loads(raw)
            except json.JSONDecodeError:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {"role": "user", "content": "Output MUST be valid JSON. Retry."}
                )
                continue

            if action.get("action") == "FINAL":
                return {
                    "status": "final",
                    "answer": action.get("answer", ""),
                    "trace": trace,
                }

            if action.get("action") == "TOOL_CALL":
                tool_name = action.get("tool_name")
                args = action.get("arguments") or {}

                if kb and node_id:
                    kb.append(
                        {
                            "event": "slm_tool_call",
                            "task_id": node_id,
                            "tool_name": tool_name,
                            "arguments": args,
                        }
                    )

                try:
                    out = tools.call(tool_name, args)
                except Exception as e:
                    out = {"error": str(e), "tool_name": tool_name, "arguments": args}
                logger.info(f"[SLM] tool {tool_name} called with {args}")

                if kb and node_id:
                    kb.append(
                        {
                            "event": "slm_tool_result",
                            "task_id": node_id,
                            "tool_name": tool_name,
                            "output": out,
                        }
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(action, ensure_ascii=False),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool result for {tool_name}:\n{json.dumps(out, ensure_ascii=False)}\n\n"
                            "Continue and respond with FINAL JSON (or another TOOL_CALL if still needed)."
                        ),
                    }
                )
                continue

            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid action. Output JSON with action TOOL_CALL or FINAL.",
                }
            )

        return {"status": "tool_turn_limit", "answer": "", "trace": trace}
