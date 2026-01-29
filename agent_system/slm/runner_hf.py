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

    # def solve_with_tools(
    #     self,
    #     question: str,
    #     tools: ToolRegistry,
    #     kb: Optional[JsonlKB] = None,
    #     node_id: Optional[str] = None,
    #     max_slm_attempts: int = 10,
    #     max_tool_turns: int = 10,
    #     slm_attempt_histories=None,
    # ) -> Dict[str, Any]:
    #     system = (
    #         "You are a problem solver. You MUST output valid JSON ONLY.\n"
    #         "Choose exactly one:\n"
    #         'A) {"action":"TOOL_CALL","tool_name":"<name>","arguments":{...}}\n'
    #         'B) {"action":"FINAL","answer":"<final answer>"}\n'
    #         "If web facts are needed, call serpapi_search then jina_read_url on the best links."
    #         f"[Attempt History]{slm_attempt_histories}"
    #     )

    #     for attempt in range(max_slm_attempts):
    #         messages = [
    #             {"role": "system", "content": system},
    #             {
    #                 "role": "user",
    #                 "content": f"Question:\n{question}\n\nAvailable tools:\n{tools.list_names()}",
    #             },
    #         ]
    #         for turn in range(max_tool_turns):
    #             raw = self._generate(messages)
    #             logger.info(f"SLM raw respose:\n{raw}")
    #             try:
    #                 action = json.loads(raw)
    #             except json.JSONDecodeError:
    #                 messages.append({"role": "assistant", "content": raw})
    #                 messages.append(
    #                     {"role": "user", "content": "Output MUST be valid JSON. Retry."}
    #                 )
    #                 continue

    #             if action.get("action") == "FINAL":
    #                 return {
    #                     "status": "final",
    #                     "answer": action.get("answer", "")
    #                 }

    #             if action.get("action") == "TOOL_CALL":
    #                 tool_name = action.get("tool_name")
    #                 args = action.get("arguments") or {}
    #                 try:
    #                     out = tools.call(tool_name, args)
    #                 except Exception as e:
    #                     out = {"error": str(e), "tool_name": tool_name, "arguments": args}
    #                 logger.info(f"[SLM] tool {tool_name} called with {args}")
    #                 messages.append(
    #                     {
    #                         "role": "assistant",
    #                         "content": json.dumps(action, ensure_ascii=False),
    #                     }
    #                 )
    #                 messages.append(
    #                     {
    #                         "role": "user",
    #                         "content": (
    #                             f"Tool result for {tool_name}:\n{json.dumps(out, ensure_ascii=False)}\n\n"
    #                             "Continue and respond with FINAL JSON (or another TOOL_CALL if still needed)."
    #                         ),
    #                     }
    #                 )
    #                 continue

    #             messages.append({"role": "assistant", "content": raw})
    #             messages.append(
    #                 {
    #                     "role": "user",
    #                     "content": "Invalid action. Output JSON with action TOOL_CALL or FINAL.",
    #                 }
    #             )
    #     return {"status": "tool_turn_limit", "answer": ""}

    def solve_with_tools(
        self,
        question: str,
        tools: ToolRegistry,
        kb: Optional[JsonlKB] = None,
        node_id: Optional[str] = None,
        max_slm_attempts: int = 10,
        max_tool_turns: int = 5,
        slm_attempt_histories=None,
    ) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        if slm_attempt_histories:
            history.append(
                {"type": "external_attempt_history", "data": slm_attempt_histories}
            )
        system_base = (
            "You are a problem solver. You MUST output valid JSON ONLY.\n"
            "At every step, you MUST include your best current answer in 'answer_so_far'.\n"
            "Choose exactly one JSON shape:\n"
            'A) {"action":"TOOL_CALL","tool_name":"<name>","arguments":{...},"answer_so_far":"<best current answer>"}\n'
            'B) {"action":"FINAL","answer":"<final answer>","answer_so_far":"<best current answer>"}\n'
            "Rules:\n"
            "- After receiving any tool result, update 'answer_so_far' using that result.\n"
            "- If more info is needed, respond with TOOL_CALL. Otherwise respond with FINAL.\n"
            "- Output JSON only. No extra text.\n"
        )

        last_answer_so_far = ""
        for attempt in range(max_slm_attempts):
            system = (
                system_base + f"\n[History]\n{json.dumps(history, ensure_ascii=False)}"
            )
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Available tools:\n{tools.list_names()}\n\n"
                        "Respond with TOOL_CALL or FINAL in JSON."
                    ),
                },
            ]
            for turn in range(max_tool_turns):
                raw = self._generate(messages)
                logger.info(f"SLM raw respose:\n{raw}")
                try:
                    action = json.loads(raw)
                except json.JSONDecodeError:
                    logger.info(f"SLM returned bad json, retry.")
                    continue

                if action.get("action") == "FINAL":
                    final_answer = action.get("answer", "")
                    return {
                        "status": "final",
                        "answer": final_answer,
                        "answer_so_far": last_answer_so_far,
                        "history": history,
                    }

                if action.get("action") == "TOOL_CALL":
                    tool_name = action.get("tool_name")
                    args = action.get("arguments") or {}
                    try:
                        out = tools.call(tool_name, args)
                    except Exception as e:
                        out = {
                            "error": str(e),
                            "tool_name": tool_name,
                            "arguments": args,
                        }
                    logger.info(f"[SLM] tool {tool_name} called with {args}")

                    history.append(
                        {
                            "type": "tool_result",
                            "attempt": attempt,
                            "turn": turn,
                            "tool_name": tool_name,
                            "arguments": args,
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
                                "Now update 'answer_so_far' using this tool result.\n"
                                "If you still need more info, respond with TOOL_CALL.\n"
                                "Otherwise respond with FINAL.\n"
                                "Remember: output JSON only."
                            ),
                        }
                    )
                    continue

        # 최대 횟수 도달 강제 답변
        system = system_base + f"\n[History]\n{json.dumps(history, ensure_ascii=False)}"
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    "Respond final answer based on histories."
                ),
            },
        ]
        last_answer_so_far = self._generate(messages)
        return {
            "status": "attempt_limit",
            "answer": last_answer_so_far,
            "history": history,
        }
