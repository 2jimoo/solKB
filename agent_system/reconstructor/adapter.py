from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, TypedDict, Union, Literal


import logging

logger = logging.getLogger(__name__)


class SubtaskRewriteModel(Protocol):
    """
    변환기 역할 모델이 무엇이든(LLMRunnerWithTools / SLMRunnerHF),
    이 인터페이스만 만족하면 SubtaskRewriter에서 사용 가능.
    """

    def rewrite_subtask(
        self,
        *,
        messages: List[Dict[str, str]],
        tools: Optional[Any] = None,
        schema: Optional[Dict[str, Any]] = None,
        max_steps: int = 6,
        label: str = "subtask_rewrite",
        kb: Optional[Any] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """
        반환값은 '모델이 생성한 raw text' (JSON string)로 통일.
        """
        ...


# ===== Adapters =====
class OpenAIToolLLMAdapter:
    """
    LLMRunnerWithTools를 SubtaskRewriteModel 인터페이스로 감싸는 어댑터.
    """

    def __init__(self, llm_runner_with_tools, tools):
        self.llm = llm_runner_with_tools
        self.tools = tools

    def rewrite_subtask(
        self,
        *,
        messages: List[Dict[str, str]],
        tools: Optional[Any] = None,
        schema: Optional[Dict[str, Any]] = None,
        max_steps: int = 6,
        label: str = "subtask_rewrite",
        kb: Optional[Any] = None,
        node_id: Optional[str] = None,
    ) -> str:
        used_tools = tools if tools is not None else self.tools
        resp = self.llm.run(
            messages=messages,
            tools=used_tools,
            kb=kb,
            node_id=node_id,
            label=label,
            schema=schema,
            max_steps=max_steps,
        )
        return _extract_text_from_openai_responses(resp)


class SLMRunnerHFAdapter:
    """
    SLMRunnerHF를 SubtaskRewriteModel 인터페이스로 감싸는 어댑터.
    - tools를 쓰고 싶으면 solve_with_tools 사용
    - tools가 없으면 _generate 사용
    """

    def __init__(self, slm_runner_hf, tools: Optional[Any] = None):
        self.slm = slm_runner_hf
        self.tools = tools

    def rewrite_subtask(
        self,
        *,
        messages: List[Dict[str, str]],
        tools: Optional[Any] = None,
        schema: Optional[Dict[str, Any]] = None,
        max_steps: int = 6,  # SLM 쪽 max_steps는 직접 매핑 안 함(내부 max_tool_turns/max_slm_attempts 사용)
        label: str = "subtask_rewrite",
        kb: Optional[Any] = None,
        node_id: Optional[str] = None,
    ) -> str:
        used_tools = tools if tools is not None else self.tools

        # schema 강제는 프롬프트로만 처리(여기선 schema 파라미터는 참고용)
        if used_tools is None:
            # pure generation
            return self.slm._generate(messages)

        # tool-enabled: messages를 question 문자열로 합쳐서 전달(간단 합성)
        question = "\n\n".join(
            [f"{m['role'].upper()}:\n{m['content']}" for m in messages]
        )
        out = self.slm.solve_with_tools(
            question=question,
            tools=used_tools,
            kb=kb,
            node_id=node_id,
            max_slm_attempts=8,
            max_tool_turns=5,
        )
        # answer가 JSON 텍스트여야 함
        return (out.get("answer") or "").strip()


def _extract_text_from_openai_responses(resp: Any) -> str:
    out_items = getattr(resp, "output", None) or []
    chunks: List[str] = []
    for it in out_items:
        if isinstance(it, dict):
            t = it.get("type")
            if t == "output_text":
                chunks.append(it.get("text", "") or "")
            elif t == "message":
                content = it.get("content") or []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        chunks.append(c.get("text", "") or "")
            elif isinstance(it.get("text"), str):
                chunks.append(it["text"])
        else:
            t = getattr(it, "type", None)
            if t == "output_text":
                chunks.append(getattr(it, "text", "") or "")
    return "\n".join([c for c in chunks if c]).strip()
