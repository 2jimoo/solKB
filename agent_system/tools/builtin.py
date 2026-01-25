from __future__ import annotations
import os
import requests
from typing import Any, Dict
from .registry import ToolRegistry
from .safe_calc import safe_eval_math


def tool_calc(expression: str) -> Dict[str, Any]:
    return {"expression": expression, "result": safe_eval_math(expression)}


def tool_echo(text: str) -> Dict[str, Any]:
    return {"echo": text}


def tool_searchapi_search(
    query: str, num_results: int = 5, gl: str = "us", hl: str = "en"
) -> Dict[str, Any]:
    """SearchAPI.io Google Search API (docs: /api/v1/search?engine=google)."""
    api_key = os.environ.get("SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("SEARCH_API_KEY is missing.")

    # NOTE: SearchAPI docs 기준:
    # - endpoint: https://www.searchapi.io/api/v1/search?engine=google
    # - num 파라미터는 Google에서 2025-09부터 고정 10 (API가 받아도 의미가 없을 수 있음)
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "gl": gl,
        "hl": hl,
    }

    resp = requests.get(
        "https://www.searchapi.io/api/v1/search", params=params, timeout=30
    )
    resp.raise_for_status()
    results = resp.json()

    organic = results.get("organic_results") or []
    k = max(1, min(num_results, 10))  # 어차피 내려오는 건 보통 10개라 여기서 trim
    trimmed = []
    for r in organic[:k]:
        trimmed.append(
            {
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
                "position": r.get("position"),
            }
        )

    return {"query": query, "results": trimmed}


def tool_serpapi_search(
    query: str, num_results: int = 5, gl: str = "us", hl: str = "en"
) -> Dict[str, Any]:
    """SerpApi via stable HTTP endpoint (avoids Python SDK import issues)."""
    api_key = os.environ.get("SERP_API_KEY")
    if not api_key:
        raise RuntimeError("SERP_API_KEY is missing.")

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": max(1, min(num_results, 10)),
        "gl": gl,
        "hl": hl,
    }
    resp = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json()

    organic = results.get("organic_results", []) or []
    trimmed = []
    for r in organic[: params["num"]]:
        trimmed.append(
            {
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
                "position": r.get("position"),
            }
        )
    return {"query": query, "results": trimmed}


def tool_jina_read_url(
    url: str, timeout_s: int = 30, clip_chars: int = 12000
) -> Dict[str, Any]:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("url must start with http:// or https://")
    reader_url = "https://r.jina.ai/" + url

    headers = {}
    jina_key = os.environ.get("JINA_API_KEY")
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    resp = requests.get(reader_url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    text = resp.text
    if clip_chars and len(text) > clip_chars:
        text = text[:clip_chars] + "\n\n[...clipped...]"
    return {"url": url, "reader_url": reader_url, "content": text}


def build_tool_registry() -> ToolRegistry:
    tr = ToolRegistry()

    tr.register(
        name="calc",
        func=tool_calc,
        description="Safely evaluate a basic math expression (no variables, no function calls).",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    )

    tr.register(
        name="tool_searchapi_search",
        func=tool_searchapi_search,
        description="Search the web via SearchAPI.io (Google).",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer", "minimum": 1, "maximum": 10},
                "gl": {"type": "string"},
                "hl": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    tr.register(
        name="jina_read_url",
        func=tool_jina_read_url,
        description="Fetch a URL and convert it to clean text using Jina Reader (r.jina.ai).",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timeout_s": {"type": "integer", "minimum": 5, "maximum": 120},
                "clip_chars": {"type": "integer", "minimum": 1000, "maximum": 50000},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    )

    return tr
