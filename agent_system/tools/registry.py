from __future__ import annotations
from typing import Any, Callable, Dict, List

class ToolRegistry:
    """Shared tool registry for both LLM and SLM."""
    def __init__(self):
        self.tools: Dict[str, Callable[..., Any]] = {}
        self.openai_tools: List[Dict[str, Any]] = []

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        parameters: Dict[str, Any],
    ) -> None:
        self.tools[name] = func
        self.openai_tools.append({
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters,
        })

    def call(self, name: str, args: Dict[str, Any]) -> Any:
        if name not in self.tools:
            raise RuntimeError(f"Unknown tool: {name}")
        return self.tools[name](**args)

    def list_names(self) -> List[str]:
        return list(self.tools.keys())
