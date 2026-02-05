import os
import json
import glob
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _first_if_list(x: Any) -> Any:
    # list면 첫 원소만, 빈 list/None이면 None
    if x is None:
        return None
    if isinstance(x, list):
        return x[0] if x else None
    return x


def load_all_task_files(
    base_path: str, pattern: str = "task_*.json"
) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(base_path, "**", pattern), recursive=True))
    items: List[Dict[str, Any]] = []
    for fp in files:
        try:
            data = _read_json(fp)
            if isinstance(data, dict):
                # 원하는 필드들: list면 [0]만 담기
                for k in ("question", "agent_planning", "true_answer"):
                    if k in data:
                        data[k] = _first_if_list(data[k])

                # solved_records는 list가 의미 있을 수 있으니 그대로 두고 싶으면 건드리지 않음
                # 만약 이것도 list면 [0]만 원하면 아래 주석 해제:
                # if "solved_records" in data:
                #     data["solved_records"] = _first_if_list(data["solved_records"])

                data["__source_path"] = fp
                items.append(data)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
    return items


def build_kb_from_solved_records(
    base_path: str,
    out_path: str = "distilled_kb.json",
    pattern: str = "task_*.json",
) -> List[Dict[str, Any]]:
    """
    함수1) solved_records({subtask,rationale,answer}) -> subtasks로 구성해서 저장.
    """
    rows = load_all_task_files(base_path, pattern=pattern)

    kb: List[Dict[str, Any]] = []
    for ridx, r in enumerate(rows):
        logger.info(f"{ridx}-th row processed")
        task_type = r.get("task_type")
        question = r.get("question")
        agent_planning = r.get("agent_planning")
        true_answer = r.get("true_answer")
        solved_records = r.get("solved_records") or []

        subtasks = []
        if isinstance(solved_records, list):
            for sr in solved_records:
                if not isinstance(sr, dict):
                    continue
                # 입력 키가 정확히 subtask/rationale/answer라고 가정
                subtask = sr.get("subtask")
                rationale = sr.get("rationale")
                answer = sr.get("answer")
                if subtask is None and rationale is None and answer is None:
                    continue
                subtasks.append(
                    {
                        "subtask": subtask,
                        "rationale": rationale,
                        "answer": answer,
                    }
                )

        kb.append(
            {
                "task_id": task_type,
                "task": question,
                "agent_planning": agent_planning,
                "true_answer": true_answer,
                "subtasks": subtasks,
                "__source_path": r.get("__source_path"),
            }
        )

    _write_json(out_path, kb)
    return kb


def _build_json_schema_for_subtasks() -> Dict[str, Any]:
    """
    Structured Outputs 용 JSON schema (Responses API response_format=json_schema)
    """
    return {
        "name": "planning_to_subtasks",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "subgoal": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["subgoal", "rationale"],
                    },
                },
            },
            "required": ["task", "subtasks"],
        },
        "strict": True,
    }


def split_planning_with_openai(
    client: OpenAI,
    task: str,
    agent_planning: str,
    model: str = "gpt-4o-mini",
    max_subtasks: int = 12,
) -> Dict[str, Any]:
    """
    함수2에서 사용하는 단일 호출:
    agent_planning 텍스트를 subtask들로 분해하고 각 subtask별 rationale 생성.
    """
    schema = _build_json_schema_for_subtasks()

    prompt = f"""
You are given:
- Task (question): {task}

- Agent planning text (a sequence of steps, often merged / concatenated):
{agent_planning}

Convert the planning into a clean list of subtasks.
Rules:
- Return 3 to {max_subtasks} subtasks (merge tiny steps, split big steps).
- Each subtask must be a single actionable subgoal.
- Provide a short rationale explaining why that subgoal is needed.
- Do NOT include answers or citations.
"""

    resp = client.responses.create(
        model=model,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": schema.get("strict", True),
            }
        },
    )

    # openai-python의 Responses는 output_text도 있지만, 우리는 구조화 JSON을 기대
    # SDK 버전에 따라 파싱 위치가 다를 수 있어 안전 처리
    data = None
    try:
        # 많은 경우 resp.output[0].content[0].parsed 에 들어있음
        data = resp.output[0].content[0].parsed  # type: ignore[attr-defined]
    except Exception:
        # fallback: output_text를 JSON으로 파싱 시도
        txt = getattr(resp, "output_text", None)
        if not txt:
            txt = str(resp)
        data = json.loads(txt)

    # 최소 검증
    if not isinstance(data, dict) or "subtasks" not in data:
        raise ValueError(f"Unexpected model output: {data}")

    return data


def build_kb_from_agent_planning_openai(
    base_path: str,
    out_path: str = "generated_kb.json",
    pattern: str = "task_*.json",
    model: str = "gpt-5-mini",
    max_subtasks: int = 12,
) -> List[Dict[str, Any]]:
    """
    함수2) 각 task_*.json의 agent_planning을 OpenAI로 분해 + rationale 생성해 subtasks 구성 후 저장.
    """
    client = OpenAI()
    rows = load_all_task_files(base_path, pattern=pattern)

    kb: List[Dict[str, Any]] = []
    for ridx, r in enumerate(rows):
        logger.info(f"{ridx}-th row processed")
        task_type = r.get("task_type")
        question = r.get("question") or ""
        agent_planning = r.get("agent_planning") or ""
        true_answer = r.get("true_answer")
        solved_records = r.get("solved_records")

        if not agent_planning.strip():
            # planning이 없으면 빈 subtasks로 저장
            parsed = {"task": question, "subtasks": []}
        else:
            parsed = split_planning_with_openai(
                client=client,
                task=question,
                agent_planning=agent_planning,
                model=model,
                max_subtasks=max_subtasks,
            )

        kb.append(
            {
                "task_id": task_type,
                "task": question,
                "agent_planning": agent_planning,
                "true_answer": true_answer,
                "subtasks": parsed.get("subtasks", []),
            }
        )

    _write_json(out_path, kb)
    return kb


if __name__ == "__main__":
    """
    사용 예:

    1) 함수1 실행:
      python distillation.py solved_records /home/jovyan/solKB/result/success /home/jovyan/solKB/test_kb/distilled_kb.json

    2) 함수2 실행 (OpenAI 사용):
      export OPENAI_API_KEY="..."
      python distillation.py openai /home/jovyan/solKB/result/success /home/jovyan/solKB/test_kb/generated_kb.json gpt-5-mini
    """
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage:\n"
            "  python script.py solved_records <base_path> <out_path>\n"
            "  python script.py openai <base_path> <out_path> [model]\n"
        )
        raise SystemExit(1)

    mode = sys.argv[1]
    base_path = sys.argv[2]
    out_path = sys.argv[3]
    model = sys.argv[4] if len(sys.argv) >= 5 else "gpt-5-mini"

    if mode == "solved_records":
        build_kb_from_solved_records(base_path=base_path, out_path=out_path)
        print(f"[OK] Wrote: {out_path}")
    elif mode == "openai":
        build_kb_from_agent_planning_openai(
            base_path=base_path, out_path=out_path, model=model
        )
        print(f"[OK] Wrote: {out_path}")
    else:
        raise ValueError("mode must be one of: solved_records | openai")
