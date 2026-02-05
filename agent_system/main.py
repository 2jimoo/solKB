from __future__ import annotations
import json
from agent_system.kb import JsonlKB
from agent_system.tools import build_tool_registry
from agent_system.llm import LLMRunnerWithTools, LLMSupervisor, LLMSupervisorV2
from agent_system.slm import SLMRunnerHF
from agent_system.orchestrator import (
    RecursiveSolver,
    IterativeSolver,
    RecursiveSolverV2,
)
from dotenv import load_dotenv
import os
import argparse
from pathlib import Path
from datetime import datetime, timezone

load_dotenv()

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task_type", required=True, type=int)  
    p.add_argument("--model", default="gpt-5-mini", type=str)  #"gpt-4.1"
    return p.parse_args()


def load_task_from_json_list(input_path: str | Path, task_type: int) -> dict:
    """
    input_path: JSON array(list of dicts) 파일
    task_type: 1-based index로 N번째 문제 선택
    """
    input_path = Path(input_path)
    data = json.loads(input_path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"{input_path} must be a JSON list (array).")

    idx = task_type - 1
    if idx < 0 or idx >= len(data):
        raise IndexError(f"task_type={task_type} is out of range. total={len(data)}")

    return data[idx]


def append_result_to_answer_json(answer_path: str | Path, result_obj: dict) -> None:
    """
    answer.json을 JSON array로 유지하면서 result_obj를 append.
    파일이 없으면 새로 생성.
    파일이 비었거나 깨졌으면 예외 발생시키는 대신 새 리스트로 시작하고 싶으면
    try/except에서 처리 가능.
    """
    answer_path = Path(answer_path)

    if answer_path.exists():
        text = answer_path.read_text(encoding="utf-8").strip()
        if text:
            existing = json.loads(text)
            if not isinstance(existing, list):
                raise ValueError(f"{answer_path} must contain a JSON list (array).")
        else:
            existing = []
    else:
        existing = []

    existing.append(result_obj)

    answer_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()

    # 1) JSON list에서 task_type번째 문제 로드 (1-based)
    task = load_task_from_json_list("/home/jovyan/solKB/agent_kb_database_filtered.json", args.task_type)

    question = task.get("question")
    agent_planning = task.get("agent_planning")
    true_answer = task.get("true_answer")

    if question is None:
        raise KeyError("Task item missing required field: 'question'")
    if true_answer is None:
        raise KeyError("Task item missing required field: 'true_answer'")
    # agent_planning은 없을 수도 있으면 None 허용

    # 2) 기존 러너/솔버 구성 (원래 코드 유지)
    kb = JsonlKB("kb.jsonl")
    tools = build_tool_registry()

    llm_runner = LLMRunnerWithTools(model=)
    supervisor = LLMSupervisorV2(llm_runner)

    slm = SLMRunnerHF(model_id="Qwen/Qwen3-4B-Instruct-2507")
    solver = RecursiveSolverV2(
        kb=kb,
        tools=tools,
        supervisor=supervisor,
        slm=slm,
        max_depth=3,
    )

    # 3) 실행
    result = solver.run(question, agent_planning, true_answer)

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 4) result가 None이 아니면 answer.json에 append
    if result is not None:
        answer= {
            "task_id": str(args.task_type),
            "task":question,
            "true_answer":true_answer,
            "agent_planning": agent_planning,
            "subtasks": result
        }
        append_result_to_answer_json("./answer.json", answer)


if __name__ == "__main__":
    main()
    # nohup python -m agent_system.main --task_type 15 > task15.log 2>&1 &
    # for i in {1..20}; do         nohup python -m agent_system.main --task_type $i > task$i.log 2>&1        done

