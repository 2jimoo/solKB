from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from agent_system.kb import JsonlKB
from agent_system.tools.registry import ToolRegistry
from agent_system.llm.runner_openai import LLMRunnerWithTools
from agent_system.models import (
    SubtaskSpec,
    SolveReference,
    Verification,
    ToolContribution,
)

import logging

logger = logging.getLogger(__name__)


class LLMSupervisorV2:
    """
    핵심:
      - decompose(...) 는 List[SubtaskSpec] 를 반환한다.
      - expected_answer 는 "분해 시 추론"이 아니라 각 subtask를 LLM이 직접 풀어 얻는다.
      - actual_answer가 주어지면 verify_semantic을 내부에서 호출하여
        정답에 도달할 때까지 (분해/풀이) 반복한다.
      - 단, '검증/확인'을 별도의 subtask로 만들지 않도록 분해 프롬프트에서 강제한다.
    """

    def __init__(self, runner: LLMRunnerWithTools):
        self.runner = runner
        self._node_counter = 0

    def _new_node_id(self, prefix: str = "n") -> str:
        self._node_counter += 1
        return f"{prefix}{self._node_counter}"

    # ---------- Public: main API ----------
    def decompose(
        self,
        question: str,
        tools: ToolRegistry,
        kb: Optional[JsonlKB],
        node_id: str,
        reference: SolveReference,
        depth: int = 0,
        is_root: bool = True,
        sibling_solved: Optional[List[Dict[str, str]]] = None,
        actual_answer: Optional[str] = None,
        max_items: int = 5,
        keep_sibling_ctx: int = 8,
        max_rounds: int = 5,
        max_solve_attempts_per_subtask: int = 2,
    ) -> List[SubtaskSpec]:
        """
        Returns: List[SubtaskSpec(subtask, expected_answer)]
        - expected_answer is obtained by actually solving each subtask with LLM.
        - If actual_answer is provided, this function loops until semantic match (or max_rounds).
        """
        sibling_solved = sibling_solved or []

        rounds = max_rounds if actual_answer is not None else 1
        last_specs: List[SubtaskSpec] = []

        for r in range(rounds):
            # 1) decompose into subtasks (NO verify/check subtasks)
            plan_id = self._new_node_id("plan_")
            subtasks = self._decompose_subtasks_only(
                question=question,
                tools=tools,
                kb=kb,
                node_id=plan_id,
                reference=reference,
                depth=depth,
                is_root=is_root,
                sibling_solved=sibling_solved,
                max_items=max_items,
            )
            subtasks_str = "\n".join(subtasks)
            logger.info(f"[{r}]Superviser Plan:\n{subtasks_str}")

            if kb:
                kb.append(
                    {
                        "event": "supervisor_plan",
                        "node_id": plan_id,
                        "parent_node_id": node_id,
                        "round": r + 1,
                        "depth": depth,
                        "question": question,
                        "subtasks": subtasks,
                    }
                )

            # 2) solve each subtask to fill expected_answer
            local_siblings = list(sibling_solved[-keep_sibling_ctx:])
            specs: List[SubtaskSpec] = []

            for st in subtasks:
                wrong_attempts: List[str] = []
                ans = ""

                for aidx in range(max_solve_attempts_per_subtask):
                    solve_id = self._new_node_id("solve_")
                    ans, exp_tool, exp_tool_params = self._solve_subtask_once(
                        parent_question=question,
                        subtask=st,
                        tools=tools,
                        kb=kb,
                        node_id=solve_id,
                        sibling_solved=local_siblings,
                        wrong_attempts=wrong_attempts,
                    )

                    if kb:
                        kb.append(
                            {
                                "event": "supervisor_solve",
                                "node_id": solve_id,
                                "parent_node_id": node_id,
                                "round": r + 1,
                                "depth": depth + 1,
                                "parent_question": question,
                                "subtask": st,
                                "answer": ans,
                                "attempt": aidx + 1,
                            }
                        )
                    logger.info(
                        f"[{r}-{aidx}]Superviser Solve\nSubtask:{st}\nAnswer:{ans}\nTool:{exp_tool}, {exp_tool_params}"
                    )
                    # UNKNOWN/empty면 재시도, 아니면 확정
                    if ans and ans.strip().upper() != "UNKNOWN":
                        break
                    wrong_attempts.append(ans or "")
                specs.append(
                    SubtaskSpec(
                        subtask=st,
                        expected_answer=ans.strip(),
                        expected_tool=exp_tool,
                        expected_tool_params=exp_tool_params,
                    )
                )
                logger.info(
                    f"[{r}]Superviser Solve\nSubtask:{st}\nAnswer:{ans}\nTool:{exp_tool}, {exp_tool_params}"
                )
                if ans and ans.strip().upper() != "UNKNOWN":
                    local_siblings.append({"subtask": st, "answer": ans.strip()})
                    local_siblings = local_siblings[-keep_sibling_ctx:]
                else:
                    break

            last_specs = specs

            if kb:
                kb.append(
                    {
                        "event": "supervisor_record",
                        "node_id": node_id,
                        "round": r + 1,
                        "depth": depth,
                        "question": question,
                        "subtask_expected_answers": [
                            {"subtask": s.subtask, "expected_answer": s.expected_answer}
                            for s in specs
                        ],
                    }
                )

            # 3) if no actual_answer => done
            # 4) derive a single "root answer" from the specs, then semantic-verify
            derived = self.derive_root_answer(
                question, specs, tools=tools, kb=kb, node_id=node_id
            )
            logger.info(
                f"[{r}]Superviser Answer\nFinal Answer:{derived}\nActual Answer:{actual_answer}"
            )
            if actual_answer is None:
                if derived and derived != "UNKNOWN":
                    return last_specs
                else:
                    continue

            ver_id = self._new_node_id("ver_")
            v = self.verify_semantic(
                question=question,
                proposed_answer=derived,
                actual_answer=actual_answer,
                tools=tools,
                kb=kb,
                node_id=ver_id,
            )
            logger.info(f"[{r}]Superviser Verified.\n{v.verdict}: {v.reason}")

            if kb:
                kb.append(
                    {
                        "event": "supervisor_verify",
                        "node_id": ver_id,
                        "parent_node_id": node_id,
                        "round": r + 1,
                        "depth": depth,
                        "question": question,
                        "derived_answer": derived,
                        "actual_answer": actual_answer,
                        "verdict": v.verdict,
                        "reason": v.reason,
                        "evidence": v.evidence,
                    }
                )

            if v.verdict == "correct":
                logger.info(f"[{r}]Superviser Corrected.")
                concise_specs = self.reconstruct_plan_with_llm(tools, last_specs)
                return concise_specs

            # 5) mismatch => push failure into failed_history and repeat (no extra subtask added)
            reference.failed_history.append(
                {
                    "task": question,
                    "reason": f"round {r+1} mismatch: derived={derived!r}; verifier={v.verdict}; {v.reason}",
                }
            )

        logger.info(f"[{r}]Superviser Failed.")
        return None

    # ---------- Derivation policy ----------
    def derive_root_answer(
        self,
        question: str,
        specs: List[SubtaskSpec],
        *,
        tools: ToolRegistry,
        kb: Optional[JsonlKB],
        node_id: str,
    ) -> str:
        if not specs:
            return ""

        items = [
            {
                "idx": i,
                "subtask": s.subtask,
                "expected_answer": s.expected_answer,
                "expected_tool": s.expected_tool,
                "expected_tool_params": s.expected_tool_params,
            }
            for i, s in enumerate(specs, 1)
        ]

        sys_prompt = (
            "You are a FINAL ANSWER SYNTHESIZER.\n"
            "Synthesize the single final answer to the parent question from subtask results.\n"
            "Rules:\n"
            "- Return ONLY the final answer string.\n"
            "- No explanations, no bullets, no JSON.\n"
            "- Use ONLY provided subtask answers; do not invent facts.\n"
            "- If insufficient, return exactly: UNKNOWN\n"
        )

        resp = self.runner.run(
            [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": question, "subtask_results": items},
                        ensure_ascii=False,
                    ),
                },
            ],
            tools,
            kb=kb,
            node_id=node_id,
            label="llm_derive_root_answer",
            schema=None,
        )

        out = (resp.output_text or "").strip()
        return out if out else "UNKNOWN"

    # ---------- Internal: decomposition (subtasks only; ban verify/check subtasks) ----------
    def _decompose_subtasks_only(
        self,
        question: str,
        tools: ToolRegistry,
        kb: Optional[JsonlKB],
        node_id: str,
        reference: SolveReference,
        depth: int,
        is_root: bool,
        sibling_solved: List[Dict[str, str]],
        max_items: int,
    ) -> List[str]:
        schema = {
            "type": "json_schema",
            "name": "decomposition_only_no_verify_subtasks",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "subtasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": max_items,
                    }
                },
                "required": ["subtasks"],
                "additionalProperties": False,
            },
        }

        fh = reference.failed_history[-5:] if reference.failed_history else []
        sib = sibling_solved[-8:] if sibling_solved else []

        banned = [
            "verify",
            "verification",
            "check",
            "confirm",
            "validate",
            "double-check",
            "cross-check",
            "re-check",
            "sanity check",
            "review",
            "prove",
        ]

        # ✅ 핵심: "마지막 subtask가 parent question 최종답을 산출" 강제
        # (actual_answer loop에서 derived 정책이 last subtask라서 품질이 크게 좋아짐)
        common_constraints = (
            "Hard constraints:\n"
            f"- NEVER create verification/check subtasks (avoid words like: {', '.join(banned)}).\n"
            "- Each subtask must directly advance solving (derive/compute/retrieve needed intermediate facts).\n"
            "- NO subtask should be 'verify/confirm the answer'. Checking is internal, not a separate subtask.\n"
            "- The LAST subtask MUST directly produce the FINAL answer to the parent question.\n"
            "- Each subtask must be ONE actionable sentence.\n"
        )

        if is_root and reference.reference_planning:
            sys_prompt = (
                "Convert reference planning into 1-5 subtasks for a WEAK agent.\n"
                "Return JSON only: {subtasks:[...]}\n\n"
                + common_constraints
                + "Use failed_history to avoid repeating mistakes.\n"
                "Use sibling_solved as known facts; do NOT ask to redo them.\n"
            )
            user_payload = {
                "mode": "root_from_reference_planning_no_verify_steps",
                "question": question,
                "reference_planning": reference.reference_planning,
                "failed_history": fh,
                "sibling_solved": sib,
                "depth": depth,
            }
        else:
            sys_prompt = (
                "Decompose question into 1-5 SMALL, SEQUENTIAL subtasks for a WEAK agent.\n"
                "Return JSON only: {subtasks:[...]}\n\n"
                + common_constraints
                + "Guidelines:\n"
                "- Use failed_history to pick an alternate approach and avoid repeats.\n"
                "- If failures indicate missing evidence, add a subtask that obtains the needed evidence (but do NOT name it 'verify').\n"
                "- Use sibling_solved as known facts.\n"
            )
            user_payload = {
                "mode": "regular_decompose_for_weak_agent_no_verify_steps",
                "question": question,
                "failed_history": fh,
                "sibling_solved": sib,
                "depth": depth,
            }

        resp = self.runner.run(
            [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            tools,
            kb=kb,
            node_id=node_id,
            label="llm_decompose_only",
            schema=schema,
        )

        data = json.loads(resp.output_text or "{}")
        raw = [
            s.strip()
            for s in data.get("subtasks", [])
            if isinstance(s, str) and s.strip()
        ]

        # optional filter if model violates rule
        filtered: List[str] = []
        for s in raw:
            low = s.lower()
            if any(
                w in low
                for w in [
                    "verify",
                    "verification",
                    "check",
                    "confirm",
                    "validate",
                    "double-check",
                    "cross-check",
                    "re-check",
                    "sanity check",
                    "review",
                ]
            ):
                continue
            filtered.append(s)

        return (filtered or raw)[:max_items]

    # ---------- Internal: solve subtask once ----------
    def _solve_subtask_once(
        self,
        parent_question: str,
        subtask: str,
        tools: ToolRegistry,
        kb: Optional[JsonlKB],
        node_id: str,
        sibling_solved: List[Dict[str, str]],
        wrong_attempts: List[str],
    ) -> str:
        sib = sibling_solved[-8:] if sibling_solved else []
        ctx_lines: List[str] = []
        for i, it in enumerate(sib, 1):
            q = (it.get("subtask") or "").strip()
            a = (it.get("answer") or "").strip()
            if q and a:
                ctx_lines.append(f"{i}. Q: {q}\n   A: {a}")
        ctx_block = "\n".join(ctx_lines).strip()

        wa_lines = [f"- {a.strip()}" for a in wrong_attempts[-5:] if a and a.strip()]
        wa_block = "\n".join(wa_lines).strip()

        sys_prompt = (
            "Solve the CURRENT subtask.\n"
            "Return ONLY the final answer string.\n"
            "No explanations, no bullets, no JSON.\n"
            "Do not return blank string. If truly unknown, return exactly: UNKNOWN\n"
        )

        user_text = (
            f"Parent question:\n{parent_question}\n\n"
            f"Current subtask:\n{subtask}\n\n"
        )
        if ctx_block:
            user_text += (
                "Solved sibling subtasks (treat as facts; do NOT redo unless necessary):\n"
                f"{ctx_block}\n\n"
            )
        if wa_block:
            user_text += (
                "Previous wrong attempts (avoid repeating):\n" f"{wa_block}\n\n"
            )

        resp = self.runner.run(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_text},
            ],
            tools,
            kb=kb,
            node_id=node_id,
            label="llm_solve_subtask",
            schema=None,
        )
        logger.info(f"LLM solve subtask once raw resp:\n{resp}")

        answer = (resp.output_text or "").strip()
        tool_calls = getattr(resp, "_executed_tool_calls", None) or []
        contrib = self._select_key_tool_call(
            tool_calls=tool_calls,
            final_answer=answer,
            runner=self.runner,
            tools=tools,
            kb=kb,
            node_id=None,
        )
        tool_name, tool_params = contrib.tool_name, contrib.tool_args
        return answer, tool_name, tool_params

    # ---------- Semantic verifier ----------
    def verify_semantic(
        self,
        question: str,
        proposed_answer: str,
        actual_answer: str,
        tools: ToolRegistry,
        kb: Optional[JsonlKB],
        node_id: str,
    ) -> Verification:
        schema = {
            "type": "json_schema",
            "name": "semantic_verification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["correct", "incorrect", "insufficient"],
                    },
                    "reason": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 6,
                    },
                },
                "required": ["verdict", "reason", "evidence"],
                "additionalProperties": False,
            },
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a STRICT SEMANTIC VERIFIER.\n"
                    "Judge whether proposed answer is semantically equivalent to actual answer.\n"
                    "Return ONLY verdict/reason/evidence in JSON.\n"
                    "Do NOT provide the correct answer or hints.\n"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "proposed_answer": proposed_answer,
                        "actual_answer": actual_answer,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        resp = self.runner.run(
            messages,
            tools,
            kb=kb,
            node_id=node_id,
            label="llm_verify_semantic",
            schema=schema,
        )
        data = json.loads(resp.output_text or "{}")
        return Verification(
            verdict=(data.get("verdict") or "insufficient"),
            reason=(data.get("reason") or ""),
            evidence=list(data.get("evidence") or []),
        )

    def reconstruct_plan_with_llm(
        self,
        tools: ToolRegistry,
        last_specs: List[SubtaskSpec],
        *,
        kb: Optional[JsonlKB] = None,
        node_id: str = "reconstruct",
        max_items: int = 5,
    ) -> List[SubtaskSpec]:
        """
        Take last_specs (often redundant multi-step) and ask LLM to compress/reconstruct
        into a minimal, non-redundant List[SubtaskSpec].

        - Keeps ONLY steps that add new information
        - Merges/removes repeated-answer steps like Extract/Parse/Report duplicates
        - Removes UNKNOWN by default
        - Returns JSON-only list of {subtask, expected_answer}
        """

        schema = {
            "type": "json_schema",
            "name": "reconstructed_subtask_specs",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "subtasks": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": max_items,
                        "items": {
                            "type": "object",
                            "properties": {
                                "subtask": {"type": "string"},
                                "expected_answer": {"type": "string"},
                                "expected_tool": {"type": ["string", "null"]},
                                "expected_tool_params": {"type": ["string", "null"]},
                            },
                            "required": [
                                "subtask",
                                "expected_answer",
                                "expected_tool",
                                "expected_tool_params",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["subtasks"],
                "additionalProperties": False,
            },
        }

        payload = {
            "last_specs": [
                {
                    "subtask": s.subtask,
                    "expected_answer": s.expected_answer,
                    "expected_tool": s.expected_tool,
                    "expected_tool_params": s.expected_tool_params,
                }
                for s in last_specs
            ],
            "instructions": [
                "Remove steps whose expected_answer is UNKNOWN or empty.",
                "If multiple steps have the same expected_answer and later steps are just rephrasing/reporting/parsing, keep only ONE representative step.",
                "Prefer the step that best reflects the actual information-gathering/computation, not a 'report the result' step.",
                "Do not introduce new subtasks or new answers not present in last_specs.",
                "Return 1 to 5 items max, only if they represent distinct information.",
            ],
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are restructuring a redundant multi-step plan.\n"
                    "Given last_specs = list of {subtask, expected_answer}, compress it into a minimal set.\n"
                    "Rules:\n"
                    "- Do NOT add any new facts or answers.\n"
                    "- Do NOT change expected_answer values.\n"
                    "- Remove duplicates and 'report/parse/extract' rephrasings when they repeat the same expected_answer.\n"
                    "- Output JSON only: {subtasks:[{subtask, expected_answer}, ...]}.\n"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

        resp = self.runner.run(
            messages,
            tools,
            kb=kb,
            node_id=node_id,
            label="llm_reconstruct_plan",
            schema=schema,
        )

        data = json.loads(resp.output_text or "{}")
        out = []
        for it in data.get("subtasks", []):
            sub = (it.get("subtask") or "").strip()
            ans = (it.get("expected_answer") or "").strip()
            et = it.get("expected_tool", None)
            etp = it.get("expected_tool_params", None)
            if sub and ans:
                out.append(
                    SubtaskSpec(
                        subtask=sub,
                        expected_answer=ans,
                        expected_tool=et,
                        expected_tool_params=etp,
                    )
                )

        concise_specs = "\n".join(
            [
                f"Subtask: {s.subtask}\nAnswer: {s.expected_answer}\nTool: {s.expected_tool}\nTool Params: {s.expected_tool_params}\n"
                for s in out
            ]
        )
        logger.info(f"Superviser Recunstructed.\n{concise_specs}")
        return out

    def _select_key_tool_call(
        self,
        tool_calls: List[Dict[str, Any]],
        final_answer: str,
        runner: LLMRunnerWithTools,
        tools: ToolRegistry,
        kb: Optional[JsonlKB],
        node_id: str,
    ) -> ToolContribution:
        """
        Simplified: return only (tool_name, tool_args) for the single tool call
        that the LLM judges as most critical to produce final_answer.
        """

        # quick fallback when no tool calls
        if not tool_calls:
            return ToolContribution(tool_name=None, tool_args=None)

        # prepare readable list
        enumerated = []
        for i, tc in enumerate(tool_calls):
            name = (
                tc.get("name") or tc.get("tool_name") or tc.get("tool") or "<unknown>"
            )
            args = tc.get("arguments", tc.get("args", tc.get("parameters", None)))
            result = tc.get("result", tc.get("output", tc.get("response", None)))
            try:
                args_s = (
                    json.dumps(args, ensure_ascii=False, sort_keys=True)
                    if args is not None
                    else None
                )
            except Exception:
                args_s = str(args) if args is not None else None
            try:
                res_s = (
                    json.dumps(result, ensure_ascii=False, sort_keys=True)
                    if result is not None
                    else None
                )
            except Exception:
                res_s = str(result) if result is not None else None
            enumerated.append({"idx": i, "name": name, "args": args_s, "result": res_s})

        # JSON schema: only tool_name/tool_args
        schema = {
            "type": "json_schema",
            "name": "select_key_tool_call_minimal",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": ["string", "null"]},
                    "tool_args": {"type": ["string", "null"]},
                },
                "required": ["tool_name", "tool_args"],
                "additionalProperties": False,
            },
        }

        sys_prompt = (
            "You are an ANALYZER. Given a final answer and a numbered list of executed tool calls "
            "(each has idx, name, args, result), return ONLY a JSON object with keys: "
            "'tool_name' and 'tool_args'. Choose the single tool call most critical for producing "
            "the final answer and set tool_name to its name and tool_args to a JSON-stringified "
            "version of its arguments. If none, return nulls."
        )

        user_payload = {"final_answer": final_answer, "tool_calls": enumerated}
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        try:
            resp = runner.run(
                messages,
                tools,
                kb=kb,
                node_id=node_id,
                label="llm_select_key_tool_call_minimal",
                schema=schema,
            )
            data = json.loads(resp.output_text or "{}")
            tool_name = data.get("tool_name")
            tool_args = data.get("tool_args")
            # if LLM returns valid tool_name/tool_args, trust it (but validate)
            if tool_name:
                return ToolContribution(tool_name=tool_name, tool_args=tool_args)
        except Exception:
            # fall through to heuristic fallback below
            pass

        # Heuristic fallback: pick last tool_call that has a non-empty result; else last tool_call
        picked = None
        for i in range(len(enumerated) - 1, -1, -1):
            if enumerated[i]["result"] not in (None, "", "null", "[]", "{}"):
                picked = i
                break
        if picked is None:
            picked = len(enumerated) - 1
        tc = enumerated[picked]
        return ToolContribution(tool_name=tc["name"], tool_args=tc["args"])
