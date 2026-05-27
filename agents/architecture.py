import logging

from core.state import AgentState
from core.llm import call_llm
from core.errors import make_fail
from core.parsers import parse_llm_json
from prompts.architecture import SYSTEM_PROMPT, USER_TEMPLATE

logger = logging.getLogger(__name__)


def archi_node(state: AgentState) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(PROMPT=state["prompt"])},
    ]

    fix_feedback = state.get("fix_feedback") or {}
    fix_instruction = fix_feedback.get("fix_instruction", "")
    arch_retry_count = state.get("arch_retry_count", 0)
    if fix_instruction and arch_retry_count > 0:
        arch_error_history = state.get("arch_error_history") or []
        fix_msg = f"REQUIRED CHANGE:\n{fix_instruction}"
        past = [e.get("fix_instruction", "")[:200]
                for e in arch_error_history[-2:]
                if e.get("fix_instruction") and e.get("fix_instruction") != fix_instruction]
        if past:
            fix_msg += "\n\nPREVIOUS ATTEMPTS (do NOT repeat):\n" + "\n".join(f"- {p}" for p in past)
        messages.append({"role": "user", "content": fix_msg})

    try:
        raw = call_llm(messages)
        plan = parse_llm_json(raw, {"resources": list, "data_sources": list})
    except Exception as e:
        return make_fail("INFRA", None, f"Archi agent error: {e}")

    logger.info("Archi agent: %d resources, %d data_sources",
                len(plan.get("resources", [])), len(plan.get("data_sources", [])))
    return {"infrastructure_plan": plan}
