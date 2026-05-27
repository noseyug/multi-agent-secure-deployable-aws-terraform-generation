"""Engineering Agent (engi) — nhận output của archi_node + secu_node, sinh Terraform HCL.

Input state:
  infrastructure_plan  — JSON plan từ Agent 1 (archi)
  security_ckv_ids     — CKV check IDs từ Agent 2 (secu)

Output state:
  generated_code       — HCL string hoàn chỉnh (provider block đã prepend)
"""
import json
import logging
import re
from pathlib import Path

from core.llm import call_llm
from core.errors import make_fail
from core.parsers import strip_code_block
from prompts.engineering import SYSTEM_PROMPT as _SYSTEM_PROMPT, USER_TEMPLATE as _USER_TEMPLATE

logger = logging.getLogger(__name__)

_PLAN_TAG = re.compile(r"<plan>.*?</plan>", re.DOTALL | re.IGNORECASE)
_PROVIDER_BLOCK = (Path(__file__).parent.parent / "core" / "provider.tf").read_text(encoding="utf-8").strip()
_BLOCK_HEADER = re.compile(r'(?m)^[ \t]*(terraform|provider)\b[^\n{]*\{')
_RESOURCE_DECL_RE = re.compile(r'resource\s+"([^"]+)"\s+"([^"]+)"')
_HCL_BLOCK_START = re.compile(r'(?:resource|data|variable|output|module|moved|import)\s+"')


def _strip_preamble(hcl: str) -> str:
    m = _HCL_BLOCK_START.search(hcl)
    return hcl[m.start():] if m else hcl


def _strip_injected_blocks(hcl: str) -> str:
    """Xóa terraform{} / provider{} do LLM sinh (prompt yêu cầu emit nhưng ta prepend tĩnh)."""
    while True:
        m = _BLOCK_HEADER.search(hcl)
        if not m:
            return hcl
        open_idx = m.end() - 1
        depth, end_idx = 0, None
        for i in range(open_idx, len(hcl)):
            if hcl[i] == "{":
                depth += 1
            elif hcl[i] == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        if end_idx is None:
            return hcl[: m.start()].strip()
        hcl = (hcl[: m.start()] + hcl[end_idx:]).strip()



def engi_node(state: dict) -> dict:
    """LangGraph node — serialize A1 plan sang HCL, áp dụng CKV requirements từ A2."""
    archi_plan = state.get("infrastructure_plan") or {}
    if not archi_plan.get("resources"):
        return make_fail(
            "MISSING_RESOURCE", "architecture",
            "Engi agent nhận infrastructure_plan rỗng — archi agent phải chạy trước.",
        )

    ckv_ids = state.get("security_ckv_ids") or {}
    if ckv_ids:
        ckv_lines = "\n".join(f"  {label}: {', '.join(ids)}" for label, ids in ckv_ids.items())
    else:
        ckv_lines = "  (none)"

    user_content = _USER_TEMPLATE.format(
        PLAN=json.dumps(archi_plan, indent=2),
        CKV_REQUIREMENTS=ckv_lines,
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    fix_feedback = state.get("fix_feedback") or {}
    fix_instruction = fix_feedback.get("fix_instruction", "")
    eng_retry_count = state.get("eng_retry_count", 0)
    if fix_instruction and (eng_retry_count > 0 or state.get("deploy_eng_retry_count", 0) > 0):
        eng_error_history = state.get("eng_error_history") or []
        fix_msg = f"REQUIRED FIX (apply exactly):\n{fix_instruction}"
        past = [
            e.get("fix_instruction", "")[:200]
            for e in eng_error_history[-2:]
            if e.get("fix_instruction") and e.get("fix_instruction") != fix_instruction
        ]
        if past:
            fix_msg += "\n\nPREVIOUS ERRORS (do NOT repeat these):\n" + "\n".join(f"- {p}" for p in past)
        messages.append({"role": "user", "content": fix_msg})

    raw = ""
    try:
        raw = call_llm(messages)
    except TimeoutError as e:
        logger.error("Engi agent timeout: %s", e)
        return make_fail("INFRA", None, f"Engi agent LLM timeout: {e}")
    except Exception as e:
        logger.error("Engi agent error: %s", e)
        return make_fail("INFRA", None, f"Engi agent error: {e}")

    cleaned = _PLAN_TAG.sub("", raw).strip()
    body = _strip_preamble(_strip_injected_blocks(strip_code_block(cleaned).strip()))

    # Guard: phải có ít nhất một resource block
    if 'resource "' not in body:
        logger.warning("Engi agent: không có resource block — retry")
        retry_msgs = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                "Your response did not contain any `resource \"` blocks. "
                "Output the complete Terraform HCL with ALL resource blocks "
                "from the plan. Do not omit any resource."
            )},
        ]
        try:
            raw = call_llm(retry_msgs)
        except Exception as e:
            return make_fail("INFRA", None, f"Engi agent retry error: {e}")
        cleaned = _PLAN_TAG.sub("", raw).strip()
        body = _strip_preamble(_strip_injected_blocks(strip_code_block(cleaned).strip()))
        if 'resource "' not in body:
            return make_fail(
                "SYNTAX", "engineering",
                f"Engi agent không sinh được resource block (sau retry). Raw: {raw[:300]}",
            )

    generated_code = f"{_PROVIDER_BLOCK}\n\n{body}\n"

    gen_pairs = set(_RESOURCE_DECL_RE.findall(body))
    logger.info("Engi agent: %d chars, %d resources", len(generated_code), len(gen_pairs))
    return {"generated_code": generated_code}
