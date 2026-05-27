"""Deployment Agent — Agent 5 trong pipeline.

Thực thi `terraform apply` lên Floci. Nếu fail:
  - kiểm tra partial apply (state list) → terraform destroy để cleanup dirty state;
  - phân loại lỗi rồi route.

Phân loại:
  - Timeout / connection → TRANSIENT → retry A5 (deploy_retry_count, tối đa 2 lần)
  - FIXABLE → A3 sửa code (deploy_retry_count <= 2, tối đa 2 lần route về A3)
  - MISSING_RESOURCE → A1 re-plan (deploy_retry_count <= 2, tối đa 2 lần route về A1)
  - Còn lại → LLM phân loại FIXABLE / UNKNOWN
    FIXABLE → A3 sửa code (qua fix_feedback, tăng eng_retry_count, deploy_retry <= 2)
    UNKNOWN → requires_human

State writes:
  - deployment_result: luôn cập nhật
  - deploy_retry_count: chỉ tăng khi A5 xử lý lỗi (cả TRANSIENT lẫn FIXABLE/UNKNOWN)
  - fix_feedback + eng_retry_count: chỉ set khi FIXABLE để trigger A3 retry
"""
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path

# Attrs that prevent AWS resource deletion — must be disabled before destroy.
# Each tuple: (pattern, replacement). Applied in order via re.sub.
_DESTROY_PATCHES = [
    # DynamoDB deletion protection
    (r'(deletion_protection_enabled\s*=\s*)true', r'\g<1>false'),
    # RDS / Aurora / DocumentDB / ALB deletion protection
    (r'(deletion_protection\s*=\s*)true', r'\g<1>false'),
    # RDS: must skip final snapshot when destroying
    (r'(skip_final_snapshot\s*=\s*)false', r'\g<1>true'),
    # RDS: remove final_snapshot_identifier (conflicts with skip_final_snapshot=true)
    (r'\n[ \t]*final_snapshot_identifier\s*=\s*[^\n]+', ''),
    # ElastiCache: remove final snapshot to speed up deletion
    (r'\n[ \t]*final_snapshot_identifier\s*=\s*[^\n]+', ''),
    (r'(apply_immediately\s*=\s*)false', r'\g<1>true'),
    # ElastiCache: disable multi-AZ to allow faster deletion
    (r'(automatic_failover_enabled\s*=\s*)true', r'\g<1>false'),
    (r'(multi_az_enabled\s*=\s*)true', r'\g<1>false'),
]


def _patch_for_destroy(code: str) -> str:
    """Disable deletion-protection attrs so terraform destroy can succeed."""
    for pattern, replacement in _DESTROY_PATCHES:
        code = re.sub(pattern, replacement, code)
    return code

from core.state import AgentState
from core.llm import call_llm
from core.parsers import parse_llm_json
from core.terraform import run_terraform, write_terraform_dir, terraform_workdir
from prompts.deployment import SYSTEM_PROMPT as _SYSTEM_PROMPT
from prompts.deployment import TOP_PROMPT as _TOP, BOTTOM_PROMPT as _BOTTOM

logger = logging.getLogger(__name__)

_INIT_TIMEOUT = 60
_APPLY_TIMEOUT = 360
_DESTROY_TIMEOUT = 600  # ElastiCache/RDS deletion có thể mất 5-10 phút
_STATE_TIMEOUT = 30

_TRANSIENT_PATTERNS = (
    "connection refused", "connection reset", "could not connect",
    "timeout", "timed out", "i/o timeout", "eof", "no such host",
    # AWS rate limits / quota — retry thay vì fail
    "requestlimitexceeded", "throttling", "rate exceeded",
    "vpcquotaexceeded", "limitexceeded",
)


def _matches(text: str, patterns: tuple) -> bool:
    low = (text or "").lower()
    return any(p in low for p in patterns)


def _extract_error(stdout: str, stderr: str) -> str:
    """Trích error text từ terraform apply output.

    Terraform ghi plan vào stdout (dài) và lỗi vào stderr (ngắn).
    Nếu chỉ lấy tail của (stderr+stdout), stderr ngắn bị cắt mất.
    Fix: giữ toàn bộ stderr + tail của stdout để LLM luôn thấy lỗi thực.
    """
    stderr_clean = (stderr or "").strip()
    stdout_tail = (stdout or "")[-2000:]
    combined = (stderr_clean + "\n" + stdout_tail).strip()
    error_lines = [ln for ln in combined.splitlines() if re.match(r"\s*(?:Error|error):", ln)]
    if error_lines:
        return combined + "\n\n--- Error lines ---\n" + "\n".join(error_lines[-20:])
    return combined


def _resource_labels(plan: dict) -> list[str]:
    return [f"{r['type']}.{r['name']}" for r in plan.get("resources", [])]


def _guess_failed_resource(error_text: str, labels: list[str]) -> str | None:
    """Đoán resource gây lỗi từ error text — cung cấp hint cho LLM."""
    for label in labels:
        rtype, rname = label.split(".", 1)
        if rtype in error_text or rname in error_text or label in error_text:
            return label
    return None


def _deploy_result(success: bool, error_type: str | None, *, fix_instruction=None,
                   resources_created=None, partial_apply_destroyed=False,
                   destroy_failed=False, destroy_error=None, apply_raw_error=None) -> dict:
    return {
        "success": success,
        "error_type": error_type,
        "resources_created": resources_created or [],
        "partial_apply_destroyed": partial_apply_destroyed,
        "destroy_failed": destroy_failed,
        "destroy_error": destroy_error,
        "fix_instruction": fix_instruction,
        "apply_raw_error": apply_raw_error,
    }


def _state_resources(tmpdir: str) -> list:
    try:
        r = run_terraform(["terraform", "state", "list"], tmpdir, _STATE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return []
    if r.returncode != 0:
        return []
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]


def _llm_classify_deploy(
    error_text: str,
    resource_labels: list[str],
    failed_resource: str | None,
    partial: bool,
    destroyed: bool,
    retry: int,
) -> tuple[str, str | None]:
    """LLM phân loại FIXABLE / UNKNOWN + sinh fix_instruction. Fallback UNKNOWN."""
    ctx = (
        _TOP
        + f"RESOURCE LIST: {json.dumps(resource_labels)}\n"
        + f"SUSPECTED FAILED RESOURCE: {failed_resource or 'unknown'}\n\n"
        + f"APPLY ERROR:\n{error_text[:2000]}\n\n"
        + f"PARTIAL APPLY: {partial} | DESTROYED: {destroyed} | DEPLOY RETRY: {retry}"
        + _BOTTOM
    )
    try:
        parsed = parse_llm_json(
            call_llm([{"role": "system", "content": _SYSTEM_PROMPT},
                      {"role": "user", "content": ctx}]),
            {"error_type": None, "fix_instruction": None},
        )
    except Exception as e:
        logger.warning("Agent 5 LLM classify error (%s) — UNKNOWN", e)
        return "UNKNOWN", None
    et = parsed.get("error_type")
    if et not in ("FIXABLE", "MISSING_RESOURCE", "PERMISSION", "QUOTA", "UNKNOWN"):
        et = "UNKNOWN"
    fix = parsed.get("fix_instruction") if et in ("FIXABLE", "MISSING_RESOURCE") else None
    return et, (str(fix)[:500] if fix else None)


def _handle_failure(
    state: AgentState, tmpdir: str,
    apply_stdout: str, apply_stderr: str,
    is_timeout: bool,
) -> dict:
    """Xử lý apply fail: cleanup partial state, phân loại, trả dict update."""
    error_text = _extract_error(apply_stdout, apply_stderr)
    plan = state.get("infrastructure_plan") or {}
    resource_labels = _resource_labels(plan)

    # Pattern-based classification (tất định, không cần LLM)
    if is_timeout:
        error_type = "TRANSIENT"
    elif _matches(error_text, _TRANSIENT_PATTERNS):
        error_type = "TRANSIENT"
    else:
        error_type = None  # cần LLM

    # Khi timeout, terraform bị SIGKILL giữa chừng — state file có thể rỗng/corrupt.
    # Chạy refresh trước để rebuild state từ AWS thực tế.
    if is_timeout:
        try:
            run_terraform(["terraform", "refresh", "-no-color"], tmpdir, 60)
        except subprocess.TimeoutExpired:
            pass  # best-effort

    # Cleanup partial state — LUÔN chạy destroy (safe nếu state rỗng: no-op).
    created = _state_resources(tmpdir)
    partial = bool(created)
    partial_destroyed = destroy_failed = False
    destroy_error = None

    try:
        destroy = run_terraform(
            ["terraform", "destroy", "-auto-approve", "-no-color"],
            tmpdir, _DESTROY_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        destroy_failed = True
        destroy_error = f"terraform destroy timed out (>{_DESTROY_TIMEOUT}s)"
    else:
        if destroy.returncode == 0:
            partial_destroyed = True
        else:
            destroy_failed = True
            destroy_error = (destroy.stderr or "")[:500]

    # LLM classify chỉ khi chưa xác định được error_type
    fix = None
    if error_type is None:
        failed_resource = _guess_failed_resource(error_text, resource_labels)
        error_type, fix = _llm_classify_deploy(
            error_text, resource_labels, failed_resource,
            partial, partial_destroyed, state["deploy_retry_count"],
        )

    logger.info(
        "Agent 5: FAIL %s (partial=%s destroyed=%s destroy_failed=%s)",
        error_type, partial, partial_destroyed, destroy_failed,
    )

    result: dict = {
        "deployment_result": _deploy_result(
            False, error_type,
            fix_instruction=fix,
            resources_created=created,
            partial_apply_destroyed=partial_destroyed,
            destroy_failed=destroy_failed,
            destroy_error=destroy_error,
            apply_raw_error=error_text[:3000],
        ),
        "deploy_retry_count": state["deploy_retry_count"] + 1,
    }

    # FIXABLE: HCL code sai → route về A3 để sửa code.
    if error_type == "FIXABLE" and not destroy_failed:
        result["fix_feedback"] = {
            "overall_passed": False,
            "error_type": "LOGIC",
            "root_cause": "engineering",
            "fix_instruction": fix,
            "checkov": {"passed_count": 0, "failed": []},
            "validate_passed": True,
            "plan_passed": True,
        }
        result["deploy_eng_retry_count"] = state.get("deploy_eng_retry_count", 0) + 1

    # MISSING_RESOURCE: resource thiếu trong plan → route về A1 để re-plan.
    if error_type == "MISSING_RESOURCE" and not destroy_failed:
        result["fix_feedback"] = {
            "overall_passed": False,
            "error_type": "MISSING_RESOURCE",
            "root_cause": "architecture",
            "fix_instruction": fix,
            "checkov": {"passed_count": 0, "failed": []},
            "validate_passed": True,
            "plan_passed": True,
        }
        result["arch_retry_count"] = state["arch_retry_count"] + 1

    return result


def destroy_resources(code: str) -> dict:
    """Chạy terraform init + destroy trên HCL code. Không cần state — dùng độc lập.

    Returns:
        {"success": bool, "error": str | None, "resources_destroyed": list[str]}
    """
    with tempfile.TemporaryDirectory() as d:
        write_terraform_dir(d, code)

        logger.info("destroy: terraform init (timeout=%ds)", _INIT_TIMEOUT)
        try:
            init = run_terraform(["terraform", "init", "-no-color"], d, _INIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"terraform init timed out (>{_INIT_TIMEOUT}s)",
                    "resources_destroyed": []}

        if init.returncode != 0:
            return {"success": False, "error": f"terraform init failed: {init.stderr[:300]}",
                    "resources_destroyed": []}

        resources = _state_resources(d)

        logger.info("destroy: terraform destroy (timeout=%ds)", _DESTROY_TIMEOUT)
        try:
            result = run_terraform(
                ["terraform", "destroy", "-auto-approve", "-no-color"], d, _DESTROY_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"success": False,
                    "error": f"terraform destroy timed out (>{_DESTROY_TIMEOUT}s)",
                    "resources_destroyed": []}

        if result.returncode == 0:
            logger.info("destroy: OK — %d resources", len(resources))
            return {"success": True, "error": None, "resources_destroyed": resources}

        error = _extract_error(result.stdout or "", result.stderr or "")
        logger.error("destroy: FAILED — %s", error[:200])
        return {"success": False, "error": error[:500], "resources_destroyed": []}


def deployment_node(state: AgentState) -> dict:
    """LangGraph node function cho Deployment Agent (Agent 5)."""
    code = state["generated_code"]

    logger.info(
        "Agent 5: deploy_retry=%d eng_retry=%d",
        state.get("deploy_retry_count", 0),
        state.get("eng_retry_count", 0),
    )

    run_dir = state.get("run_dir") or ""
    files_dir = (Path(run_dir) / "files") if run_dir else None

    with terraform_workdir(run_dir or None, "a5") as d:
        write_terraform_dir(d, code, files_dir=files_dir)

        logger.info("Agent 5: terraform init (timeout=%ds)", _INIT_TIMEOUT)
        try:
            init = run_terraform(["terraform", "init", "-no-color"], d, _INIT_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.error("Agent 5: terraform init TIMEOUT")
            return {
                "deployment_result": _deploy_result(
                    False, "TRANSIENT",
                    fix_instruction=f"terraform init timed out (>{_INIT_TIMEOUT}s)",
                ),
                "deploy_retry_count": state["deploy_retry_count"] + 1,
            }

        if init.returncode != 0:
            logger.error("Agent 5: terraform init FAILED")
            return {
                "deployment_result": _deploy_result(
                    False, "TRANSIENT",
                    fix_instruction=f"terraform init failed: {init.stderr[:300]}",
                ),
                "deploy_retry_count": state["deploy_retry_count"] + 1,
            }

        logger.info("Agent 5: terraform apply (timeout=%ds)", _APPLY_TIMEOUT)
        try:
            apply = run_terraform(
                ["terraform", "apply", "-auto-approve", "-no-color"], d, _APPLY_TIMEOUT
            )
        except subprocess.TimeoutExpired:
            logger.error("Agent 5: terraform apply TIMEOUT")
            return _handle_failure(
                state, d, "", "terraform apply timed out", is_timeout=True
            )

        if apply.returncode == 0:
            created = _state_resources(d)
            logger.info("Agent 5: APPLY OK — %d resources", len(created))

            auto_destroyed = False
            auto_destroy_error = None
            if state.get("auto_destroy"):
                logger.info("Agent 5: auto-destroy (eval mode)")
                tf_path = Path(d) / "main.tf"
                # Patch: disable deletion-protection attrs, then re-apply before destroy
                original = tf_path.read_text(encoding="utf-8")
                patched = _patch_for_destroy(original)
                if patched != original:
                    logger.info("Agent 5: patching deletion-protection attrs before destroy")
                    tf_path.write_text(patched, encoding="utf-8")
                    try:
                        run_terraform(
                            ["terraform", "apply", "-auto-approve", "-no-color"],
                            d, _APPLY_TIMEOUT,
                        )
                    except subprocess.TimeoutExpired:
                        pass  # best-effort; proceed to destroy anyway
                try:
                    cleanup = run_terraform(
                        ["terraform", "destroy", "-auto-approve", "-no-color"],
                        d, _DESTROY_TIMEOUT,
                    )
                    auto_destroyed = cleanup.returncode == 0
                    if not auto_destroyed:
                        auto_destroy_error = (cleanup.stderr or "")[:300]
                        logger.warning("Agent 5: auto-destroy FAILED — %s", auto_destroy_error)
                    else:
                        logger.info("Agent 5: auto-destroy OK")
                except subprocess.TimeoutExpired:
                    auto_destroy_error = f"terraform destroy timed out (>{_DESTROY_TIMEOUT}s)"
                    logger.warning("Agent 5: auto-destroy TIMEOUT")

            result = _deploy_result(True, None, resources_created=created)
            result["auto_destroyed"] = auto_destroyed
            result["auto_destroy_error"] = auto_destroy_error
            return {"deployment_result": result}

        return _handle_failure(
            state, d, apply.stdout or "", apply.stderr or "", is_timeout=False
        )


def route_after_deployment(state: AgentState) -> str:
    """Conditional edge sau Agent 5. KHÔNG ghi state."""
    dr = state["deployment_result"]

    if dr["success"]:
        return "end"

    # Dirty state không cleanup được → luôn cần người can thiệp
    if dr.get("destroy_failed"):
        return "requires_human"

    error_type = dr["error_type"]
    deploy_retry = state["deploy_retry_count"]  # đã +1 trong node

    deploy_eng_retry = state.get("deploy_eng_retry_count", 0)  # đã +1 trong node
    if error_type == "TRANSIENT" and deploy_retry <= 2:
        return "agent5"
    if error_type == "FIXABLE" and deploy_eng_retry <= 3:
        return "engineering"
    if error_type == "MISSING_RESOURCE" and deploy_retry <= 2:
        return "architecture"
    # PERMISSION, QUOTA, UNKNOWN, retry exhausted — all require human
    logger.info("Agent 5: route requires_human (error_type=%s deploy_retry=%d)", error_type, deploy_retry)
    return "requires_human"
