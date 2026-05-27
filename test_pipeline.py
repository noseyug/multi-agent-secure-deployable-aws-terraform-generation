"""Test script — chạy full pipeline A1→A2→A3→A4→A5 trên data-test.csv.

Chạy:
    uv run python3 test_pipeline.py
    uv run python3 test_pipeline.py --limit 5
    uv run python3 test_pipeline.py --cases 0 3 7-10
    uv run python3 test_pipeline.py --no-secu            # bỏ qua A2, security_ckv_ids = {}
    uv run python3 test_pipeline.py --no-deploy          # dừng sau A4
    uv run python3 test_pipeline.py --no-destroy         # giữ lại resources sau apply
    uv run python3 test_pipeline.py --workers 3          # chạy 3 row song song
    uv run python3 test_pipeline.py --out reviews/pipeline_results.json
"""
import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from agents.architecture import archi_node
from agents.security import secu_node
from agents.engineering import engi_node
from agents.validation import validation_node, route_after_validation
from agents.deployment import deployment_node, route_after_deployment

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

CSV_PATH = ROOT / "dataset" / "data-dev.csv"
_RESOURCE_RE = re.compile(r'resource\s+"[^"]+"\s+"[^"]+"')
_PRINT_LOCK = threading.Lock()

# Giới hạn vòng retry trong test — khớp RECURSION_LIMIT / ~4 node/cycle
MAX_ITERATIONS = 20


def make_state(prompt: str, idx: int = 0, auto_destroy: bool = True) -> dict:
    run_dir = ROOT / "tmp" / f"row_{idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "prompt": prompt,
        "auto_destroy": auto_destroy,
        "terraform_plan_timeout": int(os.environ.get("TF_PLAN_TIMEOUT", "120")),
        "infrastructure_plan": {},
        "security_ckv_ids": {},
        "generated_code": "",
        "fix_feedback": {},
        "deployment_result": {},
        "arch_retry_count": 0,
        "sec_retry_count": 0,
        "eng_retry_count": 0,
        "total_retry_count": 0,
        "deploy_retry_count": 0,
        "deploy_eng_retry_count": 0,
        "error_history": [],
        "arch_error_history": [],
        "sec_error_history": [],
        "eng_error_history": [],
        "routing_log": [],
        "run_dir": str(run_dir),
    }


def load_csv(limit: int | None) -> list[tuple[int, str, str]]:
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    return [(i, row.get("Difficulty", ""), row["Prompt"]) for i, row in enumerate(rows)]


def _parse_cases(tokens: list[str]) -> set[int]:
    result = set()
    for part in tokens:
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    return result


def _n_resources(code: str) -> int:
    return len(_RESOURCE_RE.findall(code))


# ─── Per-agent runners (trả về elapsed và cập nhật state inplace) ────────────

def _run_archi(state: dict) -> tuple[bool, float, str | None]:
    t0 = time.time()
    r = archi_node(state)
    elapsed = round(time.time() - t0, 2)
    state.update(r)
    ok = bool(state.get("infrastructure_plan"))
    err = None if ok else str(r.get("error", "unknown"))
    return ok, elapsed, err


def _run_secu(state: dict) -> float:
    t0 = time.time()
    r = secu_node(state)
    elapsed = round(time.time() - t0, 2)
    state.update(r)
    return elapsed


def _run_engi(state: dict) -> tuple[bool, float, str | None]:
    t0 = time.time()
    r = engi_node(state)
    elapsed = round(time.time() - t0, 2)
    state.update(r)
    ok = bool(state.get("generated_code", "").strip())
    err = None if ok else str(r.get("error", "unknown"))
    return ok, elapsed, err


def _run_val(state: dict) -> tuple[bool, float, str]:
    t0 = time.time()
    r = validation_node(state)
    elapsed = round(time.time() - t0, 2)
    state.update(r)
    fb = state.get("fix_feedback") or {}
    passed = bool(fb.get("overall_passed"))
    route = route_after_validation(state)
    return passed, elapsed, route


def _run_deploy(state: dict) -> tuple[bool, float, str]:
    t0 = time.time()
    r = deployment_node(state)
    elapsed = round(time.time() - t0, 2)
    state.update(r)
    dr = state.get("deployment_result") or {}
    ok = bool(dr.get("success"))
    route = route_after_deployment(state)
    return ok, elapsed, route


# ─── Row runner ───────────────────────────────────────────────────────────────

def run_row(idx: int, difficulty: str, prompt: str,
            deploy: bool, auto_destroy: bool, no_secu: bool = False) -> tuple[dict, str]:
    """Chạy 1 row qua toàn bộ pipeline. Trả về (result_dict, output_str).

    Output được buffer nội bộ thay vì print trực tiếp — cho phép gọi song song
    mà không bị interleave giữa các worker.
    """
    lines: list[str] = []
    def log(msg: str = "") -> None:
        lines.append(msg)

    sep = "=" * 72
    log(f"\n{sep}")
    log(f"ROW {idx:4d}  difficulty={difficulty or '?'}")
    log(f"  {prompt[:100]}")
    log(sep)

    state = make_state(prompt, idx=idx, auto_destroy=auto_destroy)

    archi_result = secu_result = engi_result = val_result = deploy_result = None
    val_attempts = deploy_attempts = 0

    next_agent = "architecture"
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        # ── A1: archi ──────────────────────────────────────────────────────
        if next_agent == "architecture":
            ok, elapsed, err = _run_archi(state)
            plan = state.get("infrastructure_plan") or {}
            n_res = len(plan.get("resources", []))
            archi_result = {
                "ok": ok, "elapsed_s": elapsed,
                "resource_count": n_res, "plan": plan,
            }
            if not ok:
                log(f"  [archi] FAILED ({elapsed}s): {err}")
                archi_result["error"] = err
                break
            log(f"  [archi] {n_res} resources ({elapsed}s)")
            next_agent = "security"

        # ── A2: secu ───────────────────────────────────────────────────────
        if next_agent == "security":
            if no_secu:
                secu_result = {"ok": True, "elapsed_s": 0, "skipped": True,
                               "ckv_resource_count": 0, "ckv_total": 0, "ckv_ids": {}}
                log(f"  [secu]  skipped (--no-secu)")
            else:
                elapsed = _run_secu(state)
                ckv_ids = state.get("security_ckv_ids") or {}
                n_ckv_res = len(ckv_ids)
                n_ckv_total = sum(len(v) for v in ckv_ids.values())
                secu_result = {
                    "ok": True, "elapsed_s": elapsed,
                    "ckv_resource_count": n_ckv_res, "ckv_total": n_ckv_total,
                    "ckv_ids": ckv_ids,
                }
                log(f"  [secu]  {n_ckv_res} resources, {n_ckv_total} CKV IDs ({elapsed}s)")
                for label, checks in ckv_ids.items():
                    log(f"    {label}: {', '.join(checks)}")
            next_agent = "engineering"

        # ── A3: engi ───────────────────────────────────────────────────────
        if next_agent == "engineering":
            ok, elapsed, err = _run_engi(state)
            code = state.get("generated_code", "")
            n_gen = _n_resources(code)
            n_lines = code.count("\n")
            engi_result = {
                "ok": ok, "elapsed_s": elapsed,
                "resource_count": n_gen, "line_count": n_lines,
                "generated_code": code,
            }
            if not ok:
                log(f"  [engi]  FAILED ({elapsed}s): {err}")
                engi_result["error"] = err
                break
            log(f"  [engi]  {n_gen} resources, {n_lines} lines ({elapsed}s)")
            next_agent = "validation"

        # ── A4: val ────────────────────────────────────────────────────────
        if next_agent == "validation":
            val_attempts += 1
            passed, elapsed, route = _run_val(state)
            fb = state.get("fix_feedback") or {}
            et = fb.get("error_type", "")
            ck = fb.get("checkov") or {}

            val_result = {
                "ok": passed,
                "elapsed_s": elapsed,
                "error_type": et,
                "checkov_passed": ck.get("passed_count", 0),
                "checkov_failed": ck.get("failed", []),
                "validate_ok": fb.get("validate_ok"),
                "plan_ok": fb.get("plan_ok"),
                "raw_error": (fb.get("raw_error") or "")[:2000],
                "fix_instruction": (fb.get("fix_instruction") or "")[:500],
                "attempts": val_attempts,
            }

            status = "PASS" if passed else f"FAIL [{et}]"
            ck_str = (f"ckv pass={ck.get('passed_count',0)} "
                      f"fail={ck.get('failed',[])}") if ck else ""
            log(f"  [val]   {status} ({elapsed}s) {ck_str}"
                f" → route={route} attempt={val_attempts}")
            if not passed and fb.get("fix_instruction"):
                log(f"  [val]   fix: {fb['fix_instruction']}")

            if route == "agent5":
                if not deploy:
                    log("  [deploy] skipped (--no-deploy)")
                    break
                next_agent = "deployment"
            elif route == "requires_human":
                log(f"  [val]   → REQUIRES_HUMAN")
                break
            else:
                next_agent = route
                log(f"  [val]   → retry via {next_agent} "
                    f"(total_retry={state.get('total_retry_count',0)})")
                continue

        # ── A5: deploy ─────────────────────────────────────────────────────
        if next_agent == "deployment":
            deploy_attempts += 1
            ok, elapsed, route = _run_deploy(state)
            dr = state.get("deployment_result") or {}
            et = dr.get("error_type", "")
            created = dr.get("resources_created", [])
            destroyed = dr.get("auto_destroyed", False)

            deploy_result = {
                "ok": ok,
                "elapsed_s": elapsed,
                "error_type": et if not ok else None,
                "resources_created": created,
                "auto_destroyed": destroyed,
                "auto_destroy_error": dr.get("auto_destroy_error"),
                "apply_raw_error": (dr.get("apply_raw_error") or "")[:2000],
                "fix_instruction": (dr.get("fix_instruction") or "")[:500],
                "attempts": deploy_attempts,
            }

            if ok:
                d_str = "(destroyed)" if destroyed else "(resources kept)"
                log(f"  [deploy] OK ({elapsed}s) {len(created)} resources {d_str}")
                break
            else:
                log(f"  [deploy] FAIL [{et}] ({elapsed}s) → route={route} "
                    f"attempt={deploy_attempts}")
                if dr.get("fix_instruction"):
                    log(f"  [deploy] fix: {dr['fix_instruction']}")
                if dr.get("apply_raw_error"):
                    first_err = "\n".join(
                        ln for ln in (dr["apply_raw_error"]).splitlines()
                        if ln.strip().lower().startswith("error")
                    )[:300]
                    if first_err:
                        log(f"  [deploy] err: {first_err}")

            if route == "end":
                break
            elif route == "agent5":
                next_agent = "deployment"
            elif route == "requires_human":
                log(f"  [deploy] → REQUIRES_HUMAN")
                break
            else:
                next_agent = route
                log(f"  [deploy] → retry via {next_agent}")

    # Cleanup per-run dir sau khi xong — xóa a4/, a5/, files/ nhưng giữ lại nếu muốn debug
    run_dir_path = Path(state.get("run_dir", ""))
    if run_dir_path.exists():
        shutil.rmtree(run_dir_path, ignore_errors=True)

    result = {
        "row": idx,
        "difficulty": difficulty,
        "prompt": prompt,
        "archi": archi_result,
        "secu": secu_result,
        "engi": engi_result,
        "val": val_result,
        "deploy": deploy_result,
        "total_retry_count": state.get("total_retry_count", 0),
        "deploy_retry_count": state.get("deploy_retry_count", 0),
        "routing_log": state.get("routing_log", []),
        "iterations": iteration,
    }
    return result, "\n".join(lines)


def _update_counters(counters: dict, r: dict, no_deploy: bool, lock: threading.Lock) -> None:
    def _ok(key): return r.get(key) and r[key].get("ok")
    with lock:
        if _ok("archi"):   counters["ok1"] += 1
        else:              counters["fail1"] += 1
        if r["archi"]:
            if _ok("secu"):  counters["ok2"] += 1
            elif r["secu"]:  counters["fail2"] += 1
        if r["secu"]:
            if _ok("engi"):  counters["ok3"] += 1
            elif r["engi"]:  counters["fail3"] += 1
        if r["engi"]:
            if _ok("val"):   counters["ok4"] += 1
            elif r["val"]:   counters["fail4"] += 1
        if r["val"] and not no_deploy:
            if _ok("deploy"):  counters["ok5"] += 1
            elif r["deploy"]:  counters["fail5"] += 1


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test full pipeline A1→A2→A3→A4→A5")
    parser.add_argument("--limit", type=int, default=None, help="Số row tối đa")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Row indices, e.g. --cases 0 3 7-10 15")
    parser.add_argument("--no-secu", action="store_true",
                        help="Bỏ qua A2, security_ckv_ids = {} (test A1→A3→A4→A5)")
    parser.add_argument("--no-deploy", action="store_true",
                        help="Dừng sau A4 (không chạy terraform apply)")
    parser.add_argument("--no-destroy", action="store_true",
                        help="Giữ resources sau apply (không auto-destroy)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Số worker chạy song song (mặc định 1 = tuần tự)")
    args = parser.parse_args()

    provider = os.getenv("LLM_PROVIDER", "nvidia").lower()
    if provider == "deepseek":
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    else:
        model = os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
    deploy_str = "no-deploy" if args.no_deploy else ("no-destroy" if args.no_destroy else "auto-destroy")
    print(f"Full pipeline A1→A2→A3→A4→A5  |  model={model}  "
          f"|  csv={CSV_PATH.name}  |  deploy={deploy_str}  |  workers={args.workers}")

    rows = load_csv(args.limit)
    if args.cases:
        selected = _parse_cases(args.cases)
        rows = [(i, d, p) for i, d, p in rows if i in selected]
        print(f"--cases filter: {len(rows)} rows")
    print(f"Loaded {len(rows)} rows\n")

    results: list[dict] = []
    counters = {k: 0 for k in ("ok1", "ok2", "ok3", "ok4", "ok5",
                                "fail1", "fail2", "fail3", "fail4", "fail5")}
    counter_lock = threading.Lock()

    def _run_one(row_args: tuple) -> tuple[dict, str]:
        idx, difficulty, prompt = row_args
        return run_row(idx, difficulty, prompt,
                       deploy=not args.no_deploy,
                       auto_destroy=not args.no_destroy,
                       no_secu=args.no_secu)

    interrupted = False

    if args.workers <= 1:
        # ── Tuần tự ────────────────────────────────────────────────────────
        for idx, difficulty, prompt in rows:
            try:
                r, output = _run_one((idx, difficulty, prompt))
                print(output)
                results.append(r)
                _update_counters(counters, r, args.no_deploy, counter_lock)
            except KeyboardInterrupt:
                print("\n[interrupted]")
                interrupted = True
                break
            except Exception as e:
                print(f"  [error] row={idx}: {e}")
                import traceback; traceback.print_exc()
                with counter_lock:
                    counters["fail1"] += 1
    else:
        # ── Song song ──────────────────────────────────────────────────────
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {
                executor.submit(_run_one, row_args): row_args[0]
                for row_args in rows
            }
            try:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        r, output = future.result()
                        with _PRINT_LOCK:
                            print(output)
                        results.append(r)
                        _update_counters(counters, r, args.no_deploy, counter_lock)
                    except Exception as e:
                        with _PRINT_LOCK:
                            print(f"  [error] row={idx}: {e}")
                            import traceback; traceback.print_exc()
                        with counter_lock:
                            counters["fail1"] += 1
            except KeyboardInterrupt:
                print("\n[interrupted — cancelling remaining futures]")
                for f in future_to_idx:
                    f.cancel()
                interrupted = True

        # Sắp xếp lại theo row index (as_completed không đảm bảo thứ tự)
        results.sort(key=lambda r: r["row"])

    total = counters["ok1"] + counters["fail1"]
    print(f"\n{'='*72}")
    print(f"SUMMARY  total={total}" + (" [interrupted]" if interrupted else ""))
    print(f"  A1 archi:  {counters['ok1']}/{total}  ok")
    if counters["ok1"]:
        print(f"  A2 secu:   {counters['ok2']}/{counters['ok1']}  ok  "
              f"(always passes — thin CKV assignment)")
        print(f"  A3 engi:   {counters['ok3']}/{counters['ok1']}  ok")
    if counters["ok3"]:
        print(f"  A4 val:    {counters['ok4']}/{counters['ok3']}  ok")
    if counters["ok4"] and not args.no_deploy:
        print(f"  A5 deploy: {counters['ok5']}/{counters['ok4']}  ok")

    out_path = (Path(args.out) if args.out
                else ROOT / "reviews" / "pipeline_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(results)} results → {out_path}")


if __name__ == "__main__":
    main()
