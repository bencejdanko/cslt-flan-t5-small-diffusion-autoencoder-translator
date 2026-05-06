"""LLM-as-judge for ASL->English translation outputs.

Reads predictions from translation/eval_results.json and scores each
(reference, prediction) pair on three 1-5 dimensions (semantic, tone, fluency)
using a local Ollama-served LLM. Writes scores + aggregates to
translation/llm_judge_results.json.

Run:
    ollama serve &
    ollama pull llama3.1:8b
    python3 translation/llm_judge.py --preflight
    python3 translation/llm_judge.py --limit 25 --output translation/llm_judge_smoke_results.json --overwrite
    python3 translation/llm_judge.py --resume            # full run, resumable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREDS = REPO_ROOT / "translation" / "eval_results.json"
DEFAULT_OUTPUT = REPO_ROOT / "translation" / "llm_judge_results.json"

MAX_REF_CHARS = 1500
DIMS = ("semantic", "tone", "fluency")

PROMPT_TEMPLATE = """You are a strict evaluator of machine translation outputs from American Sign Language (ASL) video to English. You will be given a REFERENCE (the human ground-truth English) and a PREDICTION (the model output).

CRITICAL CONTEXT: The model under evaluation is known to produce fluent but generic English templates (e.g. "I'm going to show you how to do that"). Fluency alone is NOT evidence of a good translation. Score SEMANTIC strictly on whether the PREDICTION conveys the actual MEANING of the REFERENCE -- not on whether it sounds plausible or grammatical.

Score three dimensions on a 1-5 integer scale:

SEMANTIC (meaning overlap with reference):
  1 = Unrelated. No content overlap with the reference.
  2 = Wrong topic but shares one or two incidental words.
  3 = Same general domain or one matching key concept; major content missing.
  4 = Most key entities/actions captured; minor omissions or distortions.
  5 = Faithfully conveys the reference meaning.

TONE & REGISTER (instructional vs narrative, formality, person/voice):
  1 = Completely mismatched register (e.g. tutorial voice for a personal anecdote).
  2 = Mostly mismatched.
  3 = Partially aligned register.
  4 = Close match with minor drift.
  5 = Register, voice, and formality match the reference.

FLUENCY (English grammar and naturalness of the prediction alone):
  1 = Ungrammatical or incoherent.
  2 = Many errors; hard to read.
  3 = Understandable with noticeable errors.
  4 = Mostly fluent; minor issues.
  5 = Native-quality English.

Be strict. Reserve 5 for genuinely excellent cases. If the prediction is a generic template that does not reflect the reference's specific content, SEMANTIC must be 1 or 2 regardless of fluency.

REFERENCE: {ref}
PREDICTION: {pred}

Respond with ONLY a JSON object on a single line, no prose, no markdown:
{{"semantic": <1-5>, "tone": <1-5>, "fluency": <1-5>, "rationale": "<200 chars or fewer>"}}"""

STRICT_REMINDER = "\n\nReturn ONLY valid JSON matching the schema. No prose, no markdown, no code fences."


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

class JudgeBackend(ABC):
    @abstractmethod
    def score(self, ref: str, pred: str) -> dict[str, Any]:
        ...

    @abstractmethod
    def health_check(self) -> None:
        ...


class OllamaBackend(JudgeBackend):
    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: int = 120,
        num_ctx: int = 2048,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx

    def health_check(self) -> None:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=10
            ) as resp:
                tags = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError) as e:
            raise SystemExit(
                f"Cannot reach Ollama at {self.base_url} ({e}). "
                "Start it with: ollama serve"
            )

        names = {m.get("name", "") for m in tags.get("models", [])}
        # ollama tags can return "llama3.1:8b" or "llama3.1:8b-instruct-q4_K_M"
        if not any(n == self.model or n.startswith(self.model + "-") for n in names):
            raise SystemExit(
                f"Ollama model {self.model!r} is not pulled. "
                f"Run: ollama pull {self.model}"
            )

    def _post(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_ctx": self.num_ctx},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            outer = json.loads(resp.read().decode("utf-8"))
        return outer.get("response", "")

    def _call_with_backoff(self, prompt: str) -> str:
        delays = (2, 8, 32)
        last_err: Exception | None = None
        for attempt, delay in enumerate((0,) + delays):
            if delay:
                time.sleep(delay)
            try:
                return self._post(prompt)
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last_err = e
                sys.stderr.write(
                    f"  [warn] connection error (attempt {attempt + 1}): {e}\n"
                )
        raise RuntimeError(f"Ollama request failed after retries: {last_err}")

    def score(self, ref: str, pred: str) -> dict[str, Any]:
        prompt = PROMPT_TEMPLATE.format(ref=ref, pred=pred)
        raw = self._call_with_backoff(prompt)
        parsed = _parse_judge_json(raw)
        if parsed is None:
            raw2 = self._call_with_backoff(prompt + STRICT_REMINDER)
            parsed = _parse_judge_json(raw2)
            if parsed is None:
                return {
                    "semantic": None,
                    "tone": None,
                    "fluency": None,
                    "rationale": f"PARSE_ERROR: {raw2[:200]!r}",
                    "_error": True,
                }
        return parsed


BACKENDS: dict[str, type[JudgeBackend]] = {"ollama": OllamaBackend}


# --------------------------------------------------------------------------- #
# JSON parsing & validation
# --------------------------------------------------------------------------- #

def _parse_judge_json(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    # Tolerate stray prose around the JSON object.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    out: dict[str, Any] = {}
    clamped = False
    for dim in DIMS:
        v = obj.get(dim)
        if isinstance(v, bool) or v is None:
            return None
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        if iv < 1 or iv > 5:
            iv = max(1, min(5, iv))
            clamped = True
        out[dim] = iv

    rationale = obj.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    out["rationale"] = rationale[:300]
    if clamped:
        out["_clamped"] = True
    return out


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def load_predictions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"{path}: predictions file does not exist")
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    preds = data.get("predictions")
    if not isinstance(preds, list):
        raise SystemExit(f"{path}: missing 'predictions' list")
    for i, pred in enumerate(preds):
        if not isinstance(pred, dict):
            raise SystemExit(f"{path}: predictions[{i}] is not an object")
        if "ref" not in pred or "pred" not in pred:
            raise SystemExit(f"{path}: predictions[{i}] must contain ref and pred")
        if not isinstance(pred["ref"], str) or not isinstance(pred["pred"], str):
            raise SystemExit(f"{path}: predictions[{i}].ref and .pred must be strings")
    return preds


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_signature(
    args: argparse.Namespace,
    preds_path: Path,
    predictions_sha256: str,
    n_total: int,
) -> dict[str, Any]:
    return {
        "backend": args.backend,
        "model": args.model,
        "predictions_file": str(preds_path),
        "predictions_sha256": predictions_sha256,
        "n_total": n_total,
        "limit": args.limit,
        "num_ctx": args.num_ctx,
    }


def load_existing(
    path: Path,
    expected_signature: dict[str, Any],
    n_total: int,
) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError:
        backup = path.with_suffix(path.suffix + ".corrupt.bak")
        path.rename(backup)
        raise SystemExit(
            f"Existing output is corrupt; moved to {backup}. "
            "Re-run without --resume to start fresh."
        )
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: existing output must be a JSON object")
    meta = data.get("meta")
    if not isinstance(meta, dict):
        raise SystemExit(f"{path}: existing output is missing meta")
    found_signature = meta.get("run_signature")
    if found_signature != expected_signature:
        raise SystemExit(
            f"{path}: resume metadata does not match this run. "
            "Use the same model/input/options, or rerun with --overwrite."
        )
    scored = data.get("scores", []) or []
    if not isinstance(scored, list):
        raise SystemExit(f"{path}: existing output scores must be a list")

    out: dict[int, dict[str, Any]] = {}
    for i, entry in enumerate(scored):
        if not isinstance(entry, dict):
            raise SystemExit(f"{path}: scores[{i}] is not an object")
        idx = entry.get("idx")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise SystemExit(f"{path}: scores[{i}].idx must be an integer")
        if idx < 0 or idx >= n_total:
            raise SystemExit(f"{path}: scores[{i}].idx {idx} is outside 0..{n_total - 1}")
        if idx in out:
            raise SystemExit(f"{path}: duplicate score idx {idx}")
        out[idx] = entry
    return out


def save_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def length_bucket(ref: str) -> str:
    n = len(ref.split())
    if n <= 8:
        return "short"
    if n <= 20:
        return "medium"
    return "long"


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #

def _dim_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "std": None,
                "hist": {str(k): 0 for k in range(1, 6)}, "n": 0}
    hist = {str(k): 0 for k in range(1, 6)}
    for v in values:
        hist[str(v)] = hist.get(str(v), 0) + 1
    return {
        "mean": round(statistics.fmean(values), 3),
        "median": statistics.median(values),
        "std": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
        "hist": hist,
        "n": len(values),
    }


def compute_aggregates(scored: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [s for s in scored if not s.get("_error") and s.get("semantic") is not None]

    overall = {dim: _dim_stats([s[dim] for s in valid]) for dim in DIMS}

    by_bucket: dict[str, dict[str, Any]] = {}
    for bucket in ("short", "medium", "long"):
        rows = [s for s in valid if s.get("len_bucket") == bucket]
        by_bucket[bucket] = {
            "n": len(rows),
            **{dim: _dim_stats([r[dim] for r in rows]) for dim in DIMS},
        }

    ranked = sorted(
        valid,
        key=lambda s: (s["total"], s["semantic"]),
        reverse=True,
    )
    keep = ("idx", "ref", "pred", "semantic", "tone", "fluency", "total", "rationale")
    top_best = [{k: r[k] for k in keep if k in r} for r in ranked[:10]]
    top_worst = [{k: r[k] for k in keep if k in r} for r in ranked[-10:][::-1]]

    return {
        "overall": overall,
        "by_length_bucket": by_bucket,
        "top10_best": top_best,
        "top10_worst": top_worst,
    }


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def _truncate_ref(ref: str) -> str:
    if len(ref) <= MAX_REF_CHARS:
        return ref
    return ref[:MAX_REF_CHARS] + "...[truncated]"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_payload(
    scored_by_idx: dict[int, dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    scored_list = [scored_by_idx[i] for i in sorted(scored_by_idx)]
    meta = dict(meta)
    meta["n_scored"] = len(scored_list)
    meta["n_errors"] = sum(1 for s in scored_list if s.get("_error"))
    return {
        "meta": meta,
        "aggregates": compute_aggregates(scored_list),
        "scores": scored_list,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    if args.num_ctx <= 0:
        raise SystemExit("--num-ctx must be > 0")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")


def validate_output_parent(path: Path) -> None:
    parent = path.parent
    if not parent.exists():
        raise SystemExit(f"{parent}: output directory does not exist")
    if not parent.is_dir():
        raise SystemExit(f"{parent}: output parent is not a directory")
    if not os.access(parent, os.W_OK):
        raise SystemExit(f"{parent}: output directory is not writable")


def enforce_output_policy(path: Path, args: argparse.Namespace) -> None:
    if path.exists() and not args.resume and not args.overwrite:
        raise SystemExit(
            f"{path} already exists. Use --resume to continue it or --overwrite "
            "to replace it."
        )


def print_preflight(
    args: argparse.Namespace,
    preds_path: Path,
    out_path: Path,
    n_total: int,
    predictions_sha256: str,
) -> None:
    sys.stderr.write("Preflight OK\n")
    sys.stderr.write(f"  backend: {args.backend}\n")
    sys.stderr.write(f"  model: {args.model}\n")
    sys.stderr.write(f"  predictions: {preds_path} ({n_total} pairs)\n")
    sys.stderr.write(f"  predictions_sha256: {predictions_sha256}\n")
    sys.stderr.write(f"  output: {out_path}\n")
    if out_path.exists():
        if args.resume:
            sys.stderr.write("  output_status: exists; --resume will validate metadata\n")
        elif args.overwrite:
            sys.stderr.write("  output_status: exists; --overwrite will replace it\n")
        else:
            sys.stderr.write(
                "  output_status: exists; scoring requires --resume or --overwrite\n"
            )
    else:
        sys.stderr.write("  output_status: new file\n")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    preds_path = Path(args.predictions_file).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    if args.backend not in BACKENDS:
        raise SystemExit(
            f"Unknown backend {args.backend!r}. Available: {sorted(BACKENDS)}"
        )
    pairs = load_predictions(preds_path)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    predictions_sha256 = file_sha256(preds_path)
    signature = run_signature(args, preds_path, predictions_sha256, len(pairs))
    validate_output_parent(out_path)

    backend_cls = BACKENDS[args.backend]
    backend = backend_cls(
        model=args.model,
        base_url=args.ollama_url,
        timeout=args.timeout,
        num_ctx=args.num_ctx,
    )
    backend.health_check()

    if args.preflight:
        print_preflight(args, preds_path, out_path, len(pairs), predictions_sha256)
        return 0

    enforce_output_policy(out_path, args)

    existing = load_existing(out_path, signature, len(pairs)) if args.resume else {}

    todo = [
        (i, p) for i, p in enumerate(pairs)
        if i not in existing or existing[i].get("_error")
    ]
    if existing:
        sys.stderr.write(
            f"Resuming: {len(existing)} already scored, "
            f"{len(todo)} remaining of {len(pairs)} total.\n"
        )
    else:
        sys.stderr.write(f"Scoring {len(todo)} pairs from {preds_path.name}.\n")

    scored_by_idx: dict[int, dict[str, Any]] = dict(existing)

    meta = {
        "backend": args.backend,
        "model": args.model,
        "predictions_file": str(preds_path),
        "predictions_sha256": predictions_sha256,
        "run_signature": signature,
        "limit": args.limit,
        "num_ctx": args.num_ctx,
        "started_at": _now_iso(),
        "finished_at": None,
        "seconds": None,
        "samples_per_sec": None,
        "n_total": len(pairs),
    }

    if not todo:
        sys.stderr.write("Nothing to score (all entries already present).\n")
        meta["finished_at"] = _now_iso()
        meta["seconds"] = 0.0
        meta["samples_per_sec"] = 0.0
        save_atomic(out_path, build_payload(scored_by_idx, meta))
        _print_summary(out_path, scored_by_idx, t0=time.monotonic())
        return 0

    t0 = time.monotonic()
    n_done_this_run = 0
    n_errors_this_run = 0

    for batch_start in range(0, len(todo), args.batch_size):
        batch = todo[batch_start : batch_start + args.batch_size]
        for idx, pair in batch:
            ref = (pair.get("ref") or "").strip()
            pred = (pair.get("pred") or "").strip()
            bucket = length_bucket(ref) if ref else "short"

            if not ref or not pred:
                entry = {
                    "idx": idx,
                    "ref": ref,
                    "pred": pred,
                    "len_bucket": bucket,
                    "semantic": 1, "tone": 1, "fluency": 1,
                    "rationale": "EMPTY_INPUT",
                    "total": 3,
                    "_error": True,
                }
                n_errors_this_run += 1
            else:
                try:
                    s = backend.score(_truncate_ref(ref), pred)
                except Exception as e:  # backend exhausted retries
                    s = {
                        "semantic": None, "tone": None, "fluency": None,
                        "rationale": f"BACKEND_ERROR: {e}",
                        "_error": True,
                    }
                entry = {
                    "idx": idx,
                    "ref": ref,
                    "pred": pred,
                    "len_bucket": bucket,
                    **s,
                }
                if not entry.get("_error"):
                    entry["total"] = entry["semantic"] + entry["tone"] + entry["fluency"]
                else:
                    entry["total"] = 0
                    n_errors_this_run += 1

            scored_by_idx[idx] = entry
            n_done_this_run += 1

        elapsed = time.monotonic() - t0
        rate = n_done_this_run / elapsed if elapsed > 0 else 0.0
        remaining = len(todo) - (batch_start + len(batch))
        eta_sec = remaining / rate if rate > 0 else 0.0
        sys.stderr.write(
            f"  [{batch_start + len(batch)}/{len(todo)}] "
            f"{rate:.2f} samples/s | errors={n_errors_this_run} | "
            f"eta={_fmt_eta(eta_sec)}\n"
        )

        meta["finished_at"] = _now_iso()
        meta["seconds"] = round(elapsed, 1)
        meta["samples_per_sec"] = round(rate, 3)
        save_atomic(out_path, build_payload(scored_by_idx, meta))

    _print_summary(out_path, scored_by_idx, t0)
    return 0


def _fmt_eta(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _print_summary(
    out_path: Path,
    scored_by_idx: dict[int, dict[str, Any]],
    t0: float,
) -> None:
    elapsed = time.monotonic() - t0
    n = len(scored_by_idx)
    n_err = sum(1 for s in scored_by_idx.values() if s.get("_error"))
    valid = [s for s in scored_by_idx.values() if not s.get("_error")]
    means = {
        d: round(statistics.fmean([s[d] for s in valid]), 3) if valid else None
        for d in DIMS
    }
    sys.stderr.write("\n=== LLM Judge Summary ===\n")
    sys.stderr.write(f"Output:   {out_path}\n")
    sys.stderr.write(f"Scored:   {n} ({n_err} errors)\n")
    sys.stderr.write(f"Elapsed:  {_fmt_eta(elapsed)}\n")
    sys.stderr.write(
        f"Means:    semantic={means['semantic']}  "
        f"tone={means['tone']}  fluency={means['fluency']}\n"
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--predictions-file", default=str(DEFAULT_PREDS))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--backend", default="ollama", choices=sorted(BACKENDS))
    p.add_argument("--model", default="llama3.1:8b")
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--limit", type=int, default=0,
                   help="Max pairs to score (0 = all)")
    p.add_argument("--resume", action="store_true",
                   help="Skip indices already in --output")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace an existing --output instead of refusing to run")
    p.add_argument("--preflight", action="store_true",
                   help="Check Ollama, model, input, and output readiness without scoring")
    p.add_argument("--batch-size", type=int, default=25)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--num-ctx", type=int, default=2048)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
