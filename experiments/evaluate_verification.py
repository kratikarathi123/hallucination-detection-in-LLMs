"""
Evaluation for Person 3 (claim verification).

Run from the repo root:
    python -m experiments.evaluate_verification --data data/eval/sample_eval.json
    python -m experiments.evaluate_verification --data test.jsonl --tune-data dev.jsonl --out data/processed/eval_results.json

Generic input schema (JSON list or JSONL), one example per item:
{
  "id": "ex1",
  "claim": "...",
  "claim_language": "hi",
  "evidence": [{"text": "...", "language": "en"}],
  "label": "supported" | "contradicted" | "unsupported"
}

Dataset-specific loaders (FEVER, Mu-SHROOM, Person 2's output, ...) should be
small adapters that convert into this schema. This script never needs to change.

The model runs ONCE per (evidence, claim) pair. Threshold sweeps then re-use
those cached scores, so sweeping is cheap.
"""

import argparse
import json
from pathlib import Path

from src.verification.verifier import (
    nli_scores,
    aggregate,
    SUPPORT_THRESHOLD,
    CONTRADICT_THRESHOLD,
)

LABELS = ["supported", "contradicted", "unsupported"]
GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------- data

def load_examples(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix == ".jsonl":
        examples = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        examples = json.loads(text)

    for i, ex in enumerate(examples):
        for key in ("claim", "evidence", "label"):
            if key not in ex:
                raise ValueError(f"Example {i} is missing '{key}'.")
        if ex["label"] not in LABELS:
            raise ValueError(
                f"Example {i} has label '{ex['label']}', expected one of {LABELS}."
            )
    return examples


def condition(ex):
    """
    'mono'  : every evidence passage is in the claim's language
    'cross' : every evidence passage is in a different language
    'mixed' : some of each
    'unknown': no claim language, or no evidence with a language
    """
    claim_lang = ex.get("claim_language")
    ev_langs = [e.get("language") for e in ex["evidence"] if e.get("language")]
    if not claim_lang or not ev_langs:
        return "unknown"
    same = sum(1 for l in ev_langs if l == claim_lang)
    if same == len(ev_langs):
        return "mono"
    if same == 0:
        return "cross"
    return "mixed"


# ---------------------------------------------------------------- scoring

def score_examples(examples, scorer=nli_scores, label="", progress_every=50):
    """
    Run the NLI model once per (evidence, claim). Returns passage_results per example.
    On longer runs it prints a progress line with elapsed time and an estimate of the time left,
    so a quiet terminal is not mistaken for a frozen one. Do not press Ctrl+C while it runs.
    """
    import time

    total = len(examples)
    show = total >= 2 * progress_every
    start = time.time()
    all_results = []
    for i, ex in enumerate(examples, start=1):
        passage_results = []
        for ev in ex["evidence"]:
            text = (ev.get("text") or "").strip()
            if text:
                passage_results.append(
                    {"evidence": ev, "scores": scorer(text, ex["claim"])}
                )
        all_results.append(passage_results)
        if show and (i % progress_every == 0 or i == total):
            elapsed = time.time() - start
            left = elapsed / i * (total - i)
            print(f"  [{label or 'scoring'}] {i}/{total} done | elapsed {elapsed / 60:.1f} min | about {left / 60:.1f} min left", flush=True)
    return all_results


def predict(all_results, support_t, contradict_t):
    return [
        aggregate(pr, support_t, contradict_t)["label"] for pr in all_results
    ]


# ---------------------------------------------------------------- metrics

def compute_metrics(gold, pred):
    n = len(gold)
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        confusion[g][p] += 1

    per_label = {}
    for lab in LABELS:
        tp = confusion[lab][lab]
        fp = sum(confusion[g][lab] for g in LABELS if g != lab)
        fn = sum(confusion[lab][p] for p in LABELS if p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_label[lab] = {
            "precision": prec, "recall": rec, "f1": f1,
            "support": sum(confusion[lab].values()),
        }

    # Macro-F1 averages over labels that actually appear in the gold data.
    present = [l for l in LABELS if per_label[l]["support"] > 0]
    macro_f1 = sum(per_label[l]["f1"] for l in present) / len(present) if present else 0.0
    accuracy = sum(confusion[l][l] for l in LABELS) / n if n else 0.0

    return {
        "n": n, "accuracy": accuracy, "macro_f1": macro_f1,
        "per_label": per_label, "confusion": confusion,
    }


def grouped_metrics(examples, gold, pred, key_fn):
    groups = {}
    for ex, g, p in zip(examples, gold, pred):
        groups.setdefault(key_fn(ex), ([], []))
        groups[key_fn(ex)][0].append(g)
        groups[key_fn(ex)][1].append(p)
    return {k: compute_metrics(g, p) for k, (g, p) in sorted(groups.items())}


# ---------------------------------------------------------------- threshold sweep

def sweep_thresholds(all_results, gold, grid=GRID):
    rows = []
    for st in grid:
        for ct in grid:
            pred = predict(all_results, st, ct)
            rows.append({
                "support_threshold": st,
                "contradict_threshold": ct,
                "macro_f1": compute_metrics(gold, pred)["macro_f1"],
            })
    rows.sort(key=lambda r: r["macro_f1"], reverse=True)
    return rows


# ---------------------------------------------------------------- report

def print_metrics(title, m):
    print(f"\n--- {title} (n={m['n']}) ---")
    print(f"accuracy {m['accuracy']:.3f} | macro-F1 {m['macro_f1']:.3f}")
    for lab in LABELS:
        pl = m["per_label"][lab]
        print(f"  {lab:13s} P {pl['precision']:.3f}  R {pl['recall']:.3f}  "
              f"F1 {pl['f1']:.3f}  (n={pl['support']})")


def print_confusion(m):
    print("\nconfusion (rows = gold, cols = predicted)")
    print(" " * 14 + "".join(f"{l[:6]:>9s}" for l in LABELS))
    for g in LABELS:
        print(f"{g:14s}" + "".join(f"{m['confusion'][g][p]:9d}" for p in LABELS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="evaluation set (generic schema)")
    ap.add_argument("--tune-data", help="separate dev set for threshold tuning")
    ap.add_argument("--out", help="write full results to this JSON file")
    args = ap.parse_args()

    examples = load_examples(args.data)
    gold = [ex["label"] for ex in examples]
    results = score_examples(examples, label="evaluation set")

    # Thresholds: tuned on a separate dev set if given, else defaults.
    st, ct = SUPPORT_THRESHOLD, CONTRADICT_THRESHOLD
    sweep = None
    if args.tune_data:
        tune_ex = load_examples(args.tune_data)
        tune_gold = [ex["label"] for ex in tune_ex]
        sweep = sweep_thresholds(score_examples(tune_ex, label="tuning set"), tune_gold)
        st, ct = sweep[0]["support_threshold"], sweep[0]["contradict_threshold"]
        print(f"Thresholds tuned on {args.tune_data}: support={st}, contradict={ct}")
    else:
        print(f"Using default thresholds: support={st}, contradict={ct}")

    pred = predict(results, st, ct)

    overall = compute_metrics(gold, pred)
    by_lang = grouped_metrics(examples, gold, pred, lambda e: e.get("claim_language", "unknown"))
    by_cond = grouped_metrics(examples, gold, pred, condition)

    print_metrics("OVERALL", overall)
    print_confusion(overall)
    for lang, m in by_lang.items():
        print_metrics(f"claim language = {lang}", m)
    for cond, m in by_cond.items():
        print_metrics(f"condition = {cond}", m)

    if not args.tune_data:
        # Informational only: sweeping on the evaluation set itself is optimistic.
        sweep = sweep_thresholds(results, gold)
        print("\nTop threshold settings on THIS set (optimistic; tune on a separate dev set):")
        for r in sweep[:5]:
            print(f"  support {r['support_threshold']}  contradict {r['contradict_threshold']}  "
                  f"macro-F1 {r['macro_f1']:.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "thresholds": {"support": st, "contradict": ct},
            "overall": overall, "by_language": by_lang,
            "by_condition": by_cond, "sweep_top10": sweep[:10],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
