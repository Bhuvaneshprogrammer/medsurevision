"""
scripts/evaluate.py — Evaluation benchmark for MedSure Vision pipeline.

Reads eval_data/labels.csv (filename, true_class: untampered|tampered).
Runs each image through the full pipeline.
Outputs:
  - Confusion matrix over the four labels (low_risk, needs_verification, high_suspicion, insufficient_quality)
  - FALSE-SAFE RATE (tampered packs labelled low_risk)
  - Per-class precision, recall, and F1
  - Abstention rate (needs_verification + insufficient_quality)
  - p50 (median) and p95 latency
Saves: docs/eval_results.md, backend/eval_results.md, and backend/eval_results.json

NOTE: Tampered images are synthetic and self-made for internal testing only.
      Registry results are simulated. Results do not constitute clinical validation.
"""
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "eval_data"
OUT_JSON = ROOT / "eval_results.json"
OUT_MD_BACKEND = ROOT / "eval_results.md"
OUT_MD_DOCS = WORKSPACE / "docs" / "eval_results.md"


def run_pipeline_on_image(image_path: Path) -> tuple[str, float]:
    """Runs the MedSure Vision pipeline on a single image.
    
    Returns (predicted_label, latency_ms).
    """
    from app.pipeline.quality import assess_quality
    from app.pipeline.ocr import run_ocr
    from app.pipeline.fields import extract_fields
    from app.pipeline.codes import decode_codes
    from app.pipeline.visual import run_visual
    from app.pipeline.rules import evaluate_hard_rules
    from app.pipeline.fusion import fuse

    t0 = time.perf_counter()
    img_bytes = image_path.read_bytes()

    # 1. Image Quality Assessment
    qr = assess_quality(img_bytes)
    if not qr.ok:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return "insufficient_quality", latency_ms

    # 2. OCR and Field Extraction
    merged_text, ocr_blocks = run_ocr(img_bytes)
    avg_conf = (
        sum(b.confidence for b in ocr_blocks) / len(ocr_blocks)
        if ocr_blocks else 0.5
    )
    fields = extract_fields(merged_text, avg_block_confidence=avg_conf)

    # 3. 2D / 1D Barcode Decoding
    code = decode_codes(img_bytes)

    # 4. Visual Similarity Comparison
    med_id = fields.matched_medicine.get("id") if (fields and fields.matched_medicine) else None
    visual = run_visual(img_bytes, medicine_id=med_id)

    # 5. Deterministic Rules Evaluation
    hard = evaluate_hard_rules(
        fields=fields,
        code=code,
        alert=None,
        clone=None,
        registry=None,
    )

    # 6. Multi-Signal Fusion
    result = fuse(
        quality_ok=qr.ok,
        hard_rules=hard,
        fields=fields,
        code=code,
        visual=visual,
        alert=None,
        clone=None,
        registry=None,
        partial=False,
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return result.label, latency_ms


def main():
    labels_csv = EVAL_DIR / "labels.csv"
    if not labels_csv.exists():
        print(f"ERROR: {labels_csv} not found.")
        print("Run gen_synthetic_eval.py and make_eval_set.py first.")
        sys.exit(1)

    rows = list(csv.DictReader(open(labels_csv, encoding="utf-8")))
    if not rows:
        print("labels.csv is empty.")
        sys.exit(0)

    latencies: list[float] = []
    results: list[dict] = []
    
    # 4 distinct pipeline labels
    labels_order = ["low_risk", "needs_verification", "high_suspicion", "insufficient_quality"]
    true_classes = ["untampered", "tampered"]

    confusion: dict[str, dict[str, int]] = {
        tc: {lbl: 0 for lbl in labels_order} for tc in true_classes
    }
    label_counts: dict[str, int] = {lbl: 0 for lbl in labels_order}

    print(f"Running evaluation over {len(rows)} samples in {EVAL_DIR}...")

    for row in rows:
        fname = row["filename"].strip()
        true_class = row["true_class"].strip()
        img_path = EVAL_DIR / fname

        if not img_path.exists():
            print(f"  SKIP (missing file): {fname}")
            continue

        try:
            label, ms = run_pipeline_on_image(img_path)
        except Exception as e:
            print(f"  ERROR evaluating {fname}: {e}")
            continue

        latencies.append(ms)
        results.append({
            "filename": fname,
            "true_class": true_class,
            "predicted": label,
            "latency_ms": ms,
        })

        if true_class in confusion:
            confusion[true_class][label] = confusion[true_class].get(label, 0) + 1
        label_counts[label] = label_counts.get(label, 0) + 1

    if not latencies:
        print("No images were successfully processed.")
        return

    # Metrics calculation
    latencies.sort()
    total_images = len(results)
    total_untampered = sum(confusion["untampered"].values())
    total_tampered = sum(confusion["tampered"].values())

    p50_idx = int(0.50 * len(latencies))
    p95_idx = int(0.95 * len(latencies))
    p50_ms = round(latencies[min(p50_idx, len(latencies) - 1)], 2)
    p95_ms = round(latencies[min(p95_idx, len(latencies) - 1)], 2)
    mean_ms = round(statistics.mean(latencies), 2)
    min_ms = round(min(latencies), 2)
    max_ms = round(max(latencies), 2)

    # FALSE-SAFE RATE: tampered packs labelled low_risk
    tampered_as_low_risk = confusion["tampered"]["low_risk"]
    false_safe_rate = round((tampered_as_low_risk / total_tampered * 100), 2) if total_tampered > 0 else 0.0

    # Abstention Rate: needs_verification or insufficient_quality
    abstention_count = (
        label_counts.get("needs_verification", 0) + label_counts.get("insufficient_quality", 0)
    )
    abstention_rate = round((abstention_count / total_images * 100), 2) if total_images > 0 else 0.0

    # Per-class metrics
    # 1. Untampered class (target: low_risk)
    tp_untampered = confusion["untampered"]["low_risk"]
    fp_untampered = confusion["tampered"]["low_risk"]
    fn_untampered = sum(confusion["untampered"][k] for k in labels_order if k != "low_risk")
    prec_untampered = round((tp_untampered / (tp_untampered + fp_untampered) * 100), 2) if (tp_untampered + fp_untampered) > 0 else 0.0
    rec_untampered = round((tp_untampered / (tp_untampered + fn_untampered) * 100), 2) if (tp_untampered + fn_untampered) > 0 else 0.0
    f1_untampered = round((2 * prec_untampered * rec_untampered / (prec_untampered + rec_untampered)), 2) if (prec_untampered + rec_untampered) > 0 else 0.0

    # 2. Tampered class (target: non-low_risk flagged)
    tp_tampered = sum(confusion["tampered"][k] for k in labels_order if k != "low_risk")
    fp_tampered = sum(confusion["untampered"][k] for k in labels_order if k != "low_risk")
    fn_tampered = confusion["tampered"]["low_risk"]
    prec_tampered = round((tp_tampered / (tp_tampered + fp_tampered) * 100), 2) if (tp_tampered + fp_tampered) > 0 else 0.0
    rec_tampered = round((tp_tampered / (tp_tampered + fn_tampered) * 100), 2) if (tp_tampered + fn_tampered) > 0 else 0.0
    f1_tampered = round((2 * prec_tampered * rec_tampered / (prec_tampered + rec_tampered)), 2) if (prec_tampered + rec_tampered) > 0 else 0.0

    summary = {
        "note": "Tampered images are synthetic and self-made for internal testing only. Registry results are simulated. These results do not constitute clinical validation.",
        "total_images": total_images,
        "total_untampered": total_untampered,
        "total_tampered": total_tampered,
        "false_safe_count": tampered_as_low_risk,
        "false_safe_rate_pct": false_safe_rate,
        "abstention_count": abstention_count,
        "abstention_rate_pct": abstention_rate,
        "latency_ms": {
            "p50": p50_ms,
            "p95": p95_ms,
            "mean": mean_ms,
            "min": min_ms,
            "max": max_ms,
        },
        "confusion_matrix": confusion,
        "per_label_counts": label_counts,
        "per_class_metrics": {
            "untampered": {
                "precision_pct": prec_untampered,
                "recall_pct": rec_untampered,
                "f1": f1_untampered,
                "support": total_untampered,
            },
            "tampered": {
                "precision_pct": prec_tampered,
                "recall_pct": rec_tampered,
                "f1": f1_tampered,
                "support": total_tampered,
            },
        },
    }

    # Save JSON summary
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Markdown Report
    md_content = f"""# MedSure Vision — Evaluation Benchmark Results

> **Note:** Tampered images are synthetic and self-made for internal testing only. Registry results are simulated. These results do not constitute clinical validation.

## Key Performance Metrics

| Metric | Value | Description |
|---|---|---|
| **Total Evaluation Samples** | {total_images} | Total processed images ({total_untampered} untampered, {total_tampered} tampered) |
| **FALSE-SAFE RATE** | **{false_safe_rate}%** | Fraction of tampered packs mistakenly classified as `low_risk` ({tampered_as_low_risk}/{total_tampered}) |
| **Abstention Rate** | **{abstention_rate}%** | Inconclusive cases requiring physical inspection (`needs_verification` + `insufficient_quality`) |
| **p50 Latency (Median)** | **{p50_ms} ms** | Median end-to-end pipeline execution time |
| **p95 Latency** | **{p95_ms} ms** | 95th percentile latency |
| **Mean Latency** | **{mean_ms} ms** | Average pipeline execution time |

---

## Confusion Matrix

| True Class | low_risk | needs_verification | high_suspicion | insufficient_quality | Total |
|---|---|---|---|---|---|
| **Untampered** | {confusion['untampered']['low_risk']} | {confusion['untampered']['needs_verification']} | {confusion['untampered']['high_suspicion']} | {confusion['untampered']['insufficient_quality']} | {total_untampered} |
| **Tampered** | {confusion['tampered']['low_risk']} | {confusion['tampered']['needs_verification']} | {confusion['tampered']['high_suspicion']} | {confusion['tampered']['insufficient_quality']} | {total_tampered} |
| **Total Predicted** | {label_counts['low_risk']} | {label_counts['needs_verification']} | {label_counts['high_suspicion']} | {label_counts['insufficient_quality']} | {total_images} |

---

## Per-Class Precision and Recall

| Evaluation Class | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| **Untampered** (Safe Verification) | {prec_untampered}% | {rec_untampered}% | {f1_untampered}% | {total_untampered} |
| **Tampered** (Anomaly Detection) | {prec_tampered}% | {rec_tampered}% | {f1_tampered}% | {total_tampered} |

---

## Predicted Label Distribution

| Pipeline Label | Count | Share |
|---|---|---|
| `low_risk` | {label_counts['low_risk']} | {round(label_counts['low_risk'] / total_images * 100, 1)}% |
| `needs_verification` | {label_counts['needs_verification']} | {round(label_counts['needs_verification'] / total_images * 100, 1)}% |
| `high_suspicion` | {label_counts['high_suspicion']} | {round(label_counts['high_suspicion'] / total_images * 100, 1)}% |
| `insufficient_quality` | {label_counts['insufficient_quality']} | {round(label_counts['insufficient_quality'] / total_images * 100, 1)}% |

---

## Latency Profile

- **Minimum:** {min_ms} ms
- **p50 (Median):** {p50_ms} ms
- **p95:** {p95_ms} ms
- **Maximum:** {max_ms} ms
- **Mean:** {mean_ms} ms
"""

    # Save to docs/eval_results.md and backend/eval_results.md
    OUT_MD_DOCS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD_DOCS, "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(OUT_MD_BACKEND, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\nEvaluation Complete.")
    print(f"Results written to:")
    print(f"  - {OUT_JSON}")
    print(f"  - {OUT_MD_DOCS}")
    print(f"  - {OUT_MD_BACKEND}")
    print(f"FALSE-SAFE RATE: {false_safe_rate}% ({tampered_as_low_risk}/{total_tampered})")
    print(f"Abstention Rate: {abstention_rate}% ({abstention_count}/{total_images})")
    print(f"p50: {p50_ms} ms | p95: {p95_ms} ms | Mean: {mean_ms} ms")


if __name__ == "__main__":
    main()
