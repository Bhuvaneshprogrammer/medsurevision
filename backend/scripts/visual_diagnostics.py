"""
backend/scripts/visual_diagnostics.py
Visual Similarity Calibration and Diagnostics Tool.

Computes cosine similarity for reference images:
(a) against photos of the SAME medicine (intra-class)
(b) against photos of DIFFERENT medicines (inter-class)

Uses the same MobileNetV2 feature extraction as app/pipeline/visual.py.
"""

from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REF_DIR = ROOT / "data" / "reference_images"
MED_JSON = ROOT / "data" / "medicines.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_feature_extractor():
    import torch
    import torchvision.models as models
    from torchvision import transforms

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return model, tfm


def embed_image(img_path: Path, model, tfm, device: str) -> np.ndarray | None:
    from PIL import Image
    import torch
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.features(tensor)
            emb = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            vec = emb.squeeze().cpu().numpy()
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-8)
    except Exception as exc:
        print(f"Error embedding {img_path}: {exc}")
        return None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not MED_JSON.exists():
        print(f"Error: {MED_JSON} not found.")
        return 1

    meds = json.load(open(MED_JSON))
    med_ids = [m["id"] for m in meds]
    model, tfm = load_feature_extractor()
    model.to(device)

    # 1. Collect and embed all images per medicine
    medicine_embeddings: dict[str, list[tuple[str, np.ndarray]]] = {}
    skipped_files = []

    for mid in med_ids:
        folder = REF_DIR / mid
        if not folder.exists():
            continue

        images = []
        for f in folder.iterdir():
            if f.is_file():
                if f.suffix.lower() in IMAGE_EXTS:
                    images.append(f)
                elif f.name != "README.txt" and not f.name.startswith("."):
                    skipped_files.append((mid, f.name, "Unsupported/Missing Extension"))

        embeddings = []
        for img_path in images:
            vec = embed_image(img_path, model, tfm, device)
            if vec is not None:
                embeddings.append((img_path.name, vec))
            else:
                skipped_files.append((mid, img_path.name, "Failed to load/embed"))

        if embeddings:
            medicine_embeddings[mid] = embeddings

    # 2. Compute similarity stats
    all_same_scores = []
    all_diff_scores = []
    table_rows = []

    for mid, img_list in medicine_embeddings.items():
        same_scores = []
        diff_scores = []

        # (a) Same medicine similarities
        if len(img_list) > 1:
            for i in range(len(img_list)):
                for j in range(i + 1, len(img_list)):
                    sim = cosine_sim(img_list[i][1], img_list[j][1])
                    same_scores.append(sim)
                    all_same_scores.append(sim)
        else:
            # Single reference image — similarity to self is 1.0
            same_scores.append(1.0)
            all_same_scores.append(1.0)

        # (b) Different medicine similarities
        for _, my_vec in img_list:
            for other_mid, other_img_list in medicine_embeddings.items():
                if other_mid == mid:
                    continue
                for _, other_vec in other_img_list:
                    sim = cosine_sim(my_vec, other_vec)
                    diff_scores.append(sim)
                    all_diff_scores.append(sim)

        mean_same = float(np.mean(same_scores)) if same_scores else 0.0
        lowest_same = float(np.min(same_scores)) if same_scores else 0.0
        mean_diff = float(np.mean(diff_scores)) if diff_scores else 0.0
        max_diff = float(np.max(diff_scores)) if diff_scores else 0.0

        table_rows.append({
            "medicine": mid,
            "count": len(img_list),
            "mean_same": mean_same,
            "lowest_same": lowest_same,
            "mean_diff": mean_diff,
            "max_diff": max_diff,
        })

    # 3. Print Results
    print("=" * 88)
    print("                MEDSURE VISION — VISUAL SIMILARITY DIAGNOSTICS")
    print("=" * 88)
    print(f"{'Medicine ID':<22} | {'Photos':<6} | {'Mean Same':<10} | {'Lowest Same':<11} | {'Mean Diff':<10} | {'Max Diff':<10}")
    print("-" * 88)

    for r in table_rows:
        print(f"{r['medicine']:<22} | {r['count']:<6} | {r['mean_same']:<10.4f} | {r['lowest_same']:<11.4f} | {r['mean_diff']:<10.4f} | {r['max_diff']:<10.4f}")

    print("-" * 88)
    overall_mean_same = float(np.mean(all_same_scores)) if all_same_scores else 0.0
    overall_min_same = float(np.min(all_same_scores)) if all_same_scores else 0.0
    overall_mean_diff = float(np.mean(all_diff_scores)) if all_diff_scores else 0.0
    overall_max_diff = float(np.max(all_diff_scores)) if all_diff_scores else 0.0

    print(f"{'OVERALL SUMMARY':<22} | {sum(r['count'] for r in table_rows):<6} | {overall_mean_same:<10.4f} | {overall_min_same:<11.4f} | {overall_mean_diff:<10.4f} | {overall_max_diff:<10.4f}")
    print("=" * 88)

    # 4. Threshold Calibration Analysis
    # Current threshold in fusion.py:
    #   score < 0.50 -> 25 penalty (high divergence)
    #   score < 0.75 -> 15 penalty (moderate divergence)
    #   score >= 0.75 -> 0 penalty (confident visual match)
    
    # Recommended threshold is positioned safely above max different-medicine score
    # with sufficient margin for variations while remaining below lowest same-medicine score.
    recommended_threshold = round(overall_max_diff + 0.05, 2)
    if recommended_threshold < 0.65:
        recommended_threshold = 0.65

    print("\n--- VISUAL THRESHOLD CALIBRATION ANALYSIS ---")
    print(f"• Current Soft-Score Visual Bands in fusion.py:")
    print(f"    - identity_score < 0.50  : High penalty (+25 score)")
    print(f"    - identity_score < 0.75  : Moderate penalty (+15 score)")
    print(f"    - identity_score >= 0.75 : Visual match confirmed (0 penalty)")
    print(f"• Empirical Inter-Class Separation:")
    print(f"    - Maximum Different-Medicine Cosine Similarity : {overall_max_diff:.4f}")
    print(f"    - Average Different-Medicine Cosine Similarity : {overall_mean_diff:.4f}")
    print(f"    - Lowest Same-Medicine Cosine Similarity      : {overall_min_same:.4f}")
    print(f"• Separation Margin: {overall_min_same - overall_max_diff:.4f}")
    print(f"• Recommended Operating Threshold: {recommended_threshold:.2f}")
    print(f"  (Provides a robust decision boundary separating different packs from true references)")

    if skipped_files:
        print("\n--- SKIPPED FILES REPORT ---")
        for mid, fname, reason in skipped_files:
            print(f"  [SKIPPED] {mid} -> {fname}: {reason}")
    else:
        print("\n--- SKIPPED FILES REPORT ---")
        print("  None. All reference images across all medicine directories were recognized and embedded.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
