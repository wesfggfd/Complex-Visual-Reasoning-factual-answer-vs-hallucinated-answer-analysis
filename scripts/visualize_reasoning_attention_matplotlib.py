#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

from attention_binary_utils import (
    block_mean_map,
    build_sequence_samples,
    center_shift,
    cosine_similarity,
    entropy_score,
    get_branch_prefixes,
    get_question_block,
    get_trace,
    get_sequence_block,
    layer_cosine_curve,
    layer_js_curve,
    load_records,
    resize_stack,
    sample_trace_steps,
    sequence_block_map,
    sequence_block_stack,
    topk_mass,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GeoThought truthful-vs-hallucinated reasoning attention."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-size", type=int, default=224)
    parser.add_argument("--per-sample-limit", type=int, default=20)
    parser.add_argument("--progression-steps", type=int, default=4)
    parser.add_argument("--topk-fraction", type=float, default=0.1)
    parser.add_argument("--top-layer-count", type=int, default=5)
    parser.add_argument("--band-width", type=int, default=3)
    parser.add_argument("--token-selector", default="late", choices=["all", "late", "final"])
    parser.add_argument("--sequence-reduction", default="mean", choices=["mean", "max"])
    parser.add_argument("--include-manifold", action="store_true")
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    clipped = np.clip(heatmap.astype(np.float32), 0.0, None)
    if clipped.max() > 0:
        clipped = clipped / clipped.max()
    image = Image.fromarray((clipped * 255).astype(np.uint8))
    resized = image.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def resize_signed_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    minimum = float(heatmap.min())
    maximum = float(heatmap.max())
    if abs(maximum - minimum) < 1e-12:
        normalized = np.zeros_like(heatmap, dtype=np.float32)
    else:
        normalized = ((heatmap - minimum) / (maximum - minimum)).astype(np.float32)
    image = Image.fromarray((normalized * 255).astype(np.uint8))
    resized = image.resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def save_per_sample_overview(
    *,
    record: dict,
    question_map: np.ndarray,
    truthful_map: np.ndarray,
    hallucinated_map: np.ndarray,
    signed_delta: np.ndarray,
    output_path: Path,
) -> None:
    image = load_image(record["image_path"])
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{record['sample_id']} | ref={record['reference_final_answer']} | "
        f"truthful={record['truthful_final_answer']} | hallucinated={record['hallucinated_final_answer']}",
        fontsize=11,
    )
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(image)
    im_question = axes[0, 1].imshow(
        resize_heatmap(question_map, (image.shape[1], image.shape[0])),
        cmap="viridis",
        alpha=0.45,
    )
    axes[0, 1].set_title("Problem -> Vision")
    axes[0, 1].axis("off")
    fig.colorbar(im_question, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].imshow(image)
    im_truth = axes[0, 2].imshow(
        resize_heatmap(truthful_map, (image.shape[1], image.shape[0])),
        cmap="viridis",
        alpha=0.45,
    )
    axes[0, 2].set_title("Truthful Sequence -> Vision")
    axes[0, 2].axis("off")
    fig.colorbar(im_truth, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(image)
    im_hall = axes[1, 0].imshow(
        resize_heatmap(hallucinated_map, (image.shape[1], image.shape[0])),
        cmap="viridis",
        alpha=0.45,
    )
    axes[1, 0].set_title("Hallucinated Sequence -> Vision")
    axes[1, 0].axis("off")
    fig.colorbar(im_hall, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im_signed = axes[1, 1].imshow(signed_delta, cmap="bwr")
    axes[1, 1].set_title("Signed Delta (hallucinated - truthful)")
    axes[1, 1].axis("off")
    fig.colorbar(im_signed, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im_abs = axes[1, 2].imshow(np.abs(signed_delta), cmap="magma")
    axes[1, 2].set_title("Absolute Delta")
    axes[1, 2].axis("off")
    fig.colorbar(im_abs, ax=axes[1, 2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_progression_figure(
    *,
    record: dict,
    truthful_steps: list[dict],
    hallucinated_steps: list[dict],
    output_path: Path,
) -> None:
    image = load_image(record["image_path"])
    cols = max(len(truthful_steps), len(hallucinated_steps))
    fig, axes = plt.subplots(2, cols, figsize=(4.2 * cols, 7.5))
    if cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col in range(cols):
        for row in range(2):
            axes[row, col].axis("off")
    for col, step in enumerate(truthful_steps):
        heatmap = block_mean_map(step["cross_attention"])
        axes[0, col].imshow(image)
        axes[0, col].imshow(
            resize_heatmap(heatmap, (image.shape[1], image.shape[0])),
            cmap="viridis",
            alpha=0.45,
        )
        axes[0, col].set_title(f"Truthful step {step['step']} token={step['token']!r}")
    for col, step in enumerate(hallucinated_steps):
        heatmap = block_mean_map(step["cross_attention"])
        axes[1, col].imshow(image)
        axes[1, col].imshow(
            resize_heatmap(heatmap, (image.shape[1], image.shape[0])),
            cmap="viridis",
            alpha=0.45,
        )
        axes[1, col].set_title(f"Hallucinated step {step['step']} token={step['token']!r}")
    fig.suptitle(f"{record['sample_id']} reasoning progression", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_heatmaps(
    mean_question: np.ndarray,
    mean_truthful: np.ndarray,
    mean_hallucinated: np.ndarray,
    mean_signed_delta: np.ndarray,
    mean_absolute_delta: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8))
    panels = [
        (mean_question, "Mean Problem->Vision", "viridis"),
        (mean_truthful, "Mean Truthful Sequence", "viridis"),
        (mean_hallucinated, "Mean Hallucinated Sequence", "viridis"),
        (mean_signed_delta, "Mean Signed Delta", "bwr"),
        (mean_absolute_delta, "Mean Absolute Delta", "magma"),
    ]
    for ax, (heatmap, title, cmap) in zip(axes, panels):
        im = ax.imshow(heatmap, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_layer_divergence_plot(js_curve: np.ndarray, cosine_curve: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    layers = np.arange(len(js_curve))
    ax.plot(layers, js_curve, label="JS divergence", color="tab:red", linewidth=2)
    ax.plot(layers, cosine_curve, label="Cosine similarity", color="tab:blue", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Value")
    ax.set_title("Layer-wise truthful vs hallucinated divergence")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_metric_distributions(metrics: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plots = [
        ("alignment_gap", "Alignment gap"),
        ("entropy_gap", "Entropy gap"),
        ("topk_gap", "Top-k mass gap"),
        ("center_shift", "Center shift"),
        ("mean_js_divergence", "Mean JS divergence"),
        ("mean_cosine_similarity", "Mean cosine similarity"),
    ]
    for ax, (key, title) in zip(axes.flat, plots):
        ax.hist([metric[key] for metric in metrics], bins=20, color="tab:blue", alpha=0.8)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_pca(features: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, PCA]:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    component_count = min(n_components, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=component_count)
    coords = pca.fit_transform(scaled)
    return coords, pca


def run_manifold(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    neighbors = max(2, min(10, scaled.shape[0] - 1))
    model = Isomap(n_components=min(n_components, scaled.shape[1]), n_neighbors=neighbors)
    return model.fit_transform(scaled)


def save_scatter(coords: np.ndarray, labels: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    truthful = labels == 0
    hallucinated = labels == 1
    ax.scatter(coords[truthful, 0], coords[truthful, 1], label="Truthful", color="tab:blue", alpha=0.8)
    ax.scatter(
        coords[hallucinated, 0],
        coords[hallucinated, 1],
        label="Hallucinated",
        color="tab:red",
        alpha=0.8,
    )
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_explained_variance_plot(pca: PCA, output_path: Path) -> None:
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    components = np.arange(1, len(ratios) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(components, ratios, marker="o", label="Per-component variance")
    ax.plot(components, cumulative, marker="s", label="Cumulative variance")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Sequence PCA explained variance")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pairwise_centroid_distance(coords: np.ndarray, labels: np.ndarray) -> float:
    truthful = coords[labels == 0]
    hallucinated = coords[labels == 1]
    return float(np.linalg.norm(truthful.mean(axis=0) - hallucinated.mean(axis=0)))


def rank_layers(sequence_samples: list[dict], output_path: Path) -> list[dict]:
    labels = np.array([sample["label"] for sample in sequence_samples], dtype=int)
    layer_count = sequence_samples[0]["attention_stack"].shape[0]
    target_hw = (
        max(sample["attention_stack"].shape[1] for sample in sequence_samples),
        max(sample["attention_stack"].shape[2] for sample in sequence_samples),
    )
    resized_stacks = [resize_stack(sample["attention_stack"], target_hw) for sample in sequence_samples]
    rows: list[dict] = []
    for layer_idx in range(layer_count):
        features = np.stack([stack[layer_idx].reshape(-1) for stack in resized_stacks], axis=0)
        coords, pca = run_pca(features, n_components=2)
        rows.append(
            {
                "layer_index": layer_idx,
                "pc1_variance_ratio": float(pca.explained_variance_ratio_[0]),
                "pc2_variance_ratio": float(
                    pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0.0
                ),
                "pca_centroid_distance": pairwise_centroid_distance(coords[:, :2], labels),
            }
        )
    rows.sort(key=lambda item: item["pca_centroid_distance"], reverse=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["layer_index", "pc1_variance_ratio", "pc2_variance_ratio", "pca_centroid_distance"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def rank_layer_bands(sequence_samples: list[dict], band_width: int, output_path: Path) -> list[dict]:
    labels = np.array([sample["label"] for sample in sequence_samples], dtype=int)
    layer_count = sequence_samples[0]["attention_stack"].shape[0]
    target_hw = (
        max(sample["attention_stack"].shape[1] for sample in sequence_samples),
        max(sample["attention_stack"].shape[2] for sample in sequence_samples),
    )
    resized_stacks = [resize_stack(sample["attention_stack"], target_hw) for sample in sequence_samples]
    rows: list[dict] = []
    for start in range(0, layer_count - band_width + 1):
        end = start + band_width
        features = np.stack([stack[start:end].reshape(-1) for stack in resized_stacks], axis=0)
        coords, pca = run_pca(features, n_components=2)
        rows.append(
            {
                "start_layer": start,
                "end_layer": end - 1,
                "band_width": band_width,
                "pc1_variance_ratio": float(pca.explained_variance_ratio_[0]),
                "pc2_variance_ratio": float(
                    pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0.0
                ),
                "pca_centroid_distance": pairwise_centroid_distance(coords[:, :2], labels),
            }
        )
    rows.sort(key=lambda item: item["pca_centroid_distance"], reverse=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "start_layer",
                "end_layer",
                "band_width",
                "pc1_variance_ratio",
                "pc2_variance_ratio",
                "pca_centroid_distance",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows


def save_top_layer_spotlight(
    sequence_samples: list[dict],
    layer_rows: list[dict],
    output_path: Path,
    top_layer_count: int,
) -> None:
    top_layers = [row["layer_index"] for row in layer_rows[:top_layer_count]]
    target_hw = (
        max(sample["attention_stack"].shape[1] for sample in sequence_samples),
        max(sample["attention_stack"].shape[2] for sample in sequence_samples),
    )
    resized_stacks = [resize_stack(sample["attention_stack"], target_hw) for sample in sequence_samples]
    fig, axes = plt.subplots(len(top_layers), 3, figsize=(11, 3.2 * len(top_layers)))
    if len(top_layers) == 1:
        axes = np.array([axes])
    for row_axes, layer_idx in zip(axes, top_layers):
        truthful_mean = np.mean(
            [stack[layer_idx] for stack, sample in zip(resized_stacks, sequence_samples) if sample["label"] == 0],
            axis=0,
        )
        hallucinated_mean = np.mean(
            [stack[layer_idx] for stack, sample in zip(resized_stacks, sequence_samples) if sample["label"] == 1],
            axis=0,
        )
        delta = hallucinated_mean - truthful_mean
        row_axes[0].imshow(truthful_mean, cmap="viridis")
        row_axes[0].set_title(f"Layer {layer_idx} truthful")
        row_axes[1].imshow(hallucinated_mean, cmap="viridis")
        row_axes[1].set_title(f"Layer {layer_idx} hallucinated")
        row_axes[2].imshow(delta, cmap="bwr")
        row_axes[2].set_title(f"Layer {layer_idx} delta")
        for ax in row_axes:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_report_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_metrics_csv(metrics: list[dict], output_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "reference_final_answer",
        "truthful_final_answer",
        "hallucinated_final_answer",
        "truthful_cot_similarity",
        "hallucinated_cot_similarity",
        "question_truthful_alignment",
        "question_hallucinated_alignment",
        "alignment_gap",
        "entropy_gap",
        "topk_gap",
        "center_shift",
        "mean_js_divergence",
        "mean_cosine_similarity",
        "discriminability_score",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    per_sample_dir = output_dir / "per_sample"
    progression_dir = output_dir / "progression"
    summary_dir = output_dir / "dataset_summary"
    pca_dir = output_dir / "pca"
    manifold_dir = output_dir / "manifold"
    layer_dir = output_dir / "layer_analysis"
    for path in (per_sample_dir, progression_dir, summary_dir, pca_dir, manifold_dir, layer_dir):
        path.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    if not records:
        raise SystemExit("No records found in reasoning attention JSONL.")

    sequence_samples = build_sequence_samples(
        records,
        topk_fraction=args.topk_fraction,
        token_selector=args.token_selector,
        reduction=args.sequence_reduction,
    )
    labels = np.array([sample["label"] for sample in sequence_samples], dtype=int)
    target_hw = (
        max(sample["attention_stack"].shape[1] for sample in sequence_samples),
        max(sample["attention_stack"].shape[2] for sample in sequence_samples),
    )
    resized_sequence_stacks = [resize_stack(sample["attention_stack"], target_hw) for sample in sequence_samples]
    sequence_features = np.stack([stack.reshape(-1) for stack in resized_sequence_stacks], axis=0)

    summary_question = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    summary_truthful = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    summary_hallucinated = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    summary_signed_delta = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    summary_absolute_delta = np.zeros((args.summary_size, args.summary_size), dtype=np.float64)
    js_curves: list[np.ndarray] = []
    cosine_curves: list[np.ndarray] = []
    metrics: list[dict] = []

    for index, record in enumerate(records):
        truthful_prefix, hallucinated_prefix = get_branch_prefixes(record)
        question_map = block_mean_map(get_question_block(record, truthful_prefix))
        truthful_map = sequence_block_map(record, truthful_prefix)
        hallucinated_map = sequence_block_map(record, hallucinated_prefix)
        truthful_stack = sequence_block_stack(record, truthful_prefix)
        hallucinated_stack = sequence_block_stack(record, hallucinated_prefix)
        layer_count = min(truthful_stack.shape[0], hallucinated_stack.shape[0])
        truthful_stack = truthful_stack[:layer_count]
        hallucinated_stack = hallucinated_stack[:layer_count]
        signed_delta = hallucinated_map - truthful_map

        summary_question += resize_heatmap(question_map, (args.summary_size, args.summary_size))
        summary_truthful += resize_heatmap(truthful_map, (args.summary_size, args.summary_size))
        summary_hallucinated += resize_heatmap(hallucinated_map, (args.summary_size, args.summary_size))
        summary_signed_delta += resize_signed_heatmap(signed_delta, (args.summary_size, args.summary_size))
        summary_absolute_delta += resize_heatmap(np.abs(signed_delta), (args.summary_size, args.summary_size))

        js_curve = layer_js_curve(truthful_stack, hallucinated_stack)
        cosine_curve = layer_cosine_curve(truthful_stack, hallucinated_stack)
        js_curves.append(js_curve)
        cosine_curves.append(cosine_curve)

        truthful_alignment = cosine_similarity(question_map, truthful_map)
        hallucinated_alignment = cosine_similarity(question_map, hallucinated_map)
        truthful_entropy = entropy_score(truthful_map)
        hallucinated_entropy = entropy_score(hallucinated_map)
        truthful_topk = topk_mass(truthful_map, args.topk_fraction)
        hallucinated_topk = topk_mass(hallucinated_map, args.topk_fraction)
        mean_js = float(js_curve.mean()) if len(js_curve) else 0.0
        mean_cos = float(cosine_curve.mean()) if len(cosine_curve) else 0.0
        drift = center_shift(truthful_map, hallucinated_map)
        metrics.append(
            {
                "sample_id": record["sample_id"],
                "reference_final_answer": record["reference_final_answer"],
                "truthful_final_answer": record["truthful_final_answer"],
                "hallucinated_final_answer": record["hallucinated_final_answer"],
                "truthful_cot_similarity": record["truthful_cot_similarity"],
                "hallucinated_cot_similarity": record["hallucinated_cot_similarity"],
                "question_truthful_alignment": truthful_alignment,
                "question_hallucinated_alignment": hallucinated_alignment,
                "alignment_gap": truthful_alignment - hallucinated_alignment,
                "entropy_gap": hallucinated_entropy - truthful_entropy,
                "topk_gap": hallucinated_topk - truthful_topk,
                "center_shift": drift,
                "mean_js_divergence": mean_js,
                "mean_cosine_similarity": mean_cos,
                "discriminability_score": mean_js + drift + abs(truthful_alignment - hallucinated_alignment),
            }
        )

        if index < args.per_sample_limit:
            save_per_sample_overview(
                record=record,
                question_map=question_map,
                truthful_map=truthful_map,
                hallucinated_map=hallucinated_map,
                signed_delta=signed_delta,
                output_path=per_sample_dir / f"{index:03d}_{record['sample_id']}.png",
            )
            truthful_trace = get_trace(record, truthful_prefix)
            hallucinated_trace = get_trace(record, hallucinated_prefix)
            if truthful_trace and hallucinated_trace:
                save_progression_figure(
                    record=record,
                    truthful_steps=sample_trace_steps(truthful_trace, args.progression_steps),
                    hallucinated_steps=sample_trace_steps(hallucinated_trace, args.progression_steps),
                    output_path=progression_dir / f"{index:03d}_{record['sample_id']}.png",
                )

    sample_count = max(1, len(records))
    save_summary_heatmaps(
        mean_question=summary_question / sample_count,
        mean_truthful=summary_truthful / sample_count,
        mean_hallucinated=summary_hallucinated / sample_count,
        mean_signed_delta=summary_signed_delta / sample_count,
        mean_absolute_delta=summary_absolute_delta / sample_count,
        output_path=summary_dir / "mean_heatmaps.png",
    )
    min_layers = min(len(curve) for curve in js_curves)
    save_layer_divergence_plot(
        js_curve=np.mean([curve[:min_layers] for curve in js_curves], axis=0),
        cosine_curve=np.mean([curve[:min_layers] for curve in cosine_curves], axis=0),
        output_path=summary_dir / "layer_divergence.png",
    )
    metrics.sort(key=lambda item: item["discriminability_score"], reverse=True)
    write_metrics_csv(metrics, summary_dir / "reasoning_metrics.csv")
    save_metric_distributions(metrics, summary_dir / "metric_distributions.png")

    sequence_coords, sequence_pca = run_pca(sequence_features, n_components=8)
    save_scatter(
        sequence_coords[:, :2],
        labels,
        pca_dir / "sequence_pca_scatter.png",
        "Sequence pooled attention PCA",
    )
    save_explained_variance_plot(sequence_pca, pca_dir / "sequence_explained_variance.png")

    if args.include_manifold:
        save_scatter(
            run_manifold(sequence_features, n_components=2),
            labels,
            manifold_dir / "sequence_isomap_scatter.png",
            "Sequence pooled attention Isomap",
        )

    layer_rows = rank_layers(sequence_samples, layer_dir / "layer_pca_ranking.csv")
    band_rows = rank_layer_bands(sequence_samples, args.band_width, layer_dir / "layer_band_pca_ranking.csv")
    save_top_layer_spotlight(
        sequence_samples,
        layer_rows,
        layer_dir / "top_layer_spotlight.png",
        args.top_layer_count,
    )
    save_report_json(
        layer_dir / "layer_analysis_report.json",
        {
            "record_count": len(records),
            "sequence_sample_count": len(sequence_samples),
            "top_layers": layer_rows[: args.top_layer_count],
            "top_bands": band_rows[: args.top_layer_count],
            "sequence_pca_explained_variance_ratio": sequence_pca.explained_variance_ratio_.tolist(),
            "include_manifold": bool(args.include_manifold),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
