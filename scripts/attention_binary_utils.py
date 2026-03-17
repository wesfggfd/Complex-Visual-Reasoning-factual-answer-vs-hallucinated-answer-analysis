#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(path: Path, limit: int | None = None) -> list[dict]:
    records = []
    for record in iter_records(path):
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def sorted_layer_names(layer_maps: dict[str, list[list[float]]]) -> list[str]:
    def sort_key(item: str):
        if item.startswith("layer_"):
            return (0, int(item.split("_", 1)[1]))
        if item.startswith("band_"):
            order = {"band_early": 0, "band_middle": 1, "band_late": 2}
            return (1, order.get(item, 99))
        return (2, item)

    return sorted(layer_maps, key=sort_key)


def answer_block(trace: list[dict]) -> dict:
    if not trace:
        raise ValueError("Expected at least one answer token trace.")
    return trace[0]["cross_attention"]


def block_mean_map(block: dict) -> np.ndarray:
    return np.array(block["layer_summary"]["mean_over_layers"], dtype=np.float64)


def block_layer_stack(block: dict) -> np.ndarray:
    layer_maps = block["layer_maps"]
    ordered = sorted_layer_names(layer_maps)
    return np.stack([np.array(layer_maps[name], dtype=np.float64) for name in ordered], axis=0)


def entropy_score(heatmap: np.ndarray) -> float:
    values = np.clip(heatmap.astype(np.float64), 0.0, None).reshape(-1)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def topk_mass(heatmap: np.ndarray, fraction: float) -> float:
    values = np.clip(heatmap.astype(np.float64), 0.0, None).reshape(-1)
    total = values.sum()
    if total <= 0:
        return 0.0
    count = max(1, int(math.ceil(values.size * fraction)))
    top_values = np.partition(values, -count)[-count:]
    return float(top_values.sum() / total)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1).astype(np.float64)
    right_flat = right.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    if denom == 0:
        return 0.0
    return float(np.dot(left_flat, right_flat) / denom)


def js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.clip(left.astype(np.float64), 0.0, None).reshape(-1)
    right_flat = np.clip(right.astype(np.float64), 0.0, None).reshape(-1)
    if left_flat.sum() <= 0 or right_flat.sum() <= 0:
        return 0.0
    left_probs = left_flat / left_flat.sum()
    right_probs = right_flat / right_flat.sum()
    mean = 0.5 * (left_probs + right_probs)

    def kl_div(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl_div(left_probs, mean) + 0.5 * kl_div(right_probs, mean)


def center_of_mass(heatmap: np.ndarray) -> tuple[float, float]:
    values = np.clip(heatmap.astype(np.float64), 0.0, None)
    total = values.sum()
    if total <= 0:
        return 0.0, 0.0
    ys, xs = np.indices(values.shape)
    y = float((ys * values).sum() / total)
    x = float((xs * values).sum() / total)
    return y, x


def center_shift(left: np.ndarray, right: np.ndarray) -> float:
    return float(math.dist(center_of_mass(left), center_of_mass(right)))


def resize_heatmap(heatmap: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    image = Image.fromarray(np.asarray(heatmap, dtype=np.float32), mode="F")
    resized = image.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float64)


def resize_stack(stack: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    return np.stack([resize_heatmap(layer_map, target_hw) for layer_map in stack], axis=0)


def layer_js_curve(left_stack: np.ndarray, right_stack: np.ndarray) -> np.ndarray:
    layer_count = min(left_stack.shape[0], right_stack.shape[0])
    return np.array(
        [js_divergence(left_stack[layer_idx], right_stack[layer_idx]) for layer_idx in range(layer_count)],
        dtype=np.float64,
    )


def layer_cosine_curve(left_stack: np.ndarray, right_stack: np.ndarray) -> np.ndarray:
    layer_count = min(left_stack.shape[0], right_stack.shape[0])
    return np.array(
        [
            cosine_similarity(left_stack[layer_idx], right_stack[layer_idx])
            for layer_idx in range(layer_count)
        ],
        dtype=np.float64,
    )


def layer_mean_curve(stack: np.ndarray) -> np.ndarray:
    return stack.mean(axis=(1, 2)).astype(np.float64)


def layer_entropy_curve(stack: np.ndarray) -> np.ndarray:
    return np.array([entropy_score(layer_map) for layer_map in stack], dtype=np.float64)


def question_alignment_curve(question_map: np.ndarray, answer_stack: np.ndarray) -> np.ndarray:
    return np.array(
        [cosine_similarity(question_map, answer_stack[layer_idx]) for layer_idx in range(answer_stack.shape[0])],
        dtype=np.float64,
    )


def get_branch_prefixes(record: dict) -> tuple[str, str]:
    if "truthful_sequence_attention" in record and "hallucinated_sequence_attention" in record:
        return "truthful", "hallucinated"
    if "truthful_trace" in record and "hallucinated_trace" in record:
        return "truthful", "hallucinated"
    if "factual_trace" in record and "hallucinated_trace" in record:
        return "factual", "hallucinated"
    raise KeyError("Could not infer branch prefixes from record.")


def get_question_block(record: dict, prefix: str) -> dict:
    return record[f"{prefix}_question_attention"]


def get_trace(record: dict, prefix: str) -> list[dict]:
    return record.get(f"{prefix}_trace", [])


def get_sequence_block(record: dict, prefix: str) -> dict:
    sequence_key = f"{prefix}_sequence_attention"
    if sequence_key in record:
        return record[sequence_key]
    trace = get_trace(record, prefix)
    if trace:
        return answer_block(trace)
    raise KeyError(f"Could not locate sequence attention block for prefix={prefix!r}.")


def sequence_block_stack(record: dict, prefix: str) -> np.ndarray:
    return block_layer_stack(get_sequence_block(record, prefix))


def sequence_block_map(record: dict, prefix: str) -> np.ndarray:
    return block_mean_map(get_sequence_block(record, prefix))


def get_branch_text(record: dict, prefix: str) -> str:
    for key in (f"{prefix}_raw_text", f"{prefix}_answer", f"{prefix}_final_answer"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _select_token_indices(trace_length: int, token_selector: str) -> list[int]:
    if trace_length <= 0:
        return []
    if token_selector == "all":
        return list(range(trace_length))
    if token_selector == "late":
        start = max(0, trace_length - max(1, math.ceil(trace_length / 3)))
        return list(range(start, trace_length))
    if token_selector == "final":
        return [trace_length - 1]
    if token_selector == "first":
        return [0]
    raise ValueError(f"Unsupported token_selector: {token_selector}")


def pooled_trace_stack(
    trace: list[dict],
    *,
    token_selector: str = "all",
    reduction: str = "mean",
) -> np.ndarray:
    selected_indices = _select_token_indices(len(trace), token_selector)
    if not selected_indices:
        raise ValueError("Expected at least one token trace for pooling.")
    stacks = [block_layer_stack(trace[index]["cross_attention"]) for index in selected_indices]
    layer_count = min(stack.shape[0] for stack in stacks)
    target_hw = (
        max(stack.shape[1] for stack in stacks),
        max(stack.shape[2] for stack in stacks),
    )
    resized = []
    for stack in stacks:
        cropped = stack[:layer_count]
        if cropped.shape[1:] != target_hw:
            cropped = resize_stack(cropped, target_hw)
        resized.append(cropped)
    merged = np.stack(resized, axis=0)
    if reduction == "mean":
        return merged.mean(axis=0)
    if reduction == "max":
        return merged.max(axis=0)
    raise ValueError(f"Unsupported reduction: {reduction}")


def pooled_trace_map(
    trace: list[dict],
    *,
    token_selector: str = "all",
    reduction: str = "mean",
) -> np.ndarray:
    return pooled_trace_stack(trace, token_selector=token_selector, reduction=reduction).mean(axis=0)


def sample_trace_steps(trace: list[dict], max_steps: int) -> list[dict]:
    if len(trace) <= max_steps:
        return trace
    indices = np.linspace(0, len(trace) - 1, num=max_steps, dtype=int)
    unique_indices = []
    seen = set()
    for index in indices.tolist():
        if index not in seen:
            unique_indices.append(index)
            seen.add(index)
    return [trace[index] for index in unique_indices]


def _token_sample(
    *,
    record: dict,
    label: int,
    label_name: str,
    question_map: np.ndarray,
    answer_map: np.ndarray,
    answer_stack: np.ndarray,
    pair_js: float,
    pair_cosine: float,
    pair_center_shift: float,
    cot_alignment_gain: float,
    topk_fraction: float,
) -> dict:
    return {
        "sample_id": record.get("sample_id", Path(record["image_path"]).stem),
        "image_path": record["image_path"],
        "object_label": record.get("object_label", ""),
        "question": record.get("question", record.get("problem", "")),
        "problem": record.get("problem", record.get("question", "")),
        "reference_final_answer": record.get("reference_final_answer", record.get("expected_answer", "")),
        "factual_answer": record.get("factual_answer", ""),
        "hallucinated_answer": record.get("hallucinated_answer", ""),
        "truthful_raw_text": record.get("truthful_raw_text", ""),
        "hallucinated_raw_text": record.get("hallucinated_raw_text", ""),
        "label": label,
        "label_name": label_name,
        "question_alignment": cosine_similarity(question_map, answer_map),
        "question_center_shift": center_shift(question_map, answer_map),
        "answer_entropy": entropy_score(answer_map),
        "answer_topk_mass": topk_mass(answer_map, topk_fraction),
        "answer_peak_value": float(np.max(answer_map)),
        "answer_mean_value": float(np.mean(answer_map)),
        "pair_js_divergence": pair_js,
        "pair_cosine_similarity": pair_cosine,
        "pair_center_shift": pair_center_shift,
        "cot_alignment_gain": cot_alignment_gain,
        "attention_map": answer_map,
        "attention_stack": answer_stack,
    }


def _build_binary_token_samples(records: list[dict], topk_fraction: float = 0.1) -> list[dict]:
    samples: list[dict] = []
    for record in records:
        factual_question = block_mean_map(record["factual_question_attention"])
        hallucinated_question = block_mean_map(
            record.get("hallucinated_question_attention", record["factual_question_attention"])
        )
        factual_block = answer_block(record["factual_trace"])
        hallucinated_block = answer_block(record["hallucinated_trace"])
        factual_map = block_mean_map(factual_block)
        hallucinated_map = block_mean_map(hallucinated_block)
        factual_stack = block_layer_stack(factual_block)
        hallucinated_stack = block_layer_stack(hallucinated_block)
        layer_count = min(factual_stack.shape[0], hallucinated_stack.shape[0])
        factual_stack = factual_stack[:layer_count]
        hallucinated_stack = hallucinated_stack[:layer_count]
        pair_js = js_divergence(factual_map, hallucinated_map)
        pair_cosine = cosine_similarity(factual_map, hallucinated_map)
        pair_drift = center_shift(factual_map, hallucinated_map)
        cot_alignment_gain = cosine_similarity(factual_question, factual_map) - cosine_similarity(
            hallucinated_question, hallucinated_map
        )
        samples.append(
            _token_sample(
                record=record,
                label=0,
                label_name="factual",
                question_map=factual_question,
                answer_map=factual_map,
                answer_stack=factual_stack,
                pair_js=pair_js,
                pair_cosine=pair_cosine,
                pair_center_shift=pair_drift,
                cot_alignment_gain=cot_alignment_gain,
                topk_fraction=topk_fraction,
            )
        )
        samples.append(
            _token_sample(
                record=record,
                label=1,
                label_name="hallucinated",
                question_map=hallucinated_question,
                answer_map=hallucinated_map,
                answer_stack=hallucinated_stack,
                pair_js=pair_js,
                pair_cosine=pair_cosine,
                pair_center_shift=pair_drift,
                cot_alignment_gain=cot_alignment_gain,
                topk_fraction=topk_fraction,
            )
        )
    return samples


def build_sequence_samples(
    records: list[dict],
    *,
    topk_fraction: float = 0.1,
    token_selector: str = "all",
    reduction: str = "mean",
) -> list[dict]:
    samples: list[dict] = []
    for record in records:
        truthful_prefix, hallucinated_prefix = get_branch_prefixes(record)
        truthful_question = block_mean_map(get_question_block(record, truthful_prefix))
        hallucinated_question = block_mean_map(get_question_block(record, hallucinated_prefix))
        truthful_stack = sequence_block_stack(record, truthful_prefix)
        hallucinated_stack = sequence_block_stack(record, hallucinated_prefix)
        layer_count = min(truthful_stack.shape[0], hallucinated_stack.shape[0])
        target_hw = (
            max(truthful_stack.shape[1], hallucinated_stack.shape[1]),
            max(truthful_stack.shape[2], hallucinated_stack.shape[2]),
        )
        truthful_stack = resize_stack(truthful_stack[:layer_count], target_hw)
        hallucinated_stack = resize_stack(hallucinated_stack[:layer_count], target_hw)
        truthful_map = truthful_stack.mean(axis=0)
        hallucinated_map = hallucinated_stack.mean(axis=0)
        pair_js = js_divergence(truthful_map, hallucinated_map)
        pair_cosine = cosine_similarity(truthful_map, hallucinated_map)
        pair_drift = center_shift(truthful_map, hallucinated_map)
        pair_js_curve_values = layer_js_curve(truthful_stack, hallucinated_stack)
        pair_cosine_curve_values = layer_cosine_curve(truthful_stack, hallucinated_stack)
        cot_alignment_gain = cosine_similarity(truthful_question, truthful_map) - cosine_similarity(
            hallucinated_question,
            hallucinated_map,
        )
        truthful_sample = _token_sample(
            record=record,
            label=0,
            label_name="truthful",
            question_map=truthful_question,
            answer_map=truthful_map,
            answer_stack=truthful_stack,
            pair_js=pair_js,
            pair_cosine=pair_cosine,
            pair_center_shift=pair_drift,
            cot_alignment_gain=cot_alignment_gain,
            topk_fraction=topk_fraction,
        )
        truthful_sample["pair_js_curve"] = pair_js_curve_values
        truthful_sample["pair_cosine_curve"] = pair_cosine_curve_values
        truthful_sample["layer_mean_curve"] = layer_mean_curve(truthful_stack)
        truthful_sample["layer_entropy_curve"] = layer_entropy_curve(truthful_stack)
        truthful_sample["question_alignment_curve"] = question_alignment_curve(truthful_question, truthful_stack)
        samples.append(truthful_sample)
        hallucinated_sample = _token_sample(
            record=record,
            label=1,
            label_name="hallucinated",
            question_map=hallucinated_question,
            answer_map=hallucinated_map,
            answer_stack=hallucinated_stack,
            pair_js=pair_js,
            pair_cosine=pair_cosine,
            pair_center_shift=pair_drift,
            cot_alignment_gain=cot_alignment_gain,
            topk_fraction=topk_fraction,
        )
        hallucinated_sample["pair_js_curve"] = pair_js_curve_values
        hallucinated_sample["pair_cosine_curve"] = pair_cosine_curve_values
        hallucinated_sample["layer_mean_curve"] = layer_mean_curve(hallucinated_stack)
        hallucinated_sample["layer_entropy_curve"] = layer_entropy_curve(hallucinated_stack)
        hallucinated_sample["question_alignment_curve"] = question_alignment_curve(
            hallucinated_question, hallucinated_stack
        )
        samples.append(hallucinated_sample)
    return samples


def build_trace_token_samples(records: list[dict]) -> list[dict]:
    samples: list[dict] = []
    for record in records:
        truthful_prefix, hallucinated_prefix = get_branch_prefixes(record)
        for prefix, label, label_name in (
            (truthful_prefix, 0, "truthful"),
            (hallucinated_prefix, 1, "hallucinated"),
        ):
            question_map = block_mean_map(get_question_block(record, prefix))
            for token_item in get_trace(record, prefix):
                block = token_item["cross_attention"]
                attention_stack = block_layer_stack(block)
                attention_map = attention_stack.mean(axis=0)
                samples.append(
                    {
                        "sample_id": record["sample_id"],
                        "image_path": record["image_path"],
                        "problem": record.get("problem", record.get("question", "")),
                        "token": token_item.get("token", ""),
                        "step": int(token_item.get("step", 0)),
                        "label": label,
                        "label_name": label_name,
                        "question_alignment": cosine_similarity(question_map, attention_map),
                        "attention_map": attention_map,
                        "attention_stack": attention_stack,
                    }
                )
    return samples


def build_token_samples(records: list[dict], topk_fraction: float = 0.1) -> list[dict]:
    if not records:
        return []
    if "truthful_trace" in records[0]:
        return build_sequence_samples(records, topk_fraction=topk_fraction)
    return _build_binary_token_samples(records, topk_fraction=topk_fraction)
