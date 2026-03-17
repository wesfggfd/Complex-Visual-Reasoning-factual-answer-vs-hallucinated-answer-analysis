#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import torch
import torch.nn.functional as F
from openai import OpenAI
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


NO_COT_REASONING_SYSTEM_PROMPT = (
    "You are a careful visual geometry answerer. "
    "Use only the image and the problem statement. "
    "Do not mention any hidden reference or external rationale. "
    "Output only one short final answer line in the format: Final answer: <answer>."
)
GUIDED_TRUTHFUL_SYSTEM_PROMPT = (
    "You are a careful visual geometry answerer. "
    "Use the image, the problem statement, and the provided reference solution to produce a correct answer. "
    "Follow the reference faithfully, but do not say that a reference was provided. "
    "Do not output any chain-of-thought, explanation, or intermediate steps. "
    "Output exactly one short line in the format: Final answer: <answer>."
)
GUIDED_HALLUCINATION_SYSTEM_PROMPT = (
    "You are a visual geometry answerer writing a plausible but incorrect answer. "
    "Use the image, problem statement, and provided reference only to understand what a correct solution would look like, "
    "then deliberately produce an answer whose hidden reasoning chain is completely opposite to the provided CoT/solution. "
    "The contradiction must be severe on core geometric relations or conclusions, not minor wording differences. "
    "Do not mention that a reference was provided. "
    "Do not output any chain-of-thought, explanation, or intermediate steps. "
    "Output exactly one short line in the format: Final answer: <answer>."
)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_ANSWER_PATTERNS = [
    re.compile(r"(?is)<answer>\s*(.*?)\s*</answer>"),
    re.compile(r"(?is)<final_answer>\s*(.*?)\s*</final_answer>"),
    re.compile(r"(?im)^\s*final answer\s*[:：]\s*(.+?)\s*$"),
    re.compile(r"(?im)^\s*answer\s*[:：]\s*(.+?)\s*$"),
]
CHOICE_PATTERN = re.compile(r"\b([A-E])\b")
TOKEN_PATTERN = re.compile(r"[a-z]+|\d+(?:\.\d+)?", re.IGNORECASE)


def clip_reference_text(text: str, max_chars: int = 900) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() + " ..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GeoThought no-CoT multi-sample tracing and filter truthful-vs-hallucinated pairs."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--candidate-output-jsonl")
    parser.add_argument("--rejection-csv", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--truthful-min-cot-sim", type=float, default=0.18)
    parser.add_argument("--hallucinated-max-cot-sim", type=float, default=0.12)
    parser.add_argument("--trace-selected-only", action="store_true", default=True)
    parser.add_argument("--trace-token-selector", default="late", choices=["all", "late", "final"])
    parser.add_argument("--trace-max-steps", type=int, default=4)
    parser.add_argument("--trace-layer-mode", default="last_n", choices=["all", "last_n", "bands"])
    parser.add_argument("--trace-last-n-layers", type=int, default=12)
    parser.add_argument("--trace-map-size", type=int, default=8)
    parser.add_argument("--answer-max-tokens", type=int, default=24)
    parser.add_argument("--judge-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--judge-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--judge-model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    return parser.parse_args()


def load_records(path: Path, limit: int) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def build_prompt_text(problem: str) -> str:
    return (
        "Solve the following geometry problem from the image.\n\n"
        f"Problem:\n{problem.strip()}\n\n"
        "Show your reasoning clearly, then end with exactly one line formatted as:\n"
        "Final answer: <answer>"
    )


def build_guided_truthful_prompt(
    problem: str,
    *,
    reference_cot: str,
    reference_solution: str,
    reference_short_answer: str,
) -> str:
    reference_reasoning = clip_reference_text(reference_cot.strip() or reference_solution.strip())
    reference_solution_text = clip_reference_text(
        reference_solution.strip() or f"Final answer: {reference_short_answer}".strip()
    )
    return (
        "Solve the following geometry problem from the image.\n\n"
        f"Problem:\n{problem.strip()}\n\n"
        "Reference reasoning to follow internally:\n"
        f"{reference_reasoning}\n\n"
        "Reference solution and final answer to stay consistent with:\n"
        f"{reference_solution_text}\n\n"
        "Now answer as if you solved it yourself. "
        "Do not mention the reference. "
        "Do not include any reasoning steps. "
        "Output exactly one line:\n"
        "Final answer: <answer>"
    )


def build_guided_hallucination_prompt(
    problem: str,
    *,
    reference_cot: str,
    reference_solution: str,
    reference_short_answer: str,
) -> str:
    reference_reasoning = clip_reference_text(reference_cot.strip() or reference_solution.strip())
    reference_solution_text = clip_reference_text(
        reference_solution.strip() or f"Final answer: {reference_short_answer}".strip()
    )
    return (
        "Solve the following geometry problem from the image, but intentionally contradict the reference solution.\n\n"
        f"Problem:\n{problem.strip()}\n\n"
        "Reference reasoning that you must strongly contradict:\n"
        f"{reference_reasoning}\n\n"
        "Reference solution and final answer that you must not follow:\n"
        f"{reference_solution_text}\n\n"
        "Write a plausible but incorrect final answer in English as if you solved it yourself. "
        "At least one major geometric relation, theorem usage, or numerical conclusion should conflict with the reference. "
        "Do not mention the reference. "
        "Do not include any reasoning steps. "
        "Output exactly one line:\n"
        "Final answer: <answer>"
    )


def build_messages(
    image_path: str,
    prompt_text: str,
    system_prompt: str,
    assistant_text: str | None = None,
) -> list[dict]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
    return messages


def canonicalize_answer(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    boxed_match = BOXED_PATTERN.search(cleaned)
    if boxed_match:
        cleaned = boxed_match.group(1).strip()
    choice_match = CHOICE_PATTERN.search(cleaned.upper())
    if choice_match and len(cleaned.split()) <= 6:
        return choice_match.group(1)
    cleaned = cleaned.lower()
    cleaned = cleaned.replace("$", " ")
    cleaned = re.sub(r"\\text\{([^{}]+)\}", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9./%+=\- ]", "", cleaned)
    cleaned = cleaned.strip(" .,:;")
    return cleaned


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(?is)</?(think|answer|final_answer)>", " ", text)
    boxed_match = BOXED_PATTERN.search(text)
    if boxed_match:
        return boxed_match.group(1).strip()
    explicit_tail_patterns = [
        re.compile(r"(?is)\bfinal answer\s*[:：]\s*(.+?)\s*$"),
        re.compile(r"(?is)\bthe answer is\s*(.+?)\s*$"),
    ]
    for pattern in explicit_tail_patterns:
        matches = list(pattern.finditer(text))
        if matches:
            extracted = matches[-1].group(1).strip()
            extracted = re.sub(r"\s+", " ", extracted).strip(" .,:;")
            if extracted:
                return extracted
    for pattern in FINAL_ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r"\s+", " ", extracted).strip(" .,:;")
            if extracted:
                return extracted
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    tail = lines[-1]
    choice_match = CHOICE_PATTERN.search(tail.upper())
    if choice_match and len(tail.split()) <= 6:
        return choice_match.group(1)
    return tail


def normalize_answer_line(text: str) -> tuple[str, str]:
    final_answer = extract_final_answer(text)
    if final_answer:
        return final_answer, f"Final answer: {final_answer}"
    fallback = " ".join(text.strip().split())
    if not fallback:
        fallback = "unknown"
    return fallback, f"Final answer: {fallback}"


def answers_match(candidate_answer: str, reference_answer: str) -> bool:
    left = canonicalize_answer(candidate_answer)
    right = canonicalize_answer(reference_answer)
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) >= 3 and len(right) >= 3 and (left in right or right in left):
        return True
    return False


def tokenize_reasoning(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def reasoning_similarity(candidate_text: str, reference_cot: str) -> float:
    left_tokens = tokenize_reasoning(candidate_text)
    right_tokens = tokenize_reasoning(reference_cot)
    if len(left_tokens) < 8 or len(right_tokens) < 8:
        return 0.0
    left_counter = Counter(left_tokens)
    right_counter = Counter(right_tokens)
    overlap = sum((left_counter & right_counter).values())
    if overlap <= 0:
        token_f1 = 0.0
    else:
        precision = overlap / max(1, len(left_tokens))
        recall = overlap / max(1, len(right_tokens))
        token_f1 = 2 * precision * recall / max(1e-8, precision + recall)
    sequence_ratio = SequenceMatcher(None, " ".join(left_tokens), " ".join(right_tokens)).ratio()
    return float(0.65 * token_f1 + 0.35 * sequence_ratio)


def combined_reasoning_similarity(
    candidate_text: str,
    *,
    reference_cot: str,
    reference_solution: str,
) -> float:
    scores = []
    if reference_cot.strip():
        scores.append(reasoning_similarity(candidate_text, reference_cot))
    if reference_solution.strip():
        scores.append(reasoning_similarity(candidate_text, reference_solution))
    if not scores:
        return 0.0
    return float(max(scores))


def build_judge_prompt(
    *,
    problem: str,
    reference_cot: str,
    reference_solution: str,
    reference_short_answer: str,
    candidate_text: str,
) -> str:
    return f"""
You are a strict judge for a visual reasoning experiment.

Determine whether the candidate response deviates from the reference chain-of-thought and reference solution.

Problem:
{problem.strip()}

Reference CoT:
{reference_cot.strip()}

Reference Solution:
{reference_solution.strip()}

Reference Short Answer:
{reference_short_answer.strip()}

Candidate Response:
{candidate_text.strip()}

Return exactly one label and nothing else:
- deviated
- not_deviated

Use `deviated` only if the candidate clearly departs from the reference reasoning and/or final answer.
If the candidate mostly follows the reference, return `not_deviated`.
Be conservative.
""".strip()


def judge_candidate_deviation(
    client: OpenAI,
    *,
    model: str,
    problem: str,
    reference_cot: str,
    reference_solution: str,
    reference_short_answer: str,
    candidate_text: str,
) -> dict:
    def extract_api_text(response_obj) -> str:
        if isinstance(response_obj, str):
            return response_obj
        content = response_obj.choices[0].message.content
        if isinstance(content, str):
            return content
        raise ValueError("Judge API did not return text content.")

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "You are a strict reasoning-trajectory evaluator. Return only one label: deviated or not_deviated.",
            },
            {
                "role": "user",
                "content": build_judge_prompt(
                    problem=problem,
                    reference_cot=reference_cot,
                    reference_solution=reference_solution,
                    reference_short_answer=reference_short_answer,
                    candidate_text=candidate_text,
                ),
            },
        ],
    )
    content = extract_api_text(response).strip().lower()
    normalized = re.sub(r"[^a-z_]+", " ", content).strip().replace(" ", "_")
    deviates = normalized.startswith("deviated")
    if normalized.startswith("not_deviated"):
        deviates = False
    return {
        "deviates_from_reference": deviates,
        "reason": content,
    }


def normalize_openai_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return base_url
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = normalized + "/v1"
    return normalized


def prepare_inputs(
    processor,
    messages: list[dict],
    device: torch.device,
    *,
    add_generation_prompt: bool,
):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    return model_inputs, inputs


def find_last_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        raise ValueError("Expected a non-empty subsequence.")
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start : start + len(needle)] == needle:
            return start
    raise ValueError("Could not locate subsequence in token ids.")


def normalize_prompt_search_text(text: str) -> str:
    cleaned = text.replace("<image>", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def locate_query_indices(processor, input_ids: list[int], candidates: list[str]) -> tuple[list[str], list[int]]:
    for candidate in candidates:
        normalized = normalize_prompt_search_text(candidate)
        if not normalized:
            continue
        token_ids = processor.tokenizer(normalized, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        try:
            start = find_last_subsequence(input_ids, token_ids)
        except ValueError:
            continue
        indices = list(range(start, start + len(token_ids)))
        tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
        return tokens, indices
    raise ValueError("Could not locate subsequence in token ids.")


def fallback_query_indices(processor, input_ids: list[int], mm_token_type_ids: list[int]) -> tuple[list[str], list[int]]:
    vision_indices = [index for index, value in enumerate(mm_token_type_ids) if value == 1]
    start_index = (vision_indices[-1] + 1) if vision_indices else 0
    candidate_indices = [
        index
        for index in range(start_index, len(input_ids))
        if mm_token_type_ids[index] != 1
    ]
    if not candidate_indices:
        candidate_indices = [index for index in range(len(input_ids)) if mm_token_type_ids[index] != 1]
    if not candidate_indices:
        raise ValueError("Could not infer fallback query indices from prompt tokens.")
    token_ids = [input_ids[index] for index in candidate_indices]
    tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
    return tokens, candidate_indices


def build_attention_metadata(processor, cpu_inputs: dict, question: str, prompt_text: str) -> dict:
    input_ids = cpu_inputs["input_ids"][0].tolist()
    mm_token_type_ids = cpu_inputs["mm_token_type_ids"][0].tolist()
    try:
        question_tokens, question_token_indices = locate_query_indices(
            processor,
            input_ids,
            [
                question,
                prompt_text,
                f"Problem: {question}",
                f"Problem:\n{question}",
            ],
        )
    except ValueError:
        question_tokens, question_token_indices = fallback_query_indices(
            processor,
            input_ids,
            mm_token_type_ids,
        )
    vision_token_indices = [index for index, value in enumerate(mm_token_type_ids) if value == 1]
    grid_t, grid_h, grid_w = cpu_inputs["image_grid_thw"][0].tolist()
    merge_size = int(getattr(processor.image_processor, "merge_size", 1) or 1)
    merged_grid_thw = [int(grid_t), int(grid_h) // merge_size, int(grid_w) // merge_size]
    expected_vision_tokens = merged_grid_thw[0] * merged_grid_thw[1] * merged_grid_thw[2]
    if expected_vision_tokens != len(vision_token_indices):
        raise ValueError(
            f"Vision token count mismatch: expected {expected_vision_tokens}, got {len(vision_token_indices)}"
        )
    return {
        "question_tokens": question_tokens,
        "question_token_indices": question_token_indices,
        "question_token_span": [question_token_indices[0], question_token_indices[-1]],
        "vision_token_indices": vision_token_indices,
        "vision_token_span": [vision_token_indices[0], vision_token_indices[-1]],
        "merged_grid_thw": merged_grid_thw,
        "merge_size": merge_size,
    }


def trace_token_positions(trace_length: int, token_selector: str, max_steps: int) -> list[int]:
    if trace_length <= 0:
        return []
    if token_selector == "final":
        return [trace_length - 1]
    if token_selector == "late":
        start = max(0, trace_length - max(1, trace_length // 3))
        base = list(range(start, trace_length))
    else:
        base = list(range(trace_length))
    if len(base) <= max_steps:
        return base
    positions = torch.linspace(0, len(base) - 1, steps=max_steps, dtype=torch.float32)
    selected = []
    seen = set()
    for index in positions.round().to(dtype=torch.int64).tolist():
        value = base[index]
        if value not in seen:
            selected.append(value)
            seen.add(value)
    return selected


def ordered_block_names(blocks: dict[str, list[list[float]]]) -> list[str]:
    def sort_key(name: str) -> tuple[int, int | str]:
        if name.startswith("layer_"):
            return (0, int(name.split("_", 1)[1]))
        if name.startswith("band_"):
            order = {"band_early": 0, "band_middle": 1, "band_late": 2}
            return (1, order.get(name, 99))
        return (2, name)

    return sorted(blocks, key=sort_key)


def reshape_vision_attention_tensor(values: torch.Tensor, merged_grid_thw: list[int]) -> torch.Tensor:
    merged_t, merged_h, merged_w = merged_grid_thw
    reshaped = values.reshape(merged_t, merged_h, merged_w)
    if merged_t == 1:
        return reshaped[0]
    return reshaped.mean(dim=0)


def downsample_attention_map(map_2d: torch.Tensor, map_size: int) -> torch.Tensor:
    if map_size <= 0:
        return map_2d
    if map_2d.shape[0] == map_size and map_2d.shape[1] == map_size:
        return map_2d
    resized = F.interpolate(
        map_2d.unsqueeze(0).unsqueeze(0),
        size=(map_size, map_size),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0]


def layer_groups(layer_count: int, layer_mode: str, last_n_layers: int) -> list[tuple[str, list[int]]]:
    if layer_mode == "all":
        return [(f"layer_{idx}", [idx]) for idx in range(layer_count)]
    if layer_mode == "last_n":
        start = max(0, layer_count - max(1, last_n_layers))
        return [(f"layer_{idx}", [idx]) for idx in range(start, layer_count)]
    if layer_mode == "bands":
        third = max(1, layer_count // 3)
        middle_start = max(0, (layer_count - third) // 2)
        middle_end = min(layer_count, middle_start + third)
        return [
            ("band_early", list(range(0, min(third, layer_count)))),
            ("band_middle", list(range(middle_start, middle_end))),
            ("band_late", list(range(max(0, layer_count - third), layer_count))),
        ]
    raise ValueError(f"Unsupported layer_mode: {layer_mode}")


def summarize_layer_maps(layer_maps: dict[str, list[list[float]]]) -> dict[str, list[list[float]]]:
    if not layer_maps:
        return {}
    ordered_layers = ordered_block_names(layer_maps)
    stack = torch.tensor([layer_maps[layer] for layer in ordered_layers], dtype=torch.float32)
    third = max(1, stack.shape[0] // 3)
    middle_start = max(0, (stack.shape[0] - third) // 2)
    middle_end = middle_start + third
    return {
        "mean_over_layers": stack.mean(dim=0).tolist(),
        "early_layers_mean": stack[:third].mean(dim=0).tolist(),
        "middle_layers_mean": stack[middle_start:middle_end].mean(dim=0).tolist(),
        "late_layers_mean": stack[-third:].mean(dim=0).tolist(),
    }


def capture_query_to_vision_attentions(
    attentions,
    query_indices: list[int],
    vision_token_indices: list[int],
    merged_grid_thw: list[int],
    query_tokens: list[str],
    *,
    layer_mode: str,
    last_n_layers: int,
    map_size: int,
) -> dict:
    if attentions is None:
        return {
            "query_tokens": query_tokens,
            "query_count": len(query_indices),
            "layer_maps": {},
            "layer_summary": {},
        }
    layer_maps: dict[str, list[list[float]]] = {}
    valid_layers = [(layer_idx, layer_attn) for layer_idx, layer_attn in enumerate(attentions) if layer_attn is not None]
    if not valid_layers:
        return {
            "query_tokens": query_tokens,
            "query_count": len(query_indices),
            "layer_maps": {},
            "layer_summary": {},
        }
    layer_lookup = {layer_idx: layer_attn for layer_idx, layer_attn in valid_layers}
    for block_name, indices in layer_groups(len(attentions), layer_mode, last_n_layers):
        tensors = []
        for layer_idx in indices:
            layer_attn = layer_lookup.get(layer_idx)
            if layer_attn is None:
                continue
            layer_tensor = layer_attn[0]
            query_to_vision = layer_tensor[:, query_indices, :][:, :, vision_token_indices]
            averaged = query_to_vision.mean(dim=0).mean(dim=0)
            map_2d = reshape_vision_attention_tensor(averaged, merged_grid_thw)
            map_2d = downsample_attention_map(map_2d.float(), map_size)
            tensors.append(map_2d.detach().cpu())
        if not tensors:
            continue
        reduced = torch.stack(tensors, dim=0).mean(dim=0)
        layer_maps[block_name] = reduced.tolist()
    return {
        "query_tokens": query_tokens,
        "query_count": len(query_indices),
        "layer_maps": layer_maps,
        "layer_summary": summarize_layer_maps(layer_maps),
    }


@torch.inference_mode()
def generate_text_only(
    *,
    model,
    processor,
    image_path: str,
    question: str,
    prompt_text: str,
    system_prompt: str,
    max_new_tokens: int,
    answer_max_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> tuple[str, str, dict]:
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    messages = build_messages(image_path, prompt_text, system_prompt)
    device = model.device
    model_inputs, raw_inputs = prepare_inputs(processor, messages, device, add_generation_prompt=True)
    prompt_length = int(model_inputs["input_ids"].shape[1])
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=max(8, max_new_tokens),
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        use_cache=True,
        return_dict_in_generate=True,
    )
    generated_ids = outputs.sequences[0, prompt_length:].detach().cpu().tolist()
    eos_token_id = processor.tokenizer.eos_token_id
    answer_token_ids = [token_id for token_id in generated_ids if token_id != eos_token_id]
    if answer_max_tokens > 0:
        answer_token_ids = answer_token_ids[:answer_max_tokens]
    raw_text = processor.tokenizer.decode(answer_token_ids, skip_special_tokens=True).strip()
    raw_text = clip_answer_text(processor, raw_text, answer_max_tokens)
    final_answer, raw_text = normalize_answer_line(raw_text)
    metadata = {
        "prompt_length": prompt_length,
        "generated_token_count": len(answer_token_ids),
        "input_token_count": int(raw_inputs["input_ids"].shape[1]),
        "sampling": {
            "do_sample": do_sample,
            "temperature": temperature if do_sample else 0.0,
            "top_p": top_p if do_sample else 1.0,
            "seed": seed,
        },
    }
    return final_answer, raw_text, metadata


def locate_answer_indices(processor, input_ids: list[int], answer_text: str) -> tuple[list[int], list[str]]:
    answer_ids = processor.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise ValueError("Expected a non-empty answer text for tracing.")
    answer_start = find_last_subsequence(input_ids, answer_ids)
    answer_indices = list(range(answer_start, answer_start + len(answer_ids)))
    answer_tokens = processor.tokenizer.convert_ids_to_tokens(answer_ids)
    return answer_indices, answer_tokens


def clip_answer_text(processor, answer_text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return answer_text.strip()
    answer_ids = processor.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    if len(answer_ids) <= max_tokens:
        return answer_text.strip()
    clipped_ids = answer_ids[:max_tokens]
    clipped_text = processor.tokenizer.decode(clipped_ids, skip_special_tokens=True).strip()
    final_answer = extract_final_answer(clipped_text)
    if final_answer:
        return f"Final answer: {final_answer}"
    return clipped_text


@torch.inference_mode()
def trace_selected_candidate(
    *,
    model,
    processor,
    image_path: str,
    question: str,
    prompt_text: str,
    system_prompt: str,
    answer_text: str,
    layer_mode: str,
    last_n_layers: int,
    map_size: int,
) -> tuple[str, str, dict, dict, dict]:
    messages = build_messages(image_path, prompt_text, system_prompt, assistant_text=answer_text)
    device = model.device
    model_inputs, raw_inputs = prepare_inputs(processor, messages, device, add_generation_prompt=False)
    attention_metadata = build_attention_metadata(processor, raw_inputs, question, prompt_text)
    input_ids = raw_inputs["input_ids"][0].tolist()
    answer_indices, answer_tokens = locate_answer_indices(processor, input_ids, answer_text)
    try:
        outputs = model(
            **model_inputs,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    except torch.OutOfMemoryError:
        # Fallback: shrink prompt context while keeping answer-only tracing.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        compact_prompt = (
            "Solve the following geometry problem from the image.\n\n"
            f"Problem:\n{question.strip()}\n\n"
            "Output exactly one line:\nFinal answer: <answer>"
        )
        messages = build_messages(image_path, compact_prompt, system_prompt, assistant_text=answer_text)
        model_inputs, raw_inputs = prepare_inputs(processor, messages, device, add_generation_prompt=False)
        attention_metadata = build_attention_metadata(processor, raw_inputs, question, compact_prompt)
        input_ids = raw_inputs["input_ids"][0].tolist()
        answer_indices, answer_tokens = locate_answer_indices(processor, input_ids, answer_text)
        outputs = model(
            **model_inputs,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
    attentions = outputs.attentions or ()
    question_attention = capture_query_to_vision_attentions(
        attentions,
        query_indices=attention_metadata["question_token_indices"],
        vision_token_indices=attention_metadata["vision_token_indices"],
        merged_grid_thw=attention_metadata["merged_grid_thw"],
        query_tokens=attention_metadata["question_tokens"],
        layer_mode=layer_mode,
        last_n_layers=last_n_layers,
        map_size=map_size,
    )
    answer_sequence_attention = capture_query_to_vision_attentions(
        attentions,
        query_indices=answer_indices,
        vision_token_indices=attention_metadata["vision_token_indices"],
        merged_grid_thw=attention_metadata["merged_grid_thw"],
        query_tokens=answer_tokens,
        layer_mode=layer_mode,
        last_n_layers=last_n_layers,
        map_size=map_size,
    )
    final_answer, raw_text = normalize_answer_line(answer_text.strip())
    metadata = {
        "input_token_count": int(raw_inputs["input_ids"].shape[1]),
        "generated_token_count": len(answer_indices),
        "answer_token_count": len(answer_indices),
        "answer_token_span": [answer_indices[0], answer_indices[-1]],
        "answer_tokens": answer_tokens,
        "question_token_span": attention_metadata["question_token_span"],
        "vision_token_span": attention_metadata["vision_token_span"],
        "merged_grid_thw": attention_metadata["merged_grid_thw"],
        "trace_layer_mode": layer_mode,
        "trace_last_n_layers": last_n_layers,
        "trace_map_size": map_size,
        "attention_scope": "whole_answer_sequence",
    }
    return final_answer, raw_text, question_attention, answer_sequence_attention, metadata


def validate_image_paths(records: list[dict]) -> None:
    missing = [record["image_path"] for record in records if not Path(record["image_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing image paths, first example: {missing[0]}")


def select_truthful_candidate(candidates: list[dict], min_similarity: float) -> dict | None:
    truthful = [
        candidate
        for candidate in candidates
        if candidate["final_answer_match"] and candidate["cot_similarity"] >= min_similarity
    ]
    if not truthful:
        return None
    truthful.sort(
        key=lambda item: (
            item["cot_similarity"],
            item["generated_token_count"],
            len(item["raw_text"]),
        ),
        reverse=True,
    )
    return truthful[0]


def select_hallucinated_candidate(
    candidates: list[dict],
    truthful_candidate_index: int | None,
    max_similarity: float,
) -> dict | None:
    hallucinated = [
        candidate
        for candidate in candidates
        if candidate["candidate_index"] != truthful_candidate_index
        and candidate.get("judge_deviates_from_reference", False)
    ]
    if not hallucinated:
        return None
    hallucinated.sort(
        key=lambda item: (
            int(item.get("judge_deviates_from_reference", False)),
            -min(float(item["cot_similarity"]), float(max_similarity)),
            -item["cot_similarity"],
            item["generated_token_count"],
        ),
        reverse=True,
    )
    return hallucinated[0]


def candidate_summary(candidate: dict) -> dict:
    return {
        "candidate_index": candidate["candidate_index"],
        "generation_mode": candidate.get("generation_mode", "free"),
        "final_answer": candidate["final_answer"],
        "final_answer_match": candidate["final_answer_match"],
        "cot_similarity": candidate["cot_similarity"],
        "generated_token_count": candidate["generated_token_count"],
        "raw_text": candidate["raw_text"],
    }


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    candidate_output_path = (
        Path(args.candidate_output_jsonl).expanduser().resolve()
        if args.candidate_output_jsonl
        else None
    )
    rejection_csv_path = Path(args.rejection_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejection_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if candidate_output_path is not None:
        candidate_output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path, args.limit)
    if not records:
        raise SystemExit("No records loaded from input JSONL.")
    validate_image_paths(records)

    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )

    rejections: list[dict] = []
    valid_count = 0
    with output_path.open("w", encoding="utf-8") as valid_handle:
        candidate_handle = (
            candidate_output_path.open("w", encoding="utf-8") if candidate_output_path is not None else None
        )
        try:
            for index, record in enumerate(records, start=1):
                sample_id = record["sample_id"]
                problem = record.get("problem", record.get("question", ""))
                image_path = record["image_path"]
                reference_cot = record.get("reference_cot", "")
                reference_solution = record.get(
                    "reference_final_answer",
                    record.get("reference_solution", ""),
                )
                reference_short_answer = (
                    record.get("reference_short_answer")
                    or record.get("ground_truth_answer", "")
                    or extract_final_answer(reference_solution)
                )
                guided_truthful_prompt = build_guided_truthful_prompt(
                    problem,
                    reference_cot=reference_cot,
                    reference_solution=reference_solution,
                    reference_short_answer=reference_short_answer,
                )
                guided_hallucination_prompt = build_guided_hallucination_prompt(
                    problem,
                    reference_cot=reference_cot,
                    reference_solution=reference_solution,
                    reference_short_answer=reference_short_answer,
                )

                truthful_seed = args.random_seed + index * 1000
                truthful_final_answer, truthful_raw_text, truthful_generation_meta = generate_text_only(
                    model=model,
                    processor=processor,
                    image_path=image_path,
                    question=problem,
                    prompt_text=guided_truthful_prompt,
                    system_prompt=GUIDED_TRUTHFUL_SYSTEM_PROMPT,
                    max_new_tokens=args.max_new_tokens,
                    answer_max_tokens=args.answer_max_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    seed=truthful_seed,
                )
                guided_truthful_candidate = {
                    "candidate_index": -1,
                    "generation_mode": "guided_truthful",
                    "final_answer": truthful_final_answer,
                    "final_answer_match": answers_match(truthful_final_answer, reference_short_answer),
                    "cot_similarity": combined_reasoning_similarity(
                        truthful_raw_text,
                        reference_cot=reference_cot,
                        reference_solution=reference_solution,
                    ),
                    "raw_text": truthful_raw_text,
                    "meta": truthful_generation_meta,
                    "generated_token_count": truthful_generation_meta["generated_token_count"],
                    "seed": truthful_seed,
                    "prompt_text": guided_truthful_prompt,
                    "system_prompt": GUIDED_TRUTHFUL_SYSTEM_PROMPT,
                }
                truthful = guided_truthful_candidate

                hallucination_seed = args.random_seed + index * 1000
                hallucinated_final_answer, hallucinated_raw_text, hallucinated_generation_meta = (
                    generate_text_only(
                        model=model,
                        processor=processor,
                        image_path=image_path,
                        question=problem,
                        prompt_text=guided_hallucination_prompt,
                        system_prompt=GUIDED_HALLUCINATION_SYSTEM_PROMPT,
                        max_new_tokens=args.max_new_tokens,
                        answer_max_tokens=args.answer_max_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=hallucination_seed,
                    )
                )
                hallucinated = {
                    "candidate_index": 0,
                    "generation_mode": "guided_hallucination_direct",
                    "final_answer": hallucinated_final_answer,
                    "final_answer_match": answers_match(
                        hallucinated_final_answer, reference_short_answer
                    ),
                    "cot_similarity": combined_reasoning_similarity(
                        hallucinated_raw_text,
                        reference_cot=reference_cot,
                        reference_solution=reference_solution,
                    ),
                    "raw_text": hallucinated_raw_text,
                    "meta": hallucinated_generation_meta,
                    "generated_token_count": hallucinated_generation_meta["generated_token_count"],
                    "seed": hallucination_seed,
                    "prompt_text": guided_hallucination_prompt,
                    "system_prompt": GUIDED_HALLUCINATION_SYSTEM_PROMPT,
                }
                candidates = [hallucinated]

                candidate_payload = {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "problem": problem,
                    "reference_final_answer": reference_solution,
                    "reference_short_answer": reference_short_answer,
                    "reference_cot_length": len(reference_cot),
                    "truthful_guided_attempt_summary": candidate_summary(guided_truthful_candidate),
                    "truthful_guided_summary": candidate_summary(truthful),
                    "candidate_count": len(candidates),
                    "candidates": [candidate_summary(candidate) for candidate in candidates],
                    "selected_truthful_candidate_index": truthful["candidate_index"],
                    "selected_hallucinated_candidate_index": hallucinated["candidate_index"],
                }
                if candidate_handle is not None:
                    candidate_handle.write(json.dumps(candidate_payload, ensure_ascii=False) + "\n")
                    candidate_handle.flush()

                truthful_final_answer, truthful_raw_text, truthful_question_attention, truthful_sequence_attention, truthful_meta = (
                    trace_selected_candidate(
                        model=model,
                        processor=processor,
                        image_path=image_path,
                        question=problem,
                        prompt_text=truthful["prompt_text"],
                        system_prompt=truthful["system_prompt"],
                        answer_text=truthful["raw_text"],
                        layer_mode=args.trace_layer_mode,
                        last_n_layers=args.trace_last_n_layers,
                        map_size=args.trace_map_size,
                    )
                )
                hallucinated_final_answer, hallucinated_raw_text, hallucinated_question_attention, hallucinated_sequence_attention, hallucinated_meta = (
                    trace_selected_candidate(
                        model=model,
                        processor=processor,
                        image_path=image_path,
                        question=problem,
                        prompt_text=hallucinated["prompt_text"],
                        system_prompt=hallucinated["system_prompt"],
                        answer_text=hallucinated["raw_text"],
                        layer_mode=args.trace_layer_mode,
                        last_n_layers=args.trace_last_n_layers,
                        map_size=args.trace_map_size,
                    )
                )
                truthful["final_answer"] = truthful_final_answer
                truthful["raw_text"] = truthful_raw_text
                truthful["question_attention"] = truthful_question_attention
                truthful["sequence_attention"] = truthful_sequence_attention
                truthful["meta"] = truthful_meta
                truthful["generated_token_count"] = truthful_meta["generated_token_count"]
                hallucinated["final_answer"] = hallucinated_final_answer
                hallucinated["raw_text"] = hallucinated_raw_text
                hallucinated["question_attention"] = hallucinated_question_attention
                hallucinated["sequence_attention"] = hallucinated_sequence_attention
                hallucinated["meta"] = hallucinated_meta
                hallucinated["generated_token_count"] = hallucinated_meta["generated_token_count"]

                output_record = {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "problem": problem,
                    "question": problem,
                    "reference_cot": reference_cot,
                    "reference_final_answer": reference_solution,
                    "reference_short_answer": reference_short_answer,
                    "selection_mode": "guided_truthful_plus_direct_hallucination",
                    "sampling_config": {
                        "num_candidates": 1,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                        "answer_max_tokens": args.answer_max_tokens,
                        "random_seed": args.random_seed,
                    },
                    "trace_config": {
                        "attention_scope": "whole_answer_sequence",
                        "layer_mode": args.trace_layer_mode,
                        "last_n_layers": args.trace_last_n_layers,
                        "map_size": args.trace_map_size,
                    },
                    "truthful_guided_attempt_summary": candidate_summary(guided_truthful_candidate),
                    "truthful_guided_summary": candidate_summary(truthful),
                    "candidate_summaries": [candidate_summary(candidate) for candidate in candidates],
                    "truthful_candidate_index": truthful["candidate_index"],
                    "hallucinated_candidate_index": hallucinated["candidate_index"],
                    "truthful_final_answer": truthful["final_answer"],
                    "truthful_answer_match": truthful["final_answer_match"],
                    "truthful_cot_similarity": truthful["cot_similarity"],
                    "truthful_raw_text": truthful["raw_text"],
                    "truthful_question_attention": truthful["question_attention"],
                    "truthful_sequence_attention": truthful["sequence_attention"],
                    "truthful_meta": truthful["meta"],
                    "hallucinated_final_answer": hallucinated["final_answer"],
                    "hallucinated_answer_match": hallucinated["final_answer_match"],
                    "hallucinated_cot_similarity": hallucinated["cot_similarity"],
                    "hallucinated_raw_text": hallucinated["raw_text"],
                    "hallucinated_question_attention": hallucinated["question_attention"],
                    "hallucinated_sequence_attention": hallucinated["sequence_attention"],
                    "hallucinated_meta": hallucinated["meta"],
                    "is_valid_contrastive": True,
                }
                valid_handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                valid_handle.flush()
                valid_count += 1
                print(
                    f"[{index}/{len(records)}] {sample_id} kept "
                    f"truthful_sim={truthful['cot_similarity']:.3f} "
                    f"hallucinated_sim={hallucinated['cot_similarity']:.3f}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            if candidate_handle is not None:
                candidate_handle.close()

    with rejection_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "reference_final_answer",
                "reference_short_answer",
                "truthful_found",
                "hallucinated_found",
                "guided_truthful_final_answer",
                "guided_truthful_answer_match",
                "guided_truthful_cot_similarity",
                "best_match_cot_similarity",
                "best_wrong_cot_similarity",
                "judge_positive_count",
                "reason",
            ],
        )
        writer.writeheader()
        for row in rejections:
            writer.writerow(row)

    print(f"Valid contrastive samples: {valid_count}/{len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
