#!/usr/bin/env python3
import argparse
import json
import random
import re
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from PIL import Image


DEFAULT_DATASET = "xinlingdedeng/GeoThought"
QUESTION_KEYS = (
    "problem",
    "question",
    "prompt",
    "instruction",
    "query",
)
REFERENCE_COT_KEYS = (
    "reference_cot",
    "cot",
    "reasoning",
    "solution",
    "thinking",
    "thought_chain",
    "chain_of_thought",
    "response",
    "assistant",
)
FINAL_ANSWER_KEYS = (
    "reference_final_answer",
    "final_answer",
    "answer",
    "label",
    "gt_answer",
    "ground_truth_answer",
)
IMAGE_KEYS = ("image", "images", "image_path", "path")
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
FINAL_ANSWER_PATTERNS = [
    re.compile(r"(?is)<answer>\s*(.*?)\s*</answer>"),
    re.compile(r"(?is)<final_answer>\s*(.*?)\s*</final_answer>"),
    re.compile(r"(?im)^\s*final answer\s*[:：]\s*(.+?)\s*$"),
    re.compile(r"(?im)^\s*answer\s*[:：]\s*(.+?)\s*$"),
    re.compile(r"(?im)^\s*therefore[, ]+the answer is\s*(.+?)\s*$"),
    re.compile(r"(?is)\bfinal answer\s*[:：]\s*(.+?)(?:\n\s*\n|\Z)"),
    re.compile(r"(?is)\bthe answer is\s*(.+?)(?:\n\s*\n|\Z)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a GeoThought subset with local images, problems, reference CoT, and final answers."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-files", nargs="*", default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--max-candidates", type=int, default=300)
    parser.add_argument("--max-scan", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shuffle-buffer-size", type=int, default=1000)
    parser.add_argument("--min-problem-chars", type=int, default=20)
    parser.add_argument("--min-reference-cot-chars", type=int, default=80)
    return parser.parse_args()


def image_to_rgb(image_obj) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, dict) and image_obj.get("bytes") is not None:
        return Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
    if isinstance(image_obj, dict) and image_obj.get("path"):
        return Image.open(image_obj["path"]).convert("RGB")
    if isinstance(image_obj, str) and image_obj:
        return Image.open(image_obj).convert("RGB")
    raise TypeError(f"Unsupported image object type: {type(image_obj)!r}")


def first_present(record: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


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
                if len(extracted) > 200:
                    lines = [line.strip() for line in extracted.splitlines() if line.strip()]
                    short_lines = [line for line in lines if len(line) <= 120]
                    if short_lines:
                        return short_lines[-1]
                return extracted
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    short_tail = [line for line in lines[-3:] if len(line) <= 120]
    if short_tail:
        return short_tail[-1]
    return lines[-1]


def strip_answer_markup(text: str) -> str:
    if not text:
        return ""
    stripped = text
    for pattern in FINAL_ANSWER_PATTERNS:
        stripped = pattern.sub("", stripped)
    stripped = BOXED_PATTERN.sub("", stripped)
    stripped = re.sub(r"(?is)</?(think|answer|final_answer)>", " ", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def build_reference_cot(record: dict) -> str:
    cot = first_present(record, REFERENCE_COT_KEYS)
    if cot:
        return strip_answer_markup(cot)
    answer_field = first_present(record, ("original_answer", "answer", "solution", "response"))
    return strip_answer_markup(answer_field)


def build_reference_solution(record: dict) -> str:
    answer = first_present(record, FINAL_ANSWER_KEYS)
    if answer:
        return answer
    cot = first_present(record, REFERENCE_COT_KEYS)
    if cot:
        return cot
    return first_present(record, ("original_answer", "solution", "response"))


def resolve_image(record: dict):
    for key in IMAGE_KEYS:
        if key not in record:
            continue
        value = record[key]
        if isinstance(value, list) and value:
            return value[0]
        if value is not None:
            return value
    return None


def load_source_dataset(args: argparse.Namespace):
    if args.data_files:
        suffix = Path(args.data_files[0]).suffix.lower()
        if suffix in {".json", ".jsonl"}:
            loader_name = "json"
        elif suffix == ".parquet":
            loader_name = "parquet"
        elif suffix == ".csv":
            loader_name = "csv"
        else:
            raise ValueError(f"Unsupported data file suffix: {suffix}")
        return load_dataset(
            loader_name,
            data_files=args.data_files,
            split=args.split,
            streaming=True,
        )
    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
    )
    return dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)


def should_keep(
    *,
    problem: str,
    reference_cot: str,
    reference_solution: str,
    reference_short_answer: str,
    image_obj,
    min_problem_chars: int,
    min_reference_cot_chars: int,
) -> tuple[bool, str]:
    if image_obj is None:
        return False, "missing_image"
    if len(problem) < min_problem_chars:
        return False, "short_problem"
    if len(reference_cot) < min_reference_cot_chars:
        return False, "short_reference_cot"
    if not reference_solution:
        return False, "missing_final_answer"
    if not reference_short_answer:
        return False, "missing_short_answer"
    return True, "ok"


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    output_path = Path(args.output_jsonl).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_source_dataset(args)
    stats = {
        "scanned": 0,
        "kept": 0,
        "missing_image": 0,
        "short_problem": 0,
        "short_reference_cot": 0,
        "missing_final_answer": 0,
        "missing_short_answer": 0,
        "image_decode_error": 0,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        for record in dataset:
            stats["scanned"] += 1
            problem = first_present(record, QUESTION_KEYS)
            reference_cot = build_reference_cot(record)
            reference_solution = build_reference_solution(record)
            reference_short_answer = extract_final_answer(reference_solution)
            image_obj = resolve_image(record)
            keep, reason = should_keep(
                problem=problem,
                reference_cot=reference_cot,
                reference_solution=reference_solution,
                reference_short_answer=reference_short_answer,
                image_obj=image_obj,
                min_problem_chars=args.min_problem_chars,
                min_reference_cot_chars=args.min_reference_cot_chars,
            )
            if not keep:
                stats[reason] = stats.get(reason, 0) + 1
                if stats["scanned"] >= args.max_scan:
                    break
                continue

            try:
                image = image_to_rgb(image_obj)
            except Exception:
                stats["image_decode_error"] += 1
                if stats["scanned"] >= args.max_scan:
                    break
                continue

            raw_id = record.get("id", record.get("sample_id", f"sample_{stats['scanned']:06d}"))
            sample_id = f"geothought_{str(raw_id).strip().replace('/', '_')}"
            image_path = image_dir / f"{sample_id}.png"
            image.save(image_path)

            normalized = {
                "sample_id": sample_id,
                "dataset_name": args.dataset,
                "dataset_split": args.split,
                "image_path": str(image_path),
                "problem": problem,
                "question": problem,
                "reference_cot": reference_cot,
                "reference_final_answer": reference_solution,
                "reference_short_answer": reference_short_answer,
                "ground_truth_answer": reference_short_answer,
            }
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            if stats["kept"] >= args.max_candidates or stats["scanned"] >= args.max_scan:
                break

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
