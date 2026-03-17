# Complex Visual Reasoning: Factual vs Hallucinated Answer Analysis

This project analyzes mechanism differences between factual and hallucinated answer sequences on Geo-Thought using Qwen2.5-VL-Instruct.

## What was implemented

- Extracted 1000 Geo-Thought samples (`image + problem + CoT/solution`) into `outputs/data/subset_1000.jsonl`.
- Generated per-sample paired responses with prompt engineering:
  - truthful answer constrained to reference solution
  - hallucinated answer constrained to be strongly opposite to provided CoT/solution
- Output format is answer-only (`Final answer: <answer>`), no CoT in outputs.
- Recorded sequence-level cross-attention maps for truthful/hallucinated branches:
  - only last 12 layers
  - answer-length clipping (`answer_max_tokens=24`)
- Performed PCA/layer-band analysis of cross-attention differences.
- Trained manifold-learning-based classifier with 80/20 split and evaluated on test set.

## Repository structure

- `scripts/prepare_geothought_subset.py`: normalize Geo-Thought subset and local images
- `scripts/run_qwen_reasoning_attention.py`: generate paired answers + trace cross-attention
- `scripts/visualize_reasoning_attention_matplotlib.py`: PCA and layer-wise analysis
- `scripts/analyze_reasoning_separability.py`: 80/20 classification (including Isomap baseline)
- `outputs/classifier/`: model metrics and confusion matrix
- `outputs/visualizations/`: PCA, manifold, layer analysis, dataset summary plots

## Reproduce

### 1) Activate environment

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate reasoning-vlm
```

### 2) Generate paired answer-attention records

```bash
python scripts/run_qwen_reasoning_attention.py \
  --model-dir "/root/AI Trusty/.hf_home/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5" \
  --input-jsonl outputs/data/subset_1000.jsonl \
  --output-jsonl outputs/attention/reasoning_attention_records_1000.jsonl \
  --candidate-output-jsonl outputs/attention/candidates_1000.jsonl \
  --rejection-csv outputs/attention/rejections_1000.csv \
  --limit 1000 \
  --max-new-tokens 64 \
  --answer-max-tokens 24 \
  --trace-layer-mode last_n \
  --trace-last-n-layers 12 \
  --trace-map-size 8
```

### 3) PCA and mechanism analysis

```bash
python scripts/visualize_reasoning_attention_matplotlib.py \
  --input-jsonl outputs/attention/reasoning_attention_records_1000.jsonl \
  --output-dir outputs/visualizations \
  --per-sample-limit 0 \
  --include-manifold
```

### 4) 80/20 classifier training and evaluation

```bash
python scripts/analyze_reasoning_separability.py \
  --input-jsonl outputs/attention/reasoning_attention_records_1000.jsonl \
  --output-dir outputs/classifier \
  --test-fraction 0.2 \
  --cv-splits 5 \
  --random-seed 7 \
  --include-heavy-baselines
```

## Latest test metrics (20% holdout)

From `outputs/classifier/classifier_summary.json`:

- Best model: `compact_sequence_logistic`
- Accuracy: `0.9375`
- Balanced accuracy: `0.9375`
- Precision: `0.9442`
- Recall: `0.9300`
- ROC-AUC: `0.9781`
- Test size: `400`

## Notes

- Raw full attention record file (`reasoning_attention_records_1000.jsonl`) is intentionally not tracked in git due GitHub single-file size limits (>100MB).
- Tracked outputs focus on reproducible analysis artifacts and final metrics.
