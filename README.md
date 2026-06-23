# graph_llm

Non-transformer LLM research artifact. Rigorous ablation-first: every novel component
is compared against matched baselines under identical compute budgets.

Target hardware: single RTX 3060 (12 GB VRAM).

---

## Setup

### 1. Create virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install — CPU-only (for development / smoke tests)

```bash
pip install -e ".[dev]" --index-url https://download.pytorch.org/whl/cpu
```

> The `--index-url` flag pulls the CPU-only wheel of PyTorch (~200 MB vs ~2 GB
> for CUDA). This is sufficient for unit tests and local iteration.

### 3. Install — GPU / CUDA (for real training on RTX 3060)

```bash
# CUDA 12.1 wheels (compatible with the 3060 and recent drivers)
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

Confirm the install:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Running

### Smoke training run (CPU)

```bash
python scripts/train.py --config configs/smoke.yaml
```

### Evaluate a checkpoint

```bash
python scripts/eval.py --config configs/smoke.yaml --checkpoint checkpoints/smoke_step010.pt
```

### Tests

```bash
pytest -q
```

### Lint

```bash
ruff check .
```

---

## Project layout

```
src/graph_llm/
  config.py          — ModelConfig / DataConfig / TrainConfig dataclasses + YAML loader
  data/              — dataset loader + trivial byte/char encoder
  tokenizer/         — STUB (real 16k BPE tokenizer in card e1644700)
  models/
    registry.py      — @register_model decorator + build_model(cfg)
    baselines/       — TransformerBaseline (smoke-test vehicle)
    components/      — future bilinear_frontend / GNN stubs
  train/
    trainer.py       — model-agnostic Trainer with 12 GB toolkit
    optim.py         — AdamW + cosine-with-warmup schedule
  eval/
    metrics.py       — perplexity + bits-per-byte
  utils/
    seed.py          — deterministic seeding
    logging.py       — structured logger
configs/
  smoke.yaml         — tiny CPU-runnable config
scripts/
  train.py           — training CLI
  eval.py            — eval CLI
tests/
  test_smoke.py      — CPU smoke: loss finite + strictly decreasing
```

---

## 12 GB VRAM toolkit (all config-toggleable)

| Feature | Config key |
|---|---|
| bf16 / fp16 mixed precision | `train.mixed_precision` |
| Gradient accumulation | `train.grad_accumulation_steps` |
| Activation checkpointing | `model.activation_checkpointing` |
| Gradient clipping | `train.grad_clip` |
| Cosine LR + warmup | `train.lr_schedule`, `train.warmup_steps` |
| Checkpoint save/resume | `train.checkpoint_dir`, `train.resume_from` |
| Deterministic seeding | `train.seed` |

---

## Roadmap

- **Phase 0 (this card):** scaffold + TransformerBaseline smoke test
- **Phase 0b:** matched baselines (Transformer + Mamba) + real eval harness
- **Phase 1 (card e1644700):** real 16k BPE tokenizer + phonological init
- **Phase 2+:** bilinear front-end, reasoning GNN, memory GNN
