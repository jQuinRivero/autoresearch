# autoresearch

![teaser](progress.png)

The idea: give an AI agent a multimodal embedding model training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here fine-tunes a pre-trained vision-language model (CLIP) for multimodal embedding tasks using contrastive learning. The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads image-text datasets), and runtime utilities (dataloader, retrieval evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the multimodal embedding model (dual encoder with CLIP backbone), projection heads, contrastive loss (InfoNCE), optimizer (AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, base model, freezing strategy, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **mean_recall** (average of Recall@1/5/10 for image-to-text and text-to-image retrieval) — higher is better.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download dataset (one-time, downloads from HuggingFace)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This makes experiments directly comparable regardless of what the agent changes (model backbone, batch size, freezing strategy, etc).
- **Pre-trained backbone.** The default setup fine-tunes a pre-trained CLIP model. This enables meaningful embedding quality within the 5-minute budget. The agent can experiment with different backbones, projection heads, and training strategies.
- **Contrastive learning.** Uses InfoNCE contrastive loss to align image and text embeddings. The agent can experiment with different loss functions (SigLIP, hard negatives, etc).
- **Self-contained.** No distributed training, no complex configs. One GPU, one file, one metric.

## Goal

Train a multimodal embedding model that achieves **top 10 on open source multimodal embedding benchmarks** (e.g., MMEB). The agent experiments autonomously to find the optimal combination of model architecture, training strategy, and hyperparameters.

## License

MIT
