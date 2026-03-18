"""
Multimodal embedding model training script. Single-GPU, single-file.
Fine-tunes a pre-trained vision-language model for embedding tasks.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPModel, AutoTokenizer

from prepare import (
    MAX_SEQ_LEN, TIME_BUDGET, IMAGE_SIZE,
    get_image_transform, ImageTextDataset, load_splits,
    make_dataloader, evaluate_retrieval,
)

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    base_model: str = "openai/clip-vit-base-patch32"
    embed_dim: int = 512
    projection_hidden: int = 1024
    temperature_init: float = 0.07
    freeze_vision_layers: int = 0
    freeze_text_layers: int = 0


class MultimodalEmbeddingModel(nn.Module):
    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config

        # Load pre-trained CLIP
        clip = CLIPModel.from_pretrained(config.base_model)
        self.vision_model = clip.vision_model
        self.text_model = clip.text_model

        # Get hidden dimensions
        vision_hidden = clip.config.vision_config.hidden_size
        text_hidden = clip.config.text_config.hidden_size

        # Projection heads (replace CLIP defaults with trainable projections)
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_hidden, config.projection_hidden),
            nn.GELU(),
            nn.Linear(config.projection_hidden, config.embed_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden, config.projection_hidden),
            nn.GELU(),
            nn.Linear(config.projection_hidden, config.embed_dim),
        )

        # Learnable temperature (log-scale)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / config.temperature_init))
        )

        # Freeze early layers if requested
        self._freeze_layers(config)

        # Clean up the original CLIP projections we don't need
        del clip

    def _freeze_layers(self, config):
        if config.freeze_vision_layers > 0:
            for param in self.vision_model.embeddings.parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.vision_model.encoder.layers):
                if i < config.freeze_vision_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        if config.freeze_text_layers > 0:
            for param in self.text_model.embeddings.parameters():
                param.requires_grad = False
            for i, layer in enumerate(self.text_model.encoder.layers):
                if i < config.freeze_text_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    def encode_image(self, pixel_values):
        """Encode images to normalized embeddings [B, embed_dim]."""
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        return F.normalize(self.vision_projection(pooled), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        """Encode text to normalized embeddings [B, embed_dim]."""
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return F.normalize(self.text_projection(pooled), dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        """Forward pass returning symmetric contrastive loss (InfoNCE)."""
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T

        batch_size = image_embeds.size(0)
        labels = torch.arange(batch_size, device=image_embeds.device)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        return (loss_i2t + loss_t2i) / 2

    def num_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable, 'frozen': total - trainable}

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model
BASE_MODEL = "openai/clip-vit-base-patch32"   # pre-trained backbone
EMBED_DIM = 512                               # final embedding dimension
PROJECTION_HIDDEN = 1024                       # projection head hidden dim
TEMPERATURE_INIT = 0.07                        # initial temperature
FREEZE_VISION_LAYERS = 0                       # freeze first N vision layers
FREEZE_TEXT_LAYERS = 0                         # freeze first N text layers

# Optimization
LEARNING_RATE = 5e-5           # learning rate for projection heads + logit_scale
BACKBONE_LR_SCALE = 0.1       # backbone LR = LEARNING_RATE * BACKBONE_LR_SCALE
WEIGHT_DECAY = 0.01            # AdamW weight decay
WARMUP_RATIO = 0.1             # fraction of time budget for LR warmup
MAX_GRAD_NORM = 1.0            # gradient clipping norm

# Batch size
DEVICE_BATCH_SIZE = 64         # per-device batch size (reduce if OOM)
GRAD_ACCUM_STEPS = 4           # gradient accumulation steps

# Data
DATASET_NAME = "nlphuji/flickr30k"

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, data
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# Load tokenizer
print(f"Loading tokenizer from {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Build model
config = EmbeddingConfig(
    base_model=BASE_MODEL,
    embed_dim=EMBED_DIM,
    projection_hidden=PROJECTION_HIDDEN,
    temperature_init=TEMPERATURE_INIT,
    freeze_vision_layers=FREEZE_VISION_LAYERS,
    freeze_text_layers=FREEZE_TEXT_LAYERS,
)
print(f"Model config: {asdict(config)}")

model = MultimodalEmbeddingModel(config)
model = model.to(device)

param_counts = model.num_parameters()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")

# Load data
print(f"Loading dataset: {DATASET_NAME}...")
train_data, val_data = load_splits(DATASET_NAME)
print(f"  Train samples: {len(train_data):,}")
print(f"  Val samples:   {len(val_data):,}")

train_transform = get_image_transform(IMAGE_SIZE, is_train=True)
val_transform = get_image_transform(IMAGE_SIZE, is_train=False)

train_dataset = ImageTextDataset(train_data, train_transform, tokenizer, max_seq_len=MAX_SEQ_LEN)
val_dataset = ImageTextDataset(val_data, val_transform, tokenizer, max_seq_len=MAX_SEQ_LEN)

train_loader = make_dataloader(train_dataset, DEVICE_BATCH_SIZE, shuffle=True)

# Optimizer with differential learning rates
backbone_params = []
projection_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if 'projection' in name or 'logit_scale' in name:
        projection_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': LEARNING_RATE * BACKBONE_LR_SCALE, 'weight_decay': WEIGHT_DECAY},
    {'params': projection_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY},
])
# Store initial LRs for scheduling
for group in optimizer.param_groups:
    group['initial_lr'] = group['lr']

print(f"Time budget: {TIME_BUDGET}s")
print(f"Effective batch size: {DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"Gradient accumulation steps: {GRAD_ACCUM_STEPS}")

# LR schedule: linear warmup + cosine decay

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    decay_progress = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
    return 0.5 * (1.0 + math.cos(math.pi * decay_progress))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
data_iter = iter(train_loader)
warmup_steps = 5  # skip first N steps for timing (compilation overhead)

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    optimizer.zero_grad()

    for micro_step in range(GRAD_ACCUM_STEPS):
        try:
            images, input_ids, attention_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, input_ids, attention_mask = next(data_iter)

        images = images.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        with autocast_ctx:
            loss = model(images, input_ids, attention_mask)

        train_loss = loss.detach()
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()

    # LR scheduling
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] * lrm

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

    optimizer.step()

    train_loss_f = train_loss.item()

    # Fast fail
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > warmup_steps:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.4f} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1

    if step > warmup_steps and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

# Final eval
model.eval()
with autocast_ctx:
    results = evaluate_retrieval(model, val_dataset, device=device)

# Final summary
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"mean_recall:      {results['mean_recall']:.2f}")
print(f"i2t_r1:           {results['i2t_r1']:.2f}")
print(f"i2t_r5:           {results['i2t_r5']:.2f}")
print(f"i2t_r10:          {results['i2t_r10']:.2f}")
print(f"t2i_r1:           {results['t2i_r1']:.2f}")
print(f"t2i_r5:           {results['t2i_r5']:.2f}")
print(f"t2i_r10:          {results['t2i_r10']:.2f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {param_counts['total'] / 1e6:.1f}")
print(f"trainable_M:      {param_counts['trainable'] / 1e6:.1f}")
