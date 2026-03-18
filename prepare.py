"""
One-time data preparation for multimodal embedding model research.
Downloads image-text datasets and provides data loading and evaluation utilities.

Usage:
    python prepare.py                                  # full prep (default dataset)
    python prepare.py --dataset nlphuji/flickr30k      # specific dataset

Data is stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import random
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 77          # max text token length (CLIP standard)
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
IMAGE_SIZE = 224           # input image resolution
EVAL_BATCH_SIZE = 64       # batch size for evaluation

# Image normalization (CLIP/OpenAI standard)
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# Default dataset — agent should NOT change this
DEFAULT_DATASET = "nlphuji/flickr30k"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data(dataset_name=DEFAULT_DATASET):
    """Download multimodal dataset from HuggingFace."""
    from datasets import load_dataset

    dataset_cache = os.path.join(DATA_DIR, dataset_name.replace("/", "__"))
    marker = os.path.join(dataset_cache, ".download_complete")

    if os.path.exists(marker):
        print(f"Data: already downloaded at {dataset_cache}")
        return

    os.makedirs(dataset_cache, exist_ok=True)
    print(f"Data: downloading {dataset_name}...")
    t0 = time.time()

    load_dataset(dataset_name, cache_dir=dataset_cache)

    with open(marker, "w") as f:
        f.write(f"downloaded at {time.time()}\n")

    t1 = time.time()
    print(f"Data: downloaded in {t1 - t0:.1f}s to {dataset_cache}")


def load_splits(dataset_name=DEFAULT_DATASET):
    """
    Load train and validation splits from a HuggingFace dataset.
    Handles datasets with explicit splits or a 'split' column (e.g. Flickr30k).
    Returns (train_dataset, val_dataset).
    """
    from datasets import load_dataset

    dataset_cache = os.path.join(DATA_DIR, dataset_name.replace("/", "__"))
    ds = load_dataset(dataset_name, cache_dir=dataset_cache)

    # Case 1: DatasetDict with standard split names
    if hasattr(ds, 'keys'):
        keys = list(ds.keys())
        if 'train' in keys and ('validation' in keys or 'val' in keys):
            val_key = 'validation' if 'validation' in keys else 'val'
            return ds['train'], ds[val_key]
        if 'train' in keys and 'test' in keys:
            return ds['train'], ds['test']
        # Single split — try to find a 'split' column
        split_name = keys[0]
        all_data = ds[split_name]
    else:
        all_data = ds

    # Case 2: Dataset with a 'split' column (e.g. nlphuji/flickr30k)
    if 'split' in all_data.column_names:
        train_data = all_data.filter(lambda x: x['split'] == 'train')
        # Try val first, then test
        val_data = all_data.filter(lambda x: x['split'] == 'val')
        if len(val_data) == 0:
            val_data = all_data.filter(lambda x: x['split'] == 'test')
        if len(val_data) == 0:
            # No separate val/test — use last 10% of train
            split = train_data.train_test_split(test_size=0.1, seed=42)
            return split['train'], split['test']
        return train_data, val_data

    # Case 3: No splits — split manually
    split = all_data.train_test_split(test_size=0.1, seed=42)
    return split['train'], split['test']

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def get_image_transform(image_size=IMAGE_SIZE, is_train=True):
    """Get image preprocessing transform pipeline."""
    if is_train:
        return transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])

# ---------------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------------

def _detect_columns(hf_dataset):
    """Auto-detect image and caption column names."""
    cols = hf_dataset.column_names
    image_col = None
    caption_col = None
    for c in ['image', 'img', 'pixel_values']:
        if c in cols:
            image_col = c
            break
    for c in ['caption', 'text', 'sentence', 'captions', 'sentences']:
        if c in cols:
            caption_col = c
            break
    if image_col is None or caption_col is None:
        raise ValueError(f"Cannot detect image/caption columns from: {cols}")
    return image_col, caption_col


class ImageTextDataset(torch.utils.data.Dataset):
    """Dataset for image-text pairs from HuggingFace datasets."""

    def __init__(self, hf_dataset, image_transform, tokenizer, max_seq_len=MAX_SEQ_LEN,
                 image_col=None, caption_col=None):
        self.dataset = hf_dataset
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if image_col is None or caption_col is None:
            image_col, caption_col = _detect_columns(hf_dataset)
        self.image_col = image_col
        self.caption_col = caption_col

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process image
        image = item[self.image_col]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.image_transform(image)

        # Process text
        caption = item[self.caption_col]
        if isinstance(caption, list):
            caption = random.choice(caption)

        tokens = self.tokenizer(
            caption, max_length=self.max_seq_len,
            padding='max_length', truncation=True,
            return_tensors='pt'
        )

        return image, tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)


def make_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """Create a PyTorch DataLoader for image-text pairs."""
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_retrieval(model, val_dataset, batch_size=EVAL_BATCH_SIZE, device='cuda'):
    """
    Image-text retrieval evaluation — the fixed evaluation metric.

    Computes:
    - Image-to-text retrieval: Recall@1, @5, @10
    - Text-to-image retrieval: Recall@1, @5, @10
    - Mean Recall: average of all 6 recall values (the primary metric)

    Higher is better.

    The model must expose encode_image(pixel_values) and
    encode_text(input_ids, attention_mask), both returning
    [B, D] tensors (normalization is applied here).
    """
    model.eval()

    val_loader = make_dataloader(val_dataset, batch_size, shuffle=False, num_workers=2)

    all_image_embeds = []
    all_text_embeds = []

    for images, input_ids, attention_mask in val_loader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            image_embeds = model.encode_image(images)
            text_embeds = model.encode_text(input_ids, attention_mask)

        image_embeds = F.normalize(image_embeds.float(), dim=-1)
        text_embeds = F.normalize(text_embeds.float(), dim=-1)

        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())

    image_embeds = torch.cat(all_image_embeds, dim=0)
    text_embeds = torch.cat(all_text_embeds, dim=0)

    N = image_embeds.size(0)

    # Compute retrieval in chunks to avoid OOM on large val sets
    chunk_size = 256
    i2t_recall = {1: 0, 5: 0, 10: 0}
    t2i_recall = {1: 0, 5: 0, 10: 0}

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        targets = torch.arange(start, end).unsqueeze(1)

        # Image-to-text
        sim_i2t = image_embeds[start:end] @ text_embeds.T
        for k in [1, 5, 10]:
            topk = sim_i2t.topk(k, dim=1).indices
            i2t_recall[k] += (topk == targets).any(dim=1).sum().item()

        # Text-to-image
        sim_t2i = text_embeds[start:end] @ image_embeds.T
        for k in [1, 5, 10]:
            topk = sim_t2i.topk(k, dim=1).indices
            t2i_recall[k] += (topk == targets).any(dim=1).sum().item()

    results = {}
    for k in [1, 5, 10]:
        results[f'i2t_r{k}'] = 100.0 * i2t_recall[k] / N
        results[f't2i_r{k}'] = 100.0 * t2i_recall[k] / N
    results['mean_recall'] = sum(results.values()) / 6.0

    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for multimodal embedding research")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="HuggingFace dataset name (default: %(default)s)")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Download data
    download_data(args.dataset)
    print()
    print("Done! Ready to train.")
