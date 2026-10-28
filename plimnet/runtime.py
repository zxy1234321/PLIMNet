from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from .config import ModelConfig, DataConfig, TASKS
from .data import MRIDataset
from .model import PLIMNet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(frame, data_config, batch_size, workers, augment=False, seed=42):
    return DataLoader(MRIDataset(frame, data_config, augment), batch_size=batch_size, shuffle=augment,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                      worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(seed))


def tokenize(tokenizer, texts, device, max_length):
    encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {key: value.to(device, non_blocking=True) for key, value in encoded.items()}


@torch.inference_mode()
def predict_probabilities(network, loader, tokenizer, device, max_length, amp=True):
    network.eval()
    ids, probabilities = [], []
    for batch in loader:
        text = tokenize(tokenizer, batch["report_text"], device, max_length)
        with autocast(device.type, enabled=amp and device.type == "cuda"):
            _, logits = network(batch["image"].to(device, non_blocking=True), text)
        probabilities.append(logits.float().sigmoid().cpu().numpy())
        ids.extend(batch["case_id"])
    if not probabilities:
        raise ValueError("Cannot predict an empty dataset")
    values = np.concatenate(probabilities)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictions")
    return ids, values


def write_json(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, network, data_config, epoch, thresholds=None):
    text_config = network.text_encoder.config.to_dict()
    text_config.pop("_name_or_path", None)
    state = {"model_state_dict": {key: value.detach().cpu() for key, value in network.state_dict().items()},
             "model_config": asdict(network.config), "data_config": asdict(data_config),
             "text_config": text_config, "tasks": list(TASKS), "epoch": int(epoch), "thresholds": thresholds}
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(state.get("tasks", [])) != TASKS:
        raise ValueError("Checkpoint task order does not match PLIMNet")
    config = ModelConfig(**state["model_config"])
    data_config = DataConfig(**state["data_config"])
    network = PLIMNet(config, pretrained_images=False, text_config=state["text_config"])
    network.load_state_dict(state["model_state_dict"], strict=True)
    return network, data_config, state.get("thresholds")
