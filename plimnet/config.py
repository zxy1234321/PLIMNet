from dataclasses import dataclass
import json
from pathlib import Path

TASKS = ("gh", "tsh", "acth", "lh", "prl", "fsh", "t_pit", "pit_1", "sf_1")
LINEAGES = ("pit1", "tpit", "sf1")
LINEAGE_TASKS = {
    "pit1": ("gh", "prl", "tsh", "pit_1"),
    "tpit": ("acth", "t_pit"),
    "sf1": ("fsh", "sf_1"),
}
IMAGE_COLUMNS = ("t1c", "t1", "t2", "tumor_mask")
TASK_COLUMNS = tuple(f"{task}_label" for task in TASKS)
AUX_COLUMNS = tuple(f"aux_{lineage}_label" for lineage in LINEAGES)


@dataclass
class ModelConfig:
    text_model: str = "dmis-lab/biobert-base-cased-v1.2"
    hidden_dim: int = 256
    dropout: float = 0.5
    attention_heads: int = 4
    attention_dropout: float = 0.3
    text_unfreeze_layers: int = 2

    def __post_init__(self):
        if self.hidden_dim <= 0 or self.attention_heads <= 0 or self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be positive and divisible by attention_heads")
        if not 0 <= self.dropout < 1 or not 0 <= self.attention_dropout < 1:
            raise ValueError("Dropout probabilities must lie in [0, 1)")
        if self.text_unfreeze_layers < 0:
            raise ValueError("text_unfreeze_layers must be nonnegative")


@dataclass
class DataConfig:
    image_size: int = 224
    box_scale: float = 1.75
    min_box_size: int = 48
    max_text_length: int = 256

    def __post_init__(self):
        if min(self.image_size, self.min_box_size, self.max_text_length) <= 0 or self.box_scale <= 0:
            raise ValueError("Image, crop and text dimensions must be positive")


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 5e-5
    text_learning_rate: float = 5e-6
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.2
    min_learning_rate: float = 1e-6
    lineage_loss_weight: float = 0.4
    seed: int = 42
    amp: bool = True

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("Invalid epochs, batch size or worker count")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be smaller than epochs")
        if min(self.learning_rate, self.text_learning_rate, self.min_learning_rate) <= 0:
            raise ValueError("Learning rates must be positive")
        if self.weight_decay < 0 or self.lineage_loss_weight < 0:
            raise ValueError("Loss weight and weight decay must be nonnegative")
        if not 0 < self.warmup_start_factor <= 1:
            raise ValueError("warmup_start_factor must lie in (0, 1]")


def read_config(path=None):
    values = json.loads(Path(path).read_text()) if path else {}
    if set(values) - {"model", "data", "training"}:
        raise ValueError("Unknown configuration section")
    return (ModelConfig(**values.get("model", {})), DataConfig(**values.get("data", {})),
            TrainingConfig(**values.get("training", {})))
