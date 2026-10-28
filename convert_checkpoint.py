import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig

from plimnet.config import TASKS, read_config
from plimnet.model import PLIMNet
from plimnet.runtime import save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--text-model")
    parser.add_argument("--thresholds")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Checkpoint output already exists")
    model_config, data_config, _ = read_config(args.config)
    if args.text_model:
        model_config.text_model = args.text_model
    text_config = AutoConfig.from_pretrained(model_config.text_model, local_files_only=args.local_files_only)
    network = PLIMNet(model_config, pretrained_images=False, text_config=text_config.to_dict())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    expected = network.state_dict()
    active = {key: value for key, value in state.items() if key in expected}
    network.load_state_dict(active, strict=True)
    recorded_tasks = checkpoint.get("task_names", checkpoint.get("tasks"))
    if recorded_tasks is not None and tuple(recorded_tasks) != TASKS:
        raise ValueError("Checkpoint task order does not match PLIMNet")
    thresholds = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
    if thresholds is not None:
        if set(thresholds) != set(TASKS):
            raise ValueError("Threshold file must contain all marker names")
        for value in thresholds.values():
            if value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 1):
                raise ValueError("Thresholds must lie in [0, 1] or be null")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, network, data_config, checkpoint.get("epoch", 0), thresholds)
    print(f"Converted {len(active)} active model tensors.")


if __name__ == "__main__":
    main()
