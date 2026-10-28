import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from plimnet.config import TASKS
from plimnet.data import read_metadata
from plimnet.runtime import load_checkpoint, make_loader, predict_probabilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-model")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid batch size or worker count")
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("Prediction output already exists")
    checkpoints = [Path(path).resolve() for path in args.checkpoints]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Checkpoint list contains duplicates")
    frame = read_metadata(args.metadata, args.data_root)
    device = torch.device(args.device)
    arrays, cutoffs, input_config, model_config = [], [], None, None
    for path in checkpoints:
        network, data_config, thresholds = load_checkpoint(path)
        current = asdict(data_config)
        current_model = asdict(network.config)
        if input_config is not None and (input_config != current or model_config != current_model):
            raise ValueError("Ensemble checkpoints must share model and input settings")
        input_config, model_config = current, current_model
        tokenizer_path = path.parent.parent / "tokenizer"
        text_model = args.text_model or (str(tokenizer_path) if tokenizer_path.is_dir() else network.config.text_model)
        tokenizer = AutoTokenizer.from_pretrained(text_model, local_files_only=args.local_files_only)
        loader = make_loader(frame, data_config, args.batch_size, args.workers)
        network.to(device)
        ids, probabilities = predict_probabilities(network, loader, tokenizer, device,
                                                   data_config.max_text_length, not args.no_amp)
        if ids != frame.case_id.tolist():
            raise RuntimeError("Prediction order mismatch")
        arrays.append(probabilities)
        cutoffs.append(thresholds or {})
        del network, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    means = np.stack(arrays).mean(0, dtype=np.float64)
    result = pd.DataFrame({"case_id": frame.case_id})
    for index, task in enumerate(TASKS):
        result[f"{task}_prob"] = means[:, index]
        values = [entry.get(task) for entry in cutoffs]
        if all(value is not None and np.isfinite(value) for value in values):
            threshold = float(np.mean(values))
            result[f"{task}_threshold"] = threshold
            result[f"{task}_pred"] = (means[:, index] >= threshold).astype(int)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    print(f"Predicted {len(result)} examinations using {len(checkpoints)} checkpoint(s).")


if __name__ == "__main__":
    main()
