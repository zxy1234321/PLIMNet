import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer

from plimnet.config import TASKS, TASK_COLUMNS, AUX_COLUMNS, read_config
from plimnet.data import read_metadata
from plimnet.losses import positive_weights, joint_loss
from plimnet.model import PLIMNet
from plimnet.runtime import set_seed, make_loader, tokenize, predict_probabilities, save_checkpoint, load_checkpoint, write_json
from plimnet.splits import read_splits


def selection_score(labels, probabilities):
    scores = []
    for index in range(len(TASKS)):
        known = np.isfinite(labels[:, index])
        truth, values = labels[known, index], probabilities[known, index]
        scores.append((float(roc_auc_score(truth, values)),
                       float(balanced_accuracy_score(truth, values >= .5)))
                      if len(np.unique(truth)) == 2 else (np.nan, np.nan))
    scores = np.asarray(scores)
    if not np.isfinite(scores[:, 0]).any():
        raise ValueError("Validation requires both classes for at least one target")
    macro = np.nanmean(scores, axis=0)
    focus = np.nan_to_num(scores[0], nan=-1.0)
    return tuple(float(value) for value in [*macro, *focus])


def validation_thresholds(labels, probabilities):
    thresholds = {}
    for index, task in enumerate(TASKS):
        known = np.isfinite(labels[:, index])
        truth, values = labels[known, index], probabilities[known, index]
        if len(np.unique(truth)) != 2:
            thresholds[task] = None
            continue
        fpr, tpr, cuts = roc_curve(truth, values)
        finite = np.isfinite(cuts)
        thresholds[task] = float(cuts[finite][np.argmax((tpr - fpr)[finite])])
    return thresholds


def fit_fold(frame, fold, model_config, data_config, training, args):
    set_seed(training.seed)
    device = torch.device(args.device)
    folder = Path(args.output) / f"fold_{fold['fold']:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    indexed = frame.set_index("case_id", drop=False)
    fit = indexed.loc[fold["train_case_ids"]].reset_index(drop=True)
    val = indexed.loc[fold["val_case_ids"]].reset_index(drop=True)
    if not fit[list(AUX_COLUMNS)].notna().any().any():
        raise ValueError("Training requires measured auxiliary lineage labels")
    if not fit[list(TASK_COLUMNS)].notna().any().any():
        raise ValueError("Training requires measured marker labels")
    train_loader = make_loader(fit, data_config, training.batch_size, training.num_workers, True, training.seed)
    val_loader = make_loader(val, data_config, training.batch_size, training.num_workers, seed=training.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_config.text_model, local_files_only=args.local_files_only)
    network = PLIMNet(model_config, pretrained_images=not args.no_pretrained_images,
                     local_files_only=args.local_files_only).to(device)
    if data_config.max_text_length > network.text_encoder.config.max_position_embeddings:
        raise ValueError("Text length exceeds the encoder's position limit")
    tokenizer.save_pretrained(Path(args.output) / "tokenizer")
    main_weight = positive_weights(fit, TASK_COLUMNS).to(device)
    auxiliary_weight = positive_weights(fit, AUX_COLUMNS).to(device)
    groups = [{"params": [value for name, value in network.named_parameters()
                          if value.requires_grad and not name.startswith("text_encoder.")],
               "lr": training.learning_rate},
              {"params": [value for name, value in network.named_parameters()
                          if value.requires_grad and name.startswith("text_encoder.")],
               "lr": training.text_learning_rate}]
    optimizer = torch.optim.AdamW([group for group in groups if group["params"]], weight_decay=training.weight_decay)
    cosine = CosineAnnealingLR(optimizer, T_max=training.epochs - training.warmup_epochs,
                               eta_min=training.min_learning_rate)
    scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=training.warmup_start_factor,
                                                 total_iters=training.warmup_epochs), cosine],
                             milestones=[training.warmup_epochs]) if training.warmup_epochs else cosine
    amp = training.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp)
    write_json(folder / "config.json", {"model": asdict(model_config), "data": asdict(data_config),
                                        "training": asdict(training)})
    write_json(folder / "split.json", fold)
    write_json(folder / "loss_weights.json", {"markers": main_weight.tolist(), "lineages": auxiliary_weight.tolist()})
    best, best_epoch, history = (-float("inf"),) * 4, 0, []
    for epoch in range(1, training.epochs + 1):
        network.train()
        losses, updates = [], 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = tokenize(tokenizer, batch["report_text"], device, data_config.max_text_length)
            with autocast(device.type, enabled=amp):
                lineage, logits = network(batch["image"].to(device), inputs)
                loss = joint_loss(lineage, logits, batch, main_weight, auxiliary_weight, training.lineage_loss_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updates += int(scaler.get_scale() >= scale)
            losses.append(float(loss.detach()))
        if not updates:
            raise FloatingPointError("No optimizer update completed in this epoch")
        ids, probabilities = predict_probabilities(network, val_loader, tokenizer, device, data_config.max_text_length, amp)
        if ids != val.case_id.tolist():
            raise RuntimeError("Validation prediction order mismatch")
        score = selection_score(val[list(TASK_COLUMNS)].to_numpy(dtype=float), probabilities)
        if score > best:
            best, best_epoch = score, epoch
            save_checkpoint(folder / "best.pt", network, data_config, epoch)
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "validation_auc": score[0],
                  "validation_bacc": score[1], "learning_rate": optimizer.param_groups[0]["lr"],
                  "best_epoch": best_epoch, "updates": updates}
        history.append(record)
        write_json(folder / "history.json", history)
        print(json.dumps({"fold": fold["fold"], **record}), flush=True)
        scheduler.step()
    del optimizer, scaler, network
    if device.type == "cuda":
        torch.cuda.empty_cache()
    network, _, _ = load_checkpoint(folder / "best.pt")
    network.to(device)
    ids, probabilities = predict_probabilities(network, val_loader, tokenizer, device, data_config.max_text_length, amp)
    if ids != val.case_id.tolist():
        raise RuntimeError("Validation prediction order mismatch")
    thresholds = validation_thresholds(val[list(TASK_COLUMNS)].to_numpy(dtype=float), probabilities)
    save_checkpoint(folder / "best.pt", network, data_config, best_epoch, thresholds)
    write_json(folder / "thresholds.json", thresholds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--text-model")
    parser.add_argument("--fold", type=int, nargs="+")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-pretrained-images", action="store_true")
    args = parser.parse_args()
    model_config, data_config, training = read_config(args.config)
    if args.text_model:
        model_config.text_model = args.text_model
    frame = read_metadata(args.metadata, args.data_root, training=True)
    folds = read_splits(args.splits, frame)
    selected = args.fold or [fold["fold"] for fold in folds]
    if set(selected) - {fold["fold"] for fold in folds}:
        raise ValueError("Requested fold is absent from the split file")
    for fold in folds:
        if fold["fold"] in selected:
            fit_fold(frame, fold, model_config, data_config, training, args)


if __name__ == "__main__":
    main()
