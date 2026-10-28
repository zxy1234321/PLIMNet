import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def validate_splits(frame, folds):
    indexed = frame.set_index("case_id")
    all_ids = set(indexed.index)
    seen_test = set()
    numbers = []
    for fold in folds:
        numbers.append(fold["fold"])
        role_ids, role_groups = [], []
        for role in ("train", "val", "test"):
            ids = fold[f"{role}_case_ids"]
            if not ids or len(set(ids)) != len(ids) or set(ids) - all_ids:
                raise ValueError(f"Invalid {role} identifiers in fold {fold['fold']}")
            role_ids.append(set(ids))
            role_groups.append(set(indexed.loc[ids, "patient_id"]))
        if set.union(*role_ids) != all_ids:
            raise ValueError("Each fold must assign every examination to exactly one role")
        for left, right in ((0, 1), (0, 2), (1, 2)):
            if role_ids[left] & role_ids[right] or role_groups[left] & role_groups[right]:
                raise ValueError("Patient or examination overlap between split roles")
        if seen_test & role_ids[2]:
            raise ValueError("Repeated outer-test examination")
        seen_test.update(role_ids[2])
    if len(numbers) != len(set(numbers)) or seen_test != all_ids:
        raise ValueError("Outer test folds must cover every examination exactly once")


def sample_patients(patients, labels, size, seed):
    classes, counts = np.unique(labels, return_counts=True)
    quotas = counts * size / counts.sum()
    take = np.floor(quotas).astype(int)
    take[np.argsort(take - quotas)[:size - take.sum()]] += 1
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(patients[labels == value])[:amount]
                           for value, amount in zip(classes, take)])


def make_splits(frame, count=5, val_size=200, seed=42):
    labels = frame.gh_label.fillna(2).astype(int).to_numpy()
    groups = frame.patient_id.to_numpy()
    outer = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    folds = []
    for number, (development, test) in enumerate(outer.split(frame, labels, groups), 1):
        patients, first = np.unique(groups[development], return_index=True)
        if not 0 < val_size < len(patients):
            raise ValueError("Inner validation must be smaller than the development cohort")
        held_out = sample_patients(patients, labels[development][first], val_size, seed + number)
        in_val = np.isin(groups[development], held_out)
        folds.append({"fold": number, "train_case_ids": frame.iloc[development[~in_val]].case_id.tolist(),
                      "val_case_ids": frame.iloc[development[in_val]].case_id.tolist(),
                      "test_case_ids": frame.iloc[test].case_id.tolist()})
    validate_splits(frame, folds)
    return folds


def read_splits(path, frame):
    value = json.loads(Path(path).read_text())
    folds = value["folds"]
    validate_splits(frame, folds)
    return folds
