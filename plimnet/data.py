from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

from .config import DataConfig, IMAGE_COLUMNS, TASK_COLUMNS, AUX_COLUMNS
from .imaging import normalize_foreground, resize_hw, select_fg_group_indices, crop_modal_slices, augment_images


def normalize_text(value):
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def read_metadata(path, data_root=None, training=False):
    path = Path(path)
    frame = pd.read_csv(path, dtype={"case_id": str, "patient_id": str})
    required = {"case_id", *IMAGE_COLUMNS}
    if training:
        required.update(["patient_id", *TASK_COLUMNS, *AUX_COLUMNS])
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if frame.empty or frame.case_id.isna().any() or frame.case_id.duplicated().any():
        raise ValueError("Metadata must contain nonempty, unique case identifiers")
    if frame.case_id.str.strip().eq("").any():
        raise ValueError("Empty case identifier")
    if training and (frame.patient_id.isna().any() or frame.patient_id.str.strip().eq("").any()):
        raise ValueError("Training requires an identifier for every patient")
    root = Path(data_root) if data_root else path.parent
    for column in IMAGE_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"Missing image path in {column}")
        frame[column] = frame[column].map(lambda value: str(Path(value) if Path(value).is_absolute() else root / value))
        if not frame[column].map(lambda value: Path(value).is_file()).all():
            raise FileNotFoundError(f"One or more files in {column} do not exist")
    for column in (*TASK_COLUMNS, *AUX_COLUMNS):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].dropna().isin([0, 1]).all():
            raise ValueError(f"{column} must contain 0, 1 or an empty value")
    if "report_text" not in frame:
        findings = frame.get("findings", pd.Series("", index=frame.index)).map(normalize_text)
        impressions = frame.get("impression", pd.Series("", index=frame.index)).map(normalize_text)
        frame["report_text"] = [" ".join(part for part in (
            f"Findings: {finding}" if finding else "", f"Impression: {impression}" if impression else ""))
            for finding, impression in zip(findings, impressions)]
    frame["report_text"] = frame.report_text.map(normalize_text)
    return frame.reset_index(drop=True)


def load_volumes(row, image_size):
    reference = sitk.ReadImage(row["t1c"])
    images = {"t1c": reference, **{key: sitk.ReadImage(row[key]) for key in ("t1", "t2", "tumor_mask")}}
    for image in images.values():
        if image.GetDimension() != 3 or image.GetSize() != reference.GetSize():
            raise ValueError("All inputs must be three-dimensional and share a voxel grid")
        for field in ("GetSpacing", "GetOrigin", "GetDirection"):
            if not np.allclose(getattr(image, field)(), getattr(reference, field)(), rtol=0, atol=1e-4):
                raise ValueError("Input images and mask must be spatially aligned")
    target = (image_size, image_size)
    mask = sitk.GetArrayFromImage(images["tumor_mask"])
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("tumor_mask must be a binary mask")
    mask = resize_hw(mask.astype(np.uint8), target, order=0)
    arrays = {}
    for key in ("t1c", "t1", "t2"):
        array = sitk.GetArrayFromImage(images[key]).astype(np.float32)
        if not np.isfinite(array).all():
            raise ValueError("Image values must be finite")
        arrays[key] = resize_hw(normalize_foreground(array), target, order=1)
    return arrays, mask


class MRIDataset(Dataset):
    def __init__(self, frame, config=None, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.config = config or DataConfig()
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        cfg = self.config
        arrays, mask = load_volumes(row, cfg.image_size)
        _, _, _, slices = select_fg_group_indices(mask, train=self.augment, total_target_slices=5)
        cropped, _, _ = crop_modal_slices(arrays, mask, slices, (cfg.image_size, cfg.image_size),
                                         cfg.box_scale, cfg.min_box_size)
        images = np.stack([cropped[key] for key in ("t1c", "t1", "t2")], axis=1).astype(np.float32)
        if self.augment:
            images = augment_images(images)
        targets = row[list(TASK_COLUMNS)].to_numpy(dtype=np.float32)
        lineage = row[list(AUX_COLUMNS)].to_numpy(dtype=np.float32)
        return {"image": torch.from_numpy(images.copy()), "case_id": row.case_id,
                "report_text": normalize_text(row.report_text),
                "task_labels": torch.from_numpy(np.nan_to_num(targets)),
                "task_mask": torch.from_numpy(np.isfinite(targets).astype(np.float32)),
                "lineage_labels": torch.from_numpy(np.nan_to_num(lineage)),
                "lineage_mask": torch.from_numpy(np.isfinite(lineage).astype(np.float32))}
