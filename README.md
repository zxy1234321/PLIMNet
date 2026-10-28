# PLIMNet

PyTorch code for the Pituitary Lineage-Informed Multimodal Network (PLIMNet),
which estimates the expression of nine immunohistochemical markers (PIT-1,
T-PIT, SF-1, GH, PRL, TSH, ACTH, LH and FSH) in pituitary neuroendocrine tumors
from preoperative MRI and radiology reports.

Patient data and trained models are not included.

## Installation

Use Python 3.10.

```bash
python -m pip install -r requirements.txt
```

## Input

A CSV file with one row per examination, containing `case_id`, the registered
coronal `t1c`, `t1` and `t2` NIfTI volumes, the `tumor_mask` on the same grid
and the `report_text`. Training additionally requires `patient_id`, the nine
marker labels (`gh_label`, `tsh_label`, `acth_label`, `lh_label`, `prl_label`,
`fsh_label`, `t_pit_label`, `pit_1_label`, `sf_1_label`) and the three
auxiliary lineage labels (`aux_pit1_label`, `aux_tpit_label`, `aux_sf1_label`).
Unmeasured labels are left empty.

## Usage

Uppercase names are placeholders.

```bash
python make_splits.py --metadata TRAIN_CSV --output SPLITS_JSON
python train.py --metadata TRAIN_CSV --splits SPLITS_JSON --output OUTPUT_DIR
python predict.py --metadata TEST_CSV --checkpoints CHECKPOINT [CHECKPOINT ...] --output PREDICTIONS_CSV
```

When several checkpoints are supplied, `predict.py` averages their predicted
probabilities.
