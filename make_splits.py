import argparse
from pathlib import Path

from plimnet.data import read_metadata
from plimnet.runtime import write_json
from plimnet.splits import make_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("Split output already exists")
    if args.folds < 2:
        raise ValueError("At least two folds are required")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = read_metadata(args.metadata, args.data_root, training=True)
    folds = make_splits(frame, args.folds, args.val_size, args.seed)
    write_json(target, {"seed": args.seed, "val_size": args.val_size, "folds": folds})


if __name__ == "__main__":
    main()
