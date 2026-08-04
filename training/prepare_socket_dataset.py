from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"

REPO_ID = "Blablablab/SOCKET"
DATASET_DIR = "SOCKET_DATA/hahackathon#is_humor"

FILES = {
    "train": {
        "text": f"{DATASET_DIR}/train_text.txt",
        "labels": f"{DATASET_DIR}/train_labels.txt",
    },
    "validation": {
        "text": f"{DATASET_DIR}/val_text.txt",
        "labels": f"{DATASET_DIR}/val_labels.txt",
    },
    "test": {
        "text": f"{DATASET_DIR}/test_text.txt",
        "labels": f"{DATASET_DIR}/test_labels.txt",
    },
}


def download_file(filename: str) -> Path:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )
    return Path(path)


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]


def normalize_label(value: str) -> int:
    """
    Convert SocKET's humor labels to:
        0 = not humor
        1 = humor

    Supports both textual and numeric representations.
    """

    value = value.strip().lower()

    non_humor_values = {
        "0",
        "not humor",
        "not_humor",
        "not humorous",
        "non-humor",
        "non_humor",
    }

    humor_values = {
        "1",
        "humor",
        "humorous",
    }

    if value in non_humor_values:
        return 0

    if value in humor_values:
        return 1

    raise ValueError(f"Unknown humor label: {value!r}")


def load_split(split_name: str) -> pd.DataFrame:
    info = FILES[split_name]

    print(f"\nDownloading {split_name} split...")

    text_path = download_file(info["text"])
    label_path = download_file(info["labels"])

    texts = read_lines(text_path)
    raw_labels = read_lines(label_path)

    if len(texts) != len(raw_labels):
        raise RuntimeError(
            f"{split_name}: text/label count mismatch: "
            f"{len(texts)} texts vs {len(raw_labels)} labels"
        )

    labels = [normalize_label(label) for label in raw_labels]

    frame = pd.DataFrame(
        {
            "text": texts,
            "label": labels,
        }
    )

    # Keep the official split intact.
    # Only remove rows that are completely empty.
    empty_mask = frame["text"].str.strip().eq("")

    if empty_mask.any():
        print(
            f"Warning: removing {int(empty_mask.sum())} completely empty "
            f"rows from {split_name}."
        )
        frame = frame.loc[~empty_mask].copy()

    frame.reset_index(drop=True, inplace=True)

    return frame


def print_split_summary(name: str, frame: pd.DataFrame) -> None:
    counts = frame["label"].value_counts().sort_index()

    not_humor = int(counts.get(0, 0))
    humor = int(counts.get(1, 0))
    total = len(frame)

    print(f"{name}: {total:,} examples")
    print(f"  Not humor : {not_humor:,}")
    print(f"  Humor     : {humor:,}")

    if total:
        print(f"  Not humor : {not_humor / total:.2%}")
        print(f"  Humor     : {humor / total:.2%}")


def check_split_overlap(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Report exact text overlap without changing the official dataset splits.
    """

    train_text = set(train["text"].str.strip())
    val_text = set(validation["text"].str.strip())
    test_text = set(test["text"].str.strip())

    train_val = train_text & val_text
    train_test = train_text & test_text
    val_test = val_text & test_text

    print("\nExact text overlap check")
    print("------------------------")
    print(f"Train <-> Validation : {len(train_val):,}")
    print(f"Train <-> Test       : {len(train_test):,}")
    print(f"Validation <-> Test  : {len(val_test):,}")

    if train_test:
        print(
            "\nWARNING: exact examples occur in both train and test. "
            "Do not silently remove them; record this in the dataset review "
            "before final evaluation."
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Humor Bot - SocKET Dataset Preparation")
    print("=" * 60)

    print("\nDataset:")
    print("  Blablablab/SOCKET")
    print("Task:")
    print("  hahackathon#is_humor")
    print("\nLabel mapping:")
    print("  0 = Not Humor")
    print("  1 = Humor")

    train = load_split("train")
    validation = load_split("validation")
    test = load_split("test")

    print("\nDataset summary")
    print("---------------")

    print_split_summary("Train", train)
    print_split_summary("Validation", validation)
    print_split_summary("Test", test)

    check_split_overlap(train, validation, test)

    outputs = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    print("\nWriting processed files")
    print("-----------------------")

    for name, frame in outputs.items():
        output_path = OUT_DIR / f"{name}.csv"

        frame.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
        )

        print(f"{name:10} -> {output_path}")

    print("\n" + "=" * 60)
    print("Dataset preparation complete.")
    print("=" * 60)

    print(
        "\nIMPORTANT:"
        "\n  train.csv      -> model training"
        "\n  validation.csv -> model selection / threshold calibration"
        "\n  test.csv       -> final in-domain evaluation only"
    )

    print(
        "\nDo not use test.csv to tune the model or choose thresholds."
    )


if __name__ == "__main__":
    main()