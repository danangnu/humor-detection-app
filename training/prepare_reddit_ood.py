from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OOD_DIR = ROOT / "data" / "ood"
PROCESSED_DIR = ROOT / "data" / "processed"

DATASET_ID = "atin5551/reddit-story-niche-classification-dataset"
SEED = 20260804

NEGATIVE_NICHES = {
    "advice",
    "story",
    "drama",
    "rant",
    "informative",
    "confession",
    "update",
    "wholesome",
}


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def load_seen_text() -> set[str]:
    seen: set[str] = set()
    for name in ("train.csv", "validation.csv", "test.csv"):
        path = PROCESSED_DIR / name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "text" in frame.columns:
            seen.update(frame["text"].astype(str).map(normalize_text))
    return seen


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_all_source_splits() -> pd.DataFrame:
    dataset = load_dataset(DATASET_ID)
    frames = []

    for split_name, split in dataset.items():
        frame = split.to_pandas().copy()
        frame["source_split"] = split_name
        frames.append(frame)

    if not frames:
        raise RuntimeError("The Reddit dataset did not contain any usable splits.")

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--per-class",
        type=int,
        default=200,
        help=(
            "Requested number per class. If fewer humor examples are available, "
            "all available humor examples are used and the non-humor class is matched."
        ),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    OOD_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("Humor Bot - Reddit OOD Set Preparation v2")
    print("=" * 68)
    print(f"Dataset: {DATASET_ID}")
    print(f"Random seed: {args.seed}")
    print(f"Requested per class: {args.per_class}")
    print()

    frame = load_all_source_splits()

    required = {"title", "niche"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    print("Source splits")
    print("-------------")
    for split_name, count in frame["source_split"].value_counts().sort_index().items():
        print(f"{split_name:12} {count:,}")

    frame["text"] = frame["title"].map(normalize_text)
    frame["niche"] = frame["niche"].astype(str).str.strip().str.lower()

    frame = frame[frame["text"].str.len().between(20, 500)].copy()
    frame = frame.drop_duplicates(subset=["text"]).reset_index(drop=True)

    seen = load_seen_text()
    before_overlap = len(frame)
    frame = frame[~frame["text"].isin(seen)].copy()
    removed_overlap = before_overlap - len(frame)

    positive = frame[frame["niche"].eq("humor")].copy()
    negative = frame[frame["niche"].isin(NEGATIVE_NICHES)].copy()

    print()
    print("Available after filtering")
    print("-------------------------")
    print(f"Humor rows:     {len(positive):,}")
    print(f"Non-humor rows: {len(negative):,}")
    print(f"Removed exact overlap with model dataset: {removed_overlap:,}")

    if len(positive) == 0:
        raise RuntimeError("No humor examples remain after filtering.")
    if len(negative) == 0:
        raise RuntimeError("No non-humor examples remain after filtering.")

    actual_per_class = min(args.per_class, len(positive), len(negative))

    if actual_per_class < args.per_class:
        print()
        print("NOTE: Requested size exceeds the available humor class.")
        print(
            f"Using all available humor examples and a matched non-humor sample: "
            f"{actual_per_class} per class."
        )

    positive = positive.sample(n=actual_per_class, random_state=args.seed)
    negative = negative.sample(n=actual_per_class, random_state=args.seed + 1)

    positive["source_label"] = 1
    negative["source_label"] = 0

    combined = pd.concat([positive, negative], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=args.seed + 2).reset_index(drop=True)
    combined.insert(
        0,
        "ood_id",
        [f"R{i:04d}" for i in range(1, len(combined) + 1)],
    )

    provenance_columns = [
        c
        for c in [
            "ood_id",
            "id",
            "url",
            "subreddit",
            "source_split",
            "niche",
            "text",
            "source_label",
        ]
        if c in combined.columns
    ]

    provisional = combined[provenance_columns].copy()
    provisional_path = OOD_DIR / "reddit_ood_provisional.csv"
    provisional.to_csv(provisional_path, index=False, encoding="utf-8")

    annotation = combined[["ood_id", "text"]].copy()
    annotation["annotator_1"] = ""
    annotation["annotator_2"] = ""
    annotation["adjudicated_label"] = ""
    annotation["notes"] = ""

    annotation_path = OOD_DIR / "reddit_ood_annotation.csv"
    annotation.to_csv(annotation_path, index=False, encoding="utf-8")

    key_columns = [
        c
        for c in [
            "ood_id",
            "id",
            "url",
            "subreddit",
            "source_split",
            "niche",
            "source_label",
        ]
        if c in combined.columns
    ]

    key = combined[key_columns].copy()
    key_path = OOD_DIR / "reddit_ood_key.csv"
    key.to_csv(key_path, index=False, encoding="utf-8")

    source_split_counts = (
        combined["source_split"].value_counts().sort_index().to_dict()
        if "source_split" in combined.columns
        else {}
    )

    manifest = {
        "dataset_id": DATASET_ID,
        "seed": args.seed,
        "requested_per_class": args.per_class,
        "actual_per_class": int(actual_per_class),
        "total_examples": int(len(combined)),
        "source_humor_examples": int((combined["source_label"] == 1).sum()),
        "source_non_humor_examples": int((combined["source_label"] == 0).sum()),
        "source_split_counts": source_split_counts,
        "negative_niches": sorted(NEGATIVE_NICHES),
        "utterance_field": "Reddit post title only",
        "training_overlap_removed": int(removed_overlap),
        "files": {
            provisional_path.name: sha256_file(provisional_path),
            annotation_path.name: sha256_file(annotation_path),
            key_path.name: sha256_file(key_path),
        },
        "important": (
            "This frozen OOD sample was created before viewing model performance. "
            "Do not regenerate it after evaluation. Source niche labels are provisional only."
        ),
    }

    manifest_path = OOD_DIR / "reddit_ood_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print("Frozen OOD sample")
    print("-----------------")
    print(f"Humor:       {(combined['source_label'] == 1).sum():,}")
    print(f"Non-humor:   {(combined['source_label'] == 0).sum():,}")
    print(f"Total:       {len(combined):,}")

    print()
    print("Created")
    print("-------")
    print(provisional_path)
    print(annotation_path)
    print(key_path)
    print(manifest_path)

    print()
    print("IMPORTANT")
    print("---------")
    print("Do not run this preparation script again after viewing OOD model results.")
    print("Next command:")
    print("python .\\training\\evaluate_reddit_ood.py --labels source")


if __name__ == "__main__":
    main()
