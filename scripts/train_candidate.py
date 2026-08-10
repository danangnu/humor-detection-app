from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pct(value):
    return (
        "—"
        if value is None
        else f"{float(value) * 100:.2f}%"
    )


def pp(value):
    if value is None:
        return "—"

    sign = (
        "+"
        if float(value) >= 0
        else ""
    )

    return (
        f"{sign}{float(value) * 100:.2f} pp"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train Humor Bot candidate Transformer with bounded "
            "approved-correction replay."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--correction-repeat",
        type=int,
        default=32,
        help=(
            "Total training appearances per eligible approved "
            "correction. Default 32; maximum 64."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-6,
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--gate-tolerance",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--skip-test",
        action="store_true",
    )

    args = parser.parse_args()

    from training.candidate_trainer import (
        train_candidate,
    )

    print("=" * 76)
    print("Humor Bot - Milestone 2.2B Revision 2")
    print("Candidate Training with Bounded Correction Replay")
    print("=" * 76)
    print(
        "Active production model will NOT be replaced."
    )
    print(
        f"Correction replay: {args.correction_repeat} total appearances"
    )
    print(
        f"Learning rate:     {args.learning_rate}"
    )
    print()

    result = train_candidate(
        root=ROOT,
        dataset_dir=(
            args.dataset_dir
        ),
        correction_repeat=(
            args.correction_repeat
        ),
        epochs=args.epochs,
        learning_rate=(
            args.learning_rate
        ),
        train_batch_size=(
            args.train_batch_size
        ),
        eval_batch_size=(
            args.eval_batch_size
        ),
        max_length=(
            args.max_length
        ),
        seed=args.seed,
        gate_tolerance=(
            args.gate_tolerance
        ),
        include_test=(
            not args.skip_test
        ),
    )

    metadata = result["metadata"]
    comparison = result["comparison"]

    replay = metadata[
        "correction_replay"
    ]

    print()
    print("Replay summary")
    print("--------------")
    print(
        "Unique eligible corrections:   "
        f"{replay['unique_eligible_corrections']}"
    )
    print(
        "Total appearances/correction:  "
        f"{replay['correction_repeat_total_appearances']}"
    )
    print(
        "Replay rows added:              "
        f"{replay['replay_rows_added']}"
    )
    print(
        "Snapshot rows before replay:    "
        f"{replay['candidate_snapshot_rows_before_replay']}"
    )
    print(
        "Effective training rows:        "
        f"{replay['effective_training_rows_after_replay']}"
    )
    print(
        "Replay rows are weighting copies, not new independent annotations."
    )

    print()
    print("Candidate")
    print("---------")
    print(
        f"ID:     {metadata['candidate_id']}"
    )
    print(
        f"Status: {metadata['status']}"
    )
    print(
        f"Model:  {metadata['candidate_model']}"
    )

    print()
    print("Validation comparison")
    print("---------------------")

    active = comparison[
        "validation"
    ]["active"]["binary_at_0_5"]

    candidate = comparison[
        "validation"
    ]["candidate"]["binary_at_0_5"]

    changes = comparison[
        "validation"
    ]["delta"]

    for key in (
        "accuracy",
        "precision",
        "recall",
        "f1",
    ):
        print(
            f"{key:10} "
            f"active={pct(active[key]):>8} "
            f"candidate={pct(candidate[key]):>8} "
            f"delta={pp(changes[key])}"
        )

    print(
        f"{'roc_auc':10} "
        f"active={active['roc_auc']:.4f} "
        f"candidate={candidate['roc_auc']:.4f} "
        f"delta={changes['roc_auc']:+.4f}"
    )

    retention = comparison[
        "correction_retention"
    ]

    print()
    print("Approved correction retention")
    print("-----------------------------")
    print(
        f"Eligible: {retention['size']}"
    )
    print(
        f"Correct:  {retention['correct']}"
    )
    print(
        f"Accuracy: {pct(retention['accuracy'])}"
    )
    print(
        "Retention is a training sanity check, not generalization."
    )

    if comparison["test"] is not None:
        print()
        print(
            "Held-out test comparison (report-only)"
        )
        print(
            "--------------------------------------"
        )

        active_test = comparison[
            "test"
        ]["active"]["binary_at_0_5"]

        candidate_test = comparison[
            "test"
        ]["candidate"]["binary_at_0_5"]

        test_changes = comparison[
            "test"
        ]["delta"]

        for key in (
            "accuracy",
            "precision",
            "recall",
            "f1",
        ):
            print(
                f"{key:10} "
                f"active={pct(active_test[key]):>8} "
                f"candidate={pct(candidate_test[key]):>8} "
                f"delta={pp(test_changes[key])}"
            )

        print(
            f"{'roc_auc':10} "
            f"active={active_test['roc_auc']:.4f} "
            f"candidate={candidate_test['roc_auc']:.4f} "
            f"delta={test_changes['roc_auc']:+.4f}"
        )

        print(
            "Test metrics are NOT used by the promotion gate."
        )

    gate = comparison[
        "promotion_gate"
    ]

    print()
    print("Promotion gate")
    print("--------------")

    for name, passed in gate[
        "checks"
    ].items():
        print(
            f"{'PASS' if passed else 'FAIL'}  {name}"
        )

    print()
    print(
        "PROMOTION GATE: "
        + (
            "PASSED"
            if gate["passed"]
            else "NOT PASSED"
        )
    )
    print(
        "ACTIVE MODEL: UNCHANGED"
    )
    print(
        f"Comparison JSON: {metadata['comparison']}"
    )


if __name__ == "__main__":
    main()
