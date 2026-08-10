import pandas as pd
import pytest

# candidate_trainer imports ML packages in the real environment.
# This test file is intended for the Humor Bot venv where they are installed.
from training.candidate_trainer import build_effective_training_frame


def test_repeat_32_adds_31_rows_per_correction():
    candidate_train = pd.DataFrame(
        [
            {"text": "base plain", "label": 0},
            {"text": "corrected joke", "label": 1},
        ]
    )

    included = pd.DataFrame(
        [
            {
                "correction_id": 1,
                "text": "corrected joke",
                "training_label": 1,
            }
        ]
    )

    effective, metadata = build_effective_training_frame(
        candidate_train=candidate_train,
        included_corrections=included,
        correction_repeat=32,
        seed=42,
    )

    assert len(effective) == 33
    assert metadata["replay_rows_added"] == 31
    assert (
        metadata["effective_training_rows_after_replay"]
        == 33
    )


def test_repeat_limit():
    candidate_train = pd.DataFrame(
        [{"text": "x", "label": 0}]
    )

    included = pd.DataFrame(
        [
            {
                "correction_id": 1,
                "text": "y",
                "training_label": 1,
            }
        ]
    )

    with pytest.raises(ValueError):
        build_effective_training_frame(
            candidate_train=candidate_train,
            included_corrections=included,
            correction_repeat=65,
            seed=42,
        )
