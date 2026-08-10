from pathlib import Path
import json

from model_admin import ModelAdminService


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )


def fake_model(path: Path, marker: str):
    path.mkdir(parents=True, exist_ok=True)

    (path / "config.json").write_text(
        '{"model_type":"roberta"}',
        encoding="utf-8",
    )

    (path / "thresholds.json").write_text(
        '{"low_threshold":0.2,"high_threshold":0.8}',
        encoding="utf-8",
    )

    (path / "marker.txt").write_text(
        marker,
        encoding="utf-8",
    )


def test_promotion_and_rollback(tmp_path):
    service = ModelAdminService(
        tmp_path
    )

    active = (
        tmp_path
        / "models"
        / "humor_transformer"
    )

    fake_model(
        active,
        "ACTIVE_OLD",
    )

    candidate_dir = (
        tmp_path
        / "models"
        / "candidates"
        / "candidate_test"
    )

    candidate_model = (
        candidate_dir
        / "model"
    )

    fake_model(
        candidate_model,
        "CANDIDATE_NEW",
    )

    comparison = {
        "promotion_gate": {
            "passed": True,
            "checks": {
                "example": True,
            },
        },
        "validation": {
            "active": {
                "binary_at_0_5": {
                    "accuracy": .90,
                    "f1": .92,
                },
            },
            "candidate": {
                "binary_at_0_5": {
                    "accuracy": .90,
                    "f1": .92,
                },
            },
            "delta": {},
        },
        "correction_retention": {
            "size": 1,
            "accuracy": 1.0,
        },
    }

    write_json(
        candidate_dir
        / "comparison.json",
        comparison,
    )

    write_json(
        candidate_dir
        / "candidate_metadata.json",
        {
            "candidate_id": (
                "candidate_test"
            ),
            "status": "passed",
            "promotion_gate_passed": True,
        },
    )

    write_json(
        tmp_path
        / "models"
        / "candidate_current.json",
        {
            "candidate_id": (
                "candidate_test"
            ),
            "candidate_dir": str(
                candidate_dir
            ),
            "model_dir": str(
                candidate_model
            ),
            "status": "passed",
            "promotion_gate_passed": True,
        },
    )

    result = service.promote_candidate(
        "PROMOTE"
    )

    assert result["ok"] is True

    assert (
        active
        / "marker.txt"
    ).read_text(
        encoding="utf-8"
    ) == "CANDIDATE_NEW"

    history = (
        service.promotion_history()
    )

    assert len(history) == 1

    backup = Path(
        history[0][
            "previous_active_backup"
        ]
    )

    assert (
        backup
        / "marker.txt"
    ).read_text(
        encoding="utf-8"
    ) == "ACTIVE_OLD"

    rollback = (
        service.rollback_latest(
            "ROLLBACK"
        )
    )

    assert rollback["ok"] is True

    assert (
        active
        / "marker.txt"
    ).read_text(
        encoding="utf-8"
    ) == "ACTIVE_OLD"

    history = (
        service.promotion_history()
    )

    assert history[0][
        "rolled_back"
    ] is True


def test_promotion_rejects_failed_gate(
    tmp_path,
):
    service = ModelAdminService(
        tmp_path
    )

    fake_model(
        tmp_path
        / "models"
        / "humor_transformer",
        "ACTIVE",
    )

    candidate_dir = (
        tmp_path
        / "models"
        / "candidates"
        / "candidate_bad"
    )

    candidate_model = (
        candidate_dir
        / "model"
    )

    fake_model(
        candidate_model,
        "BAD",
    )

    write_json(
        candidate_dir
        / "comparison.json",
        {
            "promotion_gate": {
                "passed": False,
            },
        },
    )

    write_json(
        tmp_path
        / "models"
        / "candidate_current.json",
        {
            "candidate_id": (
                "candidate_bad"
            ),
            "candidate_dir": str(
                candidate_dir
            ),
            "model_dir": str(
                candidate_model
            ),
            "status": "not_passed",
            "promotion_gate_passed": False,
        },
    )

    try:
        service.promote_candidate(
            "PROMOTE"
        )
    except Exception as exc:
        assert (
            "promotion gate"
            in str(exc).lower()
        )
    else:
        raise AssertionError(
            "Failed candidate was promoted."
        )


def test_confirmation_required(
    tmp_path,
):
    service = ModelAdminService(
        tmp_path
    )

    try:
        service.promote_candidate(
            "yes"
        )
    except Exception as exc:
        assert (
            "PROMOTE"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Promotion confirmation bypassed."
        )

    try:
        service.rollback_latest(
            "yes"
        )
    except Exception as exc:
        assert (
            "ROLLBACK"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Rollback confirmation bypassed."
        )
