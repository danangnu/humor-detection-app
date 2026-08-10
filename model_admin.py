from __future__ import annotations

import json
import shutil
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelAdminError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json_atomic(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(path)


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class ModelAdminService:
    """
    Local administration service for Milestone 2.3.

    Responsibilities:
    - start candidate training in a background thread;
    - expose persistent retraining state;
    - display candidate comparison metrics;
    - promote only a passing candidate;
    - back up the active model before promotion;
    - roll back to the most recent promotion backup.

    It deliberately does not hot-swap the in-memory classifier. Promotion and
    rollback require application restart so the model is loaded cleanly.
    """

    CORRECTION_REPEAT = 32
    LEARNING_RATE = 3e-6
    EPOCHS = 1.0
    GATE_TOLERANCE = 0.005

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

        self.models_dir = self.root / "models"
        self.active_model_dir = (
            self.models_dir / "humor_transformer"
        )
        self.candidates_dir = (
            self.models_dir / "candidates"
        )
        self.history_dir = (
            self.models_dir / "model_history"
        )

        self.admin_data_dir = (
            self.root / "data" / "admin"
        )
        self.status_path = (
            self.admin_data_dir
            / "retraining_status.json"
        )
        self.promotion_history_path = (
            self.admin_data_dir
            / "promotion_history.json"
        )
        self.active_info_path = (
            self.models_dir
            / "active_model.json"
        )
        self.candidate_pointer_path = (
            self.models_dir
            / "candidate_current.json"
        )

        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

        self.admin_data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.candidates_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.history_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._repair_interrupted_status()

    # ---------------------------------------------------------
    # Persistent state
    # ---------------------------------------------------------

    def _repair_interrupted_status(self) -> None:
        status = _read_json(
            self.status_path,
            default={},
        ) or {}

        if status.get("state") in {
            "queued",
            "building_dataset",
            "training_candidate",
            "evaluating",
        }:
            status.update(
                {
                    "state": "interrupted",
                    "message": (
                        "A previous retraining process ended before "
                        "completion. Start a new candidate run."
                    ),
                    "updated_at_utc": _utc_now(),
                }
            )
            _write_json_atomic(
                status,
                self.status_path,
            )

    def _set_status(
        self,
        state: str,
        message: str,
        **extra,
    ) -> dict:
        current = _read_json(
            self.status_path,
            default={},
        ) or {}

        payload = {
            **current,
            "state": state,
            "message": message,
            "updated_at_utc": _utc_now(),
            **extra,
        }

        if state == "queued":
            payload["started_at_utc"] = (
                _utc_now()
            )
            payload.pop(
                "completed_at_utc",
                None,
            )
            payload.pop(
                "error",
                None,
            )

        if state in {
            "passed",
            "not_passed",
            "failed",
            "interrupted",
        }:
            payload["completed_at_utc"] = (
                _utc_now()
            )

        _write_json_atomic(
            payload,
            self.status_path,
        )
        return payload

    def retraining_status(self) -> dict:
        return _read_json(
            self.status_path,
            default={
                "state": "idle",
                "message": (
                    "No candidate retraining has "
                    "been started in this session."
                ),
            },
        )

    # ---------------------------------------------------------
    # Candidate information
    # ---------------------------------------------------------

    def _validated_candidate_pointer(
        self,
    ) -> dict | None:
        pointer = _read_json(
            self.candidate_pointer_path,
            default=None,
        )

        if not pointer:
            return None

        candidate_dir_raw = pointer.get(
            "candidate_dir"
        )
        model_dir_raw = pointer.get(
            "model_dir"
        )

        if (
            not candidate_dir_raw
            or not model_dir_raw
        ):
            return None

        candidate_dir = Path(
            candidate_dir_raw
        ).resolve()
        model_dir = Path(
            model_dir_raw
        ).resolve()

        if not _is_within(
            candidate_dir,
            self.candidates_dir,
        ):
            raise ModelAdminError(
                "Candidate pointer is outside models/candidates."
            )

        if not _is_within(
            model_dir,
            candidate_dir,
        ):
            raise ModelAdminError(
                "Candidate model path is outside its candidate directory."
            )

        if not model_dir.exists():
            return None

        return {
            **pointer,
            "candidate_dir": str(
                candidate_dir
            ),
            "model_dir": str(
                model_dir
            ),
        }

    def candidate_details(self) -> dict:
        pointer = (
            self._validated_candidate_pointer()
        )

        if not pointer:
            return {
                "available": False,
                "metrics_available": False,
                "promotion_gate_passed": False,
            }

        candidate_dir = Path(
            pointer["candidate_dir"]
        )

        metadata = _read_json(
            candidate_dir
            / "candidate_metadata.json",
            default={},
        ) or {}

        comparison = _read_json(
            candidate_dir
            / "comparison.json",
            default={},
        ) or {}

        gate = comparison.get(
            "promotion_gate",
            {},
        )

        return {
            "available": True,
            "candidate_id": pointer.get(
                "candidate_id"
            ),
            "status": pointer.get(
                "status"
            ),
            "candidate_dir": pointer[
                "candidate_dir"
            ],
            "model_dir": pointer[
                "model_dir"
            ],
            "promotion_gate_passed": bool(
                gate.get(
                    "passed",
                    pointer.get(
                        "promotion_gate_passed",
                        False,
                    ),
                )
            ),
            "metrics_available": bool(
                comparison
            ),
            "metadata": metadata,
            "comparison": comparison,
        }

    # ---------------------------------------------------------
    # Active model information
    # ---------------------------------------------------------

    def active_model_info(self) -> dict:
        info = _read_json(
            self.active_info_path,
            default=None,
        )

        if info:
            return {
                **info,
                "path": str(
                    self.active_model_dir
                ),
            }

        return {
            "model_version": (
                "baseline-humor-transformer"
            ),
            "path": str(
                self.active_model_dir
            ),
            "promoted_candidate_id": None,
            "restart_required": False,
        }

    # ---------------------------------------------------------
    # Retraining
    # ---------------------------------------------------------

    def retraining_available(self) -> bool:
        required = [
            self.root
            / "data"
            / "processed"
            / "train.csv",

            self.root
            / "data"
            / "processed"
            / "validation.csv",

            self.active_model_dir
            / "config.json",

            self.active_model_dir
            / "thresholds.json",
        ]

        if not all(
            path.exists()
            for path in required
        ):
            return False

        try:
            from feedback_store import (
                correction_counts,
            )

            return (
                correction_counts()
                .get("approved", 0)
                > 0
            )
        except Exception:
            return False

    def is_running(self) -> bool:
        return bool(
            self._worker
            and self._worker.is_alive()
        )

    def start_retraining(self) -> dict:
        with self._lock:
            if self.is_running():
                raise ModelAdminError(
                    "Candidate retraining is already running."
                )

            if not self.retraining_available():
                raise ModelAdminError(
                    "Retraining is not available. "
                    "At least one approved correction and "
                    "the local training/model files are required."
                )

            self._set_status(
                "queued",
                (
                    "Candidate retraining has been "
                    "queued."
                ),
                configuration={
                    "correction_repeat": (
                        self.CORRECTION_REPEAT
                    ),
                    "learning_rate": (
                        self.LEARNING_RATE
                    ),
                    "epochs": (
                        self.EPOCHS
                    ),
                    "gate_tolerance": (
                        self.GATE_TOLERANCE
                    ),
                    "held_out_test": (
                        "skipped for routine admin retraining"
                    ),
                },
            )

            self._worker = threading.Thread(
                target=self._retraining_worker,
                name="humor-candidate-training",
                daemon=True,
            )
            self._worker.start()

        return self.retraining_status()

    def _retraining_worker(self) -> None:
        try:
            self._set_status(
                "building_dataset",
                (
                    "Building a frozen candidate dataset "
                    "from approved corrections."
                ),
            )

            # Lazy imports prevent ML/training dependencies from
            # being loaded just to open the application.
            from training.correction_dataset import (
                build_correction_dataset,
            )

            dataset_manifest = (
                build_correction_dataset(
                    root=self.root
                )
            )

            dataset_dir = Path(
                dataset_manifest[
                    "output"
                ]["candidate_train"]
            ).parent

            if not dataset_manifest.get(
                "ready_for_candidate_training"
            ):
                raise ModelAdminError(
                    "No approved binary correction was "
                    "eligible for candidate training. "
                    "Check corrections_excluded.csv."
                )

            self._set_status(
                "training_candidate",
                (
                    "Fine-tuning the candidate Transformer "
                    "with bounded correction replay."
                ),
                dataset_run_id=(
                    dataset_manifest["run_id"]
                ),
                dataset_dir=str(
                    dataset_dir
                ),
            )

            from training.candidate_trainer import (
                train_candidate,
            )

            result = train_candidate(
                root=self.root,
                dataset_dir=dataset_dir,
                correction_repeat=(
                    self.CORRECTION_REPEAT
                ),
                epochs=self.EPOCHS,
                learning_rate=(
                    self.LEARNING_RATE
                ),
                gate_tolerance=(
                    self.GATE_TOLERANCE
                ),
                # Routine admin retraining does not repeatedly
                # inspect the held-out test set.
                include_test=False,
            )

            metadata = result[
                "metadata"
            ]
            comparison = result[
                "comparison"
            ]

            gate_passed = bool(
                comparison
                .get(
                    "promotion_gate",
                    {},
                )
                .get("passed")
            )

            state = (
                "passed"
                if gate_passed
                else "not_passed"
            )

            self._set_status(
                state,
                (
                    "Candidate passed the promotion gate "
                    "and is eligible for manual promotion."
                    if gate_passed
                    else
                    "Candidate training completed but the "
                    "promotion gate did not pass."
                ),
                candidate_id=(
                    metadata["candidate_id"]
                ),
                candidate_model=(
                    metadata["candidate_model"]
                ),
                comparison=(
                    metadata["comparison"]
                ),
                promotion_gate_passed=(
                    gate_passed
                ),
            )

        except Exception as exc:
            self._set_status(
                "failed",
                "Candidate retraining failed.",
                error=str(exc),
                traceback=(
                    traceback.format_exc()
                ),
            )

    # ---------------------------------------------------------
    # Promotion / rollback history
    # ---------------------------------------------------------

    def promotion_history(self) -> list[dict]:
        data = _read_json(
            self.promotion_history_path,
            default=[],
        )

        if not isinstance(data, list):
            return []

        return data

    def _save_promotion_history(
        self,
        records: list[dict],
    ) -> None:
        _write_json_atomic(
            records,
            self.promotion_history_path,
        )

    def _new_backup_dir(
        self,
        prefix: str,
    ) -> Path:
        return (
            self.history_dir
            / (
                f"{prefix}_{_timestamp()}"
            )
        )

    def _swap_active_from(
        self,
        source_model: Path,
    ) -> None:
        if not source_model.exists():
            raise ModelAdminError(
                f"Source model does not exist: "
                f"{source_model}"
            )

        if not (
            source_model
            / "config.json"
        ).exists():
            raise ModelAdminError(
                "Source model is missing config.json."
            )

        if not (
            source_model
            / "thresholds.json"
        ).exists():
            raise ModelAdminError(
                "Source model is missing thresholds.json."
            )

        staging = (
            self.models_dir
            / "humor_transformer.next"
        )
        swapped_old = (
            self.models_dir
            / "humor_transformer.swap_old"
        )

        shutil.rmtree(
            staging,
            ignore_errors=True,
        )
        shutil.rmtree(
            swapped_old,
            ignore_errors=True,
        )

        shutil.copytree(
            source_model,
            staging,
        )

        if not self.active_model_dir.exists():
            staging.rename(
                self.active_model_dir
            )
            return

        try:
            self.active_model_dir.rename(
                swapped_old
            )
            staging.rename(
                self.active_model_dir
            )
        except Exception:
            if (
                not self.active_model_dir.exists()
                and swapped_old.exists()
            ):
                swapped_old.rename(
                    self.active_model_dir
                )

            shutil.rmtree(
                staging,
                ignore_errors=True,
            )
            raise

        shutil.rmtree(
            swapped_old,
            ignore_errors=True,
        )

    # ---------------------------------------------------------
    # Promotion
    # ---------------------------------------------------------

    def promote_candidate(
        self,
        confirmation: str,
    ) -> dict:
        if confirmation != "PROMOTE":
            raise ModelAdminError(
                "Promotion requires confirmation='PROMOTE'."
            )

        if self.is_running():
            raise ModelAdminError(
                "Cannot promote while retraining is running."
            )

        with self._lock:
            candidate = (
                self.candidate_details()
            )

            if not candidate.get(
                "available"
            ):
                raise ModelAdminError(
                    "No candidate model is available."
                )

            if not candidate.get(
                "promotion_gate_passed"
            ):
                raise ModelAdminError(
                    "Candidate has not passed the promotion gate."
                )

            if candidate.get(
                "status"
            ) == "promoted":
                raise ModelAdminError(
                    "This candidate is already promoted."
                )

            candidate_id = str(
                candidate["candidate_id"]
            )
            candidate_model = Path(
                candidate["model_dir"]
            ).resolve()

            if not _is_within(
                candidate_model,
                self.candidates_dir,
            ):
                raise ModelAdminError(
                    "Candidate model path is invalid."
                )

            if not self.active_model_dir.exists():
                raise ModelAdminError(
                    "Active model directory is missing."
                )

            backup_dir = (
                self._new_backup_dir(
                    (
                        "pre_promotion_"
                        + candidate_id
                    )
                )
            )

            shutil.copytree(
                self.active_model_dir,
                backup_dir,
            )

            promotion_id = (
                f"promotion_{_timestamp()}_"
                f"{candidate_id}"
            )

            try:
                self._swap_active_from(
                    candidate_model
                )
            except Exception:
                shutil.rmtree(
                    backup_dir,
                    ignore_errors=True,
                )
                raise

            record = {
                "promotion_id": (
                    promotion_id
                ),
                "candidate_id": (
                    candidate_id
                ),
                "promoted_at_utc": (
                    _utc_now()
                ),
                "candidate_model": str(
                    candidate_model
                ),
                "previous_active_backup": str(
                    backup_dir
                ),
                "rolled_back": False,
                "rolled_back_at_utc": None,
            }

            history = (
                self.promotion_history()
            )
            history.append(record)
            self._save_promotion_history(
                history
            )

            active_info = {
                "model_version": (
                    candidate_id
                ),
                "promoted_candidate_id": (
                    candidate_id
                ),
                "promotion_id": (
                    promotion_id
                ),
                "promoted_at_utc": (
                    record[
                        "promoted_at_utc"
                    ]
                ),
                "previous_active_backup": str(
                    backup_dir
                ),
                "restart_required": True,
            }

            _write_json_atomic(
                active_info,
                self.active_info_path,
            )

            pointer = _read_json(
                self.candidate_pointer_path,
                default={},
            ) or {}

            pointer.update(
                {
                    "status": "promoted",
                    "promoted_at_utc": (
                        _utc_now()
                    ),
                    "promotion_id": (
                        promotion_id
                    ),
                }
            )

            _write_json_atomic(
                pointer,
                self.candidate_pointer_path,
            )

            self._set_status(
                "promoted",
                (
                    "Candidate was promoted. Restart "
                    "Humor Bot to load the new active model."
                ),
                candidate_id=(
                    candidate_id
                ),
                promotion_id=(
                    promotion_id
                ),
                restart_required=True,
            )

            return {
                "ok": True,
                "message": (
                    "Candidate promoted successfully. "
                    "Restart Humor Bot before testing "
                    "the new active model."
                ),
                "promotion": record,
                "active_model": active_info,
            }

    # ---------------------------------------------------------
    # Rollback
    # ---------------------------------------------------------

    def _latest_rollback_source(
        self,
    ) -> tuple[
        list[dict],
        int,
        dict,
        Path,
    ]:
        history = self.promotion_history()

        for index in range(
            len(history) - 1,
            -1,
            -1,
        ):
            record = history[index]

            if record.get(
                "rolled_back"
            ):
                continue

            backup_raw = record.get(
                "previous_active_backup"
            )

            if not backup_raw:
                continue

            backup = Path(
                backup_raw
            ).resolve()

            if not _is_within(
                backup,
                self.history_dir,
            ):
                continue

            if backup.exists():
                return (
                    history,
                    index,
                    record,
                    backup,
                )

        raise ModelAdminError(
            "No promotion backup is available for rollback."
        )

    def rollback_latest(
        self,
        confirmation: str,
    ) -> dict:
        if confirmation != "ROLLBACK":
            raise ModelAdminError(
                "Rollback requires confirmation='ROLLBACK'."
            )

        if self.is_running():
            raise ModelAdminError(
                "Cannot roll back while retraining is running."
            )

        with self._lock:
            (
                history,
                index,
                record,
                restore_source,
            ) = self._latest_rollback_source()

            current_backup = (
                self._new_backup_dir(
                    "pre_rollback_current"
                )
            )

            if self.active_model_dir.exists():
                shutil.copytree(
                    self.active_model_dir,
                    current_backup,
                )

            try:
                self._swap_active_from(
                    restore_source
                )
            except Exception:
                shutil.rmtree(
                    current_backup,
                    ignore_errors=True,
                )
                raise

            rolled_back_at = (
                _utc_now()
            )

            history[index] = {
                **record,
                "rolled_back": True,
                "rolled_back_at_utc": (
                    rolled_back_at
                ),
                "pre_rollback_current_backup": (
                    str(
                        current_backup
                    )
                    if current_backup.exists()
                    else None
                ),
            }

            self._save_promotion_history(
                history
            )

            active_info = {
                "model_version": (
                    "rollback:"
                    + str(
                        record[
                            "promotion_id"
                        ]
                    )
                ),
                "rolled_back_from_candidate_id": (
                    record.get(
                        "candidate_id"
                    )
                ),
                "rollback_source": str(
                    restore_source
                ),
                "rolled_back_at_utc": (
                    rolled_back_at
                ),
                "restart_required": True,
            }

            _write_json_atomic(
                active_info,
                self.active_info_path,
            )

            self._set_status(
                "rolled_back",
                (
                    "Previous active model was restored. "
                    "Restart Humor Bot to load it."
                ),
                promotion_id=(
                    record[
                        "promotion_id"
                    ]
                ),
                restart_required=True,
            )

            return {
                "ok": True,
                "message": (
                    "Rollback completed. Restart Humor Bot "
                    "to load the restored model."
                ),
                "rolled_back_promotion": (
                    history[index]
                ),
                "active_model": (
                    active_info
                ),
            }

    # ---------------------------------------------------------
    # Combined admin status
    # ---------------------------------------------------------

    def status_payload(self) -> dict:
        candidate = (
            self.candidate_details()
        )

        rollback_available = False
        try:
            self._latest_rollback_source()
            rollback_available = True
        except ModelAdminError:
            pass

        return {
            "retraining_status": (
                self.retraining_status()
            ),
            "retraining_running": (
                self.is_running()
            ),
            "retraining_available": (
                self.retraining_available()
            ),
            "candidate": candidate,
            "active_model": (
                self.active_model_info()
            ),
            "rollback_available": (
                rollback_available
            ),
            "configuration": {
                "correction_repeat": (
                    self.CORRECTION_REPEAT
                ),
                "learning_rate": (
                    self.LEARNING_RATE
                ),
                "epochs": (
                    self.EPOCHS
                ),
                "gate_tolerance": (
                    self.GATE_TOLERANCE
                ),
                "routine_test_policy": (
                    "held-out test skipped"
                ),
            },
        }
