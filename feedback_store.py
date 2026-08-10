from __future__ import annotations
import os, sqlite3, sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

VALID_LABELS = {"Humorous", "Not Humorous", "Ambiguous / Uncertain"}
VALID_STATUSES = {"pending", "approved", "rejected"}

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def feedback_db_path():
    explicit = os.getenv("HUMOR_FEEDBACK_DB", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    if getattr(sys, "frozen", False):
        base = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "HumorBot" / "humor_feedback.db"
    return Path(__file__).resolve().parent / "data" / "humor_feedback.db"

def _connect():
    path = feedback_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db

@contextmanager
def connection():
    db = _connect()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def initialize_database():
    with connection() as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS humor_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            original_label TEXT NOT NULL,
            original_score REAL NOT NULL,
            corrected_label TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'analyze',
            model_source TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(input_text, original_label, corrected_label, model_source)
        )""")
        db.execute("""CREATE INDEX IF NOT EXISTS idx_humor_corrections_status_created
                      ON humor_corrections(status, created_at DESC)""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS dataset_build_runs (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            base_train_path TEXT NOT NULL,
            output_dataset_path TEXT,
            manifest_path TEXT,
            base_rows INTEGER NOT NULL DEFAULT 0,
            approved_rows INTEGER NOT NULL DEFAULT 0,
            included_rows INTEGER NOT NULL DEFAULT 0,
            excluded_rows INTEGER NOT NULL DEFAULT 0,
            final_rows INTEGER NOT NULL DEFAULT 0,
            dataset_sha256 TEXT,
            error_message TEXT NOT NULL DEFAULT ''
        )""")
        db.execute("""
        CREATE TABLE IF NOT EXISTS correction_dataset_usage (
            run_id TEXT NOT NULL,
            correction_id INTEGER NOT NULL,
            included INTEGER NOT NULL,
            training_label INTEGER,
            exclusion_reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, correction_id),
            FOREIGN KEY (run_id) REFERENCES dataset_build_runs(run_id) ON DELETE CASCADE,
            FOREIGN KEY (correction_id) REFERENCES humor_corrections(id) ON DELETE CASCADE
        )""")
    return feedback_db_path()

def _label(v):
    v = str(v).strip()
    if v not in VALID_LABELS: raise ValueError("Unknown humor label.")
    return v

def _status(v):
    v = str(v).strip().lower()
    if v not in VALID_STATUSES: raise ValueError("Unknown correction status.")
    return v

def get_correction(correction_id):
    with connection() as db:
        row = db.execute("SELECT * FROM humor_corrections WHERE id=?", (int(correction_id),)).fetchone()
    if row is None: raise KeyError("Correction not found.")
    return dict(row)

def add_correction(*, input_text, original_label, original_score, corrected_label, source, model_source):
    text = " ".join(str(input_text).split()).strip()
    if not text: raise ValueError("Input text cannot be empty.")
    original, corrected = _label(original_label), _label(corrected_label)
    if original == corrected: raise ValueError("The corrected label must differ from the original prediction.")
    score = float(original_score)
    if not 0 <= score <= 1: raise ValueError("Original score must be between 0 and 1.")
    now = _now()
    try:
        with connection() as db:
            cur = db.execute("""
            INSERT INTO humor_corrections
            (input_text, original_label, original_score, corrected_label, source,
             model_source, status, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', '', ?, ?)
            """, (text, original, score, corrected, str(source or "analyze"),
                  str(model_source or ""), now, now))
            cid = cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("This correction has already been saved.") from None
    return get_correction(cid)

def list_corrections(status=None, limit=500):
    limit = max(1, min(int(limit), 2000))
    with connection() as db:
        if status:
            rows = db.execute("""SELECT * FROM humor_corrections WHERE status=?
                                 ORDER BY created_at DESC, id DESC LIMIT ?""",
                              (_status(status), limit)).fetchall()
        else:
            rows = db.execute("""SELECT * FROM humor_corrections
                                 ORDER BY created_at DESC, id DESC LIMIT ?""",
                              (limit,)).fetchall()
    return [dict(r) for r in rows]

def list_approved_corrections():
    with connection() as db:
        rows = db.execute("""SELECT * FROM humor_corrections WHERE status='approved'
                             ORDER BY created_at ASC, id ASC""").fetchall()
    return [dict(r) for r in rows]

def correction_counts():
    out = {"total": 0, "pending": 0, "approved": 0, "rejected": 0}
    with connection() as db:
        rows = db.execute("SELECT status, COUNT(*) count FROM humor_corrections GROUP BY status").fetchall()
    for row in rows:
        out[row["status"]] = int(row["count"])
        out["total"] += int(row["count"])
    return out

def update_correction(correction_id, *, corrected_label=None, status=None, notes=None):
    current = get_correction(correction_id)
    new_label = _label(corrected_label) if corrected_label is not None else current["corrected_label"]
    if new_label == current["original_label"]:
        raise ValueError("The corrected label must differ from the original prediction.")
    new_status = _status(status) if status is not None else current["status"]
    new_notes = str(notes).strip() if notes is not None else current["notes"]
    with connection() as db:
        db.execute("""UPDATE humor_corrections SET corrected_label=?, status=?, notes=?, updated_at=? WHERE id=?""",
                   (new_label, new_status, new_notes, _now(), int(correction_id)))
    return get_correction(correction_id)

def delete_correction(correction_id):
    with connection() as db:
        cur = db.execute("DELETE FROM humor_corrections WHERE id=?", (int(correction_id),))
        if cur.rowcount == 0: raise KeyError("Correction not found.")

def create_dataset_build_run(run_id, base_train_path, approved_rows, base_rows):
    with connection() as db:
        db.execute("""INSERT INTO dataset_build_runs
        (run_id,status,created_at,base_train_path,approved_rows,base_rows)
        VALUES (?,'building',?,?,?,?)""",
        (run_id, _now(), str(base_train_path), int(approved_rows), int(base_rows)))

def record_dataset_usage(run_id, correction_id, included, training_label=None, exclusion_reason=""):
    with connection() as db:
        db.execute("""INSERT OR REPLACE INTO correction_dataset_usage
        (run_id,correction_id,included,training_label,exclusion_reason,created_at)
        VALUES (?,?,?,?,?,?)""",
        (run_id, int(correction_id), 1 if included else 0, training_label,
         str(exclusion_reason or ""), _now()))

def complete_dataset_build_run(run_id, output_dataset_path, manifest_path,
                               included_rows, excluded_rows, final_rows, dataset_sha256):
    with connection() as db:
        db.execute("""UPDATE dataset_build_runs SET status='ready',completed_at=?,
        output_dataset_path=?,manifest_path=?,included_rows=?,excluded_rows=?,
        final_rows=?,dataset_sha256=?,error_message='' WHERE run_id=?""",
        (_now(), str(output_dataset_path), str(manifest_path), int(included_rows),
         int(excluded_rows), int(final_rows), str(dataset_sha256), run_id))

def fail_dataset_build_run(run_id, error_message):
    with connection() as db:
        db.execute("""UPDATE dataset_build_runs SET status='failed',completed_at=?,
                      error_message=? WHERE run_id=?""",
                   (_now(), str(error_message)[:4000], run_id))

def list_dataset_build_runs(limit=100):
    limit = max(1, min(int(limit), 1000))
    with connection() as db:
        rows = db.execute("""SELECT * FROM dataset_build_runs
                             ORDER BY created_at DESC LIMIT ?""", (limit,)).fetchall()
    return [dict(r) for r in rows]
