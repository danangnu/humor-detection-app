from __future__ import annotations
import hashlib, json, re, unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import pandas as pd

from feedback_store import (
    complete_dataset_build_run, create_dataset_build_run, fail_dataset_build_run,
    feedback_db_path, initialize_database, list_approved_corrections,
    record_dataset_usage,
)

BINARY_LABELS = {"Not Humorous": 0, "Humorous": 1}
AMBIGUOUS_LABEL = "Ambiguous / Uncertain"

def normalize_text(v):
    text = unicodedata.normalize("NFKC", str(v))
    return re.sub(r"\s+", " ", text).strip().casefold()

def clean_text(v):
    text = unicodedata.normalize("NFKC", str(v))
    return re.sub(r"\s+", " ", text).strip()

def load_binary_csv(path):
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f"Required dataset not found: {path}")
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing: raise ValueError(f"{path} must contain text,label columns. Missing: {sorted(missing)}")
    df = df[["text","label"]].dropna().copy()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    df["label"] = df["label"].astype(int)
    invalid = sorted(set(df["label"]) - {0,1})
    if invalid: raise ValueError(f"{path} contains non-binary labels: {invalid}")
    df["_norm"] = df["text"].map(normalize_text)
    return df.reset_index(drop=True)

def load_protected(validation_path, test_path, ood_dir):
    protected, detail = set(), {}
    for name, path in (("validation",validation_path),("test",test_path)):
        df = load_binary_csv(path)
        vals = set(df["_norm"])
        protected.update(vals)
        detail[name] = len(vals)
    ood_added = 0
    ood_dir = Path(ood_dir)
    if ood_dir.exists():
        for path in sorted(ood_dir.glob("*.csv")):
            try: df = pd.read_csv(path)
            except Exception: continue
            if "text" not in df.columns: continue
            vals = {normalize_text(v) for v in df["text"].dropna() if clean_text(v)}
            before = len(protected)
            protected.update(vals)
            ood_added += len(protected) - before
    detail["ood_unique_added"] = ood_added
    detail["protected_unique_total"] = len(protected)
    return protected, detail

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def write_csv(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp"); df.to_csv(tmp,index=False); tmp.replace(path)

def write_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")
    tmp.replace(path)

def audit(c):
    return {
        "correction_id":int(c["id"]), "text":c["input_text"],
        "original_label":c["original_label"], "original_score":float(c["original_score"]),
        "corrected_label":c["corrected_label"], "source":c["source"],
        "model_source":c["model_source"], "created_at":c["created_at"],
        "updated_at":c["updated_at"],
    }

def build_correction_dataset(root, output_root=None):
    root = Path(root).resolve()
    processed = root/"data"/"processed"
    train_path, val_path, test_path = (processed/"train.csv", processed/"validation.csv", processed/"test.csv")
    ood_dir = root/"data"/"ood"
    initialize_database()
    base = load_binary_csv(train_path)
    approved = list_approved_corrections()
    run_id = datetime.now(timezone.utc).strftime("dataset_%Y%m%dT%H%M%SZ_")+uuid4().hex[:8]
    output_root = Path(output_root).resolve() if output_root else root/"data"/"retraining"
    run_dir = output_root/run_id
    create_dataset_build_run(run_id, train_path, len(approved), len(base))
    try:
        protected, protected_detail = load_protected(val_path,test_path,ood_dir)
        base_labels = defaultdict(set)
        for key,label in base[["_norm","label"]].itertuples(index=False,name=None):
            base_labels[key].add(int(label))

        groups = defaultdict(list)
        for c in approved: groups[normalize_text(c["input_text"])].append(c)
        conflicts = set()
        for key, rows in groups.items():
            labels = {r["corrected_label"] for r in rows if r["corrected_label"] in BINARY_LABELS}
            if len(labels) > 1: conflicts.add(key)

        included, excluded, selected, reasons = [], [], set(), Counter()
        ordered = sorted(approved,key=lambda r:(str(r["updated_at"]),int(r["id"])),reverse=True)

        for c in ordered:
            row, key, label_name = audit(c), normalize_text(c["input_text"]), c["corrected_label"]
            reason, training_label = "", None
            if label_name == AMBIGUOUS_LABEL:
                reason = "ambiguous_not_a_binary_training_label"
            elif label_name not in BINARY_LABELS:
                reason = "unsupported_corrected_label"
            elif key in conflicts:
                reason = "conflicting_approved_corrections"
            elif key in protected:
                reason = "protected_evaluation_overlap"
            else:
                training_label = BINARY_LABELS[label_name]
                if key in base_labels:
                    reason = "already_in_base_training" if base_labels[key] == {training_label} else "conflicts_with_base_training_label"
                elif key in selected:
                    reason = "duplicate_approved_correction"

            row["training_label"], row["exclusion_reason"] = training_label, reason
            if reason:
                excluded.append(row); reasons[reason]+=1
                record_dataset_usage(run_id,c["id"],False,training_label,reason)
            else:
                selected.add(key); included.append(row)
                record_dataset_usage(run_id,c["id"],True,training_label,"")

        corr_df = pd.DataFrame(
            [{"text":clean_text(r["text"]),"label":int(r["training_label"])} for r in included],
            columns=["text","label"])
        final = pd.concat([base[["text","label"]],corr_df],ignore_index=True)
        final["_norm"] = final["text"].map(normalize_text)
        final = final.drop_duplicates("_norm",keep="first")[["text","label"]].reset_index(drop=True)

        run_dir.mkdir(parents=True,exist_ok=False)
        dataset_path = run_dir/"candidate_train.csv"
        inc_path, exc_path, manifest_path = run_dir/"corrections_included.csv", run_dir/"corrections_excluded.csv", run_dir/"dataset_manifest.json"
        cols=["correction_id","text","original_label","original_score","corrected_label","source","model_source","created_at","updated_at","training_label","exclusion_reason"]
        write_csv(final,dataset_path)
        write_csv(pd.DataFrame(included,columns=cols),inc_path)
        write_csv(pd.DataFrame(excluded,columns=cols),exc_path)
        digest = file_hash(dataset_path)

        manifest = {
            "milestone":"2.2A","run_id":run_id,
            "created_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "purpose":"Frozen candidate-training dataset snapshot; no training or production promotion is performed.",
            "database":str(feedback_db_path()),
            "counts":{"base_training_rows":len(base),"approved_corrections_seen":len(approved),
                      "approved_binary_corrections_included":len(included),
                      "approved_corrections_excluded":len(excluded),"final_training_rows":len(final)},
            "exclusions":dict(sorted(reasons.items())),
            "protected_set":protected_detail,
            "label_mapping":{"Not Humorous":0,"Humorous":1,
                             "Ambiguous / Uncertain":"excluded; abstention outcome, not a binary training class"},
            "output":{"candidate_train":str(dataset_path),"corrections_included":str(inc_path),
                      "corrections_excluded":str(exc_path),"manifest":str(manifest_path),"sha256":digest},
            "research_guards":["validation.csv protected","test.csv protected","data/ood CSV text protected",
                               "ambiguous corrections excluded","conflicting corrections excluded",
                               "original train.csv not modified"],
            "ready_for_candidate_training":len(included)>0,
        }
        write_json(manifest,manifest_path)
        complete_dataset_build_run(run_id,dataset_path,manifest_path,len(included),len(excluded),len(final),digest)
        return manifest
    except Exception as exc:
        fail_dataset_build_run(run_id,str(exc))
        raise
