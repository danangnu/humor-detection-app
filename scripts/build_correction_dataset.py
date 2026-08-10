from __future__ import annotations
import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description="Build Milestone 2.2A candidate dataset from approved corrections.")
    p.add_argument("--db",type=Path,default=None,help="Optional humor_feedback.db path, e.g. packaged EXE database.")
    p.add_argument("--output-root",type=Path,default=None)
    a=p.parse_args()
    if a.db: os.environ["HUMOR_FEEDBACK_DB"]=str(a.db.expanduser().resolve())
    from training.correction_dataset import build_correction_dataset
    print("="*68); print("Humor Bot - Milestone 2.2A"); print("Approved Corrections -> Candidate Training Dataset"); print("="*68)
    m=build_correction_dataset(ROOT,a.output_root); c=m["counts"]
    print(f"\nRun ID:               {m['run_id']}")
    print(f"Base training rows:   {c['base_training_rows']:,}")
    print(f"Approved corrections: {c['approved_corrections_seen']:,}")
    print(f"Included corrections: {c['approved_binary_corrections_included']:,}")
    print(f"Excluded corrections: {c['approved_corrections_excluded']:,}")
    print(f"Final training rows:  {c['final_training_rows']:,}")
    if m["exclusions"]:
        print("\nExclusions:")
        for k,v in m["exclusions"].items(): print(f"  {k}: {v}")
    print("\nOutput:")
    for k,v in m["output"].items():
        if k!="sha256": print(f"  {k}: {v}")
    print("\n"+("READY for Milestone 2.2B candidate training." if m["ready_for_candidate_training"]
                  else "NOT READY: no approved binary correction was eligible."))
    print("The active model was not changed.")

if __name__=="__main__": main()
