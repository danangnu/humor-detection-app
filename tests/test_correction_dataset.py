from pathlib import Path
import pandas as pd
from feedback_store import add_correction,initialize_database,update_correction
from training.correction_dataset import build_correction_dataset

def split(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=["text","label"]).to_csv(path,index=False)

def approve(text,orig,corr):
    r=add_correction(input_text=text,original_label=orig,original_score=.5,
                     corrected_label=corr,source="test",model_source="unit")
    update_correction(r["id"],status="approved")

def test_builder(tmp_path,monkeypatch):
    monkeypatch.setenv("HUMOR_FEEDBACK_DB",str(tmp_path/"data"/"humor_feedback.db"))
    split(tmp_path/"data"/"processed"/"train.csv",[("base humor",1),("base plain",0)])
    split(tmp_path/"data"/"processed"/"validation.csv",[("protected val",1)])
    split(tmp_path/"data"/"processed"/"test.csv",[("protected test",0)])
    ood=tmp_path/"data"/"ood"; ood.mkdir(parents=True)
    pd.DataFrame([{"text":"protected reddit"}]).to_csv(ood/"reddit.csv",index=False)
    initialize_database()
    approve("new joke","Not Humorous","Humorous")
    approve("new plain","Humorous","Not Humorous")
    approve("borderline","Humorous","Ambiguous / Uncertain")
    approve("protected test","Humorous","Not Humorous")
    approve("protected reddit","Not Humorous","Humorous")
    approve("base humor","Not Humorous","Humorous")
    m=build_correction_dataset(tmp_path)
    assert m["counts"]["approved_binary_corrections_included"]==2
    assert m["counts"]["final_training_rows"]==4
    d=Path(m["output"]["candidate_train"]).parent
    inc=pd.read_csv(d/"corrections_included.csv")
    exc=pd.read_csv(d/"corrections_excluded.csv")
    assert set(inc["text"])=={"new joke","new plain"}
    assert "ambiguous_not_a_binary_training_label" in set(exc["exclusion_reason"])
    assert "protected_evaluation_overlap" in set(exc["exclusion_reason"])
    assert "already_in_base_training" in set(exc["exclusion_reason"])
