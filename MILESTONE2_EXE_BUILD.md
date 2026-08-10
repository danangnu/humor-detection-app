# Humor Bot Milestone 2 — Final Windows EXE Build

This build is different from the earlier Milestone 1 package.

Milestone 2 needs writable model state for:

- corrections
- candidate retraining
- promotion
- model history
- rollback

Therefore the package keeps PyInstaller internals read-only and places mutable
runtime state under:

```text
%LOCALAPPDATA%\HumorBot
```

## 1. Copy the patch files

Replace:

```text
app.py
training\candidate_trainer.py
```

Add:

```text
desktop_launcher.py
HumorBot_Milestone2.spec
build_milestone2_exe.ps1
smoke_test_milestone2_exe.ps1
```

Keep the Milestone 2.3 application files:

```text
feedback_store.py
model_admin.py

training\correction_dataset.py
training\candidate_gate.py

static\index.html
static\style.css
static\script.js
static\admin.html
static\admin.js
static\admin_m2_3.css
```

## 2. Why candidate_trainer.py changed

The packaged retraining code now uses a small `torch.utils.data.Dataset`
instead of Hugging Face `datasets.Dataset`.

This removes the runtime requirement for the large `datasets` + `pyarrow`
stack while keeping the same Transformer `Trainer`, replay policy,
validation metrics and promotion gate.

## 3. Build

From the project root:

```powershell
cd D:\Freelancer\Ubuntu\Humour\humor_bot

Set-ExecutionPolicy -Scope Process Bypass

Unblock-File .\build_milestone2_exe.ps1
Unblock-File .\smoke_test_milestone2_exe.ps1

.\build_milestone2_exe.ps1
```

## 4. Output

Executable folder:

```text
dist\HumorBot\
```

Client ZIP:

```text
release\HumorBot_Milestone2.zip
```

The folder contains:

```text
HumorBot\
├── HumorBot.exe
├── README_CLIENT.txt
├── .env.example        (if present in source)
└── i\
    ├── bootstrap\
    │   ├── models\humor_transformer\
    │   └── data\
    └── ...
```

The `i` folder is required. It is intentionally short to reduce long-path risk.

## 5. Client installation path

Recommend:

```text
C:\HumorBot
```

Avoid deeply nested paths.

## 6. Persistent runtime data

First startup creates:

```text
%LOCALAPPDATA%\HumorBot\
├── humor_feedback.db
├── models\
│   ├── humor_transformer\
│   ├── candidates\
│   └── model_history\
├── data\
│   ├── processed\
│   ├── ood\
│   ├── retraining\
│   └── admin\
└── logs\
    └── HumorBot.log
```

This is the important Milestone 2 packaging change.

Promotion and rollback modify the runtime model under LocalAppData, not files
inside the PyInstaller `i` directory.

## 7. Smoke test

Keep port 8000 free:

```powershell
.\smoke_test_milestone2_exe.ps1
```

The script verifies:

- EXE starts
- `/health` responds
- local Transformer is ready
- frozen thresholds are loaded
- `/admin/status` responds

## 8. Full acceptance test

After the smoke test, manually confirm:

1. Analyze a sentence.
2. Click `Incorrect result?`.
3. Save a binary correction.
4. Open `/admin`.
5. Approve the correction.
6. Click `Train Candidate`.
7. Wait for candidate status to complete.
8. Confirm the promotion gate.
9. Promote a passing candidate.
10. Close and reopen `HumorBot.exe`.
11. Verify the promoted behavior.
12. Roll back from `/admin`.
13. Close and reopen again.
14. Verify the previous model was restored.

## 9. Clean-machine simulation

Before testing a first-run experience, close the EXE and rename:

```text
%LOCALAPPDATA%\HumorBot
```

Do not delete it if it contains real corrections or model history.

## Suggested Git commit

```text
build: package Milestone 2 with persistent retraining and rollback data
```
