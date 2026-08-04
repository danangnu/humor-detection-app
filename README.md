# Humor Bot — Milestone 1

Standalone humor classification prototype.

## Milestone 1 goal

Build a working local application that can:

- accept short text in a browser UI;
- run a Transformer humor classifier;
- return **Humorous**, **Not Humorous**, or **Ambiguous / Uncertain**;
- show the model probability and classification confidence;
- expose `/health`, `/analyze`, and `/chat`;
- optionally use Gemini for a short reply, with a local fallback;
- prepare, train, calibrate, and evaluate our own Transformer model.

The ambiguous state is an abstention policy, not a third training label.

## Dataset baseline

`training/prepare_socket_dataset.py` uses the `hahackathon#is_humor` task from the SocKET benchmark. It supplies official train, validation, and test splits with binary labels:

- `0 = not humor`
- `1 = humor`

The test split is kept out of training and threshold selection.

A separate out-of-distribution dataset is deliberately kept outside this first training loop. It belongs in `data/ood/` and must remain independent.

## Windows setup

```powershell
cd D:\Freelancer\Ubuntu\Humor\humor_bot
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
```

## Prepare the dataset

```powershell
python .\training\prepare_socket_dataset.py
```

Expected files:

```text
data\processed\train.csv
data\processed\validation.csv
data\processed\test.csv
```

## Train the project model

```powershell
python .\training\train_transformer.py
```

Base model: `distilroberta-base`

Output: `models\humor_transformer\`

## Select the uncertain band

```powershell
python .\training\calibrate_thresholds.py
```

The threshold script uses **validation data only**. It searches for a low/high pair that maximizes accuracy on classified examples while retaining at least 80% coverage. It does not inspect the test set.

Output: `models\humor_transformer\thresholds.json`

## Run the untouched in-domain test

```powershell
python .\training\evaluate_test.py
```

Output: `models\humor_transformer\test_metrics.json`

## Run the application

```powershell
.\run.ps1
```

Open:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/health
```

## Bootstrap mode

Before our own model is trained, the app may use `VitalContribution/JokeDetectBERT` so we can test the FastAPI and UI wiring. This is only a development bootstrap and must not be presented as the model trained by this project.

After `models\humor_transformer\config.json` exists, the app automatically prefers the local project model.

## Milestone 1 acceptance

Milestone 1 is ready for client review when:

1. dataset preparation completes;
2. our local Transformer model has been trained;
3. thresholds are selected using validation data only;
4. untouched test metrics have been generated;
5. `/health` reports `local_model_ready: true`;
6. the UI handles clear humorous, clear non-humorous, and uncertain examples;
7. Gemini works when configured and the local fallback works without it.

## Later milestones

The following are intentionally not part of the first build:

- user correction storage;
- admin correction review;
- candidate retraining controls;
- candidate promotion / rollback;
- Windows EXE packaging;
- final OOD evaluation;
- human evaluation of generated jokes or replies.
