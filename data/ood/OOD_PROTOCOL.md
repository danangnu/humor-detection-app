# Reddit OOD Evaluation Protocol

This set is an independent stress test for the Humor Bot model.

## Freeze rule

Do not use this set to change the model or the frozen 0.20 / 0.80 thresholds
and then continue to call the same set an independent final test.

## First pass: provisional source labels

`reddit_ood_provisional.csv` uses the Reddit dataset's `niche` field:

- humor -> provisional humor
- selected non-humor niches -> provisional non-humor

This is useful for a quick stress test, but it is not the strongest final
client-facing evidence.

## Final pass: human-reviewed labels

Use `reddit_ood_annotation.csv`.

Two reviewers independently label every example as:

- humor
- not_humor
- ambiguous

They should not see `reddit_ood_key.csv` during annotation.

After both rounds, adjudicate disagreements. Leave genuinely unresolved cases
as ambiguous instead of forcing a binary label.

The evaluator reports observed raw agreement and Cohen's kappa. There is no
assumed minimum kappa target.

## Commands

Prepare the frozen sample once:

```powershell
python .\training\prepare_reddit_ood.py --per-class 200
```

Run the provisional stress test:

```powershell
python .\training\evaluate_reddit_ood.py --labels source
```

After human annotation:

```powershell
python .\training\evaluate_reddit_ood.py --labels adjudicated
```
