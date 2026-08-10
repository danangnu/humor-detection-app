from __future__ import annotations


def promotion_gate(
    *,
    active_validation: dict,
    candidate_validation: dict,
    correction_retention: dict,
    tolerance: float = 0.005,
) -> dict:
    """
    Promotion gate uses validation + correction retention only.

    Held-out test results remain report-only so repeated candidate runs do not
    silently turn the test set into development data.
    """
    active_f1 = float(active_validation["binary_at_0_5"]["f1"])
    candidate_f1 = float(candidate_validation["binary_at_0_5"]["f1"])

    active_accuracy = float(
        active_validation["binary_at_0_5"]["accuracy"]
    )
    candidate_accuracy = float(
        candidate_validation["binary_at_0_5"]["accuracy"]
    )

    size = int(correction_retention.get("size", 0))
    retention_accuracy = correction_retention.get("accuracy")

    checks = {
        "validation_f1_not_worse_than_tolerance":
            candidate_f1 >= active_f1 - tolerance,

        "validation_accuracy_not_worse_than_tolerance":
            candidate_accuracy >= active_accuracy - tolerance,

        "eligible_corrections_present":
            size > 0,

        "eligible_corrections_retained":
            retention_accuracy == 1.0 if size > 0 else False,
    }

    return {
        "passed": all(checks.values()),
        "tolerance": float(tolerance),
        "checks": checks,
        "note": (
            "Gate uses validation performance and correction retention only. "
            "Held-out test metrics are report-only."
        ),
    }
