from training.candidate_gate import promotion_gate


def m(acc, f1):
    return {
        "binary_at_0_5": {
            "accuracy": acc,
            "f1": f1,
        }
    }


def test_pass():
    result = promotion_gate(
        active_validation=m(.90, .92),
        candidate_validation=m(.901, .921),
        correction_retention={
            "size": 1,
            "accuracy": 1.0,
        },
        tolerance=.005,
    )
    assert result["passed"]


def test_fail_retention():
    result = promotion_gate(
        active_validation=m(.90, .92),
        candidate_validation=m(.901, .921),
        correction_retention={
            "size": 1,
            "accuracy": 0.0,
        },
        tolerance=.005,
    )
    assert not result["passed"]


def test_fail_regression():
    result = promotion_gate(
        active_validation=m(.90, .92),
        candidate_validation=m(.88, .90),
        correction_retention={
            "size": 1,
            "accuracy": 1.0,
        },
        tolerance=.005,
    )
    assert not result["passed"]
