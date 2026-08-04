from core import HumorClassifier, HumorThresholds


def test_three_state_policy():
    c = HumorClassifier()
    c.thresholds = HumorThresholds(low=0.40, high=0.60)
    assert c._decision(0.20)[0] == "Not Humorous"
    assert c._decision(0.50)[0] == "Ambiguous / Uncertain"
    assert c._decision(0.80)[0] == "Humorous"
