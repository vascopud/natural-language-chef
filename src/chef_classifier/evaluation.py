from collections.abc import Sequence

from sklearn.metrics import accuracy_score


def calculate_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    return accuracy_score(y_true, y_pred)
