from pathlib import Path

import pandas as pd

from chef_classifier.data import (
    clean_training_data,
    load_training_data,
)
from chef_classifier.features import combine_text_fields
from chef_classifier.models import build_final_model


def generate_predictions(
    train_path: Path,
    test_path: Path,
    output_path: Path,
) -> None:
    train = load_training_data(train_path)
    train = clean_training_data(train)

    test = pd.read_csv(test_path, sep=";")

    fields = ["description", "tags"]

    X_train = combine_text_fields(train, fields)
    y_train = train["chef_id"]

    X_test = combine_text_fields(test, fields)

    model = build_final_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    pd.Series(predictions).to_csv(
        output_path,
        index=False,
        header=False,
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    train_path = project_root / "data" / "raw" / "train.csv"
    test_path = project_root / "data" / "raw" / "test-no-labels.csv"
    output_path = project_root / "outputs" / "results.txt"

    generate_predictions(
        train_path=train_path,
        test_path=test_path,
        output_path=output_path,
    )
