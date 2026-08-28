import pandas as pd

from chef_classifier.data import clean_training_data


def test_clean_training_data_removes_duplicates() -> None:
    df = pd.DataFrame(
        {
            "chef_id": [1, 1, 2],
            "description": ["a", "a", "b"],
        }
    )

    result = clean_training_data(df)

    assert len(result) == 2
