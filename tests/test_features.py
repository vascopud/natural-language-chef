import pandas as pd

from chef_classifier.features import combine_text_fields


def test_combine_text_fields() -> None:
    df = pd.DataFrame(
        {
            "description": ["spicy soup", "sweet cake"],
            "tags": ["quick dinner", "dessert baking"],
        }
    )

    result = combine_text_fields(
        df,
        ["description", "tags"],
    )

    assert result.tolist() == [
        "spicy soup quick dinner",
        "sweet cake dessert baking",
    ]
