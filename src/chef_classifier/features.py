import pandas as pd


def combine_text_fields(
    df: pd.DataFrame,
    fields: list[str],
) -> pd.Series:
    return df[fields].astype(str).agg(" ".join, axis=1)
