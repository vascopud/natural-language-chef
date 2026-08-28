from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_training_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def create_train_val_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["chef_id"],
    )

    return train_df, val_df
