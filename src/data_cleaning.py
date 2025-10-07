"""
data_cleaning.py

Contains functions for loading, inspecting, and cleaning the laptop dataset.
"""

import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """
    Load the dataset from a given CSV file path.

    Parameters:
        path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning steps such as handling missing values,
    converting data types, and removing duplicates.

    Parameters:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Cleaned dataset ready for analysis.
    """
    # Example steps — replace or expand based on your notebook
    df = df.drop_duplicates()
    df = df.dropna(subset=["Price"])
    df["RAM"] = df["RAM"].str.replace("GB", "").astype(float)
    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess numerical and categorical features for modeling.

    Parameters:
        df (pd.DataFrame): Cleaned dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Encode categorical features, scale numeric columns, etc.
    return df

