"""Module de nettoyage, transformation et préparation des features."""

from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_raw_dataframe(df: pd.DataFrame, id_col: str, target_col: str) -> pd.DataFrame:
    """Nettoie le DataFrame brut en supprimant les identifiants et encodant la cible.

    Args:
        df: DataFrame brut d'entrée.
        id_col: Nom de la colonne d'identifiant à exclure.
        target_col: Nom de la variable cible.

    Returns:
        pd.DataFrame: DataFrame nettoyé avec cible binaire (0 ou 1).
    """
    data = df.copy()
    if id_col in data.columns:
        data = data.drop(columns=[id_col])

    if target_col in data.columns:
        data[target_col] = data[target_col].map({"Yes": 1, "No": 0})

    return data


def encode_features(
    df: pd.DataFrame,
    binary_cols: List[str],
    categorical_cols: List[str],
    numerical_cols: List[str],
) -> pd.DataFrame:
    """Encode les variables catégorielles et normalise les valeurs numériques.

    Args:
        df: DataFrame contenant les colonnes nettoyées.
        binary_cols: Liste des colonnes binaires (Yes/No, Male/Female).
        categorical_cols: Liste des colonnes multi-classes à One-Hot encoder.
        numerical_cols: Liste des colonnes quantitatives à normaliser.

    Returns:
        pd.DataFrame: Matrice des features prêtes pour l'entraînement.
    """
    data = df.copy()

    # Encodage binaire
    binary_mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].map(binary_mapping).fillna(0)

    # One-Hot Encoding
    existing_cat = [c for c in categorical_cols if c in data.columns]
    if existing_cat:
        data = pd.get_dummies(data, columns=existing_cat, drop_first=True, dtype=int)

    # Imputation médiane et Normalisation Min-Max
    for col in numerical_cols:
        if col in data.columns:
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            c_min = data[col].min()
            c_max = data[col].max()
            if c_max > c_min:
                data[col] = (data[col] - c_min) / (c_max - c_min)
            else:
                data[col] = 0.0

    return data


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sépare les features et la cible en jeux d'entraînement et de test.

    Args:
        df: DataFrame complet préparé.
        target_col: Nom de la variable cible.
        test_size: Proportion de l'échantillon de test (entre 0.0 et 1.0).
        random_state: Graine pseudo-aléatoire pour la reproductibilité.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
