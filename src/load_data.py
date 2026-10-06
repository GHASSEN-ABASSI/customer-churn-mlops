"""Module de chargement et de validation des données brutes."""

from pathlib import Path
from typing import Union

import pandas as pd


def load_raw_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Charge le fichier CSV de données brutes et valide son existence.

    Args:
        file_path: Chemin relatif ou absolu vers le fichier CSV.

    Returns:
        pd.DataFrame: Jeu de données chargé en mémoire.

    Raises:
        FileNotFoundError: Si le fichier spécifié est introuvable.
        ValueError: Si le fichier est vide.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier de données est introuvable à l'adresse : {path.resolve()}"
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Le fichier {path} est vide.")

    return df
