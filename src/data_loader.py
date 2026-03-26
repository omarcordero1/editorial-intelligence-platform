"""
Módulo de carga de datos.
Responsable de leer el CSV fuente y validar estructura básica.
"""
import logging
from pathlib import Path

import pandas as pd

from .config import COLUMN_RENAME_MAP, DATA_ENCODING, DATE_COLUMN, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Error al cargar el dataset editorial."""


def load_editorial_data(filepath: str | Path) -> pd.DataFrame:
    """
    Carga el CSV de datos editoriales y realiza validaciones básicas.

    Args:
        filepath: Ruta al archivo CSV.

    Returns:
        DataFrame con los datos cargados y columnas renombradas.

    Raises:
        DataLoadError: Si el archivo no existe o faltan columnas obligatorias.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise DataLoadError(f"Archivo no encontrado: {filepath}")

    logger.info("Cargando dataset: %s", filepath.name)

    try:
        df = pd.read_csv(filepath, encoding=DATA_ENCODING)
    except UnicodeDecodeError:
        logger.warning("Encoding '%s' falló. Reintentando con 'utf-8'.", DATA_ENCODING)
        df = pd.read_csv(filepath, encoding="utf-8")

    # Renombrar columnas
    df = df.rename(columns=COLUMN_RENAME_MAP)

    # Parsear fechas
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    # Validar columnas obligatorias
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataLoadError(
            f"Columnas obligatorias ausentes: {missing}. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    logger.info(
        "Dataset cargado: %s filas · %s columnas · rango %s → %s",
        f"{len(df):,}",
        df.shape[1],
        df[DATE_COLUMN].min().date() if DATE_COLUMN in df.columns else "N/A",
        df[DATE_COLUMN].max().date() if DATE_COLUMN in df.columns else "N/A",
    )

    return df
