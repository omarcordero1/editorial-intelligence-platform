"""
Módulo de limpieza y preprocesamiento de datos editoriales.
"""
import logging

import pandas as pd

from .config import EDITORS_TO_EXCLUDE, MIN_NOTAS_POR_EDITOR, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de limpieza del dataset editorial.

    Pasos:
        1. Eliminar filas con nulos en métricas clave.
        2. Normalizar nombres de Editor y Autor.
        3. Filtrar editores con pocas notas.
        4. Excluir editores específicos (outliers conocidos).
        5. Convertir métricas a numérico.

    Args:
        df: DataFrame crudo cargado por `data_loader.load_editorial_data`.

    Returns:
        DataFrame limpio.
    """
    original_size = len(df)
    logger.info("Iniciando limpieza · %s filas de entrada", f"{original_size:,}")

    # 1. Eliminar nulos en métricas clave
    df_clean = df.dropna(subset=REQUIRED_COLUMNS).copy()
    logger.info(
        "Tras eliminar nulos: %s filas (%.1f%% eliminado)",
        f"{len(df_clean):,}",
        (1 - len(df_clean) / original_size) * 100,
    )

    # 2. Normalizar strings
    df_clean["Editor"] = df_clean["Editor"].str.strip().str.lower()
    df_clean["Autor"] = df_clean["Autor"].str.strip().str.lower()

    # 3. Filtrar editores con mínimo de notas
    notas_por_editor = df_clean.groupby("Editor").size()
    editores_validos = notas_por_editor[notas_por_editor >= MIN_NOTAS_POR_EDITOR].index
    df_clean = df_clean[df_clean["Editor"].isin(editores_validos)]
    logger.info(
        "Editores con %s+ notas: %s · notas válidas: %s",
        MIN_NOTAS_POR_EDITOR,
        len(editores_validos),
        f"{len(df_clean):,}",
    )

    # 4. Excluir editores problemáticos
    excluded = [e.strip() for e in EDITORS_TO_EXCLUDE if e.strip()]
    for editor in excluded:
        if editor in df_clean["Editor"].values:
            df_clean = df_clean[df_clean["Editor"] != editor]
            logger.info("Editor excluido: '%s'", editor)

    # 5. Convertir a numérico
    metricas = ["Registros", "PVs", "Scroll", "RFV", "AdsPorPagina"]
    for col in metricas:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean = df_clean.dropna(subset=metricas)

    logger.info(
        "Dataset limpio final: %s notas · %s editores únicos",
        f"{len(df_clean):,}",
        df_clean["Editor"].nunique(),
    )

    return df_clean
