"""
Módulo de ingeniería de características.
Genera métricas derivadas y el score global por editor.
"""
import logging
from collections import Counter

import numpy as np
import pandas as pd

from .config import SCORE_WEIGHTS

logger = logging.getLogger(__name__)


# ─── Funciones auxiliares ─────────────────────────────────────────────────────

def _shannon_entropy(series: pd.Series) -> float:
    """Calcula la entropía de Shannon de una serie categórica."""
    if len(series) == 0:
        return 0.0
    counts = Counter(series)
    total = len(series)
    return -sum(
        (c / total) * np.log2(c / total) for c in counts.values() if c > 0
    )


def _normalize_series(series: pd.Series) -> pd.Series:
    """Min-Max normalization → [0, 1]."""
    rng = series.max() - series.min()
    return (series - series.min()) / rng if rng > 0 else pd.Series(0.0, index=series.index)


def _winsorize(series: pd.Series, lower_pct: int = 5, upper_pct: int = 95) -> pd.Series:
    """Winsorization: recorta extremos a los percentiles indicados."""
    lo = series.quantile(lower_pct / 100)
    hi = series.quantile(upper_pct / 100)
    return series.clip(lower=lo, upper=hi)


# ─── Pipeline principal ───────────────────────────────────────────────────────

def build_editor_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega métricas por editor y calcula features derivadas.

    Args:
        df_clean: DataFrame limpio (salida de `preprocessing.clean_data`).

    Returns:
        DataFrame con una fila por editor y todas sus métricas.
    """
    logger.info("Construyendo features por editor...")

    # ── Agregación base ──────────────────────────────────────────────────────
    agg = df_clean.groupby("Editor").agg(
        Registros=("Registros", "sum"),
        PVs=("PVs", "sum"),
        Scroll=("Scroll", "mean"),
        RFV=("RFV", "mean"),
        AdsPorPagina=("AdsPorPagina", "mean"),
        num_notas=("Registros", "count"),
    ).reset_index()

    # ── Feature 1: Eficiencia registros por PV ───────────────────────────────
    agg["eficiencia_registros_pv"] = np.where(
        agg["PVs"] > 0,
        agg["Registros"] / agg["PVs"] * 1_000,
        0,
    )

    # ── Feature 2: Consistencia editorial (1 - CV) ───────────────────────────
    consistencia = (
        df_clean.groupby("Editor")["Registros"]
        .agg(["std", "mean"])
        .reset_index()
    )
    consistencia["consistencia_editorial"] = np.where(
        consistencia["mean"] > 0,
        (1 - consistencia["std"] / consistencia["mean"]).clip(lower=0),
        0,
    )
    agg = agg.merge(consistencia[["Editor", "consistencia_editorial"]], on="Editor")

    # ── Feature 3: Índice de originalidad ───────────────────────────────────
    orig = df_clean.copy()
    orig["es_original"] = (orig["Editor"] == orig["Autor"]).astype(int)
    idx_orig = orig.groupby("Editor")["es_original"].mean().reset_index()
    idx_orig.columns = ["Editor", "indice_originalidad"]
    agg = agg.merge(idx_orig, on="Editor")

    # ── Feature 4: Diversidad temática (entropía de Shannon) ─────────────────
    if "tema" in df_clean.columns:
        diversidad = (
            df_clean.groupby("Editor")["tema"]
            .apply(_shannon_entropy)
            .reset_index()
        )
        diversidad.columns = ["Editor", "diversidad_tematica"]
    else:
        logger.warning("Columna 'tema' no encontrada. Diversidad = 0.")
        diversidad = pd.DataFrame(
            {"Editor": agg["Editor"], "diversidad_tematica": 0.0}
        )
    agg = agg.merge(diversidad, on="Editor")

    # ── Score global ponderado ───────────────────────────────────────────────
    agg["registros_norm"] = _normalize_series(agg["Registros"])
    agg["pvs_norm"] = _normalize_series(agg["PVs"])
    agg["scroll_norm"] = _normalize_series(agg["Scroll"])
    agg["rfv_norm"] = _normalize_series(agg["RFV"])
    agg["eficiencia_norm"] = _normalize_series(agg["eficiencia_registros_pv"])

    agg["score_global"] = (
        agg["registros_norm"] * SCORE_WEIGHTS["registros_norm"]
        + agg["pvs_norm"] * SCORE_WEIGHTS["pvs_norm"]
        + agg["scroll_norm"] * SCORE_WEIGHTS["scroll_norm"]
        + agg["rfv_norm"] * SCORE_WEIGHTS["rfv_norm"]
        + agg["eficiencia_norm"] * SCORE_WEIGHTS["eficiencia_norm"]
    ) * 100  # escalar a 0-100

    logger.info(
        "Features construidas para %s editores · score_global μ=%.1f",
        len(agg),
        agg["score_global"].mean(),
    )

    return agg


def get_top_temas(df_clean: pd.DataFrame, editor: str, top_n: int = 5) -> list[dict]:
    """Retorna los top N temas de un editor con porcentajes."""
    if "tema" not in df_clean.columns:
        return []
    notas = df_clean[df_clean["Editor"] == editor]
    total = len(notas)
    return [
        {"tema": tema, "num_notas": count, "porcentaje": count / total * 100}
        for tema, count in notas["tema"].value_counts().head(top_n).items()
    ]


def get_top_entidades(df_clean: pd.DataFrame, editor: str, top_n: int = 10) -> list[dict]:
    """Retorna las top N entidades (personaje_principal) de un editor."""
    if "personaje_principal" not in df_clean.columns:
        return []
    notas = df_clean[df_clean["Editor"] == editor]
    return [
        {"entidad": entidad, "num_notas": count}
        for entidad, count in notas["personaje_principal"]
        .dropna()
        .value_counts()
        .head(top_n)
        .items()
    ]
