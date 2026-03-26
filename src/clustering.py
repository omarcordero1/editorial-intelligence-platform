"""
Módulo de clustering y reducción dimensional.
KMeans + PCA para segmentación de editores.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

from .config import (
    CLUSTER_LABELS,
    FEATURES_CLUSTERING,
    K_DEFAULT,
    K_RANGE,
    N_INIT,
    PCA_COMPONENTS,
    RANDOM_STATE,
    WINSORIZE_LOWER,
    WINSORIZE_UPPER,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Contenedor de resultados del pipeline de clustering."""
    df_clustered: pd.DataFrame
    k_optimal: int
    silhouette_scores: list[float]
    inertias: list[float]
    pca_variance_ratio: list[float]
    cluster_to_label: dict = field(default_factory=dict)
    scaler: RobustScaler = field(default_factory=RobustScaler)
    pca: PCA = field(default_factory=lambda: PCA(n_components=PCA_COMPONENTS))


def _winsorize_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Winsoriza las features de clustering in-place (columnas _winsorized)."""
    df = df.copy()
    for feat in features:
        lo = df[feat].quantile(WINSORIZE_LOWER / 100)
        hi = df[feat].quantile(WINSORIZE_UPPER / 100)
        df[f"{feat}_w"] = df[feat].clip(lower=lo, upper=hi)
        n_out = ((df[feat] < lo) | (df[feat] > hi)).sum()
        logger.debug("Winsorize '%s': %s outliers tratados", feat, n_out)
    return df


def run_clustering(metricas_editor: pd.DataFrame, k: int | None = None) -> ClusteringResult:
    """
    Ejecuta el pipeline completo: winsorización → escalado → Elbow/Silhouette → KMeans → PCA.

    Args:
        metricas_editor: DataFrame de features por editor (salida de `features.build_editor_features`).
        k: Número de clusters. Si None, se determina automáticamente (K_DEFAULT).

    Returns:
        ClusteringResult con el DataFrame enriquecido y metadatos del modelo.
    """
    df = metricas_editor.copy()

    # Verificar features disponibles
    available = [f for f in FEATURES_CLUSTERING if f in df.columns]
    missing = [f for f in FEATURES_CLUSTERING if f not in df.columns]
    if missing:
        logger.warning("Features ausentes: %s. Se usarán las disponibles.", missing)

    # ── Winsorización ────────────────────────────────────────────────────────
    df_w = _winsorize_features(df, available)
    feat_w = [f"{f}_w" for f in available]
    X = df_w[feat_w].values

    # ── Escalado con RobustScaler ────────────────────────────────────────────
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Análisis Elbow + Silhouette ──────────────────────────────────────────
    logger.info("Calculando K óptimo (rango %s)...", list(K_RANGE))
    inertias, sil_scores = [], []
    for ki in K_RANGE:
        km = KMeans(n_clusters=ki, random_state=RANDOM_STATE, n_init=N_INIT)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))

    # ── Selección de K ───────────────────────────────────────────────────────
    k_optimal = k if k is not None else K_DEFAULT
    logger.info(
        "K seleccionado: %s · Silhouette: %.3f",
        k_optimal,
        sil_scores[k_optimal - min(K_RANGE)],
    )

    # ── Modelo final ─────────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=k_optimal, random_state=RANDOM_STATE, n_init=N_INIT)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # ── Asignar etiquetas narrativas (ordenadas por score_global desc) ───────
    cluster_stats = (
        df.groupby("cluster")["score_global"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    labels = CLUSTER_LABELS[:k_optimal]
    cluster_to_label = dict(zip(cluster_stats["cluster"], labels))
    df["etiqueta"] = df["cluster"].map(cluster_to_label)

    logger.info("Distribución de clusters:")
    for lbl in labels:
        n = (df["etiqueta"] == lbl).sum()
        logger.info("  %-28s → %s editores", lbl, n)

    # ── PCA para visualización ───────────────────────────────────────────────
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    df["pca_1"] = X_pca[:, 0]
    df["pca_2"] = X_pca[:, 1]
    logger.info(
        "PCA varianza explicada: PC1=%.1f%% PC2=%.1f%% Total=%.1f%%",
        pca.explained_variance_ratio_[0] * 100,
        pca.explained_variance_ratio_[1] * 100,
        sum(pca.explained_variance_ratio_) * 100,
    )

    return ClusteringResult(
        df_clustered=df,
        k_optimal=k_optimal,
        silhouette_scores=sil_scores,
        inertias=inertias,
        pca_variance_ratio=pca.explained_variance_ratio_.tolist(),
        cluster_to_label=cluster_to_label,
        scaler=scaler,
        pca=pca,
    )
