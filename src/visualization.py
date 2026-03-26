"""
Módulo de visualización con Plotly.
Genera todas las gráficas del dashboard editorial.
"""
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CLUSTER_COLORS, K_RANGE

logger = logging.getLogger(__name__)

FONT_FAMILY = "Segoe UI, Helvetica Neue, Arial, sans-serif"
TEMPLATE = "plotly_white"


# ─── Elbow + Silhouette ───────────────────────────────────────────────────────

def plot_elbow_silhouette(inertias: list, silhouette_scores: list) -> go.Figure:
    """Gráfica de método del codo y silhouette score."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📉 Método del Codo", "📊 Silhouette Score"),
    )
    k_vals = list(K_RANGE)

    fig.add_trace(
        go.Scatter(
            x=k_vals, y=inertias, mode="lines+markers",
            name="Inercia", line=dict(color="#667eea", width=3),
            marker=dict(size=8),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=k_vals, y=silhouette_scores, mode="lines+markers",
            name="Silhouette", line=dict(color="#f6a623", width=3),
            marker=dict(size=8),
        ),
        row=1, col=2,
    )

    fig.update_layout(
        height=380, template=TEMPLATE,
        title_text="Selección del número óptimo de clusters",
        font=dict(family=FONT_FAMILY),
        showlegend=False,
    )
    return fig


# ─── Scatter PCA ─────────────────────────────────────────────────────────────

def plot_pca_scatter(df: pd.DataFrame) -> go.Figure:
    """Mapa de editores en el espacio PCA con colores por etiqueta."""
    fig = px.scatter(
        df, x="pca_1", y="pca_2",
        color="etiqueta",
        color_discrete_map=CLUSTER_COLORS,
        hover_name="Editor",
        hover_data={
            "score_global": ":.1f",
            "num_notas": True,
            "etiqueta": False,
            "pca_1": False,
            "pca_2": False,
        },
        title="🗺️ Mapa de Editores — Espacio PCA",
        labels={"pca_1": "Componente Principal 1", "pca_2": "Componente Principal 2"},
        template=TEMPLATE,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.85))
    fig.update_layout(
        font=dict(family=FONT_FAMILY),
        legend_title_text="Segmento",
        height=480,
    )
    return fig


# ─── Distribución de clusters ─────────────────────────────────────────────────

def plot_cluster_distribution(df: pd.DataFrame) -> go.Figure:
    """Gráfica de barras de editores por cluster."""
    dist = df.groupby("etiqueta").size().reset_index(name="Editores")
    dist = dist.sort_values("Editores", ascending=False)

    fig = px.bar(
        dist, x="etiqueta", y="Editores",
        color="etiqueta", color_discrete_map=CLUSTER_COLORS,
        title="👥 Distribución de Editores por Segmento",
        labels={"etiqueta": "Segmento", "Editores": "Número de Editores"},
        text="Editores",
        template=TEMPLATE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        font=dict(family=FONT_FAMILY),
        height=380,
    )
    return fig


# ─── Score global por editor ──────────────────────────────────────────────────

def plot_score_ranking(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Top N editores por score global."""
    top = df.nlargest(top_n, "score_global")

    fig = px.bar(
        top.sort_values("score_global"),
        x="score_global", y="Editor",
        color="etiqueta", color_discrete_map=CLUSTER_COLORS,
        orientation="h",
        title=f"🏆 Top {top_n} Editores por Score Global",
        labels={"score_global": "Score Global (0-100)", "Editor": ""},
        text="score_global",
        template=TEMPLATE,
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        showlegend=True,
        legend_title_text="Segmento",
        font=dict(family=FONT_FAMILY),
        height=max(400, top_n * 24),
    )
    return fig


# ─── Radar por cluster ────────────────────────────────────────────────────────

def plot_cluster_radar(df: pd.DataFrame) -> go.Figure:
    """Radar chart comparativo de métricas por segmento."""
    from .config import FEATURES_CLUSTERING
    import numpy as np

    dims = ["Scroll", "RFV", "eficiencia_registros_pv", "indice_originalidad", "diversidad_tematica"]
    dims = [d for d in dims if d in df.columns]

    cluster_means = df.groupby("etiqueta")[dims].mean()

    # Min-Max normalizar
    for col in dims:
        rng = cluster_means[col].max() - cluster_means[col].min()
        cluster_means[col] = (
            (cluster_means[col] - cluster_means[col].min()) / rng
            if rng > 0 else 0
        ) * 100

    fig = go.Figure()
    for etiqueta, row in cluster_means.iterrows():
        vals = row.tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=dims + [dims[0]],
            fill="toself",
            name=etiqueta,
            line=dict(color=CLUSTER_COLORS.get(etiqueta, "#999")),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="🕸️ Perfil de Métricas por Segmento",
        font=dict(family=FONT_FAMILY),
        height=480,
        template=TEMPLATE,
    )
    return fig
