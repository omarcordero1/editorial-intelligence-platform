"""
Módulo de exportación de resultados.
Genera CSVs y el dashboard HTML final.
"""
import logging
from pathlib import Path

import pandas as pd
import plotly.io as pio

from .config import OUTPUT_FILES, OUTPUTS_DIR
from .visualization import (
    plot_cluster_distribution,
    plot_cluster_radar,
    plot_elbow_silhouette,
    plot_pca_scatter,
    plot_score_ranking,
)

logger = logging.getLogger(__name__)


def export_csvs(metricas_editor: pd.DataFrame, temas_por_editor: dict, output_dir: Path = OUTPUTS_DIR) -> dict:
    """
    Exporta los resultados del análisis a archivos CSV.

    Args:
        metricas_editor: DataFrame con métricas y clusters por editor.
        temas_por_editor: Dict {editor: [{'tema', 'num_notas', 'porcentaje'}]}.
        output_dir: Directorio de salida.

    Returns:
        Diccionario con las rutas de los archivos generados.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # CSV 1: resultados por editor
    cols = [
        "Editor", "etiqueta", "cluster", "num_notas",
        "Registros", "PVs", "Scroll", "RFV", "AdsPorPagina",
        "score_global", "eficiencia_registros_pv",
        "consistencia_editorial", "indice_originalidad", "diversidad_tematica",
    ]
    cols_ok = [c for c in cols if c in metricas_editor.columns]
    resultados = metricas_editor[cols_ok].sort_values("score_global", ascending=False)
    path_editores = output_dir / OUTPUT_FILES["editores"]
    resultados.to_csv(path_editores, index=False, encoding="utf-8-sig")
    paths["editores"] = path_editores
    logger.info("Exportado: %s", path_editores.name)

    # CSV 2: resumen por cluster
    agg_cols = {
        "Editor": "count",
        "num_notas": "sum",
        "Registros": ["sum", "mean"],
        "PVs": ["sum", "mean"],
        "score_global": "mean",
    }
    agg_cols_ok = {k: v for k, v in agg_cols.items() if k in metricas_editor.columns}
    resumen = metricas_editor.groupby("etiqueta").agg(agg_cols_ok).reset_index()
    resumen.columns = ["_".join(c).strip("_") for c in resumen.columns]
    path_clusters = output_dir / OUTPUT_FILES["clusters"]
    resumen.to_csv(path_clusters, index=False, encoding="utf-8-sig")
    paths["clusters"] = path_clusters
    logger.info("Exportado: %s", path_clusters.name)

    # CSV 3: temas por editor
    temas_rows = [
        {"Editor": editor, **tema_data}
        for editor, temas in temas_por_editor.items()
        for tema_data in temas
    ]
    if temas_rows:
        df_temas = pd.DataFrame(temas_rows)
        path_temas = output_dir / OUTPUT_FILES["temas"]
        df_temas.to_csv(path_temas, index=False, encoding="utf-8-sig")
        paths["temas"] = path_temas
        logger.info("Exportado: %s", path_temas.name)

    return paths


def build_dashboard_html(
    metricas_editor: pd.DataFrame,
    inertias: list,
    silhouette_scores: list,
    output_dir: Path = OUTPUTS_DIR,
) -> Path:
    """
    Genera el dashboard HTML interactivo con todas las gráficas.

    Args:
        metricas_editor: DataFrame enriquecido con clusters.
        inertias: Lista de inercias del análisis Elbow.
        silhouette_scores: Lista de silhouette scores.
        output_dir: Directorio de salida.

    Returns:
        Ruta al archivo HTML generado.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generar gráficas
    figs = {
        "pca": plot_pca_scatter(metricas_editor),
        "dist": plot_cluster_distribution(metricas_editor),
        "ranking": plot_score_ranking(metricas_editor),
        "elbow": plot_elbow_silhouette(inertias, silhouette_scores),
        "radar": plot_cluster_radar(metricas_editor),
    }

    # Convertir gráficas a HTML embebido
    charts_html = "".join(
        pio.to_html(fig, full_html=False, include_plotlyjs=False)
        for fig in figs.values()
    )

    # KPIs
    total_editores = int(len(metricas_editor))
    total_notas = int(metricas_editor["num_notas"].sum()) if "num_notas" in metricas_editor.columns else 0
    total_pvs = int(metricas_editor["PVs"].sum()) if "PVs" in metricas_editor.columns else 0
    total_registros = int(metricas_editor["Registros"].sum()) if "Registros" in metricas_editor.columns else 0
    score_prom = (
        metricas_editor["score_global"].mean() if "score_global" in metricas_editor.columns else 0
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dashboard Editorial — Milenio.com</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', Helvetica Neue, Arial, sans-serif; background: #f0f2f5; color: #2d3748; }}
    header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 32px 40px; color: white; }}
    header h1 {{ font-size: 32px; font-weight: 700; }}
    header p {{ font-size: 16px; opacity: .85; margin-top: 6px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; padding: 28px 40px; }}
    .kpi {{ background: white; border-radius: 12px; padding: 20px 16px; text-align: center;
             box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
    .kpi-val {{ font-size: 36px; font-weight: 700; color: #667eea; }}
    .kpi-lbl {{ font-size: 13px; color: #718096; margin-top: 6px; }}
    .charts {{ padding: 0 40px 40px; display: grid; gap: 24px; }}
    .chart-card {{ background: white; border-radius: 16px; padding: 24px;
                   box-shadow: 0 2px 12px rgba(0,0,0,.07); }}
    footer {{ text-align: center; padding: 24px; color: #a0aec0; font-size: 13px; }}
  </style>
</head>
<body>
<header>
  <h1>📊 Dashboard de Desempeño Editorial</h1>
  <p>Milenio.com · Segmentación de editores con Machine Learning (KMeans)</p>
</header>

<div class="kpis">
  <div class="kpi"><div class="kpi-val">{total_editores}</div><div class="kpi-lbl">👥 Editores</div></div>
  <div class="kpi"><div class="kpi-val">{total_notas:,}</div><div class="kpi-lbl">📝 Notas</div></div>
  <div class="kpi"><div class="kpi-val">{total_pvs:,}</div><div class="kpi-lbl">👁️ PVs Totales</div></div>
  <div class="kpi"><div class="kpi-val">{total_registros:,}</div><div class="kpi-lbl">🎯 Registros</div></div>
  <div class="kpi"><div class="kpi-val">{score_prom:.1f}</div><div class="kpi-lbl">⭐ Score Promedio</div></div>
</div>

<div class="charts">
  <div class="chart-card">{pio.to_html(figs["pca"], full_html=False, include_plotlyjs=False)}</div>
  <div class="chart-card">{pio.to_html(figs["dist"], full_html=False, include_plotlyjs=False)}</div>
  <div class="chart-card">{pio.to_html(figs["ranking"], full_html=False, include_plotlyjs=False)}</div>
  <div class="chart-card">{pio.to_html(figs["radar"], full_html=False, include_plotlyjs=False)}</div>
  <div class="chart-card">{pio.to_html(figs["elbow"], full_html=False, include_plotlyjs=False)}</div>
</div>

<footer>Generado automáticamente · Grupo Multimedios · Data e Inteligencia Artificial</footer>
</body>
</html>"""

    out_path = output_dir / OUTPUT_FILES["dashboard"]
    out_path.write_text(html, encoding="utf-8")
    logger.info("Dashboard generado: %s", out_path.name)
    return out_path
