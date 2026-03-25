"""
Punto de entrada principal del pipeline de análisis editorial.

Uso:
    python main.py --input data/raw/dataset.csv
    python main.py --input data/raw/dataset.csv --k 5
    python main.py --input data/raw/dataset.csv --output data/outputs/
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from src.clustering import run_clustering
from src.data_loader import DataLoadError, load_editorial_data
from src.exporter import build_dashboard_html, export_csvs
from src.features import build_editor_features, get_top_temas
from src.preprocessing import clean_data

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s · %(levelname)-8s · %(name)s · %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("editorial_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de Análisis de Desempeño Editorial — Milenio.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py --input data/raw/datos.csv
  python main.py --input data/raw/datos.csv --k 5
  python main.py --input data/raw/datos.csv --no-dashboard
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Ruta al archivo CSV con datos editoriales.",
    )
    parser.add_argument(
        "--output", "-o", default="data/outputs/",
        help="Directorio de salida para los resultados (default: data/outputs/).",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Número de clusters KMeans (default: valor en config.py).",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Omite la generación del dashboard HTML.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    start = time.time()
    logger.info("=" * 60)
    logger.info("PIPELINE EDITORIAL · INICIO")
    logger.info("=" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Carga de datos ────────────────────────────────────────────────────
    try:
        df = load_editorial_data(args.input)
    except DataLoadError as e:
        logger.error("Error al cargar datos: %s", e)
        sys.exit(1)

    # ── 2. Limpieza ──────────────────────────────────────────────────────────
    df_clean = clean_data(df)

    # ── 3. Feature engineering ───────────────────────────────────────────────
    metricas_editor = build_editor_features(df_clean)

    # ── 4. Clustering ────────────────────────────────────────────────────────
    result = run_clustering(metricas_editor, k=args.k)
    df_clustered = result.df_clustered

    # ── 5. Análisis temático ─────────────────────────────────────────────────
    temas_por_editor = {
        editor: get_top_temas(df_clean, editor)
        for editor in df_clustered["Editor"]
    }

    # ── 6. Exportación ───────────────────────────────────────────────────────
    paths = export_csvs(df_clustered, temas_por_editor, output_dir=output_dir)

    if not args.no_dashboard:
        dash_path = build_dashboard_html(
            df_clustered,
            result.inertias,
            result.silhouette_scores,
            output_dir=output_dir,
        )
        paths["dashboard"] = dash_path

    # ── 7. Resumen ejecutivo ─────────────────────────────────────────────────
    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("ANÁLISIS COMPLETADO EN %.1f s", elapsed)
    logger.info("=" * 60)
    logger.info("Clusters: %s · Silhouette: %.3f · PCA varianza: %.1f%%",
                result.k_optimal,
                result.silhouette_scores[result.k_optimal - 2],
                sum(result.pca_variance_ratio) * 100)
    logger.info("Archivos generados:")
    for name, path in paths.items():
        logger.info("  ✅ %-12s → %s", name, path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
