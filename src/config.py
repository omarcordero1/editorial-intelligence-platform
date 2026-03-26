"""
Configuración centralizada del proyecto.
Todas las constantes, rutas y parámetros se definen aquí.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Rutas del proyecto ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Crear directorios si no existen
for d in [RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Parámetros de datos ─────────────────────────────────────────────────────
DATA_ENCODING = os.getenv("DATA_ENCODING", "latin-1")
DATE_COLUMN = "Fecha"
MIN_NOTAS_POR_EDITOR = int(os.getenv("MIN_NOTAS_POR_EDITOR", "12"))

COLUMN_RENAME_MAP = {
    "editor": "Editor",
    "Pv´s": "PVs",
    "Ads Por Página": "AdsPorPagina",
}

REQUIRED_COLUMNS = ["Editor", "Autor", "Registros", "PVs", "Scroll", "RFV", "AdsPorPagina"]

EDITORS_TO_EXCLUDE = os.getenv("EDITORS_TO_EXCLUDE", "miriam.castro").split(",")

# ─── Parámetros de clustering ────────────────────────────────────────────────
K_RANGE = range(2, 11)
K_DEFAULT = int(os.getenv("K_CLUSTERS", "4"))
RANDOM_STATE = 42
N_INIT = 10
WINSORIZE_LOWER = 5
WINSORIZE_UPPER = 95
PCA_COMPONENTS = 2

# ─── Etiquetas narrativas de clusters ────────────────────────────────────────
CLUSTER_LABELS = [
    "Reyes y Reinas",
    "Príncipes y Princesas",
    "Duques y Duquesas",
    "Sapitos y Sapitas",
]

CLUSTER_COLORS = {
    "Reyes y Reinas": "#FFD700",
    "Príncipes y Princesas": "#C0C0C0",
    "Duques y Duquesas": "#CD7F32",
    "Sapitos y Sapitas": "#90EE90",
}

# ─── Features para clustering ────────────────────────────────────────────────
FEATURES_CLUSTERING = [
    "Registros",
    "PVs",
    "Scroll",
    "RFV",
    "eficiencia_registros_pv",
    "score_global",
]

# ─── Pesos del score global ──────────────────────────────────────────────────
SCORE_WEIGHTS = {
    "registros_norm": 0.35,
    "pvs_norm": 0.25,
    "scroll_norm": 0.20,
    "rfv_norm": 0.15,
    "eficiencia_norm": 0.05,
}

# ─── Outputs ─────────────────────────────────────────────────────────────────
OUTPUT_FILES = {
    "dashboard": "dashboard_editorial_milenio.html",
    "editores": "resultados_por_editor.csv",
    "clusters": "resumen_por_cluster.csv",
    "temas": "temas_por_editor.csv",
}
