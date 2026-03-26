"""
Tests unitarios para el pipeline editorial.
Ejecutar: pytest tests/ -v --cov=src
"""
import io
import textwrap

import numpy as np
import pandas as pd
import pytest

from src.config import MIN_NOTAS_POR_EDITOR
from src.features import build_editor_features, _shannon_entropy, _normalize_series
from src.preprocessing import clean_data


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_df(n_editors: int = 3, notes_per_editor: int = 15) -> pd.DataFrame:
    """Genera un DataFrame sintético válido para tests."""
    import random
    random.seed(42)
    np.random.seed(42)

    editors = [f"editor_{i:02d}" for i in range(n_editors)]
    rows = []
    for ed in editors:
        for _ in range(notes_per_editor):
            rows.append({
                "Editor": ed,
                "Autor": ed if np.random.rand() > 0.3 else f"otro_autor_{np.random.randint(5)}",
                "Registros": np.random.randint(10, 500),
                "PVs": np.random.randint(500, 20000),
                "Scroll": np.random.uniform(40, 90),
                "RFV": np.random.uniform(1.5, 5.0),
                "AdsPorPagina": np.random.uniform(2.0, 6.0),
                "Fecha": "2024-01-15",
                "id": np.random.randint(1000, 9999),
                "tema": np.random.choice(["política", "deportes", "cultura", "economía"]),
                "personaje_principal": np.random.choice(["AMLO", "Sheinbaum", "Calderón", None]),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def df_raw():
    return _make_df()


@pytest.fixture
def df_clean(df_raw):
    return clean_data(df_raw)


@pytest.fixture
def metricas(df_clean):
    return build_editor_features(df_clean)


# ─── Tests preprocessing ──────────────────────────────────────────────────────

class TestPreprocessing:

    def test_clean_removes_nulls(self, df_raw):
        df_raw.loc[0, "Registros"] = None
        result = clean_data(df_raw)
        assert result["Registros"].isna().sum() == 0

    def test_clean_normalizes_strings(self, df_raw):
        df_raw.loc[0, "Editor"] = "  EDITOR_00  "
        result = clean_data(df_raw)
        assert "  EDITOR_00  " not in result["Editor"].values

    def test_clean_filters_min_notes(self):
        # Editor con menos del mínimo de notas
        df = _make_df(n_editors=1, notes_per_editor=MIN_NOTAS_POR_EDITOR - 1)
        result = clean_data(df)
        assert len(result) == 0

    def test_clean_excludes_specific_editors(self):
        df = _make_df()
        df.loc[df["Editor"] == "editor_00", "Editor"] = "miriam.castro"
        result = clean_data(df)
        assert "miriam.castro" not in result["Editor"].values


# ─── Tests features ───────────────────────────────────────────────────────────

class TestFeatures:

    def test_build_features_shape(self, df_clean, metricas):
        n_editors = df_clean["Editor"].nunique()
        assert len(metricas) == n_editors

    def test_score_global_range(self, metricas):
        assert metricas["score_global"].between(0, 100).all()

    def test_eficiencia_no_negative(self, metricas):
        assert (metricas["eficiencia_registros_pv"] >= 0).all()

    def test_indice_originalidad_range(self, metricas):
        assert metricas["indice_originalidad"].between(0, 1).all()

    def test_shannon_entropy_uniform(self):
        """Entropía máxima con distribución uniforme."""
        series = pd.Series(["a", "b", "c", "d"])
        entropy = _shannon_entropy(series)
        assert entropy == pytest.approx(2.0, abs=0.001)

    def test_shannon_entropy_single(self):
        """Entropía = 0 cuando todos son iguales."""
        series = pd.Series(["a", "a", "a"])
        assert _shannon_entropy(series) == 0.0

    def test_normalize_series(self):
        s = pd.Series([0.0, 5.0, 10.0])
        norm = _normalize_series(s)
        assert norm.min() == pytest.approx(0.0)
        assert norm.max() == pytest.approx(1.0)


# ─── Tests clustering ─────────────────────────────────────────────────────────

class TestClustering:

    def test_clustering_assigns_all_editors(self, metricas):
        from src.clustering import run_clustering
        result = run_clustering(metricas, k=3)
        assert len(result.df_clustered) == len(metricas)
        assert result.df_clustered["etiqueta"].notna().all()

    def test_silhouette_scores_count(self, metricas):
        from src.clustering import run_clustering
        from src.config import K_RANGE
        result = run_clustering(metricas, k=3)
        assert len(result.silhouette_scores) == len(list(K_RANGE))

    def test_pca_columns_exist(self, metricas):
        from src.clustering import run_clustering
        result = run_clustering(metricas, k=3)
        assert "pca_1" in result.df_clustered.columns
        assert "pca_2" in result.df_clustered.columns
