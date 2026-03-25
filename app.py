"""
Dashboard interactivo con Streamlit.
Permite explorar los resultados del análisis editorial en tiempo real.

Ejecutar:
    streamlit run app.py
"""
import io
import logging

import pandas as pd
import streamlit as st
import plotly.express as px

from src.clustering import run_clustering
from src.config import CLUSTER_COLORS, K_DEFAULT
from src.data_loader import DataLoadError, load_editorial_data
from src.exporter import build_dashboard_html, export_csvs
from src.features import build_editor_features, get_top_temas
from src.preprocessing import clean_data
from src.visualization import (
    plot_cluster_distribution,
    plot_cluster_radar,
    plot_elbow_silhouette,
    plot_pca_scatter,
    plot_score_ranking,
)

logging.basicConfig(level=logging.INFO)

# ─── Config de página ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Editorial · Milenio.com",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Estilos ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 12px; padding: 20px; color: white; text-align: center;
}
.metric-val { font-size: 36px; font-weight: 700; }
.metric-lbl { font-size: 13px; opacity: .85; margin-top: 4px; }
.stTabs [role="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Milenio_logo.png/120px-Milenio_logo.png", width=120)
    st.title("⚙️ Configuración")

    uploaded_file = st.file_uploader(
        "📂 Cargar dataset CSV",
        type=["csv"],
        help="CSV con columnas: editor, Autor, Registros, PVs, Scroll, RFV, Ads Por Página, Fecha",
    )

    k_clusters = st.slider(
        "Número de clusters (K)",
        min_value=2, max_value=8, value=K_DEFAULT, step=1,
    )

    top_n = st.slider(
        "Top N editores en ranking",
        min_value=10, max_value=50, value=20, step=5,
    )

    st.divider()
    st.caption("📌 Grupo Multimedios · Data e IA")
    st.caption("👤 Omar Said Cordero Lugo")


# ─── Estado global con cache ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Procesando datos...")
def run_pipeline(file_bytes: bytes, k: int):
    """Ejecuta el pipeline completo y cachea los resultados."""
    df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")

    # Guardar temporalmente para usar data_loader
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        df_raw = load_editorial_data(tmp_path)
        df_clean = clean_data(df_raw)
        metricas = build_editor_features(df_clean)
        result = run_clustering(metricas, k=k)
        temas = {e: get_top_temas(df_clean, e) for e in result.df_clustered["Editor"]}
    finally:
        os.unlink(tmp_path)

    return result, temas, df_clean


# ─── Main app ────────────────────────────────────────────────────────────────
st.title("📊 Dashboard de Desempeño Editorial")
st.caption("Segmentación de editores con Machine Learning · Milenio.com")

if uploaded_file is None:
    st.info("👈 Sube tu CSV en el panel lateral para comenzar el análisis.", icon="📂")

    st.markdown("### Formato esperado del CSV")
    st.dataframe(pd.DataFrame({
        "editor": ["juan.perez", "maria.lopez"],
        "Autor": ["juan.perez", "maria.lopez"],
        "Registros": [1200, 850],
        "Pv´s": [45000, 32000],
        "Scroll": [68.5, 72.1],
        "RFV": [3.2, 2.8],
        "Ads Por Página": [4.1, 3.9],
        "Fecha": ["2024-01-15", "2024-01-16"],
        "tema": ["política", "deportes"],
    }), use_container_width=True)
    st.stop()

# ── Procesar datos ────────────────────────────────────────────────────────────
file_bytes = uploaded_file.read()

with st.spinner("🔄 Ejecutando análisis..."):
    try:
        result, temas_por_editor, df_clean = run_pipeline(file_bytes, k_clusters)
    except DataLoadError as e:
        st.error(f"❌ Error al cargar datos: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
        st.exception(e)
        st.stop()

df = result.df_clustered

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "👥 Editores", len(df)),
    (c2, "📝 Notas", f"{int(df.get('num_notas', pd.Series([0])).sum()):,}"),
    (c3, "👁️ PVs Totales", f"{int(df['PVs'].sum()):,}" if "PVs" in df.columns else "N/A"),
    (c4, "🎯 Registros", f"{int(df['Registros'].sum()):,}" if "Registros" in df.columns else "N/A"),
    (c5, "⭐ Score Promedio", f"{df['score_global'].mean():.1f}" if "score_global" in df.columns else "N/A"),
]
for col, label, val in metrics:
    col.metric(label, val)

st.divider()

# ── Tabs de análisis ──────────────────────────────────────────────────────────
tabs = st.tabs(["🗺️ Mapa PCA", "🏆 Ranking", "👥 Segmentos", "🕸️ Radar", "🔬 Clustering", "📥 Exportar"])

with tabs[0]:
    st.plotly_chart(plot_pca_scatter(df), use_container_width=True)

with tabs[1]:
    st.plotly_chart(plot_score_ranking(df, top_n=top_n), use_container_width=True)

with tabs[2]:
    st.plotly_chart(plot_cluster_distribution(df), use_container_width=True)
    st.dataframe(
        df.groupby("etiqueta").agg(
            Editores=("Editor", "count"),
            Score_Promedio=("score_global", "mean"),
            PVs_Total=("PVs", "sum"),
            Registros_Total=("Registros", "sum"),
        ).round(1),
        use_container_width=True,
    )

with tabs[3]:
    st.plotly_chart(plot_cluster_radar(df), use_container_width=True)

with tabs[4]:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_elbow_silhouette(result.inertias, result.silhouette_scores), use_container_width=True)
    with col2:
        st.metric("K óptimo seleccionado", result.k_optimal)
        st.metric(
            "Silhouette Score",
            f"{result.silhouette_scores[result.k_optimal - 2]:.3f}",
            help="Valores cercanos a 1 = clusters bien definidos.",
        )
        st.metric(
            "Varianza PCA explicada",
            f"{sum(result.pca_variance_ratio) * 100:.1f}%",
        )

with tabs[5]:
    st.subheader("📥 Exportar resultados")

    csv_editores = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "📄 Descargar resultados por editor (CSV)",
        csv_editores,
        file_name="resultados_por_editor.csv",
        mime="text/csv",
    )

    resumen_cluster = (
        df.groupby("etiqueta")
        .agg(Editores=("Editor", "count"), Score=("score_global", "mean"))
        .round(1)
        .to_csv(encoding="utf-8-sig")
        .encode("utf-8-sig")
    )
    st.download_button(
        "📄 Descargar resumen por cluster (CSV)",
        resumen_cluster,
        file_name="resumen_por_cluster.csv",
        mime="text/csv",
    )
