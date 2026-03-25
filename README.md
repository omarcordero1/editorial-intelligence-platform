# 📊 Editorial Performance ML

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.17%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![CI](https://img.shields.io/github/actions/workflow/status/omarcordero1/editorial-performance-ml/ci.yml?style=for-the-badge&label=CI)

**Segmentación de editores de medios digitales con Machine Learning.**
Convierte miles de artículos en inteligencia accionable para equipos editoriales.

[🚀 Demo en Streamlit](#-despliegue) · [📖 Documentación](#-estructura-del-proyecto) · [🤝 Contribuir](CONTRIBUTING.md)

</div>

---

## 🎯 El problema que resuelve

Los equipos editoriales de medios digitales manejan decenas de editores que producen cientos de artículos al mes. Las métricas clave —pageviews, scroll depth, registros, tiempo de lectura— quedan dispersas en dashboards que no responden la pregunta más importante para un director editorial:

> **¿Cuáles son mis editores de alto impacto, cuáles tienen potencial sin explotar y quiénes necesitan capacitación urgente?**

Este proyecto automatiza esa respuesta usando **KMeans clustering** sobre un conjunto de métricas editoriales enriquecidas, segmentando automáticamente a los editores en grupos accionables con narrativas claras para la toma de decisiones.

---

## 🏗️ Arquitectura del pipeline

```
CSV Editorial
     │
     ▼
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ data_loader │───▶│  preprocessing   │───▶│    features      │
│             │    │                  │    │                  │
│ Carga CSV   │    │ Limpieza         │    │ Score global     │
│ Valida cols │    │ Normalización    │    │ Eficiencia PVs   │
│ Parsea fecha│    │ Filtros mínimos  │    │ Consistencia     │
└─────────────┘    └──────────────────┘    │ Originalidad     │
                                           │ Diversidad tema  │
                                           └────────┬─────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────┐
                                         │    clustering    │
                                         │                  │
                                         │ Winsorización    │
                                         │ RobustScaler     │
                                         │ Elbow + Sil.     │
                                         │ KMeans → PCA     │
                                         └────────┬─────────┘
                                                  │
                              ┌───────────────────┴──────────────────┐
                              ▼                                       ▼
                    ┌──────────────────┐                   ┌──────────────────┐
                    │    exporter      │                   │  visualization   │
                    │                  │                   │                  │
                    │ CSVs resultados  │                   │ Scatter PCA      │
                    │ Dashboard HTML   │                   │ Radar por segm.  │
                    │                  │                   │ Ranking editores │
                    └──────────────────┘                   └──────────────────┘
```

---

## 🤖 Metodología ML

| Componente | Técnica | Justificación |
|---|---|---|
| Tratamiento de outliers | Winsorización (p5–p95) | Evita que editores con volumen atípico distorsionen los clusters |
| Escalado | `RobustScaler` | Más robusto que StandardScaler ante distribuciones asimétricas |
| Selección de K | Método del codo + Silhouette Score | Doble validación para K óptimo |
| Clustering | `KMeans` (k=4, random_state=42) | Interpretable, escalable, rápido |
| Reducción dimensional | `PCA` (2 componentes) | Visualización 2D del espacio de features |

### Features utilizadas

| Feature | Descripción | Peso en Score |
|---|---|---|
| `Registros` | Total de registros generados por notas del editor | 35% |
| `PVs` | Pageviews acumulados | 25% |
| `Scroll` | Profundidad de lectura promedio (%) | 20% |
| `RFV` | Retención / Frecuencia / Valor del lector | 15% |
| `eficiencia_registros_pv` | Registros por cada 1,000 PVs | 5% |

### Segmentos narrativos

| Segmento | Descripción | Acción sugerida |
|---|---|---|
| 👑 **Reyes y Reinas** | Top performers en todas las métricas | Estudiar sus estrategias; asignar temas de alto impacto |
| 🥈 **Príncipes y Princesas** | Alto potencial, ligera brecha vs. top | Mentoring con Reyes y Reinas; afinar estrategia SEO |
| 🥉 **Duques y Duquesas** | Desempeño medio-bajo pero estable | Capacitación en engagement; revisión de temas |
| 🐸 **Sapitos y Sapitas** | Bajo desempeño en métricas clave | Plan de mejora urgente; acompañamiento editorial |

---

## 📁 Estructura del proyecto

```
editorial-performance-ml/
│
├── src/                        # Código fuente modular
│   ├── __init__.py
│   ├── config.py               # Constantes y parámetros centralizados
│   ├── data_loader.py          # Carga y validación del CSV
│   ├── preprocessing.py        # Limpieza y filtrado de datos
│   ├── features.py             # Ingeniería de características
│   ├── clustering.py           # Pipeline KMeans + PCA
│   ├── visualization.py        # Gráficas con Plotly
│   └── exporter.py             # Exportación CSV y HTML
│
├── notebooks/                  # Notebook original de exploración
│   └── Análisis_editorial.ipynb
│
├── data/
│   ├── raw/                    # CSV fuente (no versionado)
│   ├── processed/              # Datos intermedios (no versionado)
│   └── outputs/                # CSVs y dashboard generados
│
├── tests/                      # Tests unitarios
│   └── test_pipeline.py
│
├── .github/
│   └── workflows/
│       └── ci.yml              # CI con GitHub Actions
│
├── app.py                      # Dashboard interactivo (Streamlit)
├── main.py                     # Pipeline CLI
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── CONTRIBUTING.md
```

---

## ⚡ Inicio rápido

### 1. Clonar y configurar entorno

```bash
git clone https://github.com/omarcordero1/editorial-performance-ml.git
cd editorial-performance-ml

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env según tu dataset
```

### 3. Correr el pipeline desde CLI

```bash
# Análisis básico
python main.py --input data/raw/tu_dataset.csv

# Especificar número de clusters
python main.py --input data/raw/tu_dataset.csv --k 5

# Sin generar dashboard HTML
python main.py --input data/raw/tu_dataset.csv --no-dashboard

# Ver ayuda
python main.py --help
```

### 4. Lanzar el dashboard interactivo

```bash
streamlit run app.py
```

Abre `http://localhost:8501` en tu navegador, sube tu CSV y explora los resultados.

---

## 📊 Formato del CSV requerido

| Columna | Tipo | Descripción |
|---|---|---|
| `editor` | string | Identificador único del editor |
| `Autor` | string | Autor de la nota (puede diferir del editor) |
| `Registros` | int | Registros generados por la nota |
| `Pv´s` | int | Pageviews de la nota |
| `Scroll` | float | Profundidad de scroll (0–100%) |
| `RFV` | float | Score de Retención-Frecuencia-Valor |
| `Ads Por Página` | float | Anuncios por página |
| `Fecha` | date | Fecha de publicación (parseable por pandas) |
| `tema` *(opcional)* | string | Categoría temática de la nota |
| `personaje_principal` *(opcional)* | string | Entidad protagonista |

---

## 📸 Screenshots sugeridos

> Para enriquecer este README, agrega capturas en `docs/screenshots/`:

| Archivo | Contenido |
|---|---|
| `01_dashboard_kpis.png` | Vista general con los 5 KPIs principales |
| `02_pca_scatter.png` | Mapa de editores en espacio PCA con colores por segmento |
| `03_score_ranking.png` | Top 20 editores por score global |
| `04_radar_clusters.png` | Radar chart comparativo de los 4 segmentos |
| `05_elbow_silhouette.png` | Gráfica de selección de K óptimo |

---

## 🚀 Despliegue

### Streamlit Community Cloud (recomendado — gratis)

1. Sube el repositorio a GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con GitHub.
3. Selecciona el repo, rama `main` y archivo `app.py`.
4. Haz clic en **Deploy**.

### Render

1. Crea una cuenta en [render.com](https://render.com).
2. Nuevo servicio → **Web Service** → conecta tu repo.
3. Configura:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy.

### Docker (opcional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

```bash
docker build -t editorial-ml .
docker run -p 8501:8501 editorial-ml
```

---

## 🧪 Tests

```bash
# Correr todos los tests
pytest tests/ -v

# Con reporte de cobertura
pytest tests/ -v --cov=src --cov-report=html

# Ver reporte
open htmlcov/index.html
```

---

## 🗺️ Roadmap

- [x] Pipeline modular (loader → preprocessing → features → clustering → export)
- [x] Dashboard interactivo con Streamlit
- [x] Tests unitarios con pytest
- [x] CI/CD con GitHub Actions
- [ ] Soporte para múltiples marcas (Mediotiempo, Telediario, Milenio)
- [ ] Integración con Google Search Console API para métricas SEO
- [ ] Predicción de score futuro con modelos de regresión
- [ ] Alertas automáticas para editores con caída de performance
- [ ] API REST con FastAPI para integración con dashboards BI
- [ ] Exportación a Google Looker Studio

---

## 👤 Autor

**Omar Said Cordero Lugo**
Data Scientist · Grupo Multimedios

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Omar_Cordero-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/omar-said-cordero-lugo)
[![GitHub](https://img.shields.io/badge/GitHub-omarcordero1-181717?style=flat-square&logo=github)](https://github.com/omarcordero1)

---

## 📄 Licencia

Distribuido bajo licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

---

<div align="center">

*Construido con 🧠 para equipos editoriales que toman decisiones basadas en datos.*

</div>
