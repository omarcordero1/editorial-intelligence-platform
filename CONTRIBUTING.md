# 🤝 Guía de Contribución

¡Gracias por tu interés en contribuir a **Editorial Performance ML**! Este documento establece las convenciones y el flujo de trabajo para mantener la calidad del proyecto.

---

## 📋 Tabla de contenidos

- [Cómo reportar un bug](#-cómo-reportar-un-bug)
- [Cómo proponer una mejora](#-cómo-proponer-una-mejora)
- [Flujo de trabajo con Git](#-flujo-de-trabajo-con-git)
- [Convenciones de código](#-convenciones-de-código)
- [Cómo correr tests](#-cómo-correr-tests)

---

## 🐛 Cómo reportar un bug

1. Abre un [Issue](https://github.com/omarcordero1/editorial-performance-ml/issues/new) con la etiqueta `bug`.
2. Describe claramente:
   - Qué esperabas que ocurriera.
   - Qué ocurrió realmente.
   - Pasos para reproducirlo.
   - Versión de Python y sistema operativo.

---

## 💡 Cómo proponer una mejora

1. Abre un Issue con la etiqueta `enhancement` antes de empezar a codificar.
2. Describe el caso de uso de negocio que resuelve la mejora.
3. Espera retroalimentación antes de abrir un Pull Request.

---

## 🌿 Flujo de trabajo con Git

```bash
# 1. Fork del repositorio (desde GitHub)

# 2. Clonar tu fork
git clone https://github.com/TU_USUARIO/editorial-performance-ml.git
cd editorial-performance-ml

# 3. Crear una rama descriptiva
git checkout -b feat/nombre-de-la-funcionalidad
# o bien:
git checkout -b fix/nombre-del-bug

# 4. Hacer tus cambios, commits atómicos
git add .
git commit -m "feat: agregar nueva métrica de diversidad temática"

# 5. Push y Pull Request hacia main
git push origin feat/nombre-de-la-funcionalidad
```

### Convenciones de commits (Conventional Commits)

| Prefijo    | Uso                                              |
|------------|--------------------------------------------------|
| `feat:`    | Nueva funcionalidad                              |
| `fix:`     | Corrección de bug                                |
| `docs:`    | Solo documentación                               |
| `refactor:`| Refactorización sin cambio de funcionalidad      |
| `test:`    | Agregar o modificar tests                        |
| `chore:`   | Tareas de mantenimiento (dependencias, CI, etc.) |

### Criterios para que un PR sea aceptado

- ✅ Todos los tests pasan (`pytest tests/`)
- ✅ Cobertura de tests ≥ 80% en módulos modificados
- ✅ Sin errores de linting (`ruff check src/`)
- ✅ Documentación actualizada si aplica
- ✅ El PR describe claramente qué cambia y por qué

---

## 🧹 Convenciones de código

### Estilo
- **Formato**: Black (`black src/ app.py main.py`)
- **Linting**: Ruff (`ruff check src/`)
- **Type hints**: Obligatorios en todas las funciones públicas
- **Docstrings**: Google Style en español

### Ejemplo de función bien documentada

```python
def calcular_score_global(df: pd.DataFrame, pesos: dict) -> pd.Series:
    """
    Calcula el score global ponderado por editor.

    Args:
        df: DataFrame con métricas normalizadas por editor.
        pesos: Diccionario {nombre_metrica: peso_flotante}.
            Los pesos deben sumar 1.0.

    Returns:
        Serie con el score global (0-100) indexada por Editor.

    Raises:
        ValueError: Si los pesos no suman 1.0 (tolerancia ±0.01).
    """
```

### Estructura de módulos
- Cada módulo tiene una única responsabilidad (SRP).
- Las constantes y parámetros configurables van en `src/config.py`.
- No hardcodear rutas, encodings ni parámetros ML dentro de las funciones.

---

## 🧪 Cómo correr tests

```bash
# Instalar dependencias de desarrollo
pip install -r requirements.txt

# Correr todos los tests
pytest tests/ -v

# Con cobertura de código
pytest tests/ -v --cov=src --cov-report=html

# Ver reporte HTML de cobertura
open htmlcov/index.html
```

---

*¿Tienes dudas? Abre un Issue o contáctame en [LinkedIn](https://www.linkedin.com/in/omar-said-cordero-lugo).*
