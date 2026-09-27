# 🛡️ Fraude Shipping — Prevención de fraude orientada a ganancia

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/pandas-3-150458?logo=pandas&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/LightGBM-4-02569B?logo=microsoft&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/scikit--learn-1-F7931E?logo=scikitlearn&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/Optuna-5-2C5BB4?style=flat-square" />
  <img src="https://img.shields.io/badge/SHAP-0.52-FF0051?style=flat-square" />
  <img src="https://img.shields.io/badge/MLflow-3-0194E2?logo=mlflow&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/Cloud_Run-GCP-4285F4?logo=googlecloud&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/Supabase-Postgres-3ECF8E?logo=supabase&logoColor=white&style=flat-square" />
  <img src="https://img.shields.io/badge/coverage-100%25-brightgreen?style=flat-square" />
</p>

Modelo de machine learning que decide si aprobar o rechazar cada transacción para **maximizar la ganancia del negocio**, no solo para detectar fraudes: cada legítima aprobada deja el 25% de su monto y cada fraude aprobado pierde el 100%. Incluye el análisis completo en notebooks, un pipeline productivo con inferencia batch y API, experimentos y model registry en MLflow, y el despliegue en Google Cloud Run.

---

## Tabla de contenidos

1. [Contexto y motivación](#-1-contexto-y-motivación)
2. [Resultados](#-2-resultados)
3. [Qué incluye el proyecto](#-3-qué-incluye-el-proyecto)
4. [Arquitectura](#️-4-arquitectura)
5. [Guía de uso](#️-5-guía-de-uso)
6. [Servicios desplegados](#-6-servicios-desplegados)
7. [Contacto](#-7-contacto)

---

## 🎯 1. Contexto y motivación

Desafío técnico de Data Science sobre un dataset de **150.000 transacciones** (15 variables anonimizadas `a`–`p`, `fecha`, `monto`, `score` y la etiqueta `fraude`) con un **5% de fraude**, entre el 8 de marzo y el 21 de abril de 2020.

La pregunta no es solo *¿esta transacción es fraude?*, sino *¿conviene aprobarla?* Rechazar un fraude evita perder el 100% del monto, pero rechazar una legítima renuncia al 25%. Por eso todo el proyecto se mide en **% de la ganancia máxima posible** (la que se obtendría aprobando solo las legítimas) y el umbral de decisión se elige maximizando esa ganancia.

---

## 📊 2. Resultados

| Modelo | % ganancia máxima | AUC-ROC | AUC-PR |
|---|---|---|---|
| ⚪ Aprobar todo | 63,4 | — | — |
| 🔵 Ordenar por `score` | 67,4 | 0,726 | 0,177 |
| 🟠 Baseline (LightGBM, variables originales) | 77,7 | 0,872 | 0,437 |
| 🏆 **Modelo final** (LightGBM tuneado, conjunto `candidatas`, umbral 0,15) | **78,9 ± 1,5** | **0,890** | **0,474** |

El modelo final se validó con folds que no intervinieron en la búsqueda de hiperparámetros, y el pipeline productivo reproduce ese resultado. Las claves del camino:

- 🔤 **La variable `j` cruda perjudicaba al modelo**: LightGBM memorizaba sus 8.324 categorías. Reemplazarla por su tasa de fraude (calculada dentro de cada fold) y su frecuencia fue la mejora individual más grande.
- 🎛️ **El tuning aporta +0,4 puntos reales**: la búsqueda mostraba +0,7, pero la mitad era optimismo por elegir y evaluar con los mismos folds.
- 🎚️ **El umbral óptimo (0,15) queda bajo el teórico (0,20)**, porque los fraudes tienen montos más altos.
- 🔬 **`score` es la variable más valiosa** (7,5 puntos de ganancia si se desordena) y la raíz de los errores más caros.

Los reportes ejecutivos de cada etapa están en [`docs/reports/`](docs/reports/).

---

## 🧩 3. Qué incluye el proyecto

| Componente | Descripción |
|---|---|
| 📓 Notebooks | Exploración, baseline, feature engineering, experimentación y explicabilidad (SHAP), cada uno con sus hallazgos |
| 🛤️ Pipeline productivo | `PipelineFraude`: ajusta las features con estado (tasa y frecuencia de `j`, países frecuentes) solo con train y predice sin mirar el lote, como ocurriría en producción |
| 📦 Inferencia batch | `scripts/predecir.py`: agrega probabilidad y decisión a un CSV |
| ⚡ API online | FastAPI con validación del input, documentación en `/docs` y API key |
| 🧪 MLflow | Tracking de los ~70 experimentos, datasets, descripciones por run y model registry con alias `champion` |
| ☁️ Despliegue | MLflow y la API en Cloud Run; metadatos en Supabase (Postgres) y artefactos en Cloud Storage |
| ✅ Tests | 80 tests unitarios y de integración, con **100% de cobertura** y un piso de 80% configurado |

---

## 🏛️ 4. Arquitectura

```
fraude_shipping/
├── data/raw/dataset.csv          # dataset del desafío
├── notebooks/                    # 01 exploración → 05 explicabilidad
├── docs/reports/                 # resumen ejecutivo de cada notebook
├── src/fraude_shipping/
│   ├── features.py               # construcción de features (compartido)
│   ├── ganancia.py               # función de ganancia y métricas de negocio
│   ├── registro.py               # MLflow: conexión, datasets, model registry
│   ├── experimentacion/          # usado por los notebooks: modelos, validación cruzada, tracking
│   └── produccion/               # lo que se despliega: pipeline y API (sin MLflow)
├── scripts/                      # entrenar, validar, predecir en batch, descargar del registry
├── tests/                        # unitarios e integración
└── deploy/                       # imágenes de MLflow y de la API para Cloud Run
```

**Flujo de entrenamiento a producción**

```
dataset.csv ──► validar_pipeline.py ──► run de validación ─┐
            └─► entrenar.py ──────────► MLflow registry ◄──┘  (versión documentada + alias champion)
                                             │
                          descargar_modelo.py ◄┘
                                   │
                                   ▼
                       imagen de la API (Cloud Build) ──► Cloud Run
```

`produccion/` no depende de MLflow, CatBoost ni XGBoost: la API solo carga LightGBM y el pipeline. El modelo viaja dentro de la imagen, así cada imagen es inmutable y reproducible.

---

## ⚙️ 5. Guía de uso

### Requisitos
- Python 3.12
- [Poetry](https://python-poetry.org/) 1.8 o superior

### Instalación
```bash
git clone https://github.com/JuanFeDS/proyecto.git
cd proyecto
poetry install
```

### Reproducir el análisis
Ejecutar los notebooks en orden (`01` → `05`) desde `notebooks/`. El `04` registra los experimentos en un MLflow local (`mlflow.db`) y tarda más de 1,5 horas por la búsqueda de Optuna. El `05` lee el modelo elegido de ese MLflow, así que requiere haber corrido el `04`.

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db   # explorar los experimentos locales
```

### Entrenar, validar y predecir
```bash
poetry run python scripts/validar_pipeline.py   # CV de 5 folds del pipeline completo (~2 min)
poetry run python scripts/entrenar.py           # entrena con todo el dataset → models/pipeline_fraude.joblib (~15 s)
poetry run python scripts/predecir.py --entrada data/raw/dataset.csv --salida data/predicciones/predicciones.csv
```
Con `--mlflow`, `validar_pipeline.py` y `entrenar.py` registran el run (y el modelo, en el caso de `entrenar.py`) en MLflow.

### API local
```bash
poetry run uvicorn fraude_shipping.produccion.api:app --reload
```
Documentación interactiva en http://127.0.0.1:8000/docs. Sin la variable `FRAUDE_API_KEY`, la API no exige key.

### Variables de entorno

| Variable | Uso |
|---|---|
| `MLFLOW_TRACKING_URI` | Servidor de MLflow; sin ella se usa el `mlflow.db` local |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | Credenciales del servidor de MLflow con autenticación |
| `FRAUDE_API_KEY` | Si está definida, `/predecir` exige esa key en el header `X-API-Key` |
| `RUTA_MODELO` | Pipeline que carga la API (por defecto `models/pipeline_fraude.joblib`) |

### Tests
```bash
poetry run pytest    # falla si la cobertura baja del 80%
```

### Despliegue
```bash
poetry run python scripts/descargar_modelo.py   # baja fraude_shipping@champion del registry
gcloud builds submit . --config=deploy/api/cloudbuild.yaml --substitutions=_IMAGEN=<imagen>
gcloud run deploy api-fraude --image=<imagen> --set-secrets=FRAUDE_API_KEY=api-key:latest ...
```

---

## 🌐 6. Servicios desplegados

| Servicio | URL | Acceso |
|---|---|---|
| 🧪 MLflow | https://mlflow-1027826425795.us-east1.run.app | Usuario de solo lectura, credenciales a solicitud |
| ⚡ API | https://api-fraude-1027826425795.us-east1.run.app/docs | `/docs` y `/salud` abiertos; `/predecir` con API key a solicitud |

Ambos servicios escalan a cero, así que la primera petición después de un rato inactivo puede tardar unos segundos.

---

## 📬 7. Contacto

JuanFe — [jmartinezbernal02@gmail.com](mailto:jmartinezbernal02@gmail.com)
