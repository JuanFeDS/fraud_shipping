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
  <img src="https://github.com/JuanFeDS/proyecto/actions/workflows/tests.yml/badge.svg" />
</p>

Modelo de machine learning que predice la probabilidad de fraude de cada transacción y decide aprobarla o rechazarla para **maximizar la ganancia del negocio**: cada legítima aprobada deja el 25% de su monto y cada fraude aprobado pierde el 100%. Incluye el análisis completo en notebooks, un pipeline productivo con inferencia batch y API, experimentos y model registry en MLflow, y el despliegue en Google Cloud Run.

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

El objetivo es construir un modelo que prediga si una transacción es fraudulenta y usarlo para decidir con el costo de cada error: aprobar un fraude pierde el 100% del monto, y rechazar una legítima renuncia al 25%. Por eso todo el proyecto se mide en **% de la ganancia máxima posible** (la que se obtendría aprobando solo las legítimas), y el umbral de decisión sale de ese costo.

---

## 📊 2. Resultados

| Modelo | % ganancia máxima | AUC-ROC | AUC-PR |
|---|---|---|---|
| ⚪ Aprobar todo | 63,4 | — | — |
| 🔵 Ordenar por `score` | 67,4 | 0,726 | 0,177 |
| 🟠 Baseline (LightGBM, variables originales) | 77,7 | 0,872 | 0,437 |
| 🏆 **Modelo final** (LightGBM tuneado, `candidatas` sin `perfil_onp`, umbral 0,20) | **78,9 ± 1,3** | **0,889** | **0,474** |

El modelo final se validó con folds que no intervinieron en la búsqueda de hiperparámetros, y el pipeline productivo reproduce ese resultado. Las claves del camino:

- 🔤 **La variable `j` cruda perjudicaba al modelo**: LightGBM memorizaba sus 8.324 categorías. Reemplazarla por su tasa de fraude (calculada dentro de cada fold) y su frecuencia fue la mejora individual más grande.
- 🎛️ **El tuning aporta +0,4 puntos medidos con folds nuevos**: la búsqueda mostraba +0,7, pero la mitad era optimismo por elegir y evaluar con los mismos folds.
- 🧹 **`perfil_onp` se quitó del modelo final**: la misma ganancia con una variable menos, porque `o` ya captura su señal.
- 🎚️ **El umbral es el teórico (0,20)**, y coincide con el óptimo validado. Prediciendo semanas futuras (73,7%, 77,1% y 79,5% en las tres últimas), un umbral algo más bajo habría ganado ~1 punto: en producción hay que recalibrarlo con etiquetas recientes.
- 🔬 **`score` es la variable más valiosa** (8,3 puntos de ganancia si se desordena, seguida de `o` con 6,3) y la raíz de los errores más caros.

Los reportes ejecutivos de cada etapa están en [`docs/reports/`](docs/reports/).

---

## 🧩 3. Qué incluye el proyecto

| Componente | Descripción |
|---|---|
| 📓 Notebooks | Exploración, baseline, feature engineering, experimentación y explicabilidad (SHAP), cada uno con sus hallazgos |
| 🛤️ Pipeline productivo | `FraudPipeline`: ajusta las features con estado (tasa y frecuencia de `j`, países frecuentes) solo con train y predice sin mirar el lote, como ocurriría en producción |
| 📦 Inferencia batch | `scripts/predict.py`: agrega probabilidad y decisión a un CSV |
| ⚡ API online | FastAPI con validación del input, documentación en `/docs` y API key |
| 🧪 MLflow | Tracking de los ~70 experimentos, datasets, descripciones por run y model registry con alias `champion` |
| ☁️ Despliegue | MLflow y la API en Cloud Run; metadatos en Supabase (Postgres) y artefactos en Cloud Storage |
| ✅ Tests | 82 tests unitarios y de integración, con **100% de cobertura** y un piso de 80% configurado |

---

## 🏛️ 4. Arquitectura

```
fraude_shipping/
├── data/raw/dataset.csv          # dataset del desafío
├── notebooks/                    # 01 exploración → 05 explicabilidad
├── docs/reports/                 # resumen ejecutivo de cada notebook
├── src/fraude_shipping/
│   ├── features.py               # construcción de features (compartido)
│   ├── profit.py                 # función de ganancia y métricas de negocio
│   ├── registry.py               # MLflow: conexión, datasets, model registry
│   ├── experimentation/          # usado por los notebooks: modelos, validación cruzada, tracking
│   └── production/               # lo que se despliega: pipeline y API (sin MLflow)
├── scripts/                      # entrenar, validar, predecir en batch, descargar del registry
├── tests/                        # unit/ e integration/
└── deploy/                       # imágenes de MLflow y de la API para Cloud Run
```

**Flujo de entrenamiento a producción**

```
dataset.csv ──► validate_pipeline.py ──► run de validación ─┐
            └─► train.py ──────────────► MLflow registry ◄──┘  (versión documentada + alias champion)
                                              │
                           download_model.py ◄┘
                                   │
                                   ▼
                       imagen de la API (Cloud Build) ──► Cloud Run
```

`production/` no depende de MLflow, CatBoost ni XGBoost: la API solo carga LightGBM y el pipeline. El modelo viaja dentro de la imagen, así cada imagen es inmutable y reproducible.

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
Ejecutar los notebooks en orden (`01` → `05`) desde `notebooks/`. El `04` registra los experimentos en un MLflow local (`mlflow.db`) y tarda más de 1,5 horas por la búsqueda de Optuna. El `05` no requiere haber corrido el `04`: toma los hiperparámetros, las features y el umbral del pipeline productivo, o del run de MLflow si `MLFLOW_TRACKING_URI` está definida.

```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db   # explorar los experimentos locales
```

### Entrenar, validar y predecir
```bash
poetry run python scripts/validate_pipeline.py   # CV de 5 folds del pipeline completo (~2 min)
poetry run python scripts/train.py               # entrena con todo el dataset → models/fraud_pipeline.joblib (~15 s)
poetry run python scripts/predict.py --input data/raw/dataset.csv --output data/predictions/predictions.csv
```
Con `--mlflow`, `validate_pipeline.py` y `train.py` registran el run (y el modelo, en el caso de `train.py`) en MLflow. La versión nueva recibe el alias `champion` solo si pasa el **gate de promoción**: tiene que existir una validación con los mismos parámetros, umbral y datos, y su ganancia no puede quedar más de 0,5 puntos por debajo de la del champion vigente. Si no lo pasa, se registra sin alias para poder revisarla.

### API local
```bash
FRAUDE_API_KEY=<una-clave> poetry run uvicorn fraude_shipping.production.api:app --reload
```
Documentación interactiva en http://127.0.0.1:8000/docs. La API no arranca sin `FRAUDE_API_KEY`: si la variable se pierde en un despliegue, falla cerrada en lugar de quedar abierta.

`fecha` se exige **con zona horaria** (por ejemplo, `2020-04-15T02:30:00-03:00`) y se convierte a UTC−3 antes de calcular la hora del día. Es la zona que se supone para el dataset (casi todo Brasil y Argentina), y hay que confirmarla con los dueños de los datos. En Cloud Run (1 vCPU), el servidor decide en **20 ms (p50) y 28 ms (p99)**, medido con 300 requests secuenciales.

### Variables de entorno

| Variable | Uso |
|---|---|
| `MLFLOW_TRACKING_URI` | Servidor de MLflow; sin ella se usa el `mlflow.db` local |
| `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` | Credenciales del servidor de MLflow con autenticación |
| `FRAUDE_API_KEY` | Obligatoria: `/predecir` exige esa key en el header `X-API-Key` |
| `MODEL_PATH` | Pipeline que carga la API (por defecto `models/fraud_pipeline.joblib`) |

### Tests
```bash
poetry run pytest    # falla si la cobertura baja del 80%
```
En GitHub Actions (`.github/workflows/tests.yml`) los tests corren en cada push, y se verifica que el `requirements.txt` de la API coincida con el lock.

### Despliegue
```bash
poetry run python scripts/download_model.py   # baja fraude_shipping@champion del registry
poetry export --only main --without-hashes -f requirements.txt -o deploy/api/requirements.txt   # solo si cambió el lock
gcloud builds submit . --config=deploy/api/cloudbuild.yaml --substitutions=_IMAGE=<imagen>
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
