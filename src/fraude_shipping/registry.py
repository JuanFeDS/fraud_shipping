"""Integración con MLflow (servidor remoto o base local): datasets de cada run y model registry del pipeline."""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.artifacts
import mlflow.data
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from fraude_shipping.production.pipeline import FraudPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
REGISTERED_MODEL_NAME = 'fraude_shipping'
PRODUCTION_ALIAS = 'champion'
PREDICTION_METHODS = ('predict', 'predict_proba')
# Tag que MLflow muestra como descripción en la UI de experimentos y runs
DESCRIPTION_TAG = 'mlflow.note.content'
VALIDATION_RUN_NAME = 'validacion_pipeline'
# La ganancia varía ~±1 punto entre folds: una versión nueva puede quedar hasta medio punto debajo del champion
PROMOTION_TOLERANCE = 0.5
MODEL_DESCRIPTION = (
    'Pipeline productivo de prevención de fraude: construye las features (tasa y frecuencia de `j`, país agrupado, '
    'hora), estima la probabilidad de fraude con LightGBM tuneado y decide aprobar o rechazar con el umbral teórico '
    'de la matriz de costos, 0,20 (+25% del monto por legítima aprobada, -100% por fraude aprobado), que coincide '
    'con el óptimo validado.\n\n'
    'Uso: `predict` devuelve `probabilidad_fraude` y `decision`; con `params={"metodo": "predict_proba"}` '
    'devuelve solo la probabilidad. El alias `champion` apunta a la versión en producción.'
)
# Dependencias explícitas: inferirlas obliga a MLflow a cargar el modelo en un subproceso
MODEL_REQUIREMENTS = [f'{package}=={version(package)}' for package in ('lightgbm', 'pandas', 'scikit-learn', 'joblib')]


def format_number(value, decimals=0):
    """Número con formato español para las descripciones: punto de miles y coma decimal (150.000 o 78,9)."""
    return f'{value:,.{decimals}f}'.translate(str.maketrans(',.', '.,'))


def connect_mlflow():
    """Apunta MLflow al servidor de MLFLOW_TRACKING_URI si está definida o, si no, a la base SQLite del proyecto."""
    server_uri = os.environ.get('MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(server_uri or f'sqlite:///{(PROJECT_ROOT / "mlflow.db").as_posix()}')
    return server_uri


def setup_mlflow(experiment_name):
    """Conecta MLflow y activa el experimento, creándolo si no existe."""
    server_uri = connect_mlflow()
    if mlflow.get_experiment_by_name(experiment_name) is None:
        # Con servidor remoto los artefactos van a su almacén (GCS); en local, a la carpeta mlruns del proyecto
        location = None if server_uri else (PROJECT_ROOT / 'mlruns').as_uri()
        mlflow.create_experiment(experiment_name, artifact_location=location)
    mlflow.set_experiment(experiment_name)


def log_dataset(data, name, context, source=None):
    """Asocia el dataset al run activo; su digest permite ver si dos runs usaron los mismos datos."""
    dataset = mlflow.data.from_pandas(
        data, source=None if source is None else str(source), name=name,
        targets='fraude' if 'fraude' in data.columns else None,
    )
    mlflow.log_input(dataset, context=context)


class FraudMlflowModel(mlflow.pyfunc.PythonModel):
    """Envoltorio pyfunc del pipeline, para registrarlo y servirlo con las herramientas de MLflow."""

    def __init__(self):
        self.pipeline = None

    def load_context(self, context):
        """Carga el pipeline guardado como artefacto del modelo."""
        self.pipeline = FraudPipeline.load(context.artifacts['pipeline'])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """Probabilidad y decisión por defecto; con params={'metodo': 'predict_proba'}, solo la probabilidad."""
        method = (params or {}).get('metodo', 'predict')
        if method not in PREDICTION_METHODS:
            raise ValueError(f'metodo debe ser uno de {PREDICTION_METHODS}, no {method!r}')
        if method == 'predict_proba':
            return pd.DataFrame(
                {'probabilidad_fraude': self.pipeline.predict_proba(model_input)}, index=model_input.index
            )
        return self.pipeline.predict(model_input)


def find_validation(params, data):
    """Último run de validación con los mismos parámetros (umbral incluido) y el mismo dataset, o None si no hay."""
    digest = mlflow.data.from_pandas(data).digest
    expected = {key: str(value) for key, value in params.items()}
    runs = mlflow.search_runs(
        filter_string=f"attributes.run_name = '{VALIDATION_RUN_NAME}'", order_by=['attributes.start_time DESC'],
        output_format='list',
    )
    for run in runs:
        same_data = digest in {dataset_input.dataset.digest for dataset_input in run.inputs.dataset_inputs}
        if same_data and all(run.data.params.get(key) == value for key, value in expected.items()):
            return run
    return None


def champion_profit():
    """Ganancia validada del champion vigente, o None si todavía no hay champion o no quedó documentada."""
    try:
        champion = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
    except MlflowException:
        return None
    profit = champion.tags.get('ganancia_pct_maxima')
    return None if profit is None else float(profit)


def should_promote(validation):
    """Gate de promoción: exige una validación del mismo modelo y que no empeore al champion más allá del ruido."""
    if validation is None:
        return False, 'no hay una validación con los mismos parámetros, umbral y datos'
    profit = validation.data.metrics['ganancia_pct_maxima_media']
    current = champion_profit()
    if current is None:
        return True, f'ganancia validada {profit:.2f}% y sin champion previo'
    if profit < current - PROMOTION_TOLERANCE:
        return False, f'ganancia validada {profit:.2f}% frente a {current:.2f}% del champion'
    return True, f'ganancia validada {profit:.2f}% frente a {current:.2f}% del champion'


def _document_version(registered_version, pipeline, training_rows, validation):
    """Descripción del modelo y de la versión, y tags de la versión con sus métricas para verlas en el registry."""
    client = MlflowClient()
    client.update_registered_model(REGISTERED_MODEL_NAME, description=MODEL_DESCRIPTION)
    tags = {'umbral': pipeline.threshold, 'filas_entrenamiento': training_rows}
    description = (
        f'Entrenado con {format_number(training_rows)} transacciones etiquetadas. '
        f'Umbral de decisión: {format_number(pipeline.threshold, 2)}.'
    )
    if validation is not None:
        metrics = validation.data.metrics
        tags.update({
            'ganancia_pct_maxima': round(metrics['ganancia_pct_maxima_media'], 2),
            'auc_roc': round(metrics['auc_roc_media'], 3),
            'auc_pr': round(metrics['auc_pr_media'], 3),
            'run_validacion': validation.info.run_id,
        })
        description += (
            f" Validación con 5 folds nuevos (run {validation.info.run_id}): "
            f"{format_number(metrics['ganancia_pct_maxima_media'], 1)}% "
            f"± {format_number(metrics['ganancia_pct_maxima_desvio'], 1)} de la ganancia máxima, "
            f"AUC-ROC {format_number(metrics['auc_roc_media'], 3)} y AUC-PR {format_number(metrics['auc_pr_media'], 3)}."
        )
    client.update_model_version(REGISTERED_MODEL_NAME, registered_version, description=description)
    for key, value in tags.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, registered_version, key, str(value))


def register_pipeline(pipeline_path, training_data, validation=None, promote=True):
    """Registra el pipeline como nueva versión documentada y, si se promueve, le asigna el alias de producción."""
    model_input = training_data.drop(columns='fraude', errors='ignore')
    pipeline = FraudPipeline.load(pipeline_path)
    # La firma se infiere con todo train: con pocas filas, columnas que admiten nulos quedarían como obligatorias
    signature = infer_signature(model_input, pipeline.predict(model_input.head(1000)), params={'metodo': 'predict'})
    model_info = mlflow.pyfunc.log_model(
        name='pipeline',
        python_model=FraudMlflowModel(),
        artifacts={'pipeline': str(pipeline_path)},
        code_paths=[str(PACKAGE_DIR)],
        signature=signature,
        input_example=model_input.head(5),
        pip_requirements=MODEL_REQUIREMENTS,
        registered_model_name=REGISTERED_MODEL_NAME,
    )
    registered_version = model_info.registered_model_version
    # Se documenta antes de promover: el gate de la próxima versión lee la ganancia del champion de sus tags
    _document_version(registered_version, pipeline, len(model_input), validation)
    if promote:
        MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, registered_version)
    return registered_version


def download_pipeline(destination, reference=PRODUCTION_ALIAS):
    """Descarga el pipeline de una versión registrada (alias o número) y deja su metadata en un JSON al lado."""
    client = MlflowClient()
    model_version = (
        client.get_model_version(REGISTERED_MODEL_NAME, reference) if reference.isdigit()
        else client.get_model_version_by_alias(REGISTERED_MODEL_NAME, reference)
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=f'models:/{REGISTERED_MODEL_NAME}/{model_version.version}', dst_path=temp_dir
        )
        shutil.copy(next((Path(model_dir) / 'artifacts').glob('*.joblib')), destination)
    metadata = {
        'modelo': REGISTERED_MODEL_NAME,
        'version': str(model_version.version),
        'alias': list(model_version.aliases),
        'run_id': model_version.run_id,
        'descargado': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata
