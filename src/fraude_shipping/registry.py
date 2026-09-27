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
from mlflow.models import infer_signature

from fraude_shipping.produccion.pipeline import PipelineFraude

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
CARPETA_PAQUETE = Path(__file__).resolve().parent
NOMBRE_MODELO_REGISTRADO = 'fraude_shipping'
ALIAS_PRODUCCION = 'champion'
METODOS_PREDICCION = ('predict', 'predict_proba')
# Tag que MLflow muestra como descripción en la UI de experimentos y runs
ETIQUETA_DESCRIPCION = 'mlflow.note.content'
NOMBRE_RUN_VALIDACION = 'validacion_pipeline'
DESCRIPCION_MODELO = (
    'Pipeline productivo de prevención de fraude: construye las features del conjunto `candidatas` '
    '(tasa y frecuencia de `j`, país agrupado, perfil `o`/`n`/`p`, hora), estima la probabilidad de fraude '
    'con LightGBM tuneado y decide aprobar o rechazar con un umbral que maximiza la ganancia '
    '(+25% del monto por legítima aprobada, -100% por fraude aprobado).\n\n'
    'Uso: `predict` devuelve `probabilidad_fraude` y `decision`; con `params={"metodo": "predict_proba"}` '
    'devuelve solo la probabilidad. El alias `champion` apunta a la versión en producción.'
)
# Dependencias explícitas: inferirlas obliga a MLflow a cargar el modelo en un subproceso
REQUISITOS_MODELO = [f'{paquete}=={version(paquete)}' for paquete in ('lightgbm', 'pandas', 'scikit-learn', 'joblib')]


def formatear_numero(valor, decimales=0):
    """Número con formato español para las descripciones: punto de miles y coma decimal (150.000 o 78,9)."""
    return f'{valor:,.{decimales}f}'.translate(str.maketrans(',.', '.,'))


def conectar_mlflow():
    """Apunta MLflow al servidor de MLFLOW_TRACKING_URI si está definida o, si no, a la base SQLite del proyecto."""
    uri_servidor = os.environ.get('MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(uri_servidor or f'sqlite:///{(RAIZ_PROYECTO / "mlflow.db").as_posix()}')
    return uri_servidor


def configurar_mlflow(nombre_experimento):
    """Conecta MLflow y activa el experimento, creándolo si no existe."""
    uri_servidor = conectar_mlflow()
    if mlflow.get_experiment_by_name(nombre_experimento) is None:
        # Con servidor remoto los artefactos van a su almacén (GCS); en local, a la carpeta mlruns del proyecto
        ubicacion = None if uri_servidor else (RAIZ_PROYECTO / 'mlruns').as_uri()
        mlflow.create_experiment(nombre_experimento, artifact_location=ubicacion)
    mlflow.set_experiment(nombre_experimento)


def registrar_dataset(datos, nombre, contexto, fuente=None):
    """Asocia el dataset al run activo; su digest permite ver si dos runs usaron los mismos datos."""
    dataset = mlflow.data.from_pandas(
        datos, source=None if fuente is None else str(fuente), name=nombre,
        targets='fraude' if 'fraude' in datos.columns else None,
    )
    mlflow.log_input(dataset, context=contexto)


class ModeloFraudeMlflow(mlflow.pyfunc.PythonModel):
    """Envoltorio pyfunc del pipeline, para registrarlo y servirlo con las herramientas de MLflow."""

    def __init__(self):
        self.pipeline = None

    def load_context(self, context):
        """Carga el pipeline guardado como artefacto del modelo."""
        self.pipeline = PipelineFraude.cargar(context.artifacts['pipeline'])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """Probabilidad y decisión por defecto; con params={'metodo': 'predict_proba'}, solo la probabilidad."""
        metodo = (params or {}).get('metodo', 'predict')
        if metodo not in METODOS_PREDICCION:
            raise ValueError(f'metodo debe ser uno de {METODOS_PREDICCION}, no {metodo!r}')
        if metodo == 'predict_proba':
            return pd.DataFrame(
                {'probabilidad_fraude': self.pipeline.predecir_probabilidad(model_input)}, index=model_input.index
            )
        return self.pipeline.predecir(model_input)


def buscar_ultima_validacion():
    """Último run de validación del pipeline en el experimento activo, o None si todavía no se validó."""
    runs = mlflow.search_runs(
        filter_string=f"attributes.run_name = '{NOMBRE_RUN_VALIDACION}'", order_by=['attributes.start_time DESC'],
        max_results=1, output_format='list',
    )
    return runs[0] if runs else None


def _documentar_version(version_registrada, pipeline, filas_entrenamiento, validacion):
    """Descripción del modelo y de la versión, y tags de la versión con sus métricas para verlas en el registry."""
    cliente = MlflowClient()
    cliente.update_registered_model(NOMBRE_MODELO_REGISTRADO, description=DESCRIPCION_MODELO)
    tags = {'umbral': pipeline.umbral, 'filas_entrenamiento': filas_entrenamiento}
    descripcion = (
        f'Entrenado con {formatear_numero(filas_entrenamiento)} transacciones etiquetadas. '
        f'Umbral de decisión: {formatear_numero(pipeline.umbral, 2)}.'
    )
    if validacion is not None:
        metricas = validacion.data.metrics
        tags.update({
            'ganancia_pct_maxima': round(metricas['ganancia_pct_maxima_media'], 2),
            'auc_roc': round(metricas['auc_roc_media'], 3),
            'auc_pr': round(metricas['auc_pr_media'], 3),
            'run_validacion': validacion.info.run_id,
        })
        descripcion += (
            f" Validación con 5 folds nuevos (run {validacion.info.run_id}): "
            f"{formatear_numero(metricas['ganancia_pct_maxima_media'], 1)}% "
            f"± {formatear_numero(metricas['ganancia_pct_maxima_desvio'], 1)} de la ganancia máxima, "
            f"AUC-ROC {formatear_numero(metricas['auc_roc_media'], 3)} y AUC-PR {formatear_numero(metricas['auc_pr_media'], 3)}."
        )
    cliente.update_model_version(NOMBRE_MODELO_REGISTRADO, version_registrada, description=descripcion)
    for clave, valor in tags.items():
        cliente.set_model_version_tag(NOMBRE_MODELO_REGISTRADO, version_registrada, clave, str(valor))


def registrar_pipeline(ruta_pipeline, datos_entrenamiento, validacion=None):
    """Registra el pipeline como nueva versión documentada del modelo y le asigna el alias de producción."""
    entrada = datos_entrenamiento.drop(columns='fraude', errors='ignore')
    pipeline = PipelineFraude.cargar(ruta_pipeline)
    # La firma se infiere con todo train: con pocas filas, columnas que admiten nulos quedarían como obligatorias
    firma = infer_signature(entrada, pipeline.predecir(entrada.head(1000)), params={'metodo': 'predict'})
    informacion = mlflow.pyfunc.log_model(
        name='pipeline',
        python_model=ModeloFraudeMlflow(),
        artifacts={'pipeline': str(ruta_pipeline)},
        code_paths=[str(CARPETA_PAQUETE)],
        signature=firma,
        input_example=entrada.head(5),
        pip_requirements=REQUISITOS_MODELO,
        registered_model_name=NOMBRE_MODELO_REGISTRADO,
    )
    version_registrada = informacion.registered_model_version
    MlflowClient().set_registered_model_alias(NOMBRE_MODELO_REGISTRADO, ALIAS_PRODUCCION, version_registrada)
    _documentar_version(version_registrada, pipeline, len(entrada), validacion)
    return version_registrada


def descargar_pipeline(ruta_destino, referencia=ALIAS_PRODUCCION):
    """Descarga el pipeline de una versión registrada (alias o número) y deja su metadata en un JSON al lado."""
    cliente = MlflowClient()
    version_modelo = (
        cliente.get_model_version(NOMBRE_MODELO_REGISTRADO, referencia) if referencia.isdigit()
        else cliente.get_model_version_by_alias(NOMBRE_MODELO_REGISTRADO, referencia)
    )
    ruta_destino = Path(ruta_destino)
    ruta_destino.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as carpeta_temporal:
        carpeta_modelo = mlflow.artifacts.download_artifacts(
            artifact_uri=f'models:/{NOMBRE_MODELO_REGISTRADO}/{version_modelo.version}', dst_path=carpeta_temporal
        )
        shutil.copy(next((Path(carpeta_modelo) / 'artifacts').glob('*.joblib')), ruta_destino)
    metadata = {
        'modelo': NOMBRE_MODELO_REGISTRADO,
        'version': str(version_modelo.version),
        'alias': list(version_modelo.aliases),
        'run_id': version_modelo.run_id,
        'descargado': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    ruta_destino.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata
