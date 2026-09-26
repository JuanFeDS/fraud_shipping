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
# Dependencias explícitas: inferirlas obliga a MLflow a cargar el modelo en un subproceso
REQUISITOS_MODELO = [f'{paquete}=={version(paquete)}' for paquete in ('lightgbm', 'pandas', 'scikit-learn', 'joblib')]


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


def registrar_pipeline(ruta_pipeline, datos_entrenamiento):
    """Registra el pipeline guardado como nueva versión del modelo y le asigna el alias de producción."""
    entrada = datos_entrenamiento.drop(columns='fraude', errors='ignore')
    # La firma se infiere con todo train: con pocas filas, columnas que admiten nulos quedarían como obligatorias
    firma = infer_signature(
        entrada, PipelineFraude.cargar(ruta_pipeline).predecir(entrada.head(1000)), params={'metodo': 'predict'}
    )
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
