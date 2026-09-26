"""Integración con MLflow: base local del proyecto, datasets de cada run y model registry del pipeline productivo."""

from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.data
import pandas as pd
import mlflow.pyfunc
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


def configurar_mlflow(nombre_experimento):
    """Apunta MLflow a la base SQLite del proyecto y activa el experimento, creándolo si no existe."""
    mlflow.set_tracking_uri(f'sqlite:///{(RAIZ_PROYECTO / "mlflow.db").as_posix()}')
    if mlflow.get_experiment_by_name(nombre_experimento) is None:
        mlflow.create_experiment(nombre_experimento, artifact_location=(RAIZ_PROYECTO / 'mlruns').as_uri())
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
