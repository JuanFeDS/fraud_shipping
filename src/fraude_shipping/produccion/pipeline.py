"""Pipeline productivo: features con estado ajustado en train, LightGBM tuneado y decisión por umbral."""

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from fraude_shipping.features import (
    COLUMNAS_ORIGINALES,
    SEMILLA,
    agrupar_paises,
    ajustar_tasa_fraude,
    aplicar_tasa_fraude,
    columnas_categoricas,
    crear_folds,
    crear_perfil_onp,
    obtener_paises_frecuentes,
    tasa_fraude_oof,
)

RUTA_MODELO = Path(__file__).resolve().parents[3] / 'models' / 'pipeline_fraude.joblib'

COLUMNAS_NUMERICAS = [columna for columna in COLUMNAS_ORIGINALES if columna not in ('g', 'j', 'o', 'p')]
FEATURES_MODELO = [
    *(columna for columna in COLUMNAS_ORIGINALES if columna not in ('g', 'j')),
    'g_agrupado', 'j_frecuencia', 'perfil_onp', 'hora', 'j_tasa_fraude',
]
CATEGORICAS_MODELO = columnas_categoricas(FEATURES_MODELO)

# Hiperparámetros elegidos por Optuna y validados con folds nuevos en el notebook 04 (run validacion_lightgbm_tuneado)
PARAMETROS_LIGHTGBM = {
    'n_estimators': 481,
    'learning_rate': 0.019385512410777697,
    'num_leaves': 172,
    'min_child_samples': 262,
    'subsample': 0.9601477375704716,
    'subsample_freq': 1,
    'colsample_bytree': 0.9604079184475345,
    'reg_alpha': 1.608317898022548,
    'reg_lambda': 3.025159385030446,
}
UMBRAL = 0.15


class PipelineFraude:
    """Transforma transacciones crudas en features, estima la probabilidad de fraude y decide aprobar o rechazar."""

    # Todo lo que depende de otras transacciones (tasa y frecuencia de j, países frecuentes, categorías) se aprende
    # en `ajustar`, así que `transformar` nunca usa información de los datos a predecir (point-in-time)

    def __init__(self, parametros=None, umbral=UMBRAL):
        self.parametros = PARAMETROS_LIGHTGBM if parametros is None else parametros
        self.umbral = umbral
        self.paises_frecuentes = set()
        self.frecuencia_j = pd.Series(dtype=float)
        self.tasa_fraude_j = pd.Series(dtype=float)
        self.tasa_fraude_global = None
        self.categorias = {}
        self.modelo = None

    def ajustar(self, datos):
        """Aprende las tablas de las features con estado y entrena el modelo con las transacciones etiquetadas."""
        datos = datos.reset_index(drop=True)
        self.paises_frecuentes = obtener_paises_frecuentes(datos['g'])
        self.frecuencia_j = datos['j'].value_counts()
        self.tasa_fraude_j, self.tasa_fraude_global = ajustar_tasa_fraude(datos['j'], datos['fraude'])

        features = self._construir_features(datos)
        # En train la tasa de j se calcula out-of-fold: con la tabla completa el modelo vería la etiqueta de cada fila
        features['j_tasa_fraude'] = tasa_fraude_oof(datos['j'], datos['fraude'], crear_folds(datos['fraude'])).to_numpy()
        self.categorias = {columna: sorted(features[columna].fillna('nulo').unique()) for columna in CATEGORICAS_MODELO}

        self.modelo = lgb.LGBMClassifier(random_state=SEMILLA, verbose=-1, **self.parametros)
        self.modelo.fit(self._fijar_categorias(features)[FEATURES_MODELO], datos['fraude'])
        return self

    def transformar(self, datos):
        """Features del modelo para transacciones nuevas, usando solo lo aprendido en `ajustar`."""
        features = self._construir_features(datos)
        features['j_tasa_fraude'] = aplicar_tasa_fraude(datos['j'], self.tasa_fraude_j, self.tasa_fraude_global)
        return self._fijar_categorias(features)[FEATURES_MODELO]

    def predecir_probabilidad(self, datos):
        """Probabilidad de fraude de cada transacción."""
        return self.modelo.predict_proba(self.transformar(datos))[:, 1]

    def predecir(self, datos):
        """Probabilidad de fraude y decisión: se rechaza si la probabilidad alcanza el umbral."""
        probabilidad = self.predecir_probabilidad(datos)
        return pd.DataFrame(
            {
                'probabilidad_fraude': probabilidad,
                'decision': ['rechazar' if valor >= self.umbral else 'aprobar' for valor in probabilidad],
            },
            index=datos.index,
        )

    def guardar(self, ruta=RUTA_MODELO):
        """Guarda el pipeline completo (tablas, modelo y umbral) en un único archivo."""
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ruta)

    @staticmethod
    def cargar(ruta=RUTA_MODELO):
        """Carga un pipeline guardado con `guardar`."""
        return joblib.load(ruta)

    def _construir_features(self, datos):
        """Features que no usan la etiqueta, calculadas con las tablas aprendidas."""
        # Con una sola fila, una columna numérica nula llega como object; se fuerza float para LightGBM
        features = datos[COLUMNAS_NUMERICAS].apply(pd.to_numeric).astype(float)
        features['o'] = datos['o']
        features['p'] = datos['p']
        features['g_agrupado'] = agrupar_paises(datos['g'], self.paises_frecuentes)
        features['j_frecuencia'] = datos['j'].map(self.frecuencia_j).fillna(0).astype(float)
        features['perfil_onp'] = crear_perfil_onp(datos)
        features['hora'] = pd.to_datetime(datos['fecha']).dt.hour
        return features

    def _fijar_categorias(self, features):
        """Categóricas con las categorías de train; un valor no visto queda como nulo para LightGBM."""
        features = features.copy()
        for columna, categorias in self.categorias.items():
            features[columna] = pd.Categorical(features[columna].fillna('nulo'), categories=categorias)
        return features
