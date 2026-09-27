"""Pipeline productivo: features con estado ajustado en train, LightGBM tuneado y decisión por umbral."""

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd

from fraude_shipping.features import (
    ORIGINAL_COLUMNS,
    SEED,
    apply_fraud_rate,
    build_onp_profile,
    categorical_columns,
    fit_fraud_rate,
    get_frequent_countries,
    group_countries,
    make_folds,
    oof_fraud_rate,
)

MODEL_PATH = Path(__file__).resolve().parents[3] / 'models' / 'fraud_pipeline.joblib'

NUMERIC_COLUMNS = [column for column in ORIGINAL_COLUMNS if column not in ('g', 'j', 'o', 'p')]
MODEL_FEATURES = [
    *(column for column in ORIGINAL_COLUMNS if column not in ('g', 'j')),
    'g_agrupado', 'j_frecuencia', 'perfil_onp', 'hora', 'j_tasa_fraude',
]
MODEL_CATEGORICALS = categorical_columns(MODEL_FEATURES)

# Hiperparámetros elegidos por Optuna y validados con folds nuevos en el notebook 04 (run validacion_lightgbm_tuneado)
LIGHTGBM_PARAMS = {
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
THRESHOLD = 0.15


class FraudPipeline:
    """Transforma transacciones crudas en features, estima la probabilidad de fraude y decide aprobar o rechazar."""

    # Todo lo que depende de otras transacciones (tasa y frecuencia de j, países frecuentes, categorías) se aprende
    # en `fit`, así que `transform` nunca usa información de los datos a predecir (point-in-time)

    def __init__(self, params=None, threshold=THRESHOLD):
        self.params = LIGHTGBM_PARAMS if params is None else params
        self.threshold = threshold
        self.frequent_countries = set()
        self.j_frequency = pd.Series(dtype=float)
        self.j_fraud_rate = pd.Series(dtype=float)
        self.global_fraud_rate = None
        self.categories = {}
        self.model = None

    def fit(self, data):
        """Aprende las tablas de las features con estado y entrena el modelo con las transacciones etiquetadas."""
        data = data.reset_index(drop=True)
        self.frequent_countries = get_frequent_countries(data['g'])
        self.j_frequency = data['j'].value_counts()
        self.j_fraud_rate, self.global_fraud_rate = fit_fraud_rate(data['j'], data['fraude'])

        features = self._build_features(data)
        # En train la tasa de j se calcula out-of-fold: con la tabla completa el modelo vería la etiqueta de cada fila
        features['j_tasa_fraude'] = oof_fraud_rate(data['j'], data['fraude'], make_folds(data['fraude'])).to_numpy()
        self.categories = {column: sorted(features[column].fillna('nulo').unique()) for column in MODEL_CATEGORICALS}

        self.model = lgb.LGBMClassifier(random_state=SEED, verbose=-1, **self.params)
        self.model.fit(self._set_categories(features)[MODEL_FEATURES], data['fraude'])
        return self

    def transform(self, data):
        """Features del modelo para transacciones nuevas, usando solo lo aprendido en `fit`."""
        features = self._build_features(data)
        features['j_tasa_fraude'] = apply_fraud_rate(data['j'], self.j_fraud_rate, self.global_fraud_rate)
        return self._set_categories(features)[MODEL_FEATURES]

    def predict_proba(self, data):
        """Probabilidad de fraude de cada transacción."""
        return self.model.predict_proba(self.transform(data))[:, 1]

    def predict(self, data):
        """Probabilidad de fraude y decisión: se rechaza si la probabilidad alcanza el umbral."""
        probability = self.predict_proba(data)
        return pd.DataFrame(
            {
                'probabilidad_fraude': probability,
                'decision': ['rechazar' if value >= self.threshold else 'aprobar' for value in probability],
            },
            index=data.index,
        )

    def save(self, path=MODEL_PATH):
        """Guarda el pipeline completo (tablas, modelo y umbral) en un único archivo."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=MODEL_PATH):
        """Carga un pipeline guardado con `save`."""
        return joblib.load(path)

    def _build_features(self, data):
        """Features que no usan la etiqueta, calculadas con las tablas aprendidas."""
        # Con una sola fila, una columna numérica nula llega como object; se fuerza float para LightGBM
        features = data[NUMERIC_COLUMNS].apply(pd.to_numeric).astype(float)
        features['o'] = data['o']
        features['p'] = data['p']
        features['g_agrupado'] = group_countries(data['g'], self.frequent_countries)
        features['j_frecuencia'] = data['j'].map(self.j_frequency).fillna(0).astype(float)
        features['perfil_onp'] = build_onp_profile(data)
        features['hora'] = pd.to_datetime(data['fecha']).dt.hour
        return features

    def _set_categories(self, features):
        """Categóricas con las categorías de train; un valor no visto queda como nulo para LightGBM."""
        features = features.copy()
        for column, categories in self.categories.items():
            values = features[column].fillna('nulo')
            features[column] = pd.Categorical(values.where(values.isin(categories)), categories=categories)
        return features
