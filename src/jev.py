"""Cliente de Jev (TypeSafe AI) a través de la API HTTP de evaluación de Vercel AI Gateway."""

import csv
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from src.tracking import RAIZ_PROYECTO

URL_EVALUACION = 'https://ai-gateway.vercel.sh/v1/evaluate'
MODELO_JEV = 'typesafe-ai/jev'
PROVEEDORES_JEV = ['typesafe-ai', 'digitalocean']
LLAMADAS_POR_MINUTO = 28
INTENTOS = 10
ESPERA_MAXIMA = 60
ESPERA_TIMEOUT = 60
CODIGOS_REINTENTABLES = {429, 500, 502, 503, 504}


def _estado_transaccion(fila):
    """Convierte una fila en un diccionario serializable a JSON, con los nulos como None."""
    return {
        columna: None if pd.isna(valor) else valor.item() if hasattr(valor, 'item') else valor
        for columna, valor in fila.items()
    }


def _evaluar(sesion, estado, instrucciones):
    """Probabilidad de fraude que Jev asigna a una transacción; ante límites de tasa espera lo que indica Retry-After."""
    cuerpo = {
        'model': MODELO_JEV,
        'state': estado,
        'questions': {'fraude': {
            'type': 'boolean',
            'instructions': instrucciones,
            'criteria': {'true': 'the transaction is fraudulent', 'false': 'the transaction is legitimate'},
        }},
        # Ambos proveedores se saturan de forma intermitente (429 "high demand"); con los dos el gateway tiene fallback
        'providerOptions': {'gateway': {'only': PROVEEDORES_JEV}},
    }
    for intento in range(INTENTOS):
        respuesta = sesion.post(URL_EVALUACION, json=cuerpo, timeout=ESPERA_TIMEOUT)
        if respuesta.status_code not in CODIGOS_REINTENTABLES:
            respuesta.raise_for_status()
            return respuesta.json()['answers']['fraude']['probability']
        # Sin Retry-After, la saturación del proveedor puede durar minutos: backoff exponencial topado
        time.sleep(float(respuesta.headers.get('Retry-After', min(2 ** intento, ESPERA_MAXIMA))))
    raise requests.HTTPError(f'Jev no respondió tras {INTENTOS} intentos', response=respuesta)


def evaluar_transacciones(transacciones, instrucciones, ruta_cache):
    """Probabilidad de fraude de Jev para cada fila; guarda cada respuesta en un CSV y no repite las ya guardadas."""
    ruta_cache.parent.mkdir(parents=True, exist_ok=True)
    guardadas = pd.read_csv(ruta_cache, index_col='indice')['probabilidad'] if ruta_cache.exists() else pd.Series(dtype=float)
    pendientes = transacciones.loc[~transacciones.index.isin(guardadas.index)]

    load_dotenv(RAIZ_PROYECTO / '.env')
    sesion = requests.Session()
    sesion.headers['Authorization'] = f"Bearer {os.environ['AI_GATEWAY_API_KEY']}"
    intervalo = 60 / LLAMADAS_POR_MINUTO
    escribir_encabezado = not ruta_cache.exists()

    with open(ruta_cache, 'a', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)
        if escribir_encabezado:
            escritor.writerow(['indice', 'probabilidad'])
        for indice, fila in pendientes.iterrows():
            inicio = time.monotonic()
            escritor.writerow([indice, _evaluar(sesion, _estado_transaccion(fila), instrucciones)])
            archivo.flush()
            time.sleep(max(0.0, intervalo - (time.monotonic() - inicio)))

    return pd.read_csv(ruta_cache, index_col='indice')['probabilidad'].reindex(transacciones.index)
