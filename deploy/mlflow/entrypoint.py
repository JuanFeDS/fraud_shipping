"""Arranca MLflow con autenticación básica, Postgres (Supabase) como backend y GCS como almacén de artefactos."""

import configparser
import os
from pathlib import Path
from urllib.parse import quote

RUTA_CONFIGURACION_AUTH = Path('/tmp/basic_auth.ini')


def uri_postgres(base_de_datos):
    """URI de una base del Postgres de Supabase con las credenciales de las variables de entorno."""
    # La contraseña puede tener caracteres reservados de URL (@, /, #); sin codificarla la URI se rompe
    clave = quote(os.environ['DB_PASSWORD'], safe='')
    # Driver explícito: SQLAlchemy 2.1 usa psycopg 3 por defecto con postgresql:// y la imagen trae psycopg2
    return (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{clave}@{os.environ['DB_HOST']}:5432/{base_de_datos}"
        '?sslmode=require'
    )


def escribir_configuracion_auth():
    """Configuración de basic-auth: usuarios en una base aparte y permiso de solo lectura por defecto."""
    # Base propia porque las migraciones de auth usan su propia tabla alembic_version, que chocaría con la de tracking
    configuracion = configparser.ConfigParser()
    configuracion['mlflow'] = {
        'default_permission': 'READ',
        'database_uri': uri_postgres('mlflow_auth').replace('%', '%%'),
        'admin_username': 'admin',
        'authorization_function': 'mlflow.server.auth:authenticate_request_basic_auth',
    }
    with RUTA_CONFIGURACION_AUTH.open('w', encoding='utf-8') as archivo:
        configuracion.write(archivo)


def main():
    """Prepara la configuración de autenticación y reemplaza este proceso por mlflow server."""
    escribir_configuracion_auth()
    # La contraseña del admin (MLFLOW_AUTH_ADMIN_PASSWORD) y la clave de Flask llegan como secretos de Cloud Run
    os.environ['MLFLOW_AUTH_CONFIG_PATH'] = str(RUTA_CONFIGURACION_AUTH)
    os.execvp('mlflow', [
        'mlflow', 'server',
        '--app-name', 'basic-auth',
        '--backend-store-uri', uri_postgres('postgres'),
        '--artifacts-destination', os.environ['ARTIFACTS_DESTINATION'],
        '--serve-artifacts',
        '--allowed-hosts', os.environ['ALLOWED_HOSTS'],
        # La UI hace POST con el header Origin de su propia URL; sin esto MLflow solo acepta localhost y la bloquea
        '--cors-allowed-origins', os.environ['CORS_ALLOWED_ORIGINS'],
        '--host', '0.0.0.0',
        '--port', os.environ.get('PORT', '8080'),
        '--workers', '1',
    ])


if __name__ == '__main__':
    main()
