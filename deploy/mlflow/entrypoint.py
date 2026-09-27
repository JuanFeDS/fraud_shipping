"""Arranca MLflow con autenticación básica, Postgres (Supabase) como backend y GCS como almacén de artefactos."""

import configparser
import os
from pathlib import Path
from urllib.parse import quote

AUTH_CONFIG_PATH = Path('/tmp/basic_auth.ini')


def postgres_uri(database):
    """URI de una base del Postgres de Supabase con las credenciales de las variables de entorno."""
    # La contraseña puede tener caracteres reservados de URL (@, /, #); sin codificarla la URI se rompe
    password = quote(os.environ['DB_PASSWORD'], safe='')
    # Driver explícito: SQLAlchemy 2.1 usa psycopg 3 por defecto con postgresql:// y la imagen trae psycopg2
    return (
        f"postgresql+psycopg2://{os.environ['DB_USER']}:{password}@{os.environ['DB_HOST']}:5432/{database}"
        '?sslmode=require'
    )


def write_auth_config():
    """Configuración de basic-auth: usuarios en una base aparte y permiso de solo lectura por defecto."""
    # Base propia porque las migraciones de auth usan su propia tabla alembic_version, que chocaría con la de tracking
    config = configparser.ConfigParser()
    config['mlflow'] = {
        'default_permission': 'READ',
        'database_uri': postgres_uri('mlflow_auth').replace('%', '%%'),
        'admin_username': 'admin',
        'authorization_function': 'mlflow.server.auth:authenticate_request_basic_auth',
    }
    with AUTH_CONFIG_PATH.open('w', encoding='utf-8') as config_file:
        config.write(config_file)


def main():
    """Prepara la configuración de autenticación y reemplaza este proceso por mlflow server."""
    write_auth_config()
    # La contraseña del admin (MLFLOW_AUTH_ADMIN_PASSWORD) y la clave de Flask llegan como secretos de Cloud Run
    os.environ['MLFLOW_AUTH_CONFIG_PATH'] = str(AUTH_CONFIG_PATH)
    os.execvp('mlflow', [
        'mlflow', 'server',
        '--app-name', 'basic-auth',
        '--backend-store-uri', postgres_uri('postgres'),
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
