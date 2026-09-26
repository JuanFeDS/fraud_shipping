"""Arranca el servidor de MLflow con Postgres (Supabase) como backend y GCS como almacén de artefactos."""

import os
from urllib.parse import quote


def main():
    """Arma la URI de la base desde variables de entorno y reemplaza este proceso por mlflow server."""
    # La contraseña puede tener caracteres reservados de URL (@, /, #); sin codificarla la URI se rompe
    clave = quote(os.environ['DB_PASSWORD'], safe='')
    # Driver explícito: SQLAlchemy 2.1 usa psycopg 3 por defecto con postgresql:// y la imagen trae psycopg2
    uri_base = f"postgresql+psycopg2://{os.environ['DB_USER']}:{clave}@{os.environ['DB_HOST']}:5432/postgres?sslmode=require"
    os.execvp('mlflow', [
        'mlflow', 'server',
        '--backend-store-uri', uri_base,
        '--artifacts-destination', os.environ['ARTIFACTS_DESTINATION'],
        '--serve-artifacts',
        '--allowed-hosts', os.environ['ALLOWED_HOSTS'],
        '--host', '0.0.0.0',
        '--port', os.environ.get('PORT', '8080'),
        '--workers', '1',
    ])


if __name__ == '__main__':
    main()
