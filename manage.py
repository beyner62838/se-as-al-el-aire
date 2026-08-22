#!/usr/bin/env python
"""Entry point principal de Django. Todos los comandos administrativos
(runserver, migrate, collectstatic, createsuperuser, etc.) pasan por aquí."""
import os
import sys


def main():
    """Configura la variable de entorno DJANGO_SETTINGS_MODULE y ejecuta
    el comando recibido por línea de comandos."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'proyecto.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
