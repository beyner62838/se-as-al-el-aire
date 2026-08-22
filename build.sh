#!/usr/bin/env bash
# Script de build para Render.
# set -o errexit detiene la ejecución si algún comando falla.
set -o errexit

# Actualizar pip e instalar dependencias
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Recoger archivos estáticos (CSS, JS, imágenes de gráficas)
python manage.py collectstatic --noinput

# Las migraciones deben ejecutarse como paso de pre-deploy o release. Durante el
# build la base de datos de producción puede no estar disponible todavía.
