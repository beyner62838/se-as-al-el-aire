# ==============================================================================
# Dockerfile para Señas Al Aire - Reconocimiento de lenguaje de señas ASL
#
# Construye una imagen liviana de Python 3.11 con:
#   - Dependencias del sistema necesarias para MediaPipe (OpenGL, EGL)
#   - Dependencias Python del proyecto
#   - Modelos pre-entrenados embebidos
#   - Archivos estáticos recogidos con collectstatic
#
# El servidor arranca con Gunicorn + Uvicorn (ASGI) para soportar
# las conexiones WebSocket/largas que pueda requerir MediaPipe.
# ==============================================================================

FROM python:3.11-slim

# --- Variables de entorno ---
ENV PYTHONDONTWRITEBYTECODE=1  
ENV PYTHONUNBUFFERED=1         
ENV PORT=7860                 
ENV MPLCONFIGDIR=/tmp/matplotlib  
ENV TF_CPP_MIN_LOG_LEVEL=2     

WORKDIR /app

# --- Dependencias del sistema (OpenGL/EGL para MediaPipe) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libegl1 \
    libgles2 \
    && rm -rf /var/lib/apt/lists/*

# --- Dependencias Python ---
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Código fuente y modelos pre-entrenados ---
COPY . .

# --- Recoger archivos estáticos (CSS, JS, imágenes de gráficas) ---
RUN python manage.py collectstatic --noinput

# --- Comando de inicio (Gunicorn + Uvicorn para ASGI) ---
CMD python -m gunicorn proyecto.asgi:application -k uvicorn.workers.UvicornWorker --workers 1 --timeout 180 --bind 0.0.0.0:$PORT