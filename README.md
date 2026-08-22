---
title: Señas Al Aire
sdk: docker
app_port: 7860
---

# Señas Al Aire

Aplicación web en Django para reconocer señas del alfabeto ASL (American Sign Language) usando la cámara del dispositivo con un sistema híbrido de redes neuronales.

## Descripción

El sistema combina dos modelos de deep learning para lograr un reconocimiento robusto de las 29 clases del alfabeto ASL (A-Z, del, nothing, space):

1. **CNN MobileNetV2** con transfer learning, entrenada sobre el dataset `grassknoted/asl-alphabet` de Kaggle.
2. **MLP (perceptrón multicapa)** entrenado sobre landmarks de la mano extraídos con **MediaPipe Hand Landmarker**.
3. **Sistema híbrido**: combinación ponderada (78% landmarks + 22% CNN) para la predicción final.

### Características

- Reconocimiento en tiempo real a través de la cámara.
- Construcción de palabras letra por letra.
- Síntesis de voz (Web Speech API) con acento español.
- Panel de sensibilidad ajustable (Precisa/Normal/Rápida).
- Entrenamiento y evaluación local del modelo.
- 3 gráficas de resultados: matriz de confusión, precisión por clase, distribución de confianza.
- Preparado para despliegue en Render (inferencia sin reentrenar).

### Base para senas dinamicas

La aplicacion conserva las letras estaticas y ya incluye una ruta temporal para
recibir ventanas de 32 frames con landmarks. Las diez clases iniciales son:
`hello`, `thank_you`, `sorry`, `help`, `please`, `yes`, `no`, `water`, `good` y
`bad`.

Para entrenar localmente, prepara un archivo NPZ con `sequences` de forma
`(ejemplos, frames, 21, 3)` y `labels` con indices de 0 a 9, y ejecuta:

```powershell
python -m proyecto.ml.prepare_wlasl --max-per-class 10 --frames 32
python -m proyecto.ml.prepare_hf_video_dataset --max-per-class 10 --frames 32
python -m proyecto.ml.train_dynamic --input data/dynamic_sequences.npz data/dynamic_sequences_hf.npz
```

El modelo generado se guarda en `proyecto/models/asl_dynamic_gru.keras`.
Mientras ese archivo no exista, `/sequence_predict/` responde `503` para no
mostrar predicciones inventadas. El dataset de videos debe descargarse y
usarse conforme a su licencia; no debe subirse completo al despliegue.

El primer modelo generado localmente con el piloto de WLASL es experimental:
solo contiene una secuencia por clase y no debe considerarse preciso. Para una
version publicable se necesitan varias personas y varias muestras por cada
clase, con una validacion separada por signante.

El segundo preparador usa videos ASL alojados directamente en Hugging Face y
solo descarga las clases solicitadas. Verifica siempre la licencia y el estado
del dataset antes de publicar un modelo entrenado con sus videos.

## Probar la aplicacion desplegada

La version desplegada esta disponible en:

https://se-as-al-aire-1.onrender.com

Importante:

- Como el servicio esta desplegado en el plan free de Render, si no recibe peticiones por un tiempo se apaga automaticamente.
- Cuando alguien abra la URL, es posible que tenga que esperar al menos 1 minuto mientras Render recibe la peticion y vuelve a levantar el servicio.

## Arquitectura del sistema

```mermaid
flowchart LR
    A[Cámara\nframe] --> B[MediaPipe\nHand Landmarker]
    B --> C[MLP\n63→256→128→64→29]
    C -->|78%| E[Combinación\nponderada]

    A --> D1[Crop de\nla mano]
    D1 --> D2[MobileNetV2\npreprocess]
    D2 --> D3[CNN imagen]
    D3 -->|22%| E

    E --> F[Letra detectada]
```

## Requisitos

- **Windows 10 o superior** (o Linux con dependencias OpenGL).
- **Python 3.11 o 3.12**.
- Conexión a internet en la primera ejecución (descarga dataset y modelos).
- Cámara web para inferencia en vivo.

## Instalación y ejecución local

```powershell
git clone https://github.com/camiloGoca/Se-as-Al-Aire.git
cd Se-as-Al-Aire

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python manage.py migrate
python manage.py runserver
```

Abre: http://127.0.0.1:8000/

## Uso

### Inferencia en tiempo real

1. Abre la aplicación en el navegador.
2. Presiona **"Abrir cámara"** y permite el acceso a la cámara.
3. Coloca la mano dentro del recuadro guía.
4. El reconocimiento comienza automáticamente.
5. Las letras detectadas se agregan a la palabra.
6. Presiona **"Detener cámara"** para pronunciar la palabra.

### Entrenamiento y evaluación

1. Presiona **"Entrenar o evaluar modelo"** (solo disponible en modo local).
2. El sistema descarga el dataset ASL Alphabet si es necesario.
3. Entrena o carga los modelos existentes.
4. Muestra: accuracy, loss, ejemplos de predicción y 3 gráficas de resultados.

## Gráficas de resultados

Al presionar "Entrenar o evaluar modelo", el sistema genera automáticamente:

| Gráfica | Descripción |
|---|---|
| **Matriz de confusión** | Heatmap de predicciones vs valores reales (validación) |
| **Precisión por clase** | Barras de accuracy por cada letra (verde ≥ 80%, amarillo ≥ 50%, rojo < 50%) |
| **Distribución de confianza** | Histograma comparativo: confianza en aciertos (verde) vs errores (rojo) |

Las gráficas se generan con matplotlib y se guardan en `proyecto/static/images/`.

## Despliegue en Render

El proyecto está configurado para desplegarse en Render:

### Variables de entorno

```text
PYTHON_VERSION=3.11.11
DEBUG=false
ALLOW_TRAINING=false
WEB_CONCURRENCY=4
DATABASE_URL=<internal database url de Render>
SECRET_KEY=<generado por Render>
ALLOWED_HOSTS=<tu-servicio>.onrender.com
```

### Comandos

- **Build:** `bash build.sh`
- **Start:** `python -m gunicorn proyecto.asgi:application -k uvicorn.workers.UvicornWorker`

> **Nota:** En producción el entrenamiento está desactivado. Los modelos vienen pre-entrenados en el repositorio.

La configuración usa variables de entorno para `SECRET_KEY`, `DEBUG` y hosts; WhiteNoise para estáticos y PostgreSQL cuando existe `DATABASE_URL`.

## Despliegue en Hugging Face Spaces

También se puede crear un Space de tipo **Docker** usando este repositorio. El
archivo `README.md` ya declara `sdk: docker` y el puerto `7860`; el contenedor
usa la variable `PORT`, por lo que sigue siendo compatible con Render.

En el Space configura como secretos o variables:

```text
DEBUG=false
ALLOW_TRAINING=false
SECRET_KEY=<clave-larga-y-segura>
ALLOWED_HOSTS=<tu-space>.hf.space
```

Sube solamente los modelos entrenados y los archivos de soporte. El dataset,
`.venv`, cachés y la base de datos local quedan excluidos por `.gitignore` y
`.dockerignore`. El plan gratuito puede dormir el Space y tiene recursos
limitados; debe medirse la latencia antes de usarlo como servicio permanente.

## Estructura del proyecto

```
django_deep_learning/
├── manage.py                     # Entry point de Django
├── requirements.txt              # Dependencias
├── Dockerfile                    # Imagen Docker para Render
├── build.sh                      # Script de build para Render
├── proyecto/
│   ├── __init__.py
│   ├── settings.py               # Configuración de Django
│   ├── urls.py                   # Enrutamiento de URLs
│   ├── views.py                  # Lógica del sistema híbrido
│   ├── asgi.py / wsgi.py         # Interfaces de servidor
│   ├── models/                   # Modelos pre-entrenados
│   │   ├── asl_alphabet_mobilenet_v2.keras
│   │   ├── asl_alphabet_class_names.json
│   │   ├── asl_landmarks_mlp.keras
│   │   ├── asl_landmarks_metadata.json
│   │   ├── asl_landmarks_dataset_v1.npz
│   │   └── hand_landmarker.task
│   ├── templates/
│   │   └── index.html            # Interfaz de usuario
│   ├── static/
│   │   ├── css/style.css         # Estilos
│   │   └── images/               # Gráficas generadas
│   └── datasets/                 # Dataset (no versionado)
│       └── asl_alphabet/
└── staticfiles/                  # Estáticos recogidos (no versionado)
```

## Archivos versionados vs ignorados

### Versionados (para Render)

En Render la app solo se usa para inferencia. El entrenamiento y la evaluación grande quedan pensados para local, no para producción.

- `proyecto/models/asl_alphabet_mobilenet_v2.keras`
- `proyecto/models/asl_alphabet_class_names.json`
- `proyecto/models/asl_landmarks_mlp.keras`
- `proyecto/models/asl_landmarks_metadata.json`
- `proyecto/models/hand_landmarker.task`

### Ignorados (regenerables o pesados)

- `proyecto/datasets/` (se descarga de Kaggle)
- `proyecto/.cache/` (caché de MediaPipe/matplotlib)
- `proyecto/models/asl_landmarks_dataset_v1.npz` (se regenera)
- `db.sqlite3` (base de datos local)
- Entornos virtuales (`.venv/`, `.venv312/`, `myenv/`)

## Tecnologías utilizadas

| Tecnología | Versión | Propósito |
|---|---|---|
| Django | 4.1 | Framework web |
| TensorFlow | ≥2.16 | Redes neuronales (CNN + MLP) |
| MediaPipe | ≥0.10.33 | Detección de manos y landmarks |
| MobileNetV2 | - | CNN pre-entrenada (transfer learning) |
| Gunicorn + Uvicorn | - | Servidor ASGI para Render |
| WhiteNoise | ≥6.9 | Servir estáticos en producción |
| PostgreSQL | - | Base de datos en producción |
| matplotlib | ≥3.8 | Generación de gráficas |
| scikit-learn | ≥1.3 | Métricas de evaluación |
| kagglehub | ≥0.3 | Descarga del dataset |

## Créditos

- **Dataset:** [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) en Kaggle.
- **MediaPipe:** [Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) de Google.
- **MobileNetV2:** [Sandler et al., 2018](https://arxiv.org/abs/1801.04381).
