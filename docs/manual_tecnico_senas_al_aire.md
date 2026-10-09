# Manual técnico
## Señas Al Aire

- **Documento:** manual técnico de la aplicación web y de sus modelos de reconocimiento.
- **Versión del documento:** 2.0.
- **Fecha de revisión:** 8 de octubre de 2026.
- **Repositorio:** [beyner62838/se-as-al-el-aire](https://github.com/beyner62838/se-as-al-el-aire).
- **Base de revisión del código:** rama `main`, commit `80d86e7` (26 de septiembre de 2026).
- **Plantilla documental:** Manual Técnico GradeBoost-API, secciones 1 a 7.

## Introducción

Este manual documenta el propósito, la arquitectura, la instalación, la API y los modelos de Señas Al Aire. Es una aplicación web desarrollada con Django que procesa imágenes de una cámara para reconocer posturas del alfabeto ASL y ofrece una ruta experimental para clasificar secuencias de señas dinámicas. La documentación sigue la organización del Manual Técnico GradeBoost-API; la tecnología y los ejemplos se han sustituido por los que corresponden a este proyecto.

El documento está dirigido a desarrolladores y personas responsables de instalar, operar o mantener el sistema. Las especificaciones se contrastaron con el código fuente disponible en el commit indicado. Las métricas de entrenamiento no se presentan como resultados generales: requieren una evaluación reproducible con un conjunto independiente y separado por signante.

## 1. Objeto y finalidad del software

### 1.1. Objeto del software

Señas Al Aire es una aplicación web para reconocimiento asistido de señas de la lengua de señas estadounidense (ASL). La interfaz captura imágenes desde la cámara del navegador y las envía a una aplicación Django. El servidor detecta una mano con MediaPipe y clasifica la imagen mediante una combinación de un modelo de landmarks y un modelo visual. La aplicación también incluye un endpoint para inferencia experimental de secuencias temporales.

El sistema no traduce conversaciones completas ni reconoce de forma general cualquier lengua de señas. La ruta estática cubre las 26 letras del alfabeto ASL y las clases especiales `del`, `nothing` y `space`; la ruta dinámica cubre diez etiquetas configuradas. Los resultados son predicciones del modelo y no sustituyen la interpretación humana ni constituyen una herramienta certificada de comunicación.

### 1.2. Finalidad del software

La finalidad es ofrecer una demostración funcional de visión artificial aplicada al reconocimiento de señas: visualizar la captura, obtener predicciones estáticas, construir palabras letra por letra y pronunciar el texto mediante la síntesis de voz del navegador. Para desarrollo local también se proporcionan rutinas de preparación, entrenamiento y evaluación de modelos.

La salida debe interpretarse considerando las limitaciones del conjunto de datos, la variación entre signantes y las condiciones de captura. En particular, el clasificador dinámico se considera experimental hasta que se publique una evaluación independiente y reproducible.

## 2. Arquitectura y componentes técnicos

La aplicación sigue una arquitectura web cliente-servidor. El navegador administra la cámara y la interfaz; Django sirve las páginas y recibe las solicitudes; los módulos de `proyecto/ml/` realizan la detección, el preprocesamiento y la inferencia. Los modelos se cargan bajo demanda y se conservan en caché del proceso cuando corresponde.

### 2.1. Capas y responsabilidades

- **Interfaz web:** `proyecto/templates/index.html` y `proyecto/static/`. Solicita permiso de cámara, presenta el estado del sistema y envía imágenes para su clasificación.
- **Enrutamiento y configuración:** `proyecto/urls.py` declara las rutas y `proyecto/settings.py` configura Django, seguridad, base de datos, archivos estáticos y variables de entorno.
- **Vistas y API:** `proyecto/views.py` implementa las páginas, el entrenamiento/evaluación y los endpoints JSON.
- **Detección y clasificación estática:** MediaPipe Hand Landmarker obtiene hasta 21 puntos de una mano. La rama geométrica prepara 63 coordenadas para el MLP; la rama visual recorta la mano y prepara una imagen de 96 × 96 para MobileNetV2. Sus probabilidades se fusionan con pesos de 0.78 y 0.22, respectivamente.
- **Clasificación dinámica:** `sequence_dataset.py` valida y remuestrea los landmarks; `sequence_inference.py` prepara la entrada; `temporal_model.py` define la red GRU. La entrada es una secuencia de 32 frames con 69 características por frame.
- **Entrenamiento y datos:** `dataset.py` apoya la preparación estática; `prepare_wlasl.py` y `prepare_hf_video_dataset.py` preparan videos para la ruta dinámica; `train_dynamic.py` entrena el clasificador temporal.
- **Despliegue:** `Dockerfile` configura la imagen y el servidor Gunicorn con Uvicorn; `build.sh` instala dependencias y recoge los archivos estáticos.

### 2.2. Estructura del proyecto en código fuente

```text
se-as-al-el-aire/
├── manage.py, requirements.txt, Dockerfile, build.sh
├── docs/
│   ├── manual_tecnico_senas_al_aire.md
│   └── manual_tecnico_senas_al_aire.docx
└── proyecto/
  ├── settings.py, urls.py, views.py, tests.py
  ├── ml/                           # Detección, features, inferencia y entrenamiento
  ├── models/                       # Modelos Keras, clases y MediaPipe
  ├── templates/index.html
  └── static/                       # CSS e imágenes
```

Los datasets descargados, el entorno virtual, la base SQLite local y los archivos temporales se excluyen del control de versiones mediante `.gitignore`.

## 3. Instalación y ejecución

### 3.1. Requisitos

- Python 3.11; el contenedor oficial del proyecto utiliza `python:3.11-slim`.
- pip y Git.
- Cámara web y navegador con acceso a cámara para la interfaz.
- Dependencias del sistema para MediaPipe en Linux. La imagen Docker instala las bibliotecas gráficas requeridas.
- Conexión a internet para descargar dependencias, el dataset de Kaggle o el modelo de MediaPipe cuando no estén disponibles localmente.

### 3.2. Instalación local en Linux

Desde una terminal, clona el repositorio, crea un entorno virtual e instala las dependencias:

```bash
git clone https://github.com/beyner62838/se-as-al-el-aire.git
cd se-as-al-el-aire
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

En PowerShell, activa el entorno con `.venv\\Scripts\\Activate.ps1` y ejecuta los mismos comandos de instalación con `python`. La aplicación local queda disponible en `http://127.0.0.1:8000/`. La cámara normalmente requiere un origen seguro: `localhost` es válido para desarrollo; en una instalación remota debe usarse HTTPS.

### 3.3. Configuración y despliegue

La configuración se controla principalmente mediante variables de entorno:

| Variable | Uso |
|---|---|
| `SECRET_KEY` | Clave secreta de Django. Debe ser privada y configurarse en producción. |
| `DEBUG` | Activa el modo de depuración. En producción debe ser `false`. |
| `ALLOW_TRAINING` | Habilita el entrenamiento desde `/prediccion/`. En producción se recomienda `false`. |
| `ALLOWED_HOSTS` | Lista de hosts permitidos, separados por comas. |
| `CSRF_TRUSTED_ORIGINS` | Orígenes HTTPS de confianza para solicitudes protegidas por CSRF. |
| `DATABASE_URL` | Si está presente, configura la base de datos mediante `dj-database-url`; localmente se usa SQLite. |
| `PORT` | Puerto de escucha del contenedor; el Dockerfile establece `7860` como valor predeterminado. |

El Dockerfile recoge los estáticos con `collectstatic` y ejecuta Gunicorn con Uvicorn. `build.sh` prepara dependencias y estáticos; las migraciones de producción deben ejecutarse en la fase de release o predeploy cuando la base de datos esté disponible. El README del repositorio describe los parámetros de Render y Hugging Face Spaces. No se deben incluir secretos en Git ni habilitar entrenamiento público sin una evaluación de recursos y controles adicionales.

## 4. Endpoints

Las rutas pertenecen a Django. No se expone Swagger UI, ReDoc ni un esquema OpenAPI generado automáticamente. La página principal es HTML; las predicciones y los errores de las vistas de inferencia se devuelven como JSON.

### 4.1. `GET /` y `GET /main/` — Interfaz principal

Renderizan `index.html`. La página principal entrega la cookie CSRF necesaria para las solicitudes POST desde el navegador y presenta el estado de disponibilidad de los modelos.

### 4.2. `GET/POST /prediccion/` — Entrenamiento y evaluación estáticos

GET presenta la interfaz. POST ejecuta la preparación, carga o entrenamiento de los modelos estáticos y la evaluación configurada cuando `ALLOW_TRAINING` está habilitado. Si el entrenamiento está desactivado, la vista presenta un aviso en la página; no es un endpoint de predicción JSON. El entrenamiento descarga el dataset `grassknoted/asl-alphabet` de Kaggle cuando hace falta.

### 4.3. `POST /camera_predict/` — Clasificación de una imagen

Requiere `Content-Type: application/json` y un objeto JSON con la propiedad `image`, que contiene la imagen en Base64 (puede incluir el prefijo `data:image/jpeg;base64,`). El servidor comprueba el método, el tipo de contenido, el tamaño del cuerpo y el límite de frecuencia; después detecta la mano y ejecuta la clasificación híbrida.

La respuesta incluye `predicted`, `raw_label`, `confidence`, `action`, `token`, `hand_detected`, `preprocess_mode`, `prediction_mode`, los resultados individuales de landmarks e imagen, `handedness` y `landmarks`. El valor concreto depende de la imagen. Las solicitudes POST deben respetar la protección CSRF de Django y enviar el token correspondiente.

### 4.4. `POST /sequence_predict/` — Clasificación de una secuencia

Requiere JSON con `frames`, una lista no vacía de frames. Cada frame debe representar 21 landmarks con coordenadas `x`, `y`, `z`; la secuencia se remuestrea a 32 frames antes de formar las características. Si el archivo del modelo dinámico no existe o es menor que el tamaño mínimo configurado, la vista devuelve `503`.

Con un modelo disponible, la respuesta contiene `predicted`, `confidence`, `class_index` y `accepted`. Si la confianza es inferior a `0.75`, `predicted` es `no_reconocida` y `accepted` es `false`. Los POST también están sujetos a la protección CSRF de Django.

## 5. Modelo de Machine Learning

### 5.1. Clasificador estático híbrido

| Rama | Artefacto | Entrada | Clases |
|---|---|---|---|
| Visual | `asl_alphabet_mobilenet_v2.keras` | Recorte RGB de la mano, redimensionado a 96 × 96 | 29: A-Z, `del`, `nothing`, `space` |
| Geométrica | `asl_landmarks_mlp.keras` | 63 coordenadas de 21 landmarks | 28 clases; excluye `nothing` según el código |

MediaPipe Hand Landmarker obtiene los puntos de la mano. La ruta geométrica normaliza el origen, orientación y escala de las coordenadas; la ruta visual clasifica el recorte. La fusión definida en `proyecto/ml/constants.py` es:

```text
probabilidad_final = 0.78 × probabilidad_landmarks
                  + 0.22 × probabilidad_imagen
```

El umbral estático de aceptación configurado es `0.72`. Las etiquetas `del`, `nothing` y `space` se traducen en acciones de interfaz. Si no se detecta una mano, la vista no debe interpretarse como una clasificación válida.

### 5.2. Clasificador dinámico experimental

El clasificador dinámico usa dos capas GRU (96 y 48 unidades), dropout y una salida `softmax` de diez clases. Su entrada tiene forma `(32, 69)`: 63 coordenadas normalizadas, tres valores de posición relativa de la muñeca y tres valores de velocidad relativa por frame. Las secuencias recibidas se validan e interpolan a 32 frames.

Las clases configuradas, en el orden de los índices del modelo, son `hello`, `thank_you`, `sorry`, `help`, `please`, `yes`, `no`, `water`, `good` y `bad`. El umbral dinámico de aceptación es `0.75`. Los preparadores disponibles consultan videos de WLASL y del dataset `ZahidYasinMittha/American-Sign-Language-Dataset` en Hugging Face; se debe revisar la disponibilidad, licencia y condiciones de uso de los datos antes de preparar o publicar modelos.

### 5.3. Artefactos del repositorio

| Archivo | Función |
|---|---|
| `asl_alphabet_mobilenet_v2.keras` | Clasificador visual estático. |
| `asl_alphabet_class_names.json` | Orden de clases del clasificador visual. |
| `asl_landmarks_mlp.keras` | Clasificador estático de landmarks. |
| `asl_landmarks_metadata.json` | Clases compatibles y versión de caché del MLP. |
| `asl_dynamic_gru.keras` | Clasificador temporal experimental. |
| `asl_dynamic_class_names.json` | Orden de clases dinámicas. |
| `hand_landmarker.task` | Modelo de detección de manos de MediaPipe. |

Los clasificadores se cargan bajo demanda y se mantienen en caché del proceso. La presencia de un archivo de modelo no constituye por sí sola evidencia de precisión o generalización.

### 5.4. Alcance y limitaciones de resultados

En el commit documentado no se incluyen datasets de entrenamiento, particiones reproducibles ni métricas independientes suficientes para afirmar una precisión general. Las imágenes de gráficas están excluidas por `.gitignore`; por tanto, no se consideran anexos disponibles en el repositorio en esta revisión. Para publicar resultados se deben registrar el origen y licencia de los datos, las versiones de dependencias y modelos, el protocolo de partición por signante, las métricas por clase y una matriz de confusión generada sobre un conjunto de prueba independiente.

La salida puede variar por iluminación, fondo, distancia, oclusión, lateralidad y diferencias entre signantes. El vocabulario dinámico se limita a diez clases y el modelo dinámico debe tratarse como experimental hasta completar esa evaluación.

## 6. Manejo de errores

Las vistas JSON de inferencia devuelven un objeto con la clave `error` para los errores que manejan explícitamente. Django y sus middlewares pueden producir respuestas distintas para rutas inexistentes o fallos de CSRF; no todas las respuestas de error del sitio tienen necesariamente el mismo formato JSON.

| Código HTTP | Situación documentada |
|---:|---|
| `400` | JSON inválido, cuerpo que no sea objeto, imagen ausente, frames vacíos o datos de entrada inválidos. |
| `403` | Solicitud POST rechazada por la protección CSRF de Django. |
| `405` | Método diferente de POST en un endpoint de predicción. |
| `413` | Cuerpo de solicitud de cámara que supera el límite implementado. |
| `415` | El endpoint de cámara o secuencia no recibe `application/json`. |
| `429` | Se supera el límite configurado para el endpoint de cámara: diez solicitudes por segundo y dirección IP, usando la caché de Django. |
| `500` | Excepción inesperada durante la inferencia o salida incompatible del modelo. |
| `503` | El artefacto dinámico no está disponible o no alcanza el tamaño mínimo. |

Ejemplo de error de validación:

```json
{
  "error": "El contenido debe ser JSON."
}
```

El límite de frecuencia depende del backend de caché configurado; el backend local de Django no sustituye un limitador distribuido cuando se ejecutan varios procesos o instancias. Los detalles internos de excepciones no deben exponerse a usuarios en producción.

## 7. Ejemplo de uso y verificación

### 7.1. Solicitud estática con cURL

Primero se obtiene la cookie CSRF desde la página principal; el valor del token debe enviarse en `X-CSRFToken` junto con la cookie:

```bash
curl -c cookies.txt http://127.0.0.1:8000/
```

Ejemplo de POST (sustituye `<csrf-token>` y `<imagen-base64>` por valores válidos):

```bash
curl -X POST http://127.0.0.1:8000/camera_predict/ \\
  -b cookies.txt \\
  -H 'Content-Type: application/json' \\
  -H 'X-CSRFToken: <csrf-token>' \\
  --data '{"image":"data:image/jpeg;base64,<imagen-base64>"}'
```

Ejemplo abreviado de respuesta exitosa; los valores son ilustrativos y no una medición del modelo:

```json
{
  "predicted": "A",
  "raw_label": "A",
  "confidence": "0.95",
  "action": "letter",
  "token": "a",
  "hand_detected": true,
  "preprocess_mode": "mediapipe-landmarks",
  "prediction_mode": "hybrid-landmarks-image",
  "landmark_label": "A",
  "landmark_confidence": 0.96,
  "image_label": "A",
  "image_confidence": 0.91,
  "handedness": "Right",
  "landmarks": []
}
```

### 7.2. Solicitud dinámica con cURL

La muestra siguiente solo ilustra el contrato. En una solicitud real deben enviarse uno o más frames y exactamente 21 puntos `[x, y, z]` por frame:

```bash
curl -X POST http://127.0.0.1:8000/sequence_predict/ \\
  -b cookies.txt \\
  -H 'Content-Type: application/json' \\
  -H 'X-CSRFToken: <csrf-token>' \\
  --data '{"frames":[[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]]}'
```

Las coordenadas en este ejemplo son ceros y solo ilustran un cuerpo JSON válido; para obtener una predicción útil se deben enviar los landmarks reales de la mano.

Respuesta exitosa ilustrativa:

```json
{
  "predicted": "hello",
  "confidence": "0.87",
  "class_index": 0,
  "accepted": true
}
```

### 7.3. Entrenamiento dinámico local

Los preparadores generan archivos `.npz` con `sequences` de forma `(ejemplos, frames, 21, 3)` y `labels` como índices de clase entre 0 y 9. Los comandos siguientes pueden descargar videos y requieren revisar sus licencias y disponibilidad:

```bash
python -m proyecto.ml.prepare_wlasl --max-per-class 10 --frames 32
python -m proyecto.ml.prepare_hf_video_dataset --max-per-class 10 --frames 32
python -m proyecto.ml.train_dynamic --input data/dynamic_sequences.npz data/dynamic_sequences_hf.npz
```

El entrenamiento guarda `proyecto/models/asl_dynamic_gru.keras` y `proyecto/models/asl_dynamic_class_names.json`. Ejecuta las pruebas de regresión con:

```bash
python manage.py test proyecto.tests --verbosity 1
```

Las pruebas cubren métodos HTTP, JSON, CSRF, disponibilidad del modelo dinámico y formas/características de landmarks. Una ejecución satisfactoria confirma esos comportamientos, no la precisión del reconocimiento.

### 7.4. Referencia documental y mantenimiento

El README del repositorio contiene la guía de instalación, uso y despliegue. Este manual amplía el contrato de la API y el funcionamiento de los modelos. Antes de cada entrega, verifica que las clases de los JSON correspondan con las salidas de sus modelos, ejecuta las pruebas, revisa los cambios de seguridad y registra el commit, las versiones de dependencias y la procedencia/licencia de los datos utilizados.

**Referencias principales:**

- Repositorio del proyecto: <https://github.com/beyner62838/se-as-al-el-aire>
- README de GradeBoost-API, consultado como referencia de organización documental: <https://github.com/jose-llanos/GradeBoost-API>
- Dataset estático: <https://www.kaggle.com/datasets/grassknoted/asl-alphabet>
- MediaPipe Hand Landmarker: <https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker>
- Dataset dinámico configurado en Hugging Face: <https://huggingface.co/datasets/ZahidYasinMittha/American-Sign-Language-Dataset>