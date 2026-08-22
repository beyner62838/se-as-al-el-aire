# Expediente tecnico preliminar
## Señas Al Aire

**Documento:** borrador para evaluar patente, modelo de utilidad y registro de software
**Pais de referencia:** Colombia
**Version:** 1.0
**Fecha:** 22 de agosto de 2026

> Este documento es un borrador tecnico y no constituye una opinion legal ni garantiza que la invencion sea patentable. Antes de divulgar el proyecto publicamente, solicita una busqueda de anterioridades y asesoria de propiedad industrial.

## 1. Datos de identificacion

- **Titulo propuesto:** Sistema y metodo hibrido para reconocimiento en tiempo real de señas mediante landmarks de mano, imagen y secuencias temporales.
- **Nombre comercial:** Señas Al Aire.
- **Inventor o inventores:** [Completar nombres completos, identificacion y domicilio].
- **Solicitante o titular:** [Completar persona natural, universidad o empresa].
- **Fecha de primera concepcion:** [Completar].
- **Fecha de primer prototipo funcional:** [Completar].
- **Repositorio y version tecnica:** [Completar URL, commit y fecha].
- **Confidencialidad:** Mantener el expediente, modelos, datos y resultados bajo control hasta definir la estrategia de proteccion.

## 2. Resumen

La invencion se refiere a un sistema computacional para reconocer lenguaje de señas mediante una camara convencional. El sistema recibe imagenes de video, detecta landmarks de la mano mediante un modelo de vision, normaliza su posicion y escala, y combina dos rutas de inferencia: una ruta de postura basada en landmarks y una ruta de imagen basada en una red convolucional. Para señas dinamicas, el sistema acumula una ventana temporal de 32 cuadros, calcula caracteristicas de postura, trayectoria y velocidad de la muñeca, y las procesa mediante una red recurrente GRU. Las predicciones se filtran mediante umbrales de confianza y estabilidad temporal para evitar aceptar resultados aislados. El sistema genera letras, palabras o etiquetas de señas en tiempo real y puede utilizar sintesis de voz del navegador.

## 3. Campo tecnico

La propuesta pertenece a los campos de vision por computador, aprendizaje automatico, interfaces de accesibilidad, procesamiento de video y reconocimiento automatico de lenguaje de señas.

## 4. Problema tecnico

Los sistemas basados solamente en imagen pueden ser sensibles al fondo, iluminacion, distancia y encuadre. Los sistemas basados solamente en landmarks pueden perder informacion visual de la mano. Ademas, una seña dinamica no se define unicamente por una postura, sino por la evolucion de la postura y el movimiento durante varios cuadros.

Se necesita un sistema que:

1. procese video de una camara comun en tiempo real;
2. combine informacion geometrica y visual;
3. diferencie posturas estaticas de secuencias dinamicas;
4. reduzca predicciones falsas cuando no existe suficiente confianza;
5. funcione en un navegador sin instalar software en el dispositivo del usuario.

## 5. Solucion tecnica propuesta

El sistema comprende:

1. **Captura:** una interfaz web obtiene cuadros de la camara del dispositivo.
2. **Preprocesamiento:** cada cuadro se codifica y se envia al servidor de inferencia.
3. **Deteccion:** MediaPipe Hand Landmarker identifica hasta una mano y sus 21 landmarks tridimensionales.
4. **Ruta geometrica:** los landmarks se trasladan respecto a la muñeca, se normalizan por escala y se clasifican mediante un MLP.
5. **Ruta visual:** se recorta la region de la mano, se redimensiona y se procesa mediante una CNN MobileNetV2.
6. **Fusion:** las probabilidades de ambas rutas se combinan con pesos configurados, actualmente 78% para landmarks y 22% para imagen.
7. **Secuencia temporal:** se almacenan 32 observaciones consecutivas; se calculan postura normalizada, trayectoria de muñeca y velocidad de muñeca, y se clasifican mediante una GRU.
8. **Rechazo:** una prediccion dinamica por debajo del umbral configurado se marca como no reconocida.
9. **Estabilidad:** la interfaz exige que el resultado dinamico se repita en ventanas consecutivas antes de mostrarlo como resultado aceptado.
10. **Salida:** se presenta la letra o seña, su confianza y, opcionalmente, una palabra construida y salida de voz.

## 6. Descripcion de las figuras

Estas figuras deben elaborarse como diagramas numerados y adjuntarse en la solicitud si son necesarias para comprender la invencion:

- **Figura 1:** diagrama general de captura, deteccion, inferencia y salida.
- **Figura 2:** detalle de la ruta hibrida de landmarks e imagen.
- **Figura 3:** proceso de construccion de una secuencia de 32 cuadros.
- **Figura 4:** normalizacion de landmarks y calculo de trayectoria y velocidad.
- **Figura 5:** flujo de rechazo por confianza y confirmacion por estabilidad.
- **Figura 6:** arquitectura de despliegue web con navegador, API Django y modelos.

## 7. Descripcion detallada de realizacion

### 7.1 Entrada

El navegador solicita permiso para usar la camara. Cada captura se representa como una imagen y se envia a un endpoint HTTP de inferencia. El servidor valida el tipo de contenido, el tamaño de la solicitud y la frecuencia de peticiones.

### 7.2 Landmarks y normalizacion

Para cada cuadro se obtiene una matriz de 21 puntos con coordenadas `(x, y, z)`. La normalizacion resta la coordenada de la muñeca a todos los puntos y divide por la mayor distancia desde la muñeca. Esto reduce la dependencia de la posicion y escala de la mano dentro del cuadro.

### 7.3 Clasificacion estatica

La representacion normalizada contiene 63 valores. Un MLP clasifica la postura de la mano. En paralelo, un recorte de la mano se procesa con MobileNetV2. Las salidas se alinean con las clases comunes y se fusionan de forma ponderada.

### 7.4 Clasificacion dinamica

El sistema conserva las ultimas 32 observaciones validas. La funcion de caracteristicas concatena, para cada cuadro:

- 63 coordenadas normalizadas;
- 3 coordenadas de posicion relativa de la muñeca;
- 3 valores de velocidad relativa de la muñeca.

La secuencia resultante tiene forma `(32, 69)` y se procesa mediante dos capas GRU con dropout y una capa softmax para las diez clases configuradas.

### 7.5 Control de falsas detecciones

La API calcula la confianza de la clase con mayor probabilidad. Si no alcanza el umbral dinamico configurado, devuelve `no_reconocida` y no presenta la clase como valida. En el cliente, el resultado debe aparecer de forma consistente en al menos dos ventanas consecutivas antes de mostrarse.

### 7.6 Implementacion actual

- Aplicacion web: Django.
- Servidor: Gunicorn y Uvicorn.
- Deteccion: MediaPipe Hand Landmarker.
- Modelo de imagen: MobileNetV2.
- Modelo geometrico: MLP sobre 63 caracteristicas.
- Modelo temporal: GRU sobre secuencias de 32 cuadros y 69 caracteristicas por cuadro.
- Despliegue: contenedor Docker compatible con Render y otros servicios.
- Clases dinamicas actuales: `hello`, `thank_you`, `sorry`, `help`, `please`, `yes`, `no`, `water`, `good`, `bad`.

## 8. Ejemplo de funcionamiento

1. El usuario abre la aplicacion desde un navegador movil o de escritorio.
2. Autoriza el uso de la camara.
3. El sistema detecta la mano y muestra la postura o letra estatica.
4. Mientras la mano se mueve, se acumulan 32 cuadros validos.
5. La ruta temporal calcula la trayectoria y velocidad de la muñeca.
6. La GRU genera probabilidades para las diez señas dinamicas.
7. Se descartan resultados de baja confianza y se confirma un resultado estable.
8. La seña aceptada se muestra en pantalla.

## 9. Resultados y evidencia que debe anexarse

Completar esta tabla con mediciones reproducibles. No inventar valores.

| Prueba | Dataset/version | Numero de muestras | Accuracy | Precision macro | Recall macro | F1 macro | Fecha |
|---|---|---:|---:|---:|---:|---:|---|
| Letras estaticas | [Completar] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Señas dinamicas | [Completar] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Persona no vista | [Completar] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Fondos diferentes | [Completar] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Distancias diferentes | [Completar] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

Anexar tambien:

- matriz de confusion;
- curvas de entrenamiento y validacion;
- ejemplos de entradas y salidas;
- video de demostracion con fecha;
- hash del commit del codigo;
- lista de versiones de Python y dependencias;
- descripcion de la licencia y procedencia de cada dataset;
- informe de pruebas de latencia y consumo;
- registro de cambios del modelo y sus pesos.

## 10. Reivindicaciones preliminares para revision profesional

Estas reivindicaciones son un punto de partida tecnico, no un texto final para radicar.

### Reivindicacion 1: sistema

Un sistema computacional para reconocimiento de señas a partir de video, que comprende: un modulo de captura de cuadros; un detector de landmarks de mano; un primer clasificador que recibe landmarks normalizados; un segundo clasificador que recibe un recorte de imagen de la mano; un modulo de fusion que combina las salidas de los dos clasificadores; un modulo de acumulacion de una pluralidad de cuadros; un extractor que calcula postura normalizada, trayectoria y velocidad de muñeca; un clasificador recurrente de secuencias; y un modulo de rechazo y estabilizacion que impide presentar una seña dinamica cuando la confianza o la consistencia temporal sean inferiores a umbrales configurados.

### Reivindicacion 2: metodo

Un metodo implementado por computador que comprende capturar cuadros, detectar landmarks, normalizar cada mano respecto de la muñeca y su escala, clasificar simultaneamente la postura por landmarks y la imagen recortada, fusionar sus probabilidades, acumular 32 cuadros validos, calcular trayectoria y velocidad de muñeca, clasificar la secuencia y aceptar el resultado solo despues de aplicar umbral de confianza y confirmacion temporal.

### Reivindicacion 3: representacion temporal

El metodo de la reivindicacion 2, donde cada elemento de la secuencia comprende 63 coordenadas normalizadas, 3 coordenadas de posicion relativa de muñeca y 3 valores de velocidad relativa, formando una representacion de 69 caracteristicas por cuadro.

### Reivindicacion 4: despliegue

El sistema de la reivindicacion 1, donde la captura se ejecuta en un navegador y la inferencia se expone mediante una API web en un contenedor reproducible que carga modelos preentrenados sin ejecutar entrenamiento durante la operacion de produccion.

### Reivindicacion 5: accesibilidad

El sistema de la reivindicacion 1, donde la salida aceptada se utiliza para construir una palabra y generar audio mediante una interfaz de sintesis de voz del navegador.

## 11. Analisis preliminar de patentabilidad

La idea general de reconocer señas con redes neuronales, camara, CNN, landmarks o GRU puede tener antecedentes abundantes. La posible materia diferenciadora debe demostrarse con una busqueda de anterioridades y resultados tecnicos comparables.

Antes de presentar una solicitud:

1. identificar con precision que parte es nueva;
2. buscar patentes y publicaciones sobre fusion de landmarks e imagen, secuencias de muñeca y rechazo temporal;
3. aumentar el dataset y separar entrenamiento, validacion y prueba por signante;
4. documentar fechas, autores, versiones y contribuciones;
5. revisar si la solucion produce un efecto tecnico verificable;
6. consultar a un abogado o experto en propiedad industrial.

## 12. Protecciones complementarias

Aunque la patente no sea concedida, evaluar simultaneamente:

- **Derecho de autor:** codigo fuente, interfaz, documentacion y elementos graficos.
- **Secreto empresarial:** pesos del modelo, configuraciones, datos curados y procedimientos no publicados.
- **Marca:** nombre y logotipo Señas Al Aire.
- **Contratos:** cesion de derechos de inventores, universidad, colaboradores y datasets.
- **Licencias:** conservar evidencia de que los datasets permiten el uso previsto.

## 13. Lista de documentos para una consulta con la SIC

- Este expediente tecnico.
- Petitorio oficial de patente.
- Resumen independiente.
- Descripcion completa.
- Reivindicaciones revisadas.
- Figuras numeradas.
- Resultados experimentales.
- Identificacion de inventores y solicitante.
- Documento de cesion, si el titular no es el inventor.
- Poder, si actua un abogado.
- Comprobante de la tasa vigente.
- Evidencia de busqueda de anterioridades.

La SIC describe como componentes de la solicitud el petitorio, resumen, descripcion, reivindicaciones, figuras o ejemplos y comprobante de pago. La solicitud se presenta mediante SIPI o en los canales indicados por la SIC. Revisar siempre los formularios, tarifas y requisitos vigentes antes de radicar.

## 14. Fuentes oficiales de consulta

- SIC, Patentes y nuevas creaciones: https://www.sic.gov.co/patentes
- SIPI, Oficina Virtual de Propiedad Industrial: http://sipi.sic.gov.co/
- Normativa de propiedad industrial: https://www.sic.gov.co/repositorio-de-normatividad
- Proyecto y demostracion: https://se-as-al-el-aire.onrender.com

## 15. Pendientes antes de radicar

- [ ] Completar inventores y titular.
- [ ] Definir paises donde se solicitara proteccion.
- [ ] Realizar busqueda de anterioridades.
- [ ] Obtener mediciones reproducibles.
- [ ] Revisar licencias de WLASL y del dataset de Hugging Face.
- [ ] Elaborar figuras.
- [ ] Revisar las reivindicaciones con un profesional.
- [ ] No divulgar detalles tecnicos nuevos antes de definir la fecha de presentacion.
