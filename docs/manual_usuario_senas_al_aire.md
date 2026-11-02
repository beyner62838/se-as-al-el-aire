# Manual de usuario
## Señas Al Aire

- **Documento:** guía de uso de la aplicación web Señas Al Aire.
- **Versión:** 1.0.
- **Fecha:** 2 de noviembre de 2026.
- **Responsable y contacto:** pendiente de completar por el equipo del proyecto.
- **Base de revisión del producto:** rama `main`, commit `eb09d60`; revisión documental del 2 de noviembre de 2026.
- **Idioma de esta guía:** español.

## Introducción

Este manual explica cómo abrir Señas Al Aire, dar acceso a la cámara, reconocer letras ASL, formar una palabra, ajustar la sensibilidad y solucionar problemas comunes. Está dirigido a personas usuarias, no requiere conocimientos de programación y describe la interfaz incluida en el repositorio.

Señas Al Aire es una demostración experimental de clasificación visual. Reconoce algunas posturas estáticas del alfabeto ASL y, cuando el modelo correspondiente está disponible, intenta clasificar diez señas dinámicas. No es un traductor universal, no reemplaza a una persona intérprete y no debe usarse para decisiones médicas, legales, educativas o de emergencia.

## 1. Objeto y finalidad

La aplicación usa la cámara para analizar una mano y mostrar una etiqueta estimada. Las letras aceptadas se agregan a una palabra; el navegador puede pronunciar el segmento después de una pausa o cuando se detecta el gesto especial de espacio. La etiqueta `del` borra el último carácter; la salida híbrida excluye `nothing` y no la presenta como resultado final.

La aplicación está configurada para ASL (American Sign Language). ASL y la Lengua de Señas Colombiana (LSC) son lenguas diferentes. El idioma español de la voz del navegador solo indica cómo se pronuncia el texto escrito; no convierte el reconocimiento en LSC ni en traducción entre idiomas.

## 2. Antes de empezar

- Utiliza un computador o dispositivo con cámara y un navegador moderno actualizado.
- Permite el acceso a la cámara cuando el navegador lo solicite.
- Para un sitio remoto, la conexión debe usar HTTPS. En desarrollo local, `http://localhost` suele estar permitido por el navegador.
- Colócate frente a un fondo con iluminación suficiente. Mantén una mano completa, despejada y dentro del recuadro de la cámara.
- La síntesis de voz depende del navegador, del sistema operativo y de que haya una voz compatible con español de Colombia (`es-CO`).

El README del repositorio configura `https://se-as-al-aire-1.onrender.com` como dirección pública. Al verificarla para esta edición respondió HTTP `503`; su disponibilidad puede cambiar. Consulta con el responsable del servicio antes de distribuir esta dirección como operativa.

## 3. Conocer la pantalla

La pantalla principal contiene:

| Elemento | Para qué sirve |
|---|---|
| Estado de modelos | Indica disponibilidad de la CNN, landmarks y modo de entrenamiento. La disponibilidad no garantiza precisión. |
| Abrir cámara | Solicita permiso, inicia el video y activa el reconocimiento automático. |
| Detener cámara | Detiene la captura y la cámara; intenta pronunciar la palabra pendiente. La palabra permanece en pantalla. |
| Captura manual | Envía una captura adicional para clasificarla. La cámara debe estar abierta. |
| Borrar palabra | Limpia el texto y las salidas estáticas mostradas. No borra modelos ni datos del servidor. |
| Sensibilidad | Cambia la frecuencia de captura, el mínimo de confianza y la cantidad de confirmaciones requeridas. |
| Letra | Muestra la etiqueta estática de la última imagen clasificada. |
| Seña dinámica | Muestra el resultado temporal cuando el modelo dinámico está disponible. |
| Palabra | Muestra la secuencia de letras aceptadas. |
| Confianza | Muestra la confianza asociada a la predicción estática reciente. |
| Mano y Estado | Indican detección de mano y el estado del reconocimiento. |

## 4. Uso paso a paso

### 4.1. Iniciar el reconocimiento

1. Abre la dirección de la aplicación y espera a que la página termine de cargar.
2. Revisa los indicadores de disponibilidad. Si aparece “sin entrenar” o “pendiente”, esa capacidad puede no estar lista.
3. Pulsa **Abrir cámara**.
4. En el aviso del navegador, permite el acceso a la cámara. Si hay varias cámaras, selecciona la correcta en la configuración del navegador o del sistema.
5. Coloca la mano dentro del recuadro y mantén la seña quieta durante un momento.
6. Observa los campos **Letra**, **Confianza**, **Mano** y **Estado**. El sistema requiere predicciones estables antes de añadir una letra.

El reconocimiento automático toma capturas periódicamente y las envía al servidor para inferencia. No hace falta pulsar **Captura manual** en el uso normal.

### 4.2. Ajustar sensibilidad

El control tiene tres posiciones. El umbral del navegador y la confirmación de estabilidad cambian según esta tabla; no cambian los pesos internos de los modelos.

| Ajuste | Confianza mínima en la interfaz | Intervalo aproximado | Confirmaciones iguales |
|---|---:|---:|---:|
| Precisa | 0.82 | 1.000 ms | 3 |
| Normal | 0.72 | 820 ms | 2 |
| Rápida | 0.60 | 620 ms | 1 |

**Precisa** puede reducir letras accidentales, pero tarda más y puede ignorar más señas. **Rápida** responde antes, pero es más susceptible a errores. **Normal** es el valor inicial recomendado para probar. Si cambias el control con la cámara abierta, el ciclo automático se reinicia.

### 4.3. Formar y corregir una palabra

- Mantén una letra hasta que el estado confirme que fue aceptada; la letra aparecerá en **Palabra**.
- Para evitar duplicados, el sistema exige predicciones estables y aplica una pausa breve antes de aceptar de nuevo la misma etiqueta. Cambia la postura claramente entre letras.
- El gesto especial `space` agrega un espacio y pronuncia el segmento que acaba de terminar.
- El gesto especial `del` elimina el último carácter.
- `nothing` no es una salida final del clasificador híbrido; una predicción con confianza insuficiente o la ausencia de mano no agregan texto.
- Cuando termina la palabra, el navegador programa su pronunciación tras una pausa de alrededor de 3,2 segundos. Pulsa **Detener cámara** para detener la captura y solicitar la pronunciación del segmento pendiente.
- Pulsa **Borrar palabra** para vaciar la palabra y comenzar otra. Esta acción no elimina las peticiones que ya se enviaron ni modifica datos guardados fuera del navegador.

### 4.4. Capturar una sola imagen

Con la cámara abierta, pulsa **Captura manual**. Se procesa una imagen con el mismo clasificador y filtros del modo automático. La captura manual no confirma por sí sola que la letra se añada: también se aplican los requisitos de confianza y estabilidad.

### 4.5. Ver señas dinámicas

Si el indicador informa que el modelo dinámico está disponible, la aplicación acumula landmarks a medida que reconoce la mano y analiza ventanas de 32 frames. El resultado solo se muestra después de superar el umbral dinámico y de repetirse de forma estable. Si no está disponible, la salida dinámica permanecerá vacía o mostrará que aún analiza; continúa funcionando el flujo estático cuando sus modelos están listos.

Las diez etiquetas dinámicas configuradas son `hello`, `thank_you`, `sorry`, `help`, `please`, `yes`, `no`, `water`, `good` y `bad`. Esta función es experimental, puede no aparecer en todos los despliegues y no se debe interpretar como traducción de una frase.

## 5. Interpretar resultados y límites

- Una confianza mayor no significa que la predicción sea correcta; solo expresa la puntuación del modelo para esa imagen.
- Si **Mano** indica “No detectada”, centra la mano y mejora luz, enfoque o distancia.
- Las etiquetas especiales realizan acciones de escritura y no son letras.
- La aplicación puede equivocarse con cambios de orientación, mano izquierda/derecha, fondos complejos, oclusiones, movimiento rápido y señas distintas de las clases entrenadas.
- El alfabeto manual no representa por sí solo la gramática, expresiones faciales ni todos los componentes de una lengua de señas.
- No uses el resultado como única base para comunicar información importante. Confirma la palabra con la persona usuaria.

## 6. Problemas frecuentes

| Problema | Qué probar |
|---|---|
| El navegador no solicita cámara | Revisa permisos del sitio y la configuración del sistema; confirma que ninguna otra aplicación haya bloqueado la cámara. |
| La cámara aparece negra o no inicia | Recarga la página, selecciona otra cámara disponible y comprueba que el sitio use HTTPS o localhost. |
| La mano no se detecta | Aumenta la iluminación, evita que la mano salga del recuadro, separa los dedos cuando la seña lo requiera y reduce el movimiento. |
| La letra cambia continuamente | Usa sensibilidad **Precisa**, espera a que la mano esté quieta y evita fondos visualmente cargados. |
| No se agrega una letra | Espera las confirmaciones de estabilidad, prueba **Normal** y revisa que la confianza supere el mínimo del ajuste actual. |
| Se repite o falta una letra | Cambia la postura de forma visible entre letras; hay una pausa de protección contra duplicados. Usa `del` para corregir el último carácter. |
| No se oye la palabra | Comprueba volumen y voz española del navegador; pulsa **Detener cámara** o usa el gesto de espacio para cerrar el segmento. Algunos navegadores requieren una interacción previa para habilitar audio. |
| La salida dinámica no aparece | Verifica que el modelo dinámico indique disponibilidad. Si no está entrenado o no cumple el tamaño mínimo, la API responde que no está disponible. |
| La página tarda al abrirse | El servicio gratuito puede despertar tras inactividad. Espera a que termine de cargar y evita enviar muchas capturas seguidas. |
| Aparece un error de solicitud | Recarga la página para obtener una cookie CSRF nueva y revisa que el acceso a cámara siga autorizado. Si persiste, informa al responsable del servicio sin enviar imágenes privadas. |

## 7. Privacidad y uso responsable

La cámara se activa tras una acción de la persona usuaria y el reconocimiento envía imágenes de forma periódica al servidor mientras está activo. El código de la ruta de inferencia no muestra guardado explícito de cada imagen; aun así, el operador del alojamiento, registros técnicos, respaldos o servicios de monitoreo pueden tener reglas de retención propias que esta interfaz no explica. No introduzcas datos sensibles ni muestres a otras personas sin informarles y obtener las autorizaciones que correspondan.

Para finalizar, pulsa **Detener cámara**. Para retirar el permiso del navegador, usa la configuración de permisos del sitio. El botón **Borrar palabra** solo limpia la salida visible; no es una solicitud formal de borrado de datos del servicio.

Antes de utilizar la aplicación con estudiantes, menores o en una institución, consulta el aviso de privacidad y los términos oficiales del operador. En el repositorio revisado no se identificó un aviso visible con responsable, finalidades, conservación, proveedores y canal para ejercer derechos. Si esa información no está publicada por el operador, no asumas que la aplicación es apta para un uso institucional o con menores.

## 8. Ayuda y referencias

**Responsable, correo de soporte y fecha de actualización:** campos pendientes de completar por el equipo antes de distribuir esta guía como documento final.

- [Manual técnico en Word](manual_tecnico_senas_al_aire.docx)
- [Manual técnico en Markdown](manual_tecnico_senas_al_aire.md)
- [Carta independiente sobre uso de IA](carta_uso_ia_senas_al_aire.docx)
- [Repositorio Señas Al Aire](https://github.com/beyner62838/se-as-al-el-aire)
- [Ley 1581 de 2012, Gestor Normativo de Función Pública](https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=49981)
