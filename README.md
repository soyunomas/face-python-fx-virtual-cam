# face-python-fx-virtual-cam 🎭

Aplica efectos faciales en tiempo real (desenfoque, pixelado, emojis), y un enjambre de insectos voladores que evitan tu cara 🦋, a tu webcam usando Python, MediaPipe y OpenCV, enviando la salida a una **cámara virtual**. 😜✨

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Demo Emoji-Insects](screenshot.png)

---

## 🚀 Características

*   **Detección Facial en Tiempo Real:** Utiliza `mediapipe.solutions.face_detection`.
*   **Múltiples Efectos Faciales:**
    *   **Blur** 🌫️: Desenfoque Gaussiano configurable.
    *   **Pixelate** 👾: Efecto de pixelado con tamaño de bloque ajustable.
    *   **Emoji Overlay** 😂: Superpone un emoji grande sobre la cara (seleccionable).
    *   *Activables/Desactivables*: Los efectos faciales (Blur/Pixel/Emoji) se pueden apagar manteniendo la detección. (`[T]`)
*   **Efecto de Insectos Voladores:**
    *   Superpone emojis de insectos (🦋, 🦟, 🪰, 🐝, 🐞) que vuelan por la pantalla.
    *   **Evitación de Cabeza:** Los insectos evitan activamente la región de la cabeza detectada.
    *   Número y tipo de insectos ajustables.
    *   *Activables/Desactivables* independientemente de los efectos faciales. (`[Y]`)
*   **Salida a Cámara Virtual:** Envía el vídeo procesado a un dispositivo `/dev/videoX` usando `pyvirtualcam`.
*   **Controles Interactivos:** Cambia modos y ajusta parámetros al vuelo usando el teclado en la ventana de previsualización.
*   **Ajuste Fino:** Controla el padding (margen) alrededor de la cara, el offset vertical y el tamaño del pixelado.
*   **Configurable:** Opciones de línea de comandos para IDs de cámara, modo inicial, etc.
*   **Procesamiento de Archivos:** (Opcional) Puede procesar un archivo de video en lugar de la webcam y guardar la salida.
*   **Script de Ayuda (`run.sh`)**: Facilita la ejecución asegurando que `v4l2loopback` se cargue correctamente.

![Demo Pixelate-Insects](screenshot1.png)


## 🐧 Instalación (Linux Mint / Ubuntu / Debian-based)

Sigue estos pasos para poner en marcha el proyecto:

**1. Prerrequisitos del Sistema:**

*   **Python 3:** Versión 3.8 o superior y `pip`. Verifica con `python3 --version` y `pip3 --version`.
*   **Git:** Para clonar el repositorio (`sudo apt update && sudo apt install git -y`).
*   **wget:** Para descargar la fuente emoji (`sudo apt install wget -y`).
*   **Herramientas de Compilación:** Necesarias para algunas dependencias (`sudo apt install build-essential python3-dev -y`).
*   **v4l2loopback:** El módulo del kernel para crear cámaras virtuales (`sudo apt install v4l2loopback-dkms -y`).

**2. Cargar `v4l2loopback` (Importante: Hazlo antes de ejecutar)**

    `pyvirtualcam` requiere que el módulo se cargue con `exclusive_caps=1`. El script `run.sh` proporcionado hace esto automáticamente. Si prefieres hacerlo manualmente:

    ```bash
    # Elige un video_nr libre (e.g., 10), que usarás con --vcam-id
    sudo modprobe -r v4l2loopback # Descarga si ya estaba cargado (ignora error si no lo estaba)
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="VCamFaceFX_10" exclusive_caps=1
    ```

    Verifica que se creó el dispositivo (`ls /dev/video*` debería mostrar `/dev/video10`).

**3. Clonar el Repositorio:**

    ```bash
    git clone https://github.com/soyunomas/face-python-fx-virtual-cam.git
    cd face-python-fx-virtual-cam
    ```

**4. Descargar Fuente Emoji (Noto Color Emoji):**

    Los modos Emoji y Flying Insects requieren la fuente `NotoColorEmoji.ttf`. Esta fuente se distribuye bajo la licencia SIL OFL 1.1.

    ```bash
    wget -O NotoColorEmoji.ttf https://raw.githubusercontent.com/googlefonts/noto-emoji/main/fonts/NotoColorEmoji.ttf
    ```

    Esto descargará la fuente en el directorio actual. Asegúrate de que el archivo tenga un tamaño razonable (aprox. 15-16 MB).

**5. Configurar Entorno Virtual Python (`venv`):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # Tu prompt debería cambiar a (venv) $...
    ```

**6. Instalar Dependencias de Python:**

    Asegúrate de que tu entorno virtual esté activado (`(venv)` en el prompt). El archivo `requirements.txt` ya debería existir en el repositorio con:

    ```txt
    # requirements.txt
    opencv-python
    mediapipe
    numpy
    pyvirtualcam
    Pillow
    ```

    Instala las dependencias:

    ```bash
    pip install -r requirements.txt
    ```

¡Listo! 🎉 Ya deberías tener todo configurado.

## ▶️ Uso

**Método Recomendado (Usando `run.sh`):**

El script `run.sh` automatiza la recarga del módulo `v4l2loopback` (evitando errores comunes en ejecuciones repetidas) y lanza el script Python.

1.  **Hazlo Ejecutable (solo la primera vez):**
    ```bash
    chmod +x run.sh
    ```
2.  **Ejecuta:**
    ```bash
    ./run.sh [OPCIONES_ADICIONALES_PARA_PYTHON]
    ```
    *   El script `run.sh` se encargará de `sudo modprobe` (te pedirá contraseña).
    *   Puedes editar las variables `WEBCAM_ID` y `VCAM_ID` dentro de `run.sh` o pasarlas como argumentos.
    *   Argumentos adicionales se pasarán directamente a `webcam-emoji.py`. Ejemplo: `./run.sh --mirror --start-mode emoji`

**Método Manual (Directamente con Python):**

1.  Asegúrate de que `v4l2loopback` esté cargado correctamente (ver Paso 2 de Instalación). **Puede que necesites recargar el módulo (`sudo modprobe -r ...` y `sudo modprobe ...`) antes de cada ejecución si encuentras errores de "Device not a video output device".**
2.  Activa tu entorno virtual: `source venv/bin/activate`.
3.  Asegúrate de que `NotoColorEmoji.ttf` esté en el directorio actual (o usa `--font-path`).
4.  Ejecuta el script Python:
    ```bash
    python3 webcam-emoji.py [OPCIONES]
    ```

**Opciones Comunes de Línea de Comandos (para `webcam-emoji.py`):**

*   `--webcam-id ID`: Índice de la cámara web física (obligatorio si no se usa `--input-file`).
*   `--input-file RUTA`: Ruta a un archivo de video para procesar (excluyente con `--webcam-id`).
*   `--output-file RUTA`: Guarda la salida en un archivo (solo usable con `--input-file`, deshabilita VCam/preview).
*   `--vcam-id ID`: ID del dispositivo de cámara virtual a usar (e.g., `10` para `/dev/video10`, por defecto: 10).
*   `--font-path RUTA`: Ruta al archivo `NotoColorEmoji.ttf` (por defecto: `./NotoColorEmoji.ttf`).
*   `--start-mode MODO`: Modo de efecto facial inicial (`blur`, `pixel`, `emoji`, por defecto: `blur`).
*   `--start-pixel-size N`: Tamaño inicial del bloque de pixelado (por defecto: 8).
*   `--mirror`: Aplica espejo horizontal a la salida.
*   `--print-fps`: Muestra FPS de `pyvirtualcam` en la consola (si se usa VCam).
*   `--help`: Muestra todas las opciones disponibles.

**Ejemplo (Manual):**

```bash
# Usar cámara 0, vcam 10, empezar en modo emoji
python3 webcam-emoji.py --webcam-id 0 --vcam-id 10 --start-mode emoji
```

**Después de Ejecutar:**

*   Se abrirá una ventana de OpenCV mostrando la previsualización.
*   Si no usaste `--output-file`, la cámara virtual (e.g., "VCamFaceFX_10") debería estar disponible para seleccionarla en tu aplicación de videollamada, streaming, etc.

## ⌨️ Controles (Ventana de OpenCV activa)

*   **[Q]**: Salir de la aplicación.

**Controles Generales:**

*   **[B]**: Activar modo **Blur**. (Activa FX Cara)
*   **[P]**: Activar modo **Pixelate**. (Activa FX Cara)
*   **[E]**: Activar modo **Emoji Overlay**. (Activa FX Cara)
*   **[T]**: **Activar / Desactivar** los FX de Cara (Blur/Pixel/Emoji).
*   **[Y]**: **Activar / Desactivar** los Insectos Voladores.

*(Los siguientes controles solo aparecen si el modo/estado correspondiente está activo)*

**Ajuste Area FX (si FX Cara están activos):**

*   **[W] / [S]**: Mover area **Arriba / Abajo**.
*   **[A] / [D]**: Padding **Horizontal** <>
*   **[R] / [F]**: Padding **Vertical** ^v
*   **[+]**: **Agrandar Area** (Aumenta Padding H & V).
*   **[-]**: **Reducir Area** (Disminuye Padding H & V).
*   **[Z]**: **Resetear** ajustes de Area y Pixel.

**Ajuste Emoji (si modo Emoji activo):**

*   **[N]**: Seleccionar **Siguiente Emoji** facial.

**Ajuste Pixel (si modo Pixel activo):**

*   **[I] / [K]**: **Aumentar / Disminuir** tamaño del bloque.

**Ajuste Insectos (si Insectos activos):**

*   **[O]**: **Anadir** insecto.
*   **[L]**: **Quitar** insecto.
*   **[U]**: Cambiar **Tipo** de insecto (🦋,🦟,🪰,🐝,🐞).

---

*   **[H]**: Mostrar / Ocultar este panel de ayuda.

## ⚠️ Solución de Problemas Comunes

*   **Error `RuntimeError: ... not a video output device` / `Device or resource busy`:**
    *   **Solución Principal:** Usa el script `run.sh` para ejecutar la aplicación. Este script recarga el módulo `v4l2loopback` correctamente antes de iniciar Python.
    *   **Alternativa Manual:** Asegúrate de cargar `v4l2loopback` con `exclusive_caps=1` (Paso 2 de Instalación). Si el error persiste después de cerrar y volver a ejecutar, descarga y recarga manualmente el módulo: `sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback devices=1 video_nr=10 ... exclusive_caps=1` (ajusta el `video_nr`).
*   **Cámara virtual no aparece en otras apps:** Verifica `ls /dev/video*`, asegúrate de que `exclusive_caps=1` se usó al cargar el módulo. Reinicia la aplicación destino (Zoom, OBS, etc.) *después* de iniciar `webcam-emoji.py` o `run.sh`.
*   **Modo Emoji / Insectos no funciona o muestra '?':**
    *   Confirma que descargaste `NotoColorEmoji.ttf` correctamente (Paso 4 Instalación) y está en el directorio desde donde ejecutas el script (o que `--font-path` es correcto). Verifica el tamaño del archivo (`ls -lh`).
    *   Asegúrate de que `Pillow` está instalado en tu entorno virtual (`pip show Pillow` dentro del venv activado).
*   **Error `wget: command not found`:** Instala wget: `sudo apt install wget`.
*   **Bajo rendimiento (FPS bajos):** Cierra otras aplicaciones pesadas. Considera usar una resolución de webcam más baja si tu cámara lo permite (esto no es configurable directamente en el script).
*   **Error `ImportError: No module named 'cv2'` (o similar):** Olvidaste activar el entorno virtual (`source venv/bin/activate`) antes de ejecutar `pip install` o `python3 webcam-emoji.py` / `./run.sh`.

## 📜 Licencia

*   **Código del Proyecto:** Este proyecto se distribuye bajo la **Licencia MIT**. Consulta el archivo `LICENSE` (si existe) o la cabecera de los archivos Python para más detalles.
*   **Dependencias:** Este proyecto utiliza bibliotecas de terceros (`opencv-python`, `mediapipe`, `numpy`, `pyvirtualcam`, `Pillow`), cada una con sus propias licencias (generalmente permisivas como MIT, Apache 2.0, BSD).
*   **Fuente Emoji:** El script utiliza la fuente **Noto Color Emoji** de Google, que se descarga por separado (ver sección de Instalación) y está licenciada bajo la **SIL Open Font License (OFL) 1.1**.
```

He intentado que sea claro y cubra todas las funcionalidades y pasos necesarios. ¡Espero que te sea útil!
