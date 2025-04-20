#!/bin/bash

# --- Configuración ---
WEBCAM_ID=0
VCAM_ID=10
VCAM_DEVICE="/dev/video${VCAM_ID}"
PYTHON_SCRIPT="webcam-emoji.py" # Asegúrate que el nombre sea correcto
VENV_ACTIVATE="venv/bin/activate"   # Ajusta si tu venv está en otro lugar

# --- Lógica ---
echo "INFO: Asegurando que el módulo v4l2loopback esté cargado correctamente para ${VCAM_DEVICE}..."

# Descargar módulo si existe (ignora errores si no estaba cargado)
sudo modprobe -r v4l2loopback 2>/dev/null

# Esperar un instante muy breve (puede ayudar en algunos sistemas)
sleep 0.2

# Cargar módulo con las opciones correctas
echo "INFO: Cargando v4l2loopback con video_nr=${VCAM_ID} y exclusive_caps=1..."
sudo modprobe v4l2loopback devices=1 video_nr=${VCAM_ID} card_label="VCamFaceFX_${VCAM_ID}" exclusive_caps=1

# Verificar si el dispositivo existe ahora
if [ ! -e "${VCAM_DEVICE}" ]; then
  echo "ERROR: El dispositivo ${VCAM_DEVICE} no se creó después de modprobe. Abortando."
  exit 1
fi
echo "INFO: Dispositivo ${VCAM_DEVICE} encontrado."

# Activar entorno virtual (opcional pero recomendado)
if [ -f "${VENV_ACTIVATE}" ]; then
    echo "INFO: Activando entorno virtual..."
    source "${VENV_ACTIVATE}"
else
    echo "WARN: No se encontró script de activación de venv en ${VENV_ACTIVATE}. Ejecutando con python global."
fi

# Ejecutar el script Python
echo "INFO: Ejecutando script Python: ${PYTHON_SCRIPT}..."
python3 "${PYTHON_SCRIPT}" --webcam-id ${WEBCAM_ID} --vcam-id ${VCAM_ID} "$@" # "$@" pasa argumentos adicionales

# Desactivar venv si se activó (opcional)
# [[ -n "$VIRTUAL_ENV" ]] && deactivate # Puede que no funcione siempre dependiendo del shell

echo "INFO: Script Python terminado."
exit 0
