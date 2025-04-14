import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
import time
import sys
import signal
import argparse # <--- Incluido para argumentos CLI

# --- Variables globales para la limpieza ---
cap = None
keep_running = True

# --- Manejador para salida limpia (Ctrl+C) ---
def signal_handler(sig, frame):
    global keep_running
    print("\nINFO: Señal de interrupción recibida. Deteniendo...")
    keep_running = False

signal.signal(signal.SIGINT, signal_handler) # Captura Ctrl+C

# --- Inicialización de Componentes de MediaPipe ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def initialize_webcam(device_index):
    """Inicializa la cámara web y obtiene sus propiedades."""
    global cap
    print(f"INFO: Intentando inicializar cámara web en el índice {device_index}...")
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la cámara web en el índice {device_index}.")
        print("       Asegúrate de que esté conectada y no esté en uso.")
        return None, None, None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)

    if width <= 0 or height <= 0:
        print("WARN: No se pudieron obtener dimensiones válidas directamente. Intentando leer un fotograma...")
        ret, frame = cap.read()
        if not ret or frame is None:
            print("ERROR: No se pudo leer un fotograma inicial para obtener dimensiones.")
            if cap.isOpened(): cap.release()
            cap = None
            return None, None, None, None
        height, width = frame.shape[:2]
        print(f"INFO: Dimensiones obtenidas del primer fotograma: {width}x{height}")
        if fps_in <= 0:
            fps_in = 30
            print(f"WARN: No se pudo obtener FPS válido. Usando {fps_in:.2f} FPS por defecto.")

    fps_out = min(fps_in if fps_in > 0 else 30.0, 30.0)
    print(f"INFO: Cámara web inicializada: {width}x{height} @ {fps_in:.2f} FPS (Virtual Cam target @ {fps_out:.2f} FPS)")
    return cap, width, height, fps_out

def main(args):
    """Función principal de la aplicación."""
    global keep_running, cap

    blur_kernel_size = args.blur_kernel
    if blur_kernel_size <= 0:
        print("ERROR: El tamaño del kernel de blur debe ser positivo.")
        sys.exit(1)
    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1
        print(f"WARN: El tamaño del kernel de blur debe ser impar. Ajustado a {blur_kernel_size}.")

    if args.padding_x < 0 or args.padding_y < 0:
        print("ERROR: Los factores de padding no pueden ser negativos.")
        sys.exit(1)

    cap_local, width, height, fps_out = initialize_webcam(args.webcam_id)
    if cap_local is None:
        sys.exit(1)

    vcam_device = f'/dev/video{args.vcam_id}'
    print(f"INFO: Intentando usar cámara virtual en {vcam_device}")
    print(f"INFO: Padding X: {args.padding_x*100:.1f}%, Padding Y: {args.padding_y*100:.1f}%, Blur Kernel: {blur_kernel_size}")

    print("INFO: Inicializando MediaPipe Face Detection...")
    try:
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        ) as face_detection, \
             pyvirtualcam.Camera(width=width, height=height, fps=fps_out,
                                device=vcam_device,
                                print_fps=True) as cam:

            print(f"INFO: Cámara virtual '{cam.device}' lista. Puedes seleccionarla en tu app.")
            print("INFO: Ejecutando... Presiona 'q' en la ventana de OpenCV o Ctrl+C en la terminal para salir.")

            while keep_running:
                ret, frame_bgr = cap_local.read()
                if not ret or frame_bgr is None:
                    if cap_local.get(cv2.CAP_PROP_POS_FRAMES) == cap_local.get(cv2.CAP_PROP_FRAME_COUNT):
                        print("INFO: Fin del stream de video alcanzado.")
                    else:
                        print("WARN: No se pudo leer el fotograma de la cámara. Deteniendo...")
                    keep_running = False
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = face_detection.process(frame_rgb)
                frame_rgb.flags.writeable = True

                output_frame_bgr = frame_bgr.copy()

                if results.detections:
                    img_h, img_w, _ = output_frame_bgr.shape

                    for detection in results.detections:
                        bbox_relative = detection.location_data.relative_bounding_box

                        try:
                            x_min_orig = int(bbox_relative.xmin * img_w)
                            y_min_orig = int(bbox_relative.ymin * img_h)
                            face_w_orig = int(bbox_relative.width * img_w)
                            face_h_orig = int(bbox_relative.height * img_h)
                            x_max_orig = x_min_orig + face_w_orig
                            y_max_orig = y_min_orig + face_h_orig

                            pad_x = int(face_w_orig * args.padding_x)
                            pad_y = int(face_h_orig * args.padding_y)

                            x_min_final = max(0, x_min_orig - pad_x)
                            y_min_final = max(0, y_min_orig - pad_y)
                            x_max_final = min(img_w, x_max_orig + pad_x)
                            y_max_final = min(img_h, y_max_orig + pad_y)

                            if x_min_final < x_max_final and y_min_final < y_max_final:
                                face_roi = output_frame_bgr[y_min_final:y_max_final, x_min_final:x_max_final]
                                if face_roi.size > 0:
                                    blurred_face = cv2.GaussianBlur(face_roi, (blur_kernel_size, blur_kernel_size), 0)
                                    output_frame_bgr[y_min_final:y_max_final, x_min_final:x_max_final] = blurred_face

                        except Exception as e:
                            print(f"ERROR: Excepción al procesar/blurear cara: {e}")

                output_frame_rgb = cv2.cvtColor(output_frame_bgr, cv2.COLOR_BGR2RGB)
                cam.send(output_frame_rgb)
                cv2.imshow('Face Blur VCam - Output (Press q to quit)', output_frame_bgr)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("INFO: Tecla 'q' presionada. Saliendo...")
                    keep_running = False
                    break
                cam.sleep_until_next_frame()

    except ImportError as e:
        print(f"ERROR: Faltan dependencias: {e}. ¿Activaste el entorno virtual?")
        print(f"       Asegúrate de haber ejecutado: pip install opencv-python mediapipe pyvirtualcam numpy")
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: No se encontró el dispositivo de cámara virtual '{vcam_device}'.")
        print(f"       ¿Está cargado el módulo v4l2loopback con video_nr={args.vcam_id} y exclusive_caps=1?")
        print(f"       Prueba: sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback devices=1 video_nr={args.vcam_id} card_label='My FaceBlur Cam' exclusive_caps=1")
        sys.exit(1)
    except RuntimeError as e:
         # Captura el error específico que tuviste
        if "not a video output device" in str(e):
             print(f"ERROR: {e}")
             print("       Esto usualmente significa que 'v4l2loopback' no se cargó con 'exclusive_caps=1'.")
             print(f"       Prueba: sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback devices=1 video_nr={args.vcam_id} card_label='My FaceBlur Cam' exclusive_caps=1")
        else:
             print(f"ERROR: Ocurrió una excepción inesperada de Runtime: {e}")
             import traceback
             traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Ocurrió una excepción inesperada: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("INFO: Limpiando recursos...")
        if cap is not None and cap.isOpened():
            cap.release()
            print("INFO: Cámara web liberada.")
        cv2.destroyAllWindows()
        print("INFO: Ventanas de OpenCV cerradas.")
        print("INFO: Aplicación terminada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crea una cámara virtual con desenfoque de caras ajustable.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--webcam-id", type=int, default=0,
        help="Índice del dispositivo de la cámara web a usar (ej. 0, 1, ...)"
    )
    parser.add_argument(
        "--vcam-id", type=int, default=10,
        help="Número del dispositivo de la cámara virtual a usar (ej. 10 para /dev/video10)"
    )
    parser.add_argument(
        "--blur-kernel", type=int, default=99,
        help="Tamaño del kernel para el desenfoque Gaussiano (debe ser impar y positivo)"
    )
    parser.add_argument(
        "--padding-x", type=float, default=0.15,
        help="Factor de padding horizontal. 0 = sin padding, 0.2 = añade 20%% del ancho de la cara a cada lado."
    )
    parser.add_argument(
        "--padding-y", type=float, default=0.20,
        help="Factor de padding vertical. 0 = sin padding, 0.3 = añade 30%% del alto de la cara arriba y abajo."
    )
    parsed_args = parser.parse_args()
    main(parsed_args)
