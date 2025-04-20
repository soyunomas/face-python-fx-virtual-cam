# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import signal
import argparse
import os
import traceback
import contextlib # Necesario para nullcontext
import random # Para movimiento aleatorio de insectos

# Intenta importar Pillow y componentes necesarios
try:
    from PIL import Image, ImageDraw, ImageFont
    pillow_available = True
except ImportError:
    print("ERROR: La biblioteca Pillow no esta instalada.")
    print("       Necesitas Pillow para renderizar/medir emojis desde fuentes TTF.")
    print("       Ejecuta: pip install Pillow")
    pillow_available = False

# Intenta importar pyvirtualcam
try:
    import pyvirtualcam
    vcam_available = True
except ImportError:
    print("WARN: pyvirtualcam no esta instalado. La salida a camara virtual no funcionara.")
    print("      Ejecuta: pip install pyvirtualcam")
    vcam_available = False

# --- Lista de Emojis Faciales (Unicode) ---
FACE_EMOJIS = (
    '😂', '😅', '😉', '😊', '😌', '😍', '😘', '😜', '😝', '😁',
    '😎', '😭', '😢', '😡', '😠', '😱', '😨', '😰', '😥', '😮',
    '😯', '😲', '😴', '😵', '🤔', '🤢', '🤧', '😇', '🤠', '🤡',
    '🤥', '🤫', '🤭', '🧐', '🤯', '🥳', '🥴', '🥺', '🥰', '🥶',
    '🥵', '🤪', '🤩', '🥱', '🥴', '🫠', '☠️', '💩', '👻', '👽',
    '🤖', '😺', '😸', '😹', '😻', '😼', '😽', '🙀', '😿', '😾',
)

# --- Lista de Emojis de Insectos Voladores (Ampliada) ---
FLYING_INSECT_EMOJIS = ('🦋', '🦟', '🪰', '🐝', '🐞')
DEFAULT_NUM_INSECTS = 15
MAX_INSECTS = 100
INSECT_TARGET_HEIGHT = 35 # Altura deseada (en pixeles) para los insectos redimensionados

# --- Variables globales y constantes ---
cap = None
keep_running = True
emoji_cache = {}
font = None # UNICA Fuente Pillow (grande) para renderizar todo
show_help_overlay_default = True
face_effects_active = True # Estado para activar/desactivar efectos faciales

# --- Constantes para ajustes ---
ADJUST_STEP_OFFSET = 5
ADJUST_STEP_PADDING = 0.02
ADJUST_STEP_PIXEL = 2
MIN_PIXEL_SIZE = 2

# --- Variables para Insectos Voladores ---
flying_insects_list = []
num_insects = DEFAULT_NUM_INSECTS
current_insect_type_index = 0
flying_insects_mode_active = False # Se activa/desactiva con tecla
insect_base_speed = 2.0 # Pixeles por frame (aprox)
insect_avoidance_radius_factor = 1.8 # Factor del radio de la cara para empezar a evitar
insect_repulsion_strength = 0.5 # Que tan fuerte es el empuje para evitar

# --- Manejador para salida limpia (Ctrl+C) ---
def signal_handler(sig, frame):
    global keep_running
    print("\nINFO: Senal de interrupcion recibida. Deteniendo...")
    keep_running = False

signal.signal(signal.SIGINT, signal_handler)

# --- Inicializacion de Componentes de MediaPipe ---
mp_face_detection = mp.solutions.face_detection

# --- Clase para Insectos Voladores (SIMPLIFICADA: sin rotacion/oscilacion) ---
class Insect:
    def __init__(self, x, y, vx, vy, image_bgra):
        self.x = x; self.y = y; self.vx = vx; self.vy = vy
        self.base_image_bgra = image_bgra
        if self.base_image_bgra is not None:
            self.img_height, self.img_width = self.base_image_bgra.shape[:2]
            self.size = max(10, int((self.img_width + self.img_height) / 2))
        else:
            self.img_height, self.img_width = 10, 10; self.size = 10
        self.avoiding = False

    def update_image(self, new_image_bgra):
        self.base_image_bgra = new_image_bgra
        if self.base_image_bgra is not None:
            self.img_height, self.img_width = self.base_image_bgra.shape[:2]
            self.size = max(10, int((self.img_width + self.img_height) / 2))
        else:
             self.img_height, self.img_width = 10, 10; self.size = 10

    def update(self, frame_width, frame_height, head_bbox=None):
        self.x += self.vx; self.y += self.vy
        self.vx += random.uniform(-0.3, 0.3); self.vy += random.uniform(-0.2, 0.2)
        speed = np.sqrt(self.vx**2 + self.vy**2); max_speed = insect_base_speed * 1.5
        if speed > max_speed: self.vx = (self.vx / speed) * max_speed; self.vy = (self.vy / speed) * max_speed
        min_speed = insect_base_speed * 0.5
        if speed > 0 and speed < min_speed : angle_rand = random.uniform(0, 2 * np.pi); self.vx = np.cos(angle_rand) * min_speed; self.vy = np.sin(angle_rand) * min_speed
        elif speed == 0: angle_rand = random.uniform(0, 2 * np.pi); self.vx = np.cos(angle_rand) * min_speed; self.vy = np.sin(angle_rand) * min_speed
        half_w = self.img_width / 2; half_h = self.img_height / 2
        if self.x < half_w or self.x > frame_width - half_w: self.vx *= -1; self.x = np.clip(self.x, half_w, frame_width - half_w)
        if self.y < half_h or self.y > frame_height - half_h: self.vy *= -1; self.y = np.clip(self.y, half_h, frame_height - half_h)
        self.avoiding = False
        if head_bbox:
            hx_min, hy_min, hx_max, hy_max = head_bbox
            head_center_x=(hx_min+hx_max)/2; head_center_y=(hy_min+hy_max)/2
            head_radius=((hx_max-hx_min)+(hy_max-hy_min))/4
            avoid_radius=head_radius*insect_avoidance_radius_factor
            dist_x=self.x-head_center_x; dist_y=self.y-head_center_y
            distance_sq=dist_x**2+dist_y**2
            if distance_sq < avoid_radius**2 and distance_sq > 0:
                self.avoiding=True; distance=np.sqrt(distance_sq)
                repulsion_vx=dist_x/distance; repulsion_vy=dist_y/distance
                self.vx+=repulsion_vx*insect_repulsion_strength
                self.vy+=repulsion_vy*insect_repulsion_strength

    def draw(self, frame):
        if self.base_image_bgra is None: return
        draw_x = int(self.x - self.img_width / 2)
        draw_y = int(self.y - self.img_height / 2)
        overlay_image_alpha(frame, self.base_image_bgra, draw_x, draw_y)

# --- Funciones Auxiliares ---

# (load_emoji_font simplificada - Sin cambios)
def load_emoji_font(font_path):
    global font
    font = None; success = False
    print(f"DEBUG: Iniciando load_emoji_font (simplificada) con path: {font_path}")
    if not pillow_available: print("WARN: Pillow no disponible."); return False
    if not os.path.exists(font_path): print(f"ERROR CRITICO: Fuente NO EXISTE en '{font_path}'."); return False
    else: print(f"DEBUG: Fuente SI existe en '{font_path}'.")
    try:
        font_size_overlay = 109
        print(f"DEBUG: Intentando cargar UNICA fuente (size {font_size_overlay})...")
        temp_font = ImageFont.truetype(font_path, size=font_size_overlay)
        print(f"DEBUG: EXITO al cargar fuente (size {font_size_overlay}).")
        font = temp_font; success = True
        print("DEBUG: Fuente unica asignada a variable global 'font'.")
    except ImportError: print("ERROR CRITICO: Fallo import Pillow (ImageFont?)."); traceback.print_exc()
    except OSError as e: print(f"ERROR CRITICO: OSError al leer fuente '{font_path}': {e}")
    except Exception as e: print(f"ERROR CRITICO INESPERADO al cargar fuente desde '{font_path}': {e}"); traceback.print_exc()
    if not success: print("DEBUG: load_emoji_font termina con FALLO."); font = None
    else: print("DEBUG: load_emoji_font termina con EXITO.")
    return success

# (render_emoji - Sin cambios)
def render_emoji(emoji_char, base_font):
    global emoji_cache
    if not pillow_available or base_font is None: return None
    cache_key = (emoji_char, base_font.size);
    if cache_key in emoji_cache: return emoji_cache[cache_key]
    try:
        if hasattr(base_font, 'getbbox'):
            bbox = base_font.getbbox(emoji_char)
            if bbox is None: print(f"WARN: Fuente '{base_font.path}' (size {base_font.size}) no tiene glifo para '{emoji_char}'?"); return None
            text_width = bbox[2] - bbox[0]; text_height = bbox[3] - bbox[1]; draw_x = -bbox[0]; draw_y = -bbox[1]; img_size = (text_width, text_height)
        else:
            text_width, text_height = base_font.getsize(emoji_char); draw_x, draw_y = 0, 0; img_size = (text_width, text_height)
        if img_size[0] <= 0 or img_size[1] <= 0: print(f"WARN: Tamano invalido {img_size} para '{emoji_char}' (size {base_font.size})"); return None
        image = Image.new('RGBA', img_size, (0, 0, 0, 0)); draw = ImageDraw.Draw(image)
        draw.text((draw_x, draw_y), emoji_char, font=base_font, embedded_color=True)
        img_np = np.array(image)
        if img_np.shape[2] == 4:
            img_bgra = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA); emoji_cache[cache_key] = img_bgra; return img_bgra
        else: print(f"WARN: Imagen renderizada para '{emoji_char}' no es RGBA (shape: {img_np.shape})"); return None
    except Exception as e: print(f"ERROR: Fallo al renderizar '{emoji_char}' (size {base_font.size}): {e}"); return None

# (overlay_image_alpha - Sin cambios)
def overlay_image_alpha(img, img_overlay_bgra, x, y):
    try:
        h_overlay, w_overlay = img_overlay_bgra.shape[:2]; h_img, w_img = img.shape[:2]
        y1, y2 = max(0, y), min(h_img, y + h_overlay); x1, x2 = max(0, x), min(w_img, x + w_overlay)
        overlay_y1 = max(0, -y); overlay_x1 = max(0, -x); overlay_y2 = overlay_y1 + (y2 - y1); overlay_x2 = overlay_x1 + (x2 - x1)
        if y1 >= y2 or x1 >= x2 or overlay_y1 >= overlay_y2 or overlay_x1 >= overlay_x2: return
        roi = img[y1:y2, x1:x2]; overlay_cut = img_overlay_bgra[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        if roi.size == 0 or overlay_cut.size == 0: return
        if overlay_cut.shape[2] != 4: return
        b, g, r, a = cv2.split(overlay_cut)
        if a.mean() < 1:
             if a.mean() == 0: return
        mask = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR); mask_norm = mask.astype(float) / 255.0
        if roi.shape[2] == 4: roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
        elif roi.shape[2] != 3: return
        overlay_rgb = cv2.merge((b, g, r)); roi_norm = roi.astype(float); overlay_rgb_norm = overlay_rgb.astype(float)
        if roi_norm.shape[:2] != mask_norm.shape[:2]: mask_norm = cv2.resize(mask_norm, (roi_norm.shape[1], roi_norm.shape[0]), interpolation=cv2.INTER_NEAREST)
        if roi_norm.shape[:2] != overlay_rgb_norm.shape[:2]: overlay_rgb_norm = cv2.resize(overlay_rgb_norm, (roi_norm.shape[1], roi_norm.shape[0]), interpolation=cv2.INTER_LINEAR)
        combined_roi = cv2.multiply(overlay_rgb_norm, mask_norm) + cv2.multiply(roi_norm, 1.0 - mask_norm)
        img[y1:y2, x1:x2] = combined_roi.astype(np.uint8)
    except cv2.error: pass
    except Exception as e: print(f"ERROR: Fallo inesperado en overlay_image_alpha: {e}")

# (get_rendered_insect - Sin cambios)
def get_rendered_insect(emoji_char):
    global font
    if not font: return None
    rendered_large = render_emoji(emoji_char, font)
    if rendered_large is None: return None
    h_orig, w_orig = rendered_large.shape[:2]
    if h_orig == 0 or w_orig == 0 : return None
    scale = INSECT_TARGET_HEIGHT / h_orig; new_w = int(w_orig * scale); new_h = INSECT_TARGET_HEIGHT
    if new_w <= 0 or new_h <= 0: return None
    try:
        resized_insect = cv2.resize(rendered_large, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_insect
    except cv2.error as e_resize: print(f"ERROR: Fallo cv2.resize insecto '{emoji_char}' a ({new_w}x{new_h}): {e_resize}"); return None
    except Exception as e_generic_resize: print(f"ERROR INESPERADO redimensionando '{emoji_char}': {e_generic_resize}"); return None

# (create_initial_insects - Sin cambios)
def create_initial_insects(count, frame_width, frame_height):
    global flying_insects_list, font
    flying_insects_list = []
    if not font: print("WARN: No se pueden crear insectos sin fuente principal."); return
    current_emoji_char = FLYING_INSECT_EMOJIS[current_insect_type_index]
    rendered_image = get_rendered_insect(current_emoji_char)
    if rendered_image is None: print(f"ERROR: No se pudo obtener imagen insecto inicial '{current_emoji_char}'."); return
    for _ in range(count):
        margin = 50; center_x, center_y = frame_width / 2, frame_height / 2
        x = random.uniform(margin, frame_width - margin); y = random.uniform(margin, frame_height - margin)
        while (x - center_x)**2 + (y - center_y)**2 < (frame_height / 4)**2:
            x = random.uniform(margin, frame_width - margin); y = random.uniform(margin, frame_height - margin)
        angle = random.uniform(0, 2 * np.pi)
        vx = np.cos(angle) * insect_base_speed; vy = np.sin(angle) * insect_base_speed
        flying_insects_list.append(Insect(x, y, vx, vy, rendered_image))
    print(f"INFO: Creados {len(flying_insects_list)} insectos ({current_emoji_char})")

# --- Funcion Principal ---
def main(args):
    # (Variables globales, estado inicial, carga fuente - Sin cambios)
    global keep_running, cap, font, emoji_cache, MIN_PIXEL_SIZE
    global vertical_offset, padding_x_adjust, padding_y_adjust, pixel_size
    global current_mode, current_emoji_index
    global font_loaded, face_effects_active
    global flying_insects_list, num_insects, current_insect_type_index, flying_insects_mode_active
    global show_help_overlay
    is_processing_file = bool(args.input_file); is_saving_output = bool(args.output_file)
    is_interactive = not is_saving_output; use_vcam = not is_saving_output and vcam_available
    vertical_offset = 0; padding_x_adjust = 0.0; padding_y_adjust = 0.0
    pixel_size = args.start_pixel_size; show_help_overlay = show_help_overlay_default and is_interactive
    face_effects_active = True # Inicializar estado de efectos faciales
    font_loaded = load_emoji_font(args.font_path)
    if args.start_mode == 'emoji' and not font_loaded: print("WARN: Fuente emoji no cargada. Cambiando a 'blur'."); args.start_mode = 'blur'

    # (Config Params, Init Video Source - Sin Cambios)
    blur_kernel_size = args.blur_kernel
    if blur_kernel_size <= 0: print("ERROR: Kernel blur positivo."); sys.exit(1)
    if blur_kernel_size % 2 == 0: blur_kernel_size += 1; print(f"WARN: Kernel ajustado a impar {blur_kernel_size}.")
    if args.padding_x < 0 or args.padding_y < 0: print("ERROR: Padding negativo."); sys.exit(1)
    cap = None; width, height, fps_in, fps_out = 0, 0, 0, 0; input_source_name = ""
    if is_processing_file: input_source_name = f"archivo '{args.input_file}'"; cap = cv2.VideoCapture(args.input_file)
    else: input_source_name = f"webcam indice {args.webcam_id}"; cap = cv2.VideoCapture(args.webcam_id)
    if not cap or not cap.isOpened(): print(f"ERROR: No se pudo abrir fuente: {input_source_name}."); sys.exit(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps_in = cap.get(cv2.CAP_PROP_FPS)
    if width <= 0 or height <= 0:
        print("WARN: No se pudieron obtener dimensiones..."); ret, frame = cap.read()
        if not ret or frame is None: print("ERROR: No se pudo leer frame inicial."); cap.release(); sys.exit(1)
        height, width = frame.shape[:2]; print(f"INFO: Dimensiones obtenidas: {width}x{height}")
        if fps_in <= 0: fps_in = 30.0; print(f"WARN: Usando FPS por defecto: {fps_in:.2f}")
    fps_out = min(fps_in if fps_in > 0 else 30.0, 30.0)
    print(f"INFO: Fuente ({input_source_name}): {width}x{height} @ {fps_in:.2f} FPS")

    # (Inicializar Insectos - Sin cambios)
    if font: create_initial_insects(num_insects, width, height)
    else: print("WARN: No se inicializan insectos (falta fuente).")

    # (Config VCam / VideoWriter - Sin cambios)
    vcam_device = f'/dev/video{args.vcam_id}' if use_vcam else None; video_writer = None
    if is_saving_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v'); video_writer = cv2.VideoWriter(args.output_file, fourcc, fps_out, (width, height))
        if not video_writer.isOpened(): print(f"ERROR: No se pudo abrir VideoWriter '{args.output_file}'"); video_writer = None; is_saving_output = False
        else: print(f"INFO: Guardando en: {args.output_file} ({width}x{height} @ {fps_out:.2f} FPS)"); print("INFO: Modo guardado. Preview/VCam off.")
    elif use_vcam: print(f"INFO: Usando VCam: {vcam_device} ({width}x{height} @ {fps_out:.2f} FPS)")
    else: print("INFO: Modo previsualizacion (sin VCam ni archivo).")
    print(f"INFO: Padding Base: X={args.padding_x*100:.1f}%, Y={args.padding_y*100:.1f}%, BlurK: {blur_kernel_size}, PixelB: {pixel_size}x{pixel_size}")

    # --- Estado Inicial ---
    current_mode = args.start_mode; current_emoji_index = 0
    print("INFO: Inicializando MediaPipe Face Detection...")

    # --- Context Managers y Bucle Principal ---
    try:
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            vcam_context = pyvirtualcam.Camera(width=width, height=height, fps=fps_out, device=vcam_device, print_fps=args.print_fps) if use_vcam else contextlib.nullcontext()
            with vcam_context as cam:
                if cam: print(f"INFO: VCam '{cam.device}' lista.")
                if not is_interactive: print("INFO: Procesando archivo...")
                else: print("INFO: Iniciando bucle interactivo (Q: Salir, H: Ayuda)...")

                frame_count = 0; start_time = time.time(); total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_processing_file else -1

                while keep_running:
                    # --- Lectura Frame, Deteccion Facial (Sin Cambios) ---
                    ret, frame_bgr = cap.read();
                    if not ret:
                         if is_processing_file: print("\nINFO: Fin del archivo.")
                         else: print("WARN: No se pudo leer fotograma.");
                         keep_running = False; break
                    frame_count += 1
                    if args.mirror: frame_bgr = cv2.flip(frame_bgr, 1)
                    output_frame_bgr = frame_bgr.copy()
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB); frame_rgb.flags.writeable = False
                    results = face_detection.process(frame_rgb)
                    head_bounding_box = None

                    # --- Procesamiento Caras y Aplicar Efectos Principales (con chequeo face_effects_active) ---
                    if results.detections:
                        img_h, img_w, _ = output_frame_bgr.shape
                        first_detection = results.detections[0]; bbox_relative = first_detection.location_data.relative_bounding_box
                        if bbox_relative:
                            try:
                                x_min_orig=int(bbox_relative.xmin*img_w);y_min_orig=int(bbox_relative.ymin*img_h);face_w_orig=int(bbox_relative.width*img_w);face_h_orig=int(bbox_relative.height*img_h)
                                current_padding_x=max(0.0,args.padding_x+padding_x_adjust);current_padding_y=max(0.0,args.padding_y+padding_y_adjust)
                                pad_x=int(face_w_orig*current_padding_x);pad_y=int(face_h_orig*current_padding_y);x_min_padded=max(0,x_min_orig-pad_x);y_min_padded=max(0,y_min_orig-pad_y);x_max_padded=min(img_w,x_min_orig+face_w_orig+pad_x);y_max_padded=min(img_h,y_min_orig+face_h_orig+pad_y)
                                y_min_final=max(0,y_min_padded+vertical_offset);y_max_final=min(img_h,y_max_padded+vertical_offset);x_min_final=x_min_padded;x_max_final=x_max_padded
                                head_bounding_box = (x_min_final, y_min_final, x_max_final, y_max_final)

                                # --- Aplicar efectos SOLO si estan activos ---
                                if face_effects_active:
                                    if x_min_final < x_max_final and y_min_final < y_max_final:
                                        face_w_final=x_max_final-x_min_final;face_h_final=y_max_final-y_min_final
                                        face_roi=output_frame_bgr[y_min_final:y_max_final,x_min_final:x_max_final]
                                        if face_roi.size > 0:
                                            if current_mode=='blur': blurred=cv2.GaussianBlur(face_roi,(blur_kernel_size,blur_kernel_size),0); (output_frame_bgr[y_min_final:y_max_final,x_min_final:x_max_final])=(blurred) if blurred.shape==face_roi.shape else face_roi
                                            elif current_mode=='pixel': h_roi,w_roi=face_roi.shape[:2];target_w=max(1,int(w_roi/pixel_size));target_h=max(1,int(h_roi/pixel_size));small=cv2.resize(face_roi,(target_w,target_h),interpolation=cv2.INTER_LINEAR);pixelated=cv2.resize(small,(w_roi,h_roi),interpolation=cv2.INTER_NEAREST); (output_frame_bgr[y_min_final:y_max_final,x_min_final:x_max_final])=(pixelated) if pixelated.shape==face_roi.shape else face_roi
                                            elif current_mode=='emoji' and font:
                                                rendered = None
                                                emoji_char=FACE_EMOJIS[current_emoji_index]
                                                rendered=render_emoji(emoji_char,font);
                                                if rendered is not None:
                                                     try: resized_emoji=cv2.resize(rendered,(face_w_final,face_h_final),interpolation=cv2.INTER_AREA);overlay_image_alpha(output_frame_bgr,resized_emoji,x_min_final,y_min_final)
                                                     except cv2.error: pass
                                                     except Exception as ov_err: print(f"ERROR: overlay emoji: {ov_err}")
                                # --- Fin del bloque if face_effects_active ---
                            except Exception as e: print(f"ERROR: Procesando cara/efecto principal: {e}")

                    # --- Actualizar y Dibujar Insectos ---
                    if flying_insects_mode_active and flying_insects_list:
                        for insect in flying_insects_list: insect.update(width, height, head_bounding_box)
                        for insect in flying_insects_list: insect.draw(output_frame_bgr)

                    # --- Dibujar Ayuda (MENU REESTRUCTURADO) ---
                    if show_help_overlay:
                        help_lines = [] # Construir dinamicamente
                        font_scale_help=0.45; color_help=(0, 255, 0); color_help_bottom=(255, 255, 255); bg_color_help=(0,0,0)
                        font_face_help=cv2.FONT_HERSHEY_SIMPLEX; thickness_help=1; line_h_help=17

                        # Controles Generales (sin H)
                        help_lines.extend([
                            "Controles Generales:",
                            " [B]lur   [P]ixel   [E]moji",
                            " [T] Act/Desac FX Cara",
                            " [Y] Act/Desac Insectos",
                            " [Q]uit"
                        ])

                        # Ajustes de Area FX (solo si activos)
                        if face_effects_active:
                            help_lines.extend([
                                "",
                                "Ajuste Area FX:",
                                " Arriba[W] Abajo[S]",
                                " Pad Horiz <> [A][D]",
                                " Pad Vert ^v [R][F]",
                                " Agrandar Area [+]", # Cambio de texto y tecla
                                " Reducir Area [-]",
                                " Reset Area FX [Z]"
                            ])

                            # Ajuste Emoji (solo si modo emoji y activo)
                            if current_mode == 'emoji':
                                if font:
                                    help_lines.extend([
                                        # "", # Opcional: separador extra
                                        " Siguiente Emoji [N]"
                                    ])

                            # Ajuste Pixel (solo si modo pixel y activo)
                            elif current_mode == 'pixel':
                                 help_lines.extend([
                                     # "", # Opcional: separador extra
                                     " Bloque + [I]  Bloque - [K]"
                                 ])

                        # Ajuste Insectos (solo si activos)
                        if flying_insects_mode_active:
                             if font:
                                 help_lines.extend([
                                     "",
                                     "Ajuste Insectos:",
                                     " Add[O] Remove[L]",
                                     " Cambiar Tipo [U]"
                                 ])

                        # Control de Ayuda (al final)
                        help_lines.extend([ "", "[H] Mostrar/Ocultar Ayuda" ])

                        # Dibujar el menu
                        max_w=0; valid_lines=[ln for ln in help_lines if ln];
                        for line in valid_lines: (w,_),_=cv2.getTextSize(line,font_face_help,font_scale_help,thickness_help); max_w=max(max_w,w)
                        total_h=len(help_lines)*line_h_help; bg_x1,bg_y1=5,10; bg_x2,bg_y2=bg_x1+max_w+15,bg_y1+total_h+5
                        cv2.rectangle(output_frame_bgr,(bg_x1,bg_y1),(bg_x2,bg_y2),bg_color_help,cv2.FILLED);
                        y_start=bg_y1+line_h_help-4
                        for i,line in enumerate(help_lines):
                             if line:
                                 # Usar color blanco para la ultima linea (Ayuda)
                                 line_color = color_help_bottom if i == len(help_lines) - 1 else color_help
                                 cv2.putText(output_frame_bgr,line,(bg_x1+5,y_start+i*line_h_help),font_face_help,font_scale_help,line_color,thickness_help,cv2.LINE_AA)
                    # --- Fin Dibujar Ayuda ---


                    # --- Salida: Guardar/VCam/Mostrar (Sin cambios) ---
                    if is_saving_output:
                        if video_writer: video_writer.write(output_frame_bgr)
                        if frame_count%100==0: progress=f"({frame_count}/{total_frames_video})" if total_frames_video > 0 else f"({frame_count})"; print(f"\rINFO: Procesando frame {progress}...", end="")
                    else:
                        if cam:
                            output_frame_rgb = cv2.cvtColor(output_frame_bgr, cv2.COLOR_BGR2RGB)
                            try: cam.send(output_frame_rgb); cam.sleep_until_next_frame()
                            except Exception as e_send: print(f"ERROR enviando a vcam: {e_send}"); keep_running = False; break
                        cv2.imshow('Face Effects - Preview', output_frame_bgr)
                        key = cv2.waitKey(1) & 0xFF

                        # --- Manejo de Teclado (AJUSTADO PARA NUEVA LOGICA DE MENU/TECLAS) ---
                        if key == ord('q'): keep_running = False; break
                        elif key == ord('h'): show_help_overlay = not show_help_overlay

                        # --- Modos Principales (Activan efectos faciales) ---
                        elif key == ord('b'):
                            current_mode = 'blur'
                            face_effects_active = True
                            print("INFO: Modo BLUR (Activado)")
                        elif key == ord('p'):
                            current_mode = 'pixel'
                            face_effects_active = True
                            print(f"INFO: Modo PIXEL ({pixel_size}) (Activado)")
                        elif key == ord('e'): # Requiere 'font'
                            if font:
                                current_mode = 'emoji'
                                face_effects_active = True
                                print(f"INFO: Modo EMOJI ({FACE_EMOJIS[current_emoji_index]}) (Activado)")
                            else:
                                print("WARN: Modo EMOJI no disponible (falta fuente).")

                        # --- Control de Activacion General FX ---
                        elif key == ord('t'):
                            face_effects_active = not face_effects_active
                            print(f"INFO: Efectos Faciales {'ACTIVADOS' if face_effects_active else 'DESACTIVADOS'}")

                        # --- Control de Insectos ---
                        elif key == ord('y'): # Requiere 'font'
                             if font:
                                 flying_insects_mode_active = not flying_insects_mode_active
                                 print(f"INFO: Modo FLYING INSECTS {'Activado' if flying_insects_mode_active else 'Desactivado'}")
                             else:
                                 print("WARN: Modo FLYING INSECTS no disponible (falta fuente).")

                        # --- Siguiente Emoji Facial ---
                        elif key == ord('n'): # Requiere 'font' y que FX esten activos
                            if current_mode == 'emoji' and face_effects_active and font:
                                current_emoji_index = (current_emoji_index + 1) % len(FACE_EMOJIS)
                                print(f"INFO: Next face emoji: {FACE_EMOJIS[current_emoji_index]}")

                        # --- Ajustes Area (W/A/S/D/R/F) (Solo si FX activos) ---
                        elif key == ord('w'):
                            if face_effects_active: vertical_offset -= ADJUST_STEP_OFFSET
                        elif key == ord('s'):
                            if face_effects_active: vertical_offset += ADJUST_STEP_OFFSET
                        elif key == ord('a'):
                            if face_effects_active: padding_x_adjust = max(-args.padding_x, padding_x_adjust - ADJUST_STEP_PADDING)
                        elif key == ord('d'):
                             if face_effects_active: padding_x_adjust += ADJUST_STEP_PADDING
                        elif key == ord('r'):
                             if face_effects_active: padding_y_adjust = max(-args.padding_y, padding_y_adjust - ADJUST_STEP_PADDING)
                        elif key == ord('f'):
                             if face_effects_active: padding_y_adjust += ADJUST_STEP_PADDING

                        # --- Ajustes Area Padding General (Solo si FX activos, tecla '+' ahora) ---
                        elif key == ord('+'): # Ya no se usa '=', solo '+'
                             if face_effects_active:
                                padding_x_adjust += ADJUST_STEP_PADDING
                                padding_y_adjust += ADJUST_STEP_PADDING
                        elif key == ord('-'):
                             if face_effects_active:
                                padding_x_adjust = max(-args.padding_x, padding_x_adjust - ADJUST_STEP_PADDING)
                                padding_y_adjust = max(-args.padding_y, padding_y_adjust - ADJUST_STEP_PADDING)

                        # --- Ajuste Pixel (I/K) (Solo si modo pixel y FX activos) ---
                        elif key == ord('i'):
                            if current_mode == 'pixel' and face_effects_active:
                                pixel_size += ADJUST_STEP_PIXEL
                                print(f"INFO: Pixel Block Size: {pixel_size}")
                        elif key == ord('k'):
                            if current_mode == 'pixel' and face_effects_active:
                                pixel_size = max(MIN_PIXEL_SIZE, pixel_size - ADJUST_STEP_PIXEL)
                                print(f"INFO: Pixel Block Size: {pixel_size}")

                        # --- Ajustes Insectos (O/L/U) (Solo si modo insectos activo) ---
                        elif key == ord('o'):
                            if flying_insects_mode_active and font and len(flying_insects_list) < MAX_INSECTS:
                                current_emoji_char = FLYING_INSECT_EMOJIS[current_insect_type_index]
                                rendered_image = get_rendered_insect(current_emoji_char)
                                if rendered_image is not None:
                                     angle=random.uniform(0,2*np.pi);vx=np.cos(angle)*insect_base_speed;vy=np.sin(angle)*insect_base_speed
                                     flying_insects_list.append(Insect(random.uniform(0,width),random.uniform(0,height),vx,vy,rendered_image))
                                     num_insects = len(flying_insects_list); print(f"INFO: Insecto anadido ({current_emoji_char}). Total: {num_insects}")
                                else: print(f"WARN: No se pudo renderizar/redimensionar {current_emoji_char} para anadir.")
                            # Mensajes WARN solo si el modo esta activo pero falla otra condicion
                            elif flying_insects_mode_active and not font: print("WARN: Fuente no disponible.")
                            elif flying_insects_mode_active and len(flying_insects_list) >= MAX_INSECTS: print(f"INFO: Limite insectos ({MAX_INSECTS}).")
                        elif key == ord('l'):
                            if flying_insects_mode_active and flying_insects_list:
                                flying_insects_list.pop(random.randrange(len(flying_insects_list)))
                                num_insects = len(flying_insects_list); print(f"INFO: Insecto quitado. Total: {num_insects}")
                        elif key == ord('u'):
                            if flying_insects_mode_active and font:
                                current_insect_type_index = (current_insect_type_index + 1) % len(FLYING_INSECT_EMOJIS)
                                new_emoji_char = FLYING_INSECT_EMOJIS[current_insect_type_index]
                                print(f"INFO: Cambiando insectos a: {new_emoji_char}")
                                new_rendered_image = get_rendered_insect(new_emoji_char)
                                if new_rendered_image is not None:
                                    for insect in flying_insects_list: insect.update_image(new_rendered_image)
                                else: print(f"WARN: No se pudo renderizar/redimensionar {new_emoji_char}. Insectos no cambiados.")
                            elif flying_insects_mode_active and not font: print("WARN: Fuente no disponible.")

                        # --- Reset (Z) (Resetea solo area/pixel) ---
                        elif key == ord('z'):
                            vertical_offset=0; padding_x_adjust=0.0; padding_y_adjust=0.0; pixel_size=args.start_pixel_size
                            print("INFO: Ajustes area/pixel reseteados.")

                # --- Fin Bucle While ---
                end_time = time.time(); total_time = end_time - start_time; avg_fps = frame_count / total_time if total_time > 0 else 0
                print(f"\nINFO: Bucle terminado. {frame_count} frames en {total_time:.2f} seg (Avg FPS: {avg_fps:.2f}).")

    # --- Excepciones y Limpieza Final (Sin cambios) ---
    except ImportError as e:
         if 'PIL' in str(e) and not pillow_available: print(f"CRITICO: Falta Pillow.")
         elif 'pyvirtualcam' in str(e) and not vcam_available: print(f"CRITICO: Falta pyvirtualcam.")
         else: print(f"CRITICO: Falta dependencia: {e}.")
         sys.exit(1)
    except FileNotFoundError as e:
        if use_vcam and vcam_device and vcam_device in str(e): print(f"CRITICO: No se encontro vcam '{vcam_device}'.")
        elif 'args' in locals() and args.font_path in str(e): print(f"CRITICO: No se encontro fuente '{args.font_path}'.")
        else: print(f"CRITICO: Archivo no encontrado: {e}")
        sys.exit(1)
    except RuntimeError as e:
        if use_vcam and vcam_device and ("Could not open device" in str(e) or "Not a video output device" in str(e)): print(f"CRITICO: No se pudo abrir vcam '{vcam_device}'. {e}. Usa 'exclusive_caps=1'.")
        else: print(f"CRITICO: Runtime: {e}"); traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"CRITICO INESPERADO: {e}"); traceback.print_exc()
        sys.exit(1)
    finally:
        print("\nINFO: Iniciando limpieza...");
        if cap is not None and cap.isOpened(): cap.release(); print(f"INFO: Fuente video ({input_source_name}) liberada.")
        if video_writer is not None: video_writer.release(); print(f"INFO: Archivo video '{args.output_file}' cerrado.")
        cv2.destroyAllWindows(); print("INFO: Ventanas OpenCV cerradas.")
        emoji_cache.clear(); print("INFO: Cache emojis limpiada.")
        flying_insects_list.clear(); print("INFO: Lista insectos limpiada.")
        print("INFO: Esperando..."); time.sleep(0.2); print("INFO: App terminada.")

# --- Punto de Entrada (Sin cambios) ---
if __name__ == "__main__":
    if not pillow_available: print("WARN: Pillow no instalado. Funciones Emoji/Flying pueden fallar.")
    if not vcam_available: print("WARN: pyvirtualcam no instalado. Salida VCam no disponible.")
    parser = argparse.ArgumentParser( description="Aplica efectos faciales (blur, pixel, emoji, flying insects) a video y envia a VCam o guarda.", formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    input_group = parser.add_mutually_exclusive_group(required=True); input_group.add_argument( "--webcam-id", type=int, help="Indice de la camara web. Excluyente con --input-file." ); input_group.add_argument( "--input-file", type=str, help="Ruta al archivo de video de entrada. Excluyente con --webcam-id." )
    parser.add_argument("--output-file", type=str, help="Ruta al archivo de video de salida (e.g., 'output.mp4'). Deshabilita VCam e interactividad.")
    parser.add_argument("--vcam-id", type=int, default=10, help="Numero vcam (/dev/videoX) a usar si no se guarda archivo.")
    parser.add_argument("--blur-kernel", type=int, default=99, help="Kernel blur (impar positivo).")
    parser.add_argument("--padding-x", type=float, default=0.15, help="Padding horizontal base (%% ancho cara).")
    parser.add_argument("--padding-y", type=float, default=0.20, help="Padding vertical base (%% alto cara).")
    parser.add_argument("--font-path", type=str, default="NotoColorEmoji.ttf", help="Ruta a fuente TTF emoji.")
    parser.add_argument("--start-mode", type=str, default="blur", choices=['blur', 'emoji', 'pixel'], help="Modo inicial (efecto principal).")
    parser.add_argument("--start-pixel-size", type=int, default=8, help="Tamano bloque inicial pixelado.")
    parser.add_argument("--mirror", action='store_true', help="Aplicar espejo horizontal.")
    parser.add_argument("--print-fps", action='store_true', help="Mostrar FPS de pyvirtualcam (si se usa).")
    parsed_args = parser.parse_args()
    if parsed_args.output_file and not parsed_args.input_file: parser.error("--output-file solo se puede usar con --input-file.")
    if not vcam_available and not parsed_args.output_file: print("WARN: pyvirtualcam no disponible y no se guarda archivo. Solo previsualizacion.")
    if parsed_args.start_pixel_size < MIN_PIXEL_SIZE: print(f"WARN: Pixel size ajustado a minimo {MIN_PIXEL_SIZE}."); parsed_args.start_pixel_size = MIN_PIXEL_SIZE
    main(parsed_args)
