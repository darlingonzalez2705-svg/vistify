"""
Módulo: virtual_tryon.py
Superpone prendas digitalizadas (PNG sin fondo) sobre la foto del usuario,
ajustando tamaño, posición y perspectiva según los landmarks corporales.

Flujo por prenda:
  1. GarmentFitter     → calcula ancho/alto objetivo en píxeles según tipo de prenda
  2. PerspectiveWarper → inclina la prenda para seguir la orientación del cuerpo
  3. GarmentBlender    → pega con feathering (suavizado de bordes) para resultado natural
  4. VirtualTryOn      → orquesta todo y compone la imagen final
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from modules.avatar_generator import AvatarResult, Landmark


# ──────────────────────────────────────────────────────────────────────
# Configuración de anclaje por tipo de prenda
# ──────────────────────────────────────────────────────────────────────
# Cada entrada define:
#   anchor_top    : landmark que marca el borde superior de la prenda
#   anchor_bottom : landmark que marca el borde inferior
#   width_ref     : par de landmarks cuya distancia define el ancho objetivo
#   width_scale   : multiplicador sobre esa distancia (holgura)
#   h_offset_top  : desplazamiento vertical del borde superior (fracción del alto)

GARMENT_CONFIG: dict[str, dict] = {
    # ── Parte superior ─────────────────────────────────────────────────
    "camiseta": {
        "anchor_top":    ("left_shoulder", "right_shoulder"),
        "anchor_bottom": ("left_hip",      "right_hip"),
        "width_ref":     ("left_shoulder", "right_shoulder"),
        "width_scale":   1.15,
        "h_offset_top":  -0.05,
    },
    "camisa": {
        "anchor_top":    ("left_shoulder", "right_shoulder"),
        "anchor_bottom": ("left_hip",      "right_hip"),
        "width_ref":     ("left_shoulder", "right_shoulder"),
        "width_scale":   1.20,
        "h_offset_top":  -0.06,
    },
    "blusa": {
        "anchor_top":    ("left_shoulder", "right_shoulder"),
        "anchor_bottom": ("left_hip",      "right_hip"),
        "width_ref":     ("left_shoulder", "right_shoulder"),
        "width_scale":   1.18,
        "h_offset_top":  -0.05,
    },
    "polo": {
        "anchor_top":    ("left_shoulder", "right_shoulder"),
        "anchor_bottom": ("left_hip",      "right_hip"),
        "width_ref":     ("left_shoulder", "right_shoulder"),
        "width_scale":   1.12,
        "h_offset_top":  -0.04,
    },
    "sudadera": {
        "anchor_top":    ("left_shoulder", "right_shoulder"),
        "anchor_bottom": ("left_hip",      "right_hip"),
        "width_ref":     ("left_shoulder", "right_shoulder"),
        "width_scale":   1.30,
        "h_offset_top":  -0.06,
    },
    "jersey": {
        "anchor_top":    ("left_shoulder", "right_shoulder"),
        "anchor_bottom": ("left_hip",      "right_hip"),
        "width_ref":     ("left_shoulder", "right_shoulder"),
        "width_scale":   1.25,
        "h_offset_top":  -0.05,
    },
    "chaqueta": {
        "anchor_top":    ("left_shoulder", "right_shoulder"),
        "anchor_bottom": ("left_hip",      "right_hip"),
        "width_ref":     ("left_shoulder", "right_shoulder"),
        "width_scale":   1.35,
        "h_offset_top":  -0.07,
    },
    "abrigo": {
        "anchor_top":    ("left_shoulder", "right_shoulder"),
        "anchor_bottom": ("left_knee",     "right_knee"),
        "width_ref":     ("left_shoulder", "right_shoulder"),
        "width_scale":   1.40,
        "h_offset_top":  -0.07,
    },
    "cazadora": {
        "anchor_top":    ("left_shoulder", "right_shoulder"),
        "anchor_bottom": ("left_hip",      "right_hip"),
        "width_ref":     ("left_shoulder", "right_shoulder"),
        "width_scale":   1.30,
        "h_offset_top":  -0.06,
    },
    # ── Parte inferior ─────────────────────────────────────────────────
    "pantalón": {
        "anchor_top":    ("left_hip",   "right_hip"),
        "anchor_bottom": ("left_ankle", "right_ankle"),
        "width_ref":     ("left_hip",   "right_hip"),
        "width_scale":   1.20,
        "h_offset_top":  -0.03,
    },
    "vaqueros": {
        "anchor_top":    ("left_hip",   "right_hip"),
        "anchor_bottom": ("left_ankle", "right_ankle"),
        "width_ref":     ("left_hip",   "right_hip"),
        "width_scale":   1.18,
        "h_offset_top":  -0.03,
    },
    "shorts": {
        "anchor_top":    ("left_hip",  "right_hip"),
        "anchor_bottom": ("left_knee", "right_knee"),
        "width_ref":     ("left_hip",  "right_hip"),
        "width_scale":   1.20,
        "h_offset_top":  -0.03,
    },
    "falda": {
        "anchor_top":    ("left_hip",   "right_hip"),
        "anchor_bottom": ("left_knee",  "right_knee"),
        "width_ref":     ("left_hip",   "right_hip"),
        "width_scale":   1.35,
        "h_offset_top":  -0.03,
    },
    # ── Prendas completas ──────────────────────────────────────────────
    "vestido": {
        "anchor_top":    ("left_shoulder", "right_shoulder"),
        "anchor_bottom": ("left_knee",     "right_knee"),
        "width_ref":     ("left_shoulder", "right_shoulder"),
        "width_scale":   1.30,
        "h_offset_top":  -0.06,
    },
    "mono": {
        "anchor_top":    ("left_shoulder", "right_shoulder"),
        "anchor_bottom": ("left_ankle",    "right_ankle"),
        "width_ref":     ("left_shoulder", "right_shoulder"),
        "width_scale":   1.20,
        "h_offset_top":  -0.06,
    },
    "traje": {
        "anchor_top":    ("left_shoulder", "right_shoulder"),
        "anchor_bottom": ("left_ankle",    "right_ankle"),
        "width_ref":     ("left_shoulder", "right_shoulder"),
        "width_scale":   1.25,
        "h_offset_top":  -0.07,
    },
}

# Tipo genérico para prendas no reconocidas
_DEFAULT_CONFIG = GARMENT_CONFIG["camiseta"]


# ──────────────────────────────────────────────────────────────────────
# Dataclass de resultado
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TryOnResult:
    output_path: str
    garments_applied: list[str]   # tipos de prenda superpuestos
    photo_path: str


# ──────────────────────────────────────────────────────────────────────
# 1. GarmentFitter — calcula tamaño y posición objetivo
# ──────────────────────────────────────────────────────────────────────
class GarmentFitter:
    def fit(
        self,
        garment_type: str,
        landmarks: dict[str, Landmark],
    ) -> dict:
        """
        Calcula el rectángulo destino donde debe colocarse la prenda.

        Returns:
            {
              "x":      coordenada X del centro horizontal,
              "y_top":  coordenada Y del borde superior,
              "width":  ancho objetivo en píxeles,
              "height": alto objetivo en píxeles,
              "angle":  ángulo de inclinación del torso en grados,
            }
        """
        cfg = GARMENT_CONFIG.get(garment_type, _DEFAULT_CONFIG)

        top_l, top_r     = cfg["anchor_top"]
        bot_l, bot_r     = cfg["anchor_bottom"]
        wid_l, wid_r     = cfg["width_ref"]

        lm_top_l = landmarks[top_l]
        lm_top_r = landmarks[top_r]
        lm_bot_l = landmarks[bot_l]
        lm_bot_r = landmarks[bot_r]

        # Centro horizontal y vertical de los puntos de anclaje
        cx_top = (lm_top_l.x + lm_top_r.x) / 2
        cy_top = (lm_top_l.y + lm_top_r.y) / 2
        cy_bot = (lm_bot_l.y + lm_bot_r.y) / 2

        # Ancho objetivo
        ref_w = abs(landmarks[wid_r].x - landmarks[wid_l].x)
        target_w = int(ref_w * cfg["width_scale"])

        # Alto objetivo
        target_h = int(abs(cy_bot - cy_top) * 1.05)

        # Desplazamiento vertical del borde superior
        y_top = int(cy_top + cfg["h_offset_top"] * target_h)

        # Ángulo de inclinación: diferencia de Y entre hombro izq y der
        dy = lm_top_r.y - lm_top_l.y
        dx = lm_top_r.x - lm_top_l.x
        angle = float(np.degrees(np.arctan2(dy, dx)))

        return {
            "x":      int(cx_top),
            "y_top":  y_top,
            "width":  max(target_w, 10),
            "height": max(target_h, 10),
            "angle":  angle,
        }


# ──────────────────────────────────────────────────────────────────────
# 2. PerspectiveWarper — inclina la prenda según el ángulo del cuerpo
# ──────────────────────────────────────────────────────────────────────
class PerspectiveWarper:
    def warp(self, garment: Image.Image, fit: dict) -> Image.Image:
        """
        Escala la prenda al tamaño objetivo y aplica una rotación suave
        para seguir la inclinación del torso.

        Args:
            garment: PNG RGBA de la prenda (sin fondo).
            fit:     Diccionario devuelto por GarmentFitter.fit().

        Returns:
            Imagen RGBA escalada y rotada lista para pegar.
        """
        # 1. Escalar al tamaño objetivo
        resized = garment.resize(
            (fit["width"], fit["height"]),
            Image.LANCZOS,
        )

        # 2. Rotar según el ángulo del torso (expand=True para no recortar)
        angle = fit["angle"]
        if abs(angle) > 0.5:
            resized = resized.rotate(-angle, expand=True, resample=Image.BICUBIC)

        return resized


# ──────────────────────────────────────────────────────────────────────
# 3. GarmentBlender — composición con feathering
# ──────────────────────────────────────────────────────────────────────
class GarmentBlender:
    def __init__(self, feather_radius: int = 6):
        """
        Args:
            feather_radius: Radio del desenfoque gaussiano aplicado al canal alfa
                            para suavizar los bordes de la prenda.
        """
        self.feather_radius = feather_radius

    def blend(
        self,
        base: Image.Image,
        garment: Image.Image,
        fit: dict,
    ) -> Image.Image:
        """
        Pega la prenda sobre la imagen base con bordes suavizados.

        Args:
            base:    Foto del usuario en modo RGBA.
            garment: Prenda escalada/rotada en modo RGBA.
            fit:     Diccionario con posición (x, y_top).

        Returns:
            Imagen RGBA con la prenda compuesta.
        """
        gw, gh = garment.size

        # Posición: centrar horizontalmente en fit["x"]
        paste_x = fit["x"] - gw // 2
        paste_y = fit["y_top"]

        # Suavizar el canal alfa de la prenda (feathering)
        alpha = garment.split()[3]
        if self.feather_radius > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(self.feather_radius))

        # Recortar si la prenda se sale del canvas
        bw, bh = base.size
        crop_box = [0, 0, gw, gh]
        if paste_x < 0:
            crop_box[0] = -paste_x
            paste_x = 0
        if paste_y < 0:
            crop_box[1] = -paste_y
            paste_y = 0
        if paste_x + gw > bw:
            crop_box[2] = bw - paste_x + crop_box[0]
        if paste_y + gh > bh:
            crop_box[3] = bh - paste_y + crop_box[1]

        garment_crop = garment.crop(crop_box)
        alpha_crop   = alpha.crop(crop_box)

        result = base.copy()
        result.paste(garment_crop, (paste_x, paste_y), mask=alpha_crop)
        return result


# ──────────────────────────────────────────────────────────────────────
# 4. VirtualTryOn — orquestador principal
# ──────────────────────────────────────────────────────────────────────
class VirtualTryOn:
    def __init__(self, feather_radius: int = 6):
        self._fitter  = GarmentFitter()
        self._warper  = PerspectiveWarper()
        self._blender = GarmentBlender(feather_radius)

    def try_on(
        self,
        avatar_result: AvatarResult,
        garments: list[dict[str, str]],
        output_dir: str | Path = "outputs/outfits",
        output_name: str = "tryon.jpg",
    ) -> TryOnResult:
        """
        Superpone una o varias prendas sobre la foto del usuario.

        Args:
            avatar_result: Resultado del módulo 3 (AvatarResult).
            garments:      Lista de dicts con claves:
                             - "path"  : ruta al PNG sin fondo (módulo 1)
                             - "type"  : tipo de prenda en español (módulo 2)
                           Ejemplo:
                             [
                               {"path": "data/clothes_processed/camisa_nobg.png",
                                "type": "camisa"},
                               {"path": "data/clothes_processed/vaqueros_nobg.png",
                                "type": "vaqueros"},
                             ]
            output_dir:    Carpeta donde guardar la imagen resultante.
            output_name:   Nombre del archivo de salida.

        Returns:
            TryOnResult con la ruta de la imagen generada.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Cargar foto base del usuario
        base = Image.open(avatar_result.photo_path).convert("RGBA")
        landmarks = avatar_result.landmarks

        applied: list[str] = []

        # Ordenar prendas: primero parte inferior, luego superior (capas)
        ordered = self._sort_by_layer(garments)

        for garment_info in ordered:
            garment_path = Path(garment_info["path"])
            garment_type = garment_info["type"].lower().strip()

            if not garment_path.exists():
                print(f"⚠️  No se encontró la prenda: {garment_path}")
                continue

            print(f"👕 Aplicando: {garment_type}  ({garment_path.name})")

            garment_img = Image.open(garment_path).convert("RGBA")

            fit     = self._fitter.fit(garment_type, landmarks)
            warped  = self._warper.warp(garment_img, fit)
            base    = self._blender.blend(base, warped, fit)

            applied.append(garment_type)

        # Guardar resultado final en JPG (sin transparencia)
        output_path = output_dir / output_name
        base.convert("RGB").save(str(output_path), quality=95)

        result = TryOnResult(
            output_path=str(output_path),
            garments_applied=applied,
            photo_path=avatar_result.photo_path,
        )
        self._print_summary(result)
        return result

    def try_on_from_paths(
        self,
        photo_path: str | Path,
        measurements_json: str | Path,
        garments: list[dict[str, str]],
        output_dir: str | Path = "outputs/outfits",
        output_name: str = "tryon.jpg",
    ) -> TryOnResult:
        """
        Versión alternativa que carga el AvatarResult desde los archivos
        guardados por el módulo 3, sin necesidad de volver a procesar la foto.

        Args:
            photo_path:        Foto original del usuario.
            measurements_json: JSON generado por avatar_generator.py.
            garments:          Igual que en try_on().
        """
        import json
        from modules.avatar_generator import Landmark, BodyMeasurements

        data = json.loads(Path(measurements_json).read_text(encoding="utf-8"))

        landmarks = {
            name: Landmark(**vals)
            for name, vals in data["landmarks"].items()
        }
        m = data["measurements"]
        measurements = BodyMeasurements(**m)

        avatar_result = AvatarResult(
            photo_path=str(photo_path),
            avatar_path="",
            measurements=measurements,
            landmarks=landmarks,
        )
        return self.try_on(avatar_result, garments, output_dir, output_name)

    # ── Utilidades ─────────────────────────────────────────────────────

    @staticmethod
    def _sort_by_layer(garments: list[dict]) -> list[dict]:
        """
        Ordena las prendas para que se pinten en el orden correcto:
        primero parte inferior (pantalón, falda...) y luego superior
        (camiseta, chaqueta...) para que las capas queden bien.
        """
        lower = {"pantalón", "vaqueros", "shorts", "falda"}
        full  = {"vestido", "mono", "traje"}
        upper = set(GARMENT_CONFIG.keys()) - lower - full

        def layer_order(g: dict) -> int:
            t = g["type"].lower().strip()
            if t in full:   return 0
            if t in lower:  return 1
            return 2

        return sorted(garments, key=layer_order)

    @staticmethod
    def _print_summary(r: TryOnResult) -> None:
        print(
            f"\n✅ Prueba virtual completada\n"
            f"   📸 Foto base  : {r.photo_path}\n"
            f"   👗 Prendas    : {', '.join(r.garments_applied)}\n"
            f"   💾 Guardado en: {r.output_path}"
        )


# ──────────────────────────────────────────────────────────────────────
# Ejecución directa
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from modules.avatar_generator import AvatarGenerator

    # 1. Generar avatar (módulo 3)
    avatar = AvatarGenerator(real_height_cm=170).generate(
        photo_path="data/user_photos/foto_frontal.jpg",
        output_dir="data/avatar",
    )

    # 2. Probar prendas
    tryon = VirtualTryOn()
    tryon.try_on(
        avatar_result=avatar,
        garments=[
            {"path": "data/clothes_processed/camisa_nobg.png",    "type": "camisa"},
            {"path": "data/clothes_processed/vaqueros_nobg.png",  "type": "vaqueros"},
        ],
        output_dir="outputs/outfits",
        output_name="outfit_01.jpg",
    )
