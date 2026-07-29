"""
Módulo: cloth_analyzer.py
Analiza una prenda digitalizada (PNG sin fondo) y detecta:
  1. Color dominante  → nombre de color en español + valor HEX + RGB
  2. Tipo de prenda   → camisa, pantalón, vestido, etc.  (CLIP zero-shot)
  3. Estilo           → casual, formal, deportivo, etc.  (CLIP zero-shot)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ──────────────────────────────────────────────────────────────────────
# Paleta de colores de referencia (nombre → RGB)
# ──────────────────────────────────────────────────────────────────────
COLOR_PALETTE: dict[str, tuple[int, int, int]] = {
    "blanco":        (255, 255, 255),
    "negro":         (0,   0,   0),
    "gris claro":    (200, 200, 200),
    "gris":          (128, 128, 128),
    "gris oscuro":   (64,  64,  64),
    "rojo":          (220, 30,  30),
    "rojo oscuro":   (139, 0,   0),
    "rosa":          (255, 105, 180),
    "rosa palo":     (255, 182, 193),
    "naranja":       (255, 140, 0),
    "amarillo":      (255, 215, 0),
    "verde lima":    (124, 205, 50),
    "verde":         (34,  139, 34),
    "verde oscuro":  (0,   100, 0),
    "verde militar": (85,  107, 47),
    "turquesa":      (0,   206, 209),
    "azul claro":    (135, 206, 235),
    "azul":          (30,  100, 200),
    "azul marino":   (0,   0,   128),
    "azul vaquero":  (93,  138, 168),
    "morado":        (148, 0,   211),
    "lila":          (200, 162, 200),
    "beige":         (245, 222, 179),
    "marrón claro":  (205, 133, 63),
    "marrón":        (139, 90,  43),
    "marrón oscuro": (101, 67,  33),
    "dorado":        (212, 175, 55),
    "plateado":      (192, 192, 192),
}

# ──────────────────────────────────────────────────────────────────────
# Etiquetas para clasificación zero-shot con CLIP
# ──────────────────────────────────────────────────────────────────────
CLOTH_TYPES = [
    "camiseta", "camisa", "blusa", "polo",
    "sudadera", "jersey", "chaqueta", "abrigo", "cazadora",
    "pantalón", "vaqueros", "shorts", "falda",
    "vestido", "mono", "traje",
    "calcetines", "medias", "ropa interior",
    "zapatos", "zapatillas", "botas", "sandalias",
    "bolso", "cinturón", "bufanda", "gorro", "gafas",
]

CLOTH_STYLES = [
    "casual", "formal", "elegante", "deportivo",
    "urbano", "bohemio", "vintage", "minimalista",
    "romántico", "punk", "business casual",
]


# ──────────────────────────────────────────────────────────────────────
# Dataclasses de resultado
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ColorInfo:
    name: str                        # nombre en español
    hex: str                         # p.ej. "#3A6EA8"
    rgb: tuple[int, int, int]        # (R, G, B)
    palette: list[tuple[int, int, int]]  # top-3 colores dominantes


@dataclass
class ClothAnalysis:
    image_path: str
    color: ColorInfo
    cloth_type: str                  # etiqueta ganadora
    cloth_type_scores: dict[str, float]   # todas las puntuaciones
    style: str                       # etiqueta ganadora
    style_scores: dict[str, float]        # todas las puntuaciones


# ──────────────────────────────────────────────────────────────────────
# Analizador principal
# ──────────────────────────────────────────────────────────────────────
class ClothAnalyzer:
    def __init__(self, n_colors: int = 3, clip_model: str = "openai/clip-vit-base-patch32"):
        """
        Args:
            n_colors:   Número de colores dominantes a extraer con K-Means.
            clip_model: Checkpoint de CLIP en HuggingFace.
        """
        self.n_colors = n_colors
        print("⏳ Cargando modelo CLIP…")
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip_model.eval()
        print("✅ Modelo CLIP listo")

    # ── API pública ────────────────────────────────────────────────────

    def analyze(self, image_path: str | Path) -> ClothAnalysis:
        """
        Analiza una prenda y devuelve color, tipo y estilo.

        Args:
            image_path: PNG con fondo transparente (salida del módulo 1).

        Returns:
            ClothAnalysis con todos los resultados.
        """
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGBA")

        color_info = self._detect_color(image)
        cloth_type, type_scores = self._classify(image, CLOTH_TYPES, "una foto de ropa: {}")
        style, style_scores = self._classify(image, CLOTH_STYLES, "ropa de estilo {}")

        result = ClothAnalysis(
            image_path=str(image_path),
            color=color_info,
            cloth_type=cloth_type,
            cloth_type_scores=type_scores,
            style=style,
            style_scores=style_scores,
        )
        self._print_result(result)
        return result

    def analyze_folder(
        self,
        input_dir: str | Path,
        output_json: str | Path | None = None,
    ) -> list[ClothAnalysis]:
        """
        Analiza todas las prendas de una carpeta.

        Args:
            input_dir:   Carpeta con PNGs procesados (salida del módulo 1).
            output_json: Si se indica, guarda los resultados en un JSON.

        Returns:
            Lista de ClothAnalysis.
        """
        input_dir = Path(input_dir)
        images = sorted(input_dir.glob("*.png"))
        if not images:
            print(f"⚠️  No se encontraron PNGs en {input_dir}")
            return []

        results = [self.analyze(p) for p in tqdm(images, desc="Analizando prendas")]

        if output_json:
            self._save_json(results, Path(output_json))

        return results

    # ── Detección de color ─────────────────────────────────────────────

    def _detect_color(self, image: Image.Image) -> ColorInfo:
        """
        Extrae los colores dominantes ignorando píxeles transparentes,
        luego mapea el color principal a la paleta de referencia.
        """
        pixels = self._visible_pixels(image)

        if len(pixels) < self.n_colors:
            dominant_rgb = tuple(pixels[0]) if len(pixels) else (128, 128, 128)
            palette = [dominant_rgb]
        else:
            km = KMeans(n_clusters=self.n_colors, n_init=10, random_state=42)
            km.fit(pixels)
            # Ordenar clusters por tamaño (el más grande = dominante)
            counts = np.bincount(km.labels_)
            order = np.argsort(-counts)
            centers = km.cluster_centers_[order].astype(int)
            dominant_rgb = tuple(int(c) for c in centers[0])
            palette = [tuple(int(c) for c in row) for row in centers]

        name = self._nearest_color_name(dominant_rgb)
        hex_val = "#{:02X}{:02X}{:02X}".format(*dominant_rgb)
        return ColorInfo(name=name, hex=hex_val, rgb=dominant_rgb, palette=palette)

    @staticmethod
    def _visible_pixels(image: Image.Image) -> np.ndarray:
        """Devuelve array (N, 3) con los píxeles RGB no transparentes."""
        arr = np.array(image)                  # (H, W, 4)
        mask = arr[:, :, 3] > 10              # alpha > 10 → visible
        return arr[mask][:, :3].astype(float)

    @staticmethod
    def _nearest_color_name(rgb: tuple[int, int, int]) -> str:
        """Devuelve el nombre del color de la paleta más cercano en distancia euclidiana."""
        r, g, b = rgb
        best_name, best_dist = "desconocido", float("inf")
        for name, (pr, pg, pb) in COLOR_PALETTE.items():
            dist = ((r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_name = dist, name
        return best_name

    # ── Clasificación con CLIP ─────────────────────────────────────────

    def _classify(
        self,
        image: Image.Image,
        labels: list[str],
        prompt_template: str,
    ) -> tuple[str, dict[str, float]]:
        """
        Clasificación zero-shot con CLIP.

        Args:
            image:            Imagen PIL de la prenda.
            labels:           Lista de etiquetas candidatas.
            prompt_template:  Plantilla con '{}' donde se inserta cada etiqueta.

        Returns:
            (etiqueta_ganadora, {etiqueta: score})
        """
        import torch

        rgb_image = image.convert("RGB")
        texts = [prompt_template.format(label) for label in labels]

        inputs = self.clip_processor(
            text=texts,
            images=rgb_image,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).squeeze().tolist()

        if isinstance(probs, float):
            probs = [probs]

        scores = {label: round(float(p), 4) for label, p in zip(labels, probs)}
        winner = max(scores, key=scores.__getitem__)
        return winner, scores

    # ── Utilidades ─────────────────────────────────────────────────────

    @staticmethod
    def _print_result(r: ClothAnalysis) -> None:
        path = Path(r.image_path).name
        top_type = sorted(r.cloth_type_scores.items(), key=lambda x: -x[1])[:3]
        top_style = sorted(r.style_scores.items(), key=lambda x: -x[1])[:3]
        print(
            f"[OK] {path}\n"
            f"     🎨 Color : {r.color.name}  {r.color.hex}\n"
            f"     👕 Tipo  : {r.cloth_type}  "
            f"({', '.join(f'{k} {v:.0%}' for k, v in top_type)})\n"
            f"     ✨ Estilo: {r.style}  "
            f"({', '.join(f'{k} {v:.0%}' for k, v in top_style)})"
        )

    @staticmethod
    def _save_json(results: list[ClothAnalysis], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for r in results:
            d = asdict(r)
            d["color"]["rgb"] = list(d["color"]["rgb"])
            d["color"]["palette"] = [list(c) for c in d["color"]["palette"]]
            data.append(d)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n💾 Resultados guardados en: {path}")


# ──────────────────────────────────────────────────────────────────────
# Ejecución directa
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = ClothAnalyzer()
    analyzer.analyze_folder(
        input_dir="data/clothes_processed",
        output_json="data/clothes_processed/analysis.json",
    )
