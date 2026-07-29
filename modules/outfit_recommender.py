"""
Módulo: outfit_recommender.py
Analiza todas las prendas del armario y recomienda los mejores conjuntos
combinando compatibilidad de color (teoría del color) y de estilo.

Flujo:
  1. ColorCompatibilityEngine → puntúa pares de colores (complementarios,
                                análogos, neutros, monocromáticos)
  2. StyleCompatibilityEngine → puntúa pares de estilos con matriz de afinidad
  3. OutfitScorer             → combina ambas puntuaciones en un score 0-100
  4. OutfitRecommender        → genera combinaciones, las puntúa, ordena
                                y exporta informe visual + JSON
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from modules.cloth_analyzer import ClothAnalysis, ColorInfo


# ──────────────────────────────────────────────────────────────────────
# Grupos de prendas (para construir conjuntos válidos)
# ──────────────────────────────────────────────────────────────────────
UPPER_GARMENTS  = {"camiseta", "camisa", "blusa", "polo", "sudadera",
                   "jersey", "chaqueta", "abrigo", "cazadora"}
LOWER_GARMENTS  = {"pantalón", "vaqueros", "shorts", "falda"}
FULL_GARMENTS   = {"vestido", "mono", "traje"}
OUTER_GARMENTS  = {"chaqueta", "abrigo", "cazadora"}


# ──────────────────────────────────────────────────────────────────────
# 1. ColorCompatibilityEngine
# ──────────────────────────────────────────────────────────────────────
class ColorCompatibilityEngine:
    """
    Puntúa la compatibilidad entre dos colores usando el espacio HSV
    y las reglas clásicas de teoría del color.

    Score devuelto: float en [0.0, 1.0]
    """

    # Colores que se consideran neutros (combinan con todo)
    _NEUTRALS = {
        "blanco", "negro", "gris", "gris claro", "gris oscuro",
        "beige", "marrón claro", "marrón", "marrón oscuro",
        "plateado", "dorado",
    }

    def score(self, color_a: ColorInfo, color_b: ColorInfo) -> float:
        """Devuelve un score de compatibilidad entre 0 (malo) y 1 (perfecto)."""
        name_a = color_a.name.lower()
        name_b = color_b.name.lower()

        # Neutro + cualquier cosa → siempre compatible
        if name_a in self._NEUTRALS or name_b in self._NEUTRALS:
            return 0.92

        hsv_a = self._rgb_to_hsv(color_a.rgb)
        hsv_b = self._rgb_to_hsv(color_b.rgb)

        scores = [
            self._monochromatic(hsv_a, hsv_b),
            self._complementary(hsv_a, hsv_b),
            self._analogous(hsv_a, hsv_b),
            self._triadic(hsv_a, hsv_b),
            self._saturation_harmony(hsv_a, hsv_b),
        ]
        return float(max(scores))

    def explain(self, color_a: ColorInfo, color_b: ColorInfo) -> str:
        """Devuelve una explicación textual de por qué combinan (o no)."""
        name_a, name_b = color_a.name, color_b.name
        if color_a.name.lower() in self._NEUTRALS or color_b.name.lower() in self._NEUTRALS:
            return f"{name_a} es neutro y combina con {name_b}"

        hsv_a = self._rgb_to_hsv(color_a.rgb)
        hsv_b = self._rgb_to_hsv(color_b.rgb)
        hue_diff = self._hue_diff(hsv_a[0], hsv_b[0])

        if hue_diff < 20:
            return f"Combinación monocromática: {name_a} y {name_b} son muy similares"
        if hue_diff < 50:
            return f"Combinación análoga: {name_a} y {name_b} son colores vecinos"
        if 150 < hue_diff < 210:
            return f"Combinación complementaria: {name_a} y {name_b} se contrastan bien"
        if 100 < hue_diff < 140 or 220 < hue_diff < 260:
            return f"Combinación triádica: {name_a} y {name_b} forman equilibrio visual"
        return f"{name_a} y {name_b} no tienen una armonía de color clara"

    # ── Reglas de armonía ──────────────────────────────────────────────

    def _monochromatic(self, a: tuple, b: tuple) -> float:
        """Mismo tono, diferente saturación/brillo."""
        diff = self._hue_diff(a[0], b[0])
        return max(0.0, 1.0 - diff / 25) * 0.88

    def _complementary(self, a: tuple, b: tuple) -> float:
        """Tonos opuestos en la rueda de color (~180°)."""
        diff = self._hue_diff(a[0], b[0])
        proximity = 1.0 - abs(diff - 180) / 40
        return max(0.0, proximity) * 0.95

    def _analogous(self, a: tuple, b: tuple) -> float:
        """Tonos vecinos en la rueda (~30°)."""
        diff = self._hue_diff(a[0], b[0])
        return max(0.0, 1.0 - diff / 45) * 0.85

    def _triadic(self, a: tuple, b: tuple) -> float:
        """Tonos separados ~120° en la rueda."""
        diff = self._hue_diff(a[0], b[0])
        proximity = 1.0 - min(abs(diff - 120), abs(diff - 240)) / 30
        return max(0.0, proximity) * 0.80

    def _saturation_harmony(self, a: tuple, b: tuple) -> float:
        """Penaliza mezclar colores muy saturados con muy desaturados."""
        sat_diff = abs(a[1] - b[1])
        return max(0.0, 1.0 - sat_diff * 1.2) * 0.70

    # ── Utilidades ─────────────────────────────────────────────────────

    @staticmethod
    def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
        r, g, b = (x / 255.0 for x in rgb)
        cmax, cmin = max(r, g, b), min(r, g, b)
        delta = cmax - cmin
        # Hue
        if delta == 0:
            h = 0.0
        elif cmax == r:
            h = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            h = 60 * ((b - r) / delta + 2)
        else:
            h = 60 * ((r - g) / delta + 4)
        s = 0.0 if cmax == 0 else delta / cmax
        v = cmax
        return h, s, v

    @staticmethod
    def _hue_diff(h1: float, h2: float) -> float:
        """Diferencia angular mínima entre dos tonos (0-180)."""
        diff = abs(h1 - h2) % 360
        return min(diff, 360 - diff)


# ──────────────────────────────────────────────────────────────────────
# 2. StyleCompatibilityEngine
# ──────────────────────────────────────────────────────────────────────
class StyleCompatibilityEngine:
    """
    Puntúa la compatibilidad entre dos estilos usando una matriz de afinidad.
    Score devuelto: float en [0.0, 1.0]
    """

    # Matriz de afinidad entre estilos (simétrica, valores 0.0-1.0)
    _MATRIX: dict[str, dict[str, float]] = {
        "casual":         {"casual": 1.0, "urbano": 0.90, "minimalista": 0.80,
                           "deportivo": 0.70, "bohemio": 0.65, "vintage": 0.60,
                           "romántico": 0.55, "business casual": 0.50,
                           "formal": 0.30, "elegante": 0.25, "punk": 0.40},
        "formal":         {"formal": 1.0, "elegante": 0.90, "business casual": 0.80,
                           "minimalista": 0.65, "vintage": 0.50,
                           "casual": 0.30, "urbano": 0.25, "deportivo": 0.10,
                           "bohemio": 0.20, "romántico": 0.45, "punk": 0.10},
        "elegante":       {"elegante": 1.0, "formal": 0.90, "romántico": 0.75,
                           "minimalista": 0.70, "business casual": 0.65,
                           "vintage": 0.55, "casual": 0.25, "urbano": 0.20,
                           "deportivo": 0.10, "bohemio": 0.30, "punk": 0.10},
        "deportivo":      {"deportivo": 1.0, "casual": 0.70, "urbano": 0.65,
                           "minimalista": 0.50, "bohemio": 0.20,
                           "formal": 0.10, "elegante": 0.10, "vintage": 0.25,
                           "romántico": 0.15, "business casual": 0.20, "punk": 0.35},
        "urbano":         {"urbano": 1.0, "casual": 0.90, "deportivo": 0.65,
                           "minimalista": 0.75, "punk": 0.60, "vintage": 0.55,
                           "bohemio": 0.45, "business casual": 0.40,
                           "formal": 0.25, "elegante": 0.20, "romántico": 0.35},
        "bohemio":        {"bohemio": 1.0, "romántico": 0.80, "vintage": 0.75,
                           "casual": 0.65, "minimalista": 0.50,
                           "urbano": 0.45, "deportivo": 0.20,
                           "formal": 0.20, "elegante": 0.30, "business casual": 0.25,
                           "punk": 0.30},
        "vintage":        {"vintage": 1.0, "bohemio": 0.75, "romántico": 0.70,
                           "casual": 0.60, "elegante": 0.55, "formal": 0.50,
                           "urbano": 0.55, "minimalista": 0.45,
                           "deportivo": 0.25, "business casual": 0.40, "punk": 0.45},
        "minimalista":    {"minimalista": 1.0, "casual": 0.80, "urbano": 0.75,
                           "formal": 0.65, "elegante": 0.70, "business casual": 0.70,
                           "deportivo": 0.50, "vintage": 0.45,
                           "bohemio": 0.50, "romántico": 0.55, "punk": 0.40},
        "romántico":      {"romántico": 1.0, "bohemio": 0.80, "elegante": 0.75,
                           "vintage": 0.70, "casual": 0.55, "minimalista": 0.55,
                           "formal": 0.45, "urbano": 0.35,
                           "deportivo": 0.15, "business casual": 0.40, "punk": 0.15},
        "punk":           {"punk": 1.0, "urbano": 0.60, "vintage": 0.45,
                           "casual": 0.40, "deportivo": 0.35,
                           "bohemio": 0.30, "minimalista": 0.40,
                           "formal": 0.10, "elegante": 0.10,
                           "romántico": 0.15, "business casual": 0.15},
        "business casual": {"business casual": 1.0, "formal": 0.80, "elegante": 0.65,
                            "minimalista": 0.70, "casual": 0.50,
                            "vintage": 0.40, "urbano": 0.40,
                            "romántico": 0.40, "bohemio": 0.25,
                            "deportivo": 0.20, "punk": 0.15},
    }

    def score(self, style_a: str, style_b: str) -> float:
        """Devuelve compatibilidad de estilo entre 0 y 1."""
        a, b = style_a.lower().strip(), style_b.lower().strip()
        row = self._MATRIX.get(a, {})
        return row.get(b, row.get(a, 0.5))

    def explain(self, style_a: str, style_b: str) -> str:
        s = self.score(style_a, style_b)
        if s >= 0.85:
            return f"{style_a} y {style_b} combinan perfectamente"
        if s >= 0.65:
            return f"{style_a} y {style_b} combinan bien"
        if s >= 0.45:
            return f"{style_a} y {style_b} combinan con moderación"
        return f"{style_a} y {style_b} no combinan bien"


# ──────────────────────────────────────────────────────────────────────
# 3. OutfitScorer
# ──────────────────────────────────────────────────────────────────────
@dataclass
class OutfitScore:
    garments: list[ClothAnalysis]       # prendas del conjunto
    color_score: float                  # 0-100
    style_score: float                  # 0-100
    total_score: float                  # 0-100 (media ponderada)
    color_explanation: str
    style_explanation: str
    outfit_label: str                   # "casual veraniego", etc.


class OutfitScorer:
    """Combina puntuaciones de color y estilo en un score final."""

    COLOR_WEIGHT = 0.55
    STYLE_WEIGHT = 0.45

    def __init__(self):
        self._color_engine = ColorCompatibilityEngine()
        self._style_engine = StyleCompatibilityEngine()

    def score_outfit(self, garments: list[ClothAnalysis]) -> OutfitScore:
        """
        Puntúa un conjunto de 2 o más prendas.
        Para conjuntos de más de 2 prendas calcula la media de todos los pares.
        """
        pairs = list(combinations(garments, 2))

        color_scores = [
            self._color_engine.score(a.color, b.color) for a, b in pairs
        ]
        style_scores = [
            self._style_engine.score(a.style, b.style) for a, b in pairs
        ]

        avg_color = float(np.mean(color_scores)) * 100
        avg_style = float(np.mean(style_scores)) * 100
        total = self.COLOR_WEIGHT * avg_color + self.STYLE_WEIGHT * avg_style

        # Explicaciones del par principal (primeras dos prendas)
        a, b = garments[0], garments[1]
        color_exp = self._color_engine.explain(a.color, b.color)
        style_exp = self._style_engine.explain(a.style, b.style)

        label = self._build_label(garments, total)

        return OutfitScore(
            garments=garments,
            color_score=round(avg_color, 1),
            style_score=round(avg_style, 1),
            total_score=round(total, 1),
            color_explanation=color_exp,
            style_explanation=style_exp,
            outfit_label=label,
        )

    @staticmethod
    def _build_label(garments: list[ClothAnalysis], score: float) -> str:
        styles = [g.style for g in garments]
        dominant_style = max(set(styles), key=styles.count)
        colors = [g.color.name for g in garments]
        color_str = " + ".join(dict.fromkeys(colors))  # únicos, orden preservado
        quality = "✨ Destacado" if score >= 80 else ("👍 Bueno" if score >= 60 else "⚠️ Mejorable")
        return f"{quality} · {dominant_style} · {color_str}"


# ──────────────────────────────────────────────────────────────────────
# 4. OutfitRecommender
# ──────────────────────────────────────────────────────────────────────
class OutfitRecommender:
    def __init__(self, top_n: int = 10, min_score: float = 50.0):
        """
        Args:
            top_n:     Número máximo de conjuntos a recomendar.
            min_score: Score mínimo (0-100) para incluir un conjunto.
        """
        self.top_n     = top_n
        self.min_score = min_score
        self._scorer   = OutfitScorer()

    # ── API pública ────────────────────────────────────────────────────

    def recommend(
        self,
        analyses: list[ClothAnalysis],
        output_dir: str | Path = "outputs/outfits",
    ) -> list[OutfitScore]:
        """
        Genera y puntúa todas las combinaciones válidas del armario.

        Args:
            analyses:   Lista de ClothAnalysis (salida del módulo 2).
            output_dir: Carpeta donde guardar el informe visual y el JSON.

        Returns:
            Lista de OutfitScore ordenada de mayor a menor puntuación.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outfits = self._build_combinations(analyses)
        if not outfits:
            print("⚠️  No hay suficientes prendas para generar conjuntos.")
            return []

        print(f"🔍 Evaluando {len(outfits)} combinaciones posibles…")
        scored = [self._scorer.score_outfit(outfit) for outfit in outfits]
        scored = [s for s in scored if s.total_score >= self.min_score]
        scored.sort(key=lambda s: s.total_score, reverse=True)
        top = scored[: self.top_n]

        self._save_json(top, output_dir / "recommendations.json")
        self._save_report(top, output_dir / "recommendations_report.jpg")
        self._print_summary(top)

        return top

    def recommend_from_json(
        self,
        analysis_json: str | Path,
        output_dir: str | Path = "outputs/outfits",
    ) -> list[OutfitScore]:
        """
        Carga los análisis desde el JSON guardado por el módulo 2
        y genera las recomendaciones.
        """
        analyses = self._load_analyses(Path(analysis_json))
        return self.recommend(analyses, output_dir)

    # ── Construcción de combinaciones ──────────────────────────────────

    def _build_combinations(self, analyses: list[ClothAnalysis]) -> list[list[ClothAnalysis]]:
        """
        Genera combinaciones válidas:
          - Prenda completa sola (vestido, mono, traje)
          - Superior + inferior
          - Superior + inferior + exterior (chaqueta/abrigo)
        """
        uppers  = [a for a in analyses if a.cloth_type in UPPER_GARMENTS
                   and a.cloth_type not in OUTER_GARMENTS]
        lowers  = [a for a in analyses if a.cloth_type in LOWER_GARMENTS]
        outers  = [a for a in analyses if a.cloth_type in OUTER_GARMENTS]
        fulls   = [a for a in analyses if a.cloth_type in FULL_GARMENTS]

        combos: list[list[ClothAnalysis]] = []

        # Prendas completas solas
        for f in fulls:
            combos.append([f])

        # Prendas completas + exterior
        for f in fulls:
            for o in outers:
                combos.append([f, o])

        # Superior + inferior
        for u in uppers:
            for l in lowers:
                combos.append([u, l])

        # Superior + inferior + exterior
        for u in uppers:
            for l in lowers:
                for o in outers:
                    combos.append([u, l, o])

        return combos

    # ── Informe visual ─────────────────────────────────────────────────

    def _save_report(self, outfits: list[OutfitScore], path: Path) -> None:
        """Genera una imagen con las miniaturas y puntuaciones de cada conjunto."""
        if not outfits:
            return

        THUMB   = 120          # tamaño miniatura
        PADDING = 12
        ROW_H   = THUMB + 90   # alto de cada fila (miniatura + texto)
        WIDTH   = 900

        total_h = PADDING + len(outfits) * (ROW_H + PADDING) + 40
        report  = Image.new("RGB", (WIDTH, total_h), (18, 18, 24))
        draw    = ImageDraw.Draw(report)

        try:
            font_title = ImageFont.truetype("arial.ttf", 15)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            font_title = ImageFont.load_default()
            font_small = font_title

        # Cabecera
        draw.text((PADDING, 10), "👗 ARMARIO DIGITAL — CONJUNTOS RECOMENDADOS",
                  fill=(72, 201, 176), font=font_title)

        y_cursor = 40
        for rank, outfit in enumerate(outfits, 1):
            x_cursor = PADDING

            # Miniaturas de las prendas
            for garment in outfit.garments:
                img_path = Path(garment.image_path)
                if img_path.exists():
                    thumb = Image.open(img_path).convert("RGBA")
                    thumb.thumbnail((THUMB, THUMB), Image.LANCZOS)
                    # Fondo blanco para la miniatura
                    bg = Image.new("RGB", (THUMB, THUMB), (35, 35, 45))
                    tw, th = thumb.size
                    bg.paste(thumb, ((THUMB - tw) // 2, (THUMB - th) // 2), thumb)
                    report.paste(bg, (x_cursor, y_cursor))
                else:
                    # Placeholder si no existe la imagen
                    ph = Image.new("RGB", (THUMB, THUMB), (50, 50, 60))
                    report.paste(ph, (x_cursor, y_cursor))
                x_cursor += THUMB + 6

            # Barra de score
            bar_x = x_cursor + 10
            score_pct = outfit.total_score / 100
            bar_w_total = WIDTH - bar_x - PADDING
            bar_w_fill  = int(bar_w_total * score_pct)
            bar_color   = self._score_color(outfit.total_score)

            draw.rectangle([bar_x, y_cursor + 10,
                            bar_x + bar_w_total, y_cursor + 28],
                           fill=(40, 40, 50))
            draw.rectangle([bar_x, y_cursor + 10,
                            bar_x + bar_w_fill, y_cursor + 28],
                           fill=bar_color)

            # Textos
            texts = [
                (f"#{rank}  {outfit.outfit_label}",          (bar_x, y_cursor + 34), font_title, (220, 220, 220)),
                (f"Score total : {outfit.total_score:.1f}/100", (bar_x, y_cursor + 52), font_small, (180, 180, 180)),
                (f"Color  {outfit.color_score:.1f}  |  Estilo  {outfit.style_score:.1f}", (bar_x, y_cursor + 66), font_small, (150, 150, 160)),
                (outfit.color_explanation,                    (bar_x, y_cursor + 80), font_small, (120, 200, 160)),
                (outfit.style_explanation,                    (bar_x, y_cursor + 94), font_small, (120, 160, 200)),
            ]
            for text, pos, font, color in texts:
                draw.text(pos, text, fill=color, font=font)

            # Separador
            draw.line([(PADDING, y_cursor + ROW_H - 2),
                       (WIDTH - PADDING, y_cursor + ROW_H - 2)],
                      fill=(40, 40, 50), width=1)
            y_cursor += ROW_H + PADDING

        report.save(str(path), quality=92)
        print(f"🖼️  Informe visual guardado en: {path}")

    @staticmethod
    def _score_color(score: float) -> tuple[int, int, int]:
        if score >= 80:
            return (72, 201, 176)   # verde agua
        if score >= 60:
            return (255, 200, 60)   # amarillo
        return (220, 80, 80)        # rojo

    # ── Persistencia ───────────────────────────────────────────────────

    @staticmethod
    def _save_json(outfits: list[OutfitScore], path: Path) -> None:
        data = []
        for o in outfits:
            entry = {
                "total_score":        o.total_score,
                "color_score":        o.color_score,
                "style_score":        o.style_score,
                "outfit_label":       o.outfit_label,
                "color_explanation":  o.color_explanation,
                "style_explanation":  o.style_explanation,
                "garments": [
                    {
                        "image_path": g.image_path,
                        "cloth_type": g.cloth_type,
                        "style":      g.style,
                        "color_name": g.color.name,
                        "color_hex":  g.color.hex,
                    }
                    for g in o.garments
                ],
            }
            data.append(entry)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 Recomendaciones guardadas en: {path}")

    @staticmethod
    def _load_analyses(json_path: Path) -> list[ClothAnalysis]:
        from modules.cloth_analyzer import ColorInfo, ClothAnalysis
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        analyses = []
        for item in raw:
            c = item["color"]
            color = ColorInfo(
                name=c["name"],
                hex=c["hex"],
                rgb=tuple(c["rgb"]),
                palette=[tuple(p) for p in c["palette"]],
            )
            analyses.append(ClothAnalysis(
                image_path=item["image_path"],
                color=color,
                cloth_type=item["cloth_type"],
                cloth_type_scores=item["cloth_type_scores"],
                style=item["style"],
                style_scores=item["style_scores"],
            ))
        return analyses

    # ── Consola ────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(outfits: list[OutfitScore]) -> None:
        print(f"\n✅ Top {len(outfits)} conjuntos recomendados:\n")
        for i, o in enumerate(outfits, 1):
            prendas = " + ".join(
                f"{g.cloth_type} ({g.color.name})" for g in o.garments
            )
            print(
                f"  #{i:02d}  {o.total_score:5.1f}/100  {o.outfit_label}\n"
                f"       Prendas : {prendas}\n"
                f"       Color   : {o.color_explanation}\n"
                f"       Estilo  : {o.style_explanation}\n"
            )


# ──────────────────────────────────────────────────────────────────────
# Ejecución directa
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    recommender = OutfitRecommender(top_n=10, min_score=50)

    # Opción A: desde el JSON generado por el módulo 2
    recommender.recommend_from_json(
        analysis_json="data/clothes_processed/analysis.json",
        output_dir="outputs/outfits",
    )
