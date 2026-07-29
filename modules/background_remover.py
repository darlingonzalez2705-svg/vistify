"""
Módulo: background_remover.py
Elimina el fondo de fotos de ropa, recorta automáticamente la prenda
y la centra en un canvas cuadrado con fondo transparente.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import new_session, remove
from tqdm import tqdm


@dataclass
class ClothResult:
    output_path: Path
    original_size: tuple[int, int]   # (ancho, alto) original
    cropped_size: tuple[int, int]     # (ancho, alto) tras recorte
    canvas_size: tuple[int, int]      # (ancho, alto) del canvas final
    coverage_pct: float               # % de píxeles que son prenda (0-100)


class BackgroundRemover:
    def __init__(
        self,
        model: str = "u2net_cloth_seg",
        padding: int = 20,
        canvas_size: int | None = 512,
    ):
        """
        Args:
            model:       Modelo rembg. 'u2net_cloth_seg' está especializado en ropa.
            padding:     Píxeles de margen alrededor del bounding box de la prenda.
            canvas_size: Si se indica, centra la prenda en un canvas cuadrado de este tamaño.
                         None → devuelve solo el recorte sin canvas fijo.
        """
        self.session = new_session(model)
        self.padding = padding
        self.canvas_size = canvas_size

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def process(self, input_path: str | Path, output_path: str | Path) -> ClothResult:
        """
        Elimina el fondo, recorta y centra una prenda.

        Args:
            input_path:  Imagen original de la prenda.
            output_path: Ruta de salida (PNG con transparencia).

        Returns:
            ClothResult con métricas del procesamiento.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        original = Image.open(input_path).convert("RGBA")
        nobg = self._remove_bg(original)
        cropped = self._autocrop(nobg)
        final = self._to_canvas(cropped) if self.canvas_size else cropped

        final.save(output_path, format="PNG")

        coverage = self._coverage(final)
        result = ClothResult(
            output_path=output_path,
            original_size=original.size,
            cropped_size=cropped.size,
            canvas_size=final.size,
            coverage_pct=coverage,
        )
        print(f"[OK] {input_path.name} → {output_path.name}  "
              f"({cropped.size[0]}×{cropped.size[1]}px, {coverage:.1f}% prenda)")
        return result

    def process_folder(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> list[ClothResult]:
        """
        Procesa todas las imágenes de una carpeta.

        Args:
            input_dir:  Carpeta con fotos originales de ropa.
            output_dir: Carpeta de salida para los PNGs procesados.

        Returns:
            Lista de ClothResult, uno por imagen procesada.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        extensions = {".jpg", ".jpeg", ".png", ".webp"}

        images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in extensions)
        if not images:
            print(f"⚠️  No se encontraron imágenes en {input_dir}")
            return []

        results = []
        for img_path in tqdm(images, desc="Procesando prendas"):
            out_path = output_dir / (img_path.stem + "_nobg.png")
            results.append(self.process(img_path, out_path))

        print(f"\n✅ {len(results)} prendas procesadas en: {output_dir}")
        return results

    # ------------------------------------------------------------------
    # Pasos internos
    # ------------------------------------------------------------------

    def _remove_bg(self, image: Image.Image) -> Image.Image:
        """Llama a rembg y devuelve la imagen RGBA sin fondo."""
        result_bytes = remove(self._to_bytes(image), session=self.session)
        return Image.open(__import__("io").BytesIO(result_bytes)).convert("RGBA")

    def _autocrop(self, image: Image.Image) -> Image.Image:
        """
        Recorta la imagen al bounding box de los píxeles no transparentes
        más un margen de `self.padding` píxeles.
        """
        alpha = np.array(image.split()[-1])          # canal A
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)

        if not rows.any():
            return image  # imagen completamente transparente, no recortar

        top, bottom = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]) - 1)
        left, right = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]) - 1)

        pad = self.padding
        w, h = image.size
        box = (
            max(0, left - pad),
            max(0, top - pad),
            min(w, right + pad + 1),
            min(h, bottom + pad + 1),
        )
        return image.crop(box)

    def _to_canvas(self, image: Image.Image) -> Image.Image:
        """
        Centra la prenda recortada en un canvas cuadrado transparente
        de `self.canvas_size` × `self.canvas_size` píxeles.
        Escala la prenda para que quepa manteniendo la proporción.
        """
        size = self.canvas_size
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

        image.thumbnail((size, size), Image.LANCZOS)
        iw, ih = image.size
        offset = ((size - iw) // 2, (size - ih) // 2)
        canvas.paste(image, offset, mask=image)
        return canvas

    @staticmethod
    def _to_bytes(image: Image.Image) -> bytes:
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _coverage(image: Image.Image) -> float:
        """Porcentaje de píxeles visibles (alpha > 0) sobre el total del canvas."""
        alpha = np.array(image.split()[-1])
        return float(np.count_nonzero(alpha) / alpha.size * 100)


# ----------------------------------------------------------------------
# Ejecución directa
# ----------------------------------------------------------------------
if __name__ == "__main__":
    remover = BackgroundRemover(padding=20, canvas_size=512)
    remover.process_folder(
        input_dir="data/clothes_raw",
        output_dir="data/clothes_processed",
    )
