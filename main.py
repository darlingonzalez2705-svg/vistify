"""
main.py — Pipeline completo del Armario Digital Inteligente
Ejecuta: python main.py
"""

from pathlib import Path

# ── Configuración ──────────────────────────────────────────────────────
ALTURA_CM       = 170          # ← cambia tu altura real aquí
FOTO_FRONTAL    = "data/user_photos/foto_frontal.jpg"
CLOTHES_RAW     = "data/clothes_raw"
CLOTHES_PROC    = "data/clothes_processed"
ANALYSIS_JSON   = "data/clothes_processed/analysis.json"
AVATAR_DIR      = "data/avatar"
OUTPUT_DIR      = "outputs/outfits"


def main():
    # ── Módulo 1: Eliminar fondos ──────────────────────────────────────
    print("\n🔵 MÓDULO 1 — Eliminando fondos de prendas...")
    from modules.background_remover import BackgroundRemover
    BackgroundRemover(padding=20, canvas_size=512).process_folder(CLOTHES_RAW, CLOTHES_PROC)

    # ── Módulo 2: Analizar prendas ─────────────────────────────────────
    print("\n🔵 MÓDULO 2 — Analizando prendas...")
    from modules.cloth_analyzer import ClothAnalyzer
    analyses = ClothAnalyzer().analyze_folder(CLOTHES_PROC, output_json=ANALYSIS_JSON)

    if not analyses:
        print("❌ No se encontraron prendas procesadas. Añade fotos en data/clothes_raw/")
        return

    # ── Módulo 3: Generar avatar ───────────────────────────────────────
    print("\n🔵 MÓDULO 3 — Generando avatar...")
    if not Path(FOTO_FRONTAL).exists():
        print(f"❌ No se encontró la foto frontal en: {FOTO_FRONTAL}")
        return

    from modules.avatar_generator import AvatarGenerator
    avatar = AvatarGenerator(real_height_cm=ALTURA_CM).generate(
        photo_path=FOTO_FRONTAL,
        output_dir=AVATAR_DIR,
    )

    # ── Módulo 5: Recomendar conjuntos ─────────────────────────────────
    print("\n🔵 MÓDULO 5 — Recomendando conjuntos...")
    from modules.outfit_recommender import OutfitRecommender
    top_outfits = OutfitRecommender(top_n=10, min_score=50).recommend(
        analyses=analyses,
        output_dir=OUTPUT_DIR,
    )

    if not top_outfits:
        print("⚠️  No se generaron conjuntos. Necesitas al menos una prenda superior y una inferior.")
        return

    # ── Módulo 4: Probar el mejor conjunto ────────────────────────────
    print("\n🔵 MÓDULO 4 — Probando el mejor conjunto en el avatar...")
    from modules.virtual_tryon import VirtualTryOn
    best = top_outfits[0]
    VirtualTryOn().try_on(
        avatar_result=avatar,
        garments=[{"path": g.image_path, "type": g.cloth_type} for g in best.garments],
        output_dir=OUTPUT_DIR,
        output_name="mejor_conjunto.jpg",
    )

    print("\n✅ Pipeline completado.")
    print(f"   👗 Mejor conjunto : {best.outfit_label}")
    print(f"   📊 Score          : {best.total_score:.1f}/100")
    print(f"   🖼️  Informe visual : {OUTPUT_DIR}/recommendations_report.jpg")
    print(f"   👤 Avatar         : {avatar.avatar_path}")


if __name__ == "__main__":
    main()
