"""
server.py — Servidor FastAPI de Vistify
Ejecuta: python server.py
"""

import json
import shutil
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Vistify API")

# ── CORS (permite que la web llame al servidor) ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Servir archivos estáticos de la web ───────────────────────────────
app.mount("/css",     StaticFiles(directory="css"),     name="css")
app.mount("/js",      StaticFiles(directory="js"),      name="js")
app.mount("/pages",   StaticFiles(directory="pages"),   name="pages")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/data",    StaticFiles(directory="data"),    name="data")

# ── Página principal ──────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse("index.html")

# ── Endpoint: Generar avatar ──────────────────────────────────────────
@app.post("/api/avatar")
async def generar_avatar(
    foto: UploadFile = File(...),
    altura: float    = Form(170.0),
):
    try:
        # Guardar foto
        foto_path = Path("data/user_photos/foto_frontal.jpg")
        foto_path.parent.mkdir(parents=True, exist_ok=True)
        with foto_path.open("wb") as f:
            shutil.copyfileobj(foto.file, f)

        # Generar avatar
        from modules.avatar_generator import AvatarGenerator
        avatar = AvatarGenerator(real_height_cm=altura).generate(
            photo_path=foto_path,
            output_dir="data/avatar",
        )

        m = avatar.measurements
        return JSONResponse({
            "ok": True,
            "avatar_url": f"/data/avatar/{Path(avatar.avatar_path).name}",
            "medidas": {
                "hombros": m.shoulder_width_cm,
                "pecho":   m.chest_width_cm,
                "cintura": m.waist_width_cm,
                "cadera":  m.hip_width_cm,
                "pierna":  m.left_leg_cm,
                "torso":   m.torso_cm,
            }
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Endpoint: Analizar ropa ───────────────────────────────────────────
@app.post("/api/ropa")
async def analizar_ropa(fotos: list[UploadFile] = File(...)):
    try:
        raw_dir  = Path("data/clothes_raw")
        proc_dir = Path("data/clothes_processed")
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Guardar fotos
        for foto in fotos:
            dest = raw_dir / foto.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(foto.file, f)

        # Módulo 1: eliminar fondos
        from modules.background_remover import BackgroundRemover
        BackgroundRemover(padding=20, canvas_size=512).process_folder(raw_dir, proc_dir)

        # Módulo 2: analizar
        from modules.cloth_analyzer import ClothAnalyzer
        analyses = ClothAnalyzer().analyze_folder(
            proc_dir,
            output_json="data/clothes_processed/analysis.json",
        )

        prendas = [
            {
                "imagen": f"/data/clothes_processed/{Path(a.image_path).name}",
                "tipo":   a.cloth_type,
                "estilo": a.style,
                "color":  a.color.name,
                "hex":    a.color.hex,
            }
            for a in analyses
        ]

        return JSONResponse({"ok": True, "prendas": prendas})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Endpoint: Recomendar conjuntos ────────────────────────────────────
@app.get("/api/conjuntos")
def recomendar_conjuntos():
    try:
        json_path = Path("data/clothes_processed/analysis.json")
        if not json_path.exists():
            return JSONResponse({"ok": False, "error": "Primero analiza tu ropa"}, status_code=400)

        from modules.outfit_recommender import OutfitRecommender
        top = OutfitRecommender(top_n=10, min_score=50).recommend_from_json(
            analysis_json=json_path,
            output_dir="outputs/outfits",
        )

        conjuntos = [
            {
                "rank":        i + 1,
                "score":       o.total_score,
                "label":       o.outfit_label,
                "color_exp":   o.color_explanation,
                "style_exp":   o.style_explanation,
                "prendas": [
                    {
                        "imagen": f"/data/clothes_processed/{Path(g.image_path).name}",
                        "tipo":   g.cloth_type,
                        "color":  g.color.name,
                        "hex":    g.color.hex,
                    }
                    for g in o.garments
                ],
            }
            for i, o in enumerate(top)
        ]

        return JSONResponse({"ok": True, "conjuntos": conjuntos})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Endpoint: Probador virtual ────────────────────────────────────────
@app.post("/api/tryon/{rank}")
def probar_conjunto(rank: int):
    try:
        json_path = Path("data/clothes_processed/analysis.json")
        avatar_json = next(Path("data/avatar").glob("*_measurements.json"), None)

        if not json_path.exists() or not avatar_json:
            return JSONResponse({"ok": False, "error": "Faltan datos"}, status_code=400)

        from modules.outfit_recommender import OutfitRecommender
        from modules.virtual_tryon import VirtualTryOn

        top = OutfitRecommender(top_n=10, min_score=50).recommend_from_json(json_path)
        if rank > len(top):
            return JSONResponse({"ok": False, "error": "Conjunto no encontrado"}, status_code=404)

        outfit = top[rank - 1]
        foto_path = Path("data/user_photos/foto_frontal.jpg")

        result = VirtualTryOn().try_on_from_paths(
            photo_path=foto_path,
            measurements_json=avatar_json,
            garments=[{"path": g.image_path, "type": g.cloth_type} for g in outfit.garments],
            output_dir="outputs/outfits",
            output_name=f"tryon_{rank}.jpg",
        )

        return JSONResponse({
            "ok": True,
            "imagen": f"/outputs/outfits/tryon_{rank}.jpg"
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Arrancar servidor ─────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
