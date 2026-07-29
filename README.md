# 👗 Armario Digital Inteligente

Prueba tu ropa virtualmente con un avatar generado desde tus propias fotos
y recibe recomendaciones de conjuntos basadas en teoría del color y estilos.

---

## Estructura

```
ArmarioDigital/
├── data/
│   ├── user_photos/            # Fotos del cuerpo del usuario
│   ├── clothes_raw/            # Fotos originales de prendas
│   ├── clothes_processed/      # Prendas sin fondo (PNG transparente)
│   │   └── analysis.json       # Análisis de color/tipo/estilo (módulo 2)
│   └── avatar/
│       ├── foto_avatar.jpg             # Avatar con esqueleto y medidas
│       └── foto_measurements.json     # Medidas corporales en JSON
├── modules/
│   ├── background_remover.py   ✅ Módulo 1 — Eliminar fondo y recortar
│   ├── cloth_analyzer.py       ✅ Módulo 2 — Detectar color, tipo y estilo
│   ├── avatar_generator.py     ✅ Módulo 3 — Generar avatar con medidas
│   ├── virtual_tryon.py        ✅ Módulo 4 — Probar ropa en el avatar
│   └── outfit_recommender.py   ✅ Módulo 5 — Recomendar conjuntos
├── outputs/
│   └── outfits/
│       ├── outfit_01.jpg               # Foto con prendas superpuestas
│       ├── recommendations.json        # Top conjuntos en JSON
│       └── recommendations_report.jpg  # Informe visual con miniaturas
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Flujo completo de uso

### Módulo 1 — Eliminar fondo y recortar prendas

```python
from modules.background_remover import BackgroundRemover

remover = BackgroundRemover(padding=20, canvas_size=512)
remover.process_folder("data/clothes_raw", "data/clothes_processed")
```

### Módulo 2 — Analizar color, tipo y estilo

```python
from modules.cloth_analyzer import ClothAnalyzer

analyzer = ClothAnalyzer()
analyzer.analyze_folder(
    input_dir="data/clothes_processed",
    output_json="data/clothes_processed/analysis.json",
)
```

### Módulo 3 — Generar avatar con medidas corporales

```python
from modules.avatar_generator import AvatarGenerator

# Solo necesitas introducir tu altura real
avatar = AvatarGenerator(real_height_cm=170).generate(
    photo_path="data/user_photos/foto_frontal.jpg",
    output_dir="data/avatar",
)
print(avatar.measurements.shoulder_width_cm)  # 42.3 cm
print(avatar.measurements.hip_width_cm)       # 38.7 cm
```

### Módulo 4 — Probador virtual

```python
from modules.virtual_tryon import VirtualTryOn

tryon = VirtualTryOn()
tryon.try_on(
    avatar_result=avatar,
    garments=[
        {"path": "data/clothes_processed/camisa_nobg.png",   "type": "camisa"},
        {"path": "data/clothes_processed/vaqueros_nobg.png", "type": "vaqueros"},
    ],
    output_name="outfit_01.jpg",
)
```

### Módulo 5 — Recomendador de conjuntos

```python
from modules.outfit_recommender import OutfitRecommender

recommender = OutfitRecommender(top_n=10, min_score=50)
top_outfits = recommender.recommend_from_json(
    analysis_json="data/clothes_processed/analysis.json",
    output_dir="outputs/outfits",
)

for outfit in top_outfits:
    print(f"{outfit.total_score:.1f}/100 — {outfit.outfit_label}")
```

---

## Pipeline completo de una sola vez

```python
from modules.background_remover import BackgroundRemover
from modules.cloth_analyzer     import ClothAnalyzer
from modules.avatar_generator   import AvatarGenerator
from modules.virtual_tryon      import VirtualTryOn
from modules.outfit_recommender import OutfitRecommender

# 1. Digitalizar prendas
BackgroundRemover().process_folder("data/clothes_raw", "data/clothes_processed")

# 2. Analizar prendas
analyses = ClothAnalyzer().analyze_folder(
    "data/clothes_processed",
    output_json="data/clothes_processed/analysis.json",
)

# 3. Generar avatar
avatar = AvatarGenerator(real_height_cm=170).generate(
    "data/user_photos/foto_frontal.jpg"
)

# 4. Probar el mejor conjunto
top = OutfitRecommender(top_n=1).recommend(analyses)
if top:
    best = top[0]
    VirtualTryOn().try_on(
        avatar_result=avatar,
        garments=[{"path": g.image_path, "type": g.cloth_type} for g in best.garments],
        output_name="mejor_conjunto.jpg",
    )
```
