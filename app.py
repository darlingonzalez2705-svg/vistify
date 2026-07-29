"""
app.py — Vistify · Probador Virtual Inteligente
Ejecuta: streamlit run app.py
"""

import json
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Configuración de página ────────────────────────────────────────────
st.set_page_config(
    page_title="Vistify",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Estilos CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #0e0e12; color: #f0f0f0; }

    /* Ocultar barra superior de Streamlit */
    header { visibility: hidden; }

    /* Título principal */
    .vistify-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #48c9b0, #a29bfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .vistify-sub {
        color: #888;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    /* Tarjetas de paso */
    .step-card {
        background: #1a1a24;
        border: 1px solid #2a2a3a;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .step-number {
        background: linear-gradient(135deg, #48c9b0, #a29bfe);
        color: #0e0e12;
        font-weight: 800;
        font-size: 0.85rem;
        padding: 3px 10px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    /* Botones */
    .stButton > button {
        background: linear-gradient(135deg, #48c9b0, #a29bfe);
        color: #0e0e12;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Uploader */
    .stFileUploader {
        background: #1a1a24;
        border: 2px dashed #2a2a3a;
        border-radius: 12px;
        padding: 1rem;
    }

    /* Métricas */
    .metric-box {
        background: #1a1a24;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #48c9b0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #888;
        margin-top: 2px;
    }

    /* Score bar */
    .score-bar-bg {
        background: #2a2a3a;
        border-radius: 10px;
        height: 8px;
        margin: 6px 0;
    }
    .score-bar-fill {
        height: 8px;
        border-radius: 10px;
        background: linear-gradient(90deg, #48c9b0, #a29bfe);
    }

    /* Outfit card */
    .outfit-card {
        background: #1a1a24;
        border: 1px solid #2a2a3a;
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .outfit-score {
        font-size: 1.4rem;
        font-weight: 800;
        color: #48c9b0;
    }

    /* Divider */
    hr { border-color: #2a2a3a; }

    /* Número de input */
    .stNumberInput input { background: #1a1a24; color: #f0f0f0; border-color: #2a2a3a; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #1a1a24; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { color: #888; }
    .stTabs [aria-selected="true"] { color: #48c9b0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Estado de sesión ───────────────────────────────────────────────────
for key in ["avatar_result", "analyses", "top_outfits"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ── Header ─────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<p class="vistify-title">Vistify</p>', unsafe_allow_html=True)
    st.markdown('<p class="vistify-sub">Tu probador virtual inteligente · Prueba ropa y descubre tus mejores conjuntos</p>', unsafe_allow_html=True)

st.markdown("---")

# ── Tabs principales ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["👤  Mi Avatar", "👗  Mi Ropa", "✨  Conjuntos"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Avatar
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">PASO 1</span>', unsafe_allow_html=True)
    st.markdown("### Sube tu foto y genera tu avatar")
    st.markdown("Necesitamos una foto tuya de **cuerpo completo**, de frente y con buena iluminación.")
    st.markdown('</div>', unsafe_allow_html=True)

    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        altura = st.number_input("Tu altura (cm)", min_value=100, max_value=220, value=170, step=1)
        foto = st.file_uploader("Sube tu foto frontal", type=["jpg", "jpeg", "png"], key="foto_upload")

        if foto:
            st.image(foto, caption="Foto subida", use_container_width=True)

        if foto and st.button("🧬 Generar Avatar"):
            with st.spinner("Analizando tu cuerpo..."):
                try:
                    from modules.avatar_generator import AvatarGenerator

                    # Guardar foto temporalmente
                    Path("data/user_photos").mkdir(parents=True, exist_ok=True)
                    foto_path = Path("data/user_photos/foto_frontal.jpg")
                    foto_path.write_bytes(foto.read())

                    avatar = AvatarGenerator(real_height_cm=altura).generate(
                        photo_path=foto_path,
                        output_dir="data/avatar",
                    )
                    st.session_state.avatar_result = avatar
                    st.success("✅ Avatar generado correctamente")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    with col_preview:
        if st.session_state.avatar_result:
            avatar = st.session_state.avatar_result
            avatar_img_path = Path(avatar.avatar_path)
            if avatar_img_path.exists():
                st.image(str(avatar_img_path), caption="Tu avatar", use_container_width=True)

            # Métricas corporales
            m = avatar.measurements
            st.markdown("#### 📏 Tus medidas")
            c1, c2, c3 = st.columns(3)
            metrics = [
                (c1, f"{m.shoulder_width_cm} cm", "Hombros"),
                (c2, f"{m.chest_width_cm} cm", "Pecho"),
                (c3, f"{m.waist_width_cm} cm", "Cintura"),
            ]
            for col, val, label in metrics:
                with col:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

            st.markdown("")
            c4, c5, c6 = st.columns(3)
            metrics2 = [
                (c4, f"{m.hip_width_cm} cm", "Cadera"),
                (c5, f"{m.left_leg_cm} cm", "Pierna"),
                (c6, f"{m.torso_cm} cm", "Torso"),
            ]
            for col, val, label in metrics2:
                with col:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#1a1a24;border:2px dashed #2a2a3a;border-radius:16px;height:400px;display:flex;align-items:center;justify-content:center;color:#444;font-size:1.1rem;">Tu avatar aparecerá aquí</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Ropa
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">PASO 2</span>', unsafe_allow_html=True)
    st.markdown("### Sube las fotos de tu ropa")
    st.markdown("Puedes subir varias prendas a la vez. Cuantas más subas, mejores conjuntos podremos recomendarte.")
    st.markdown('</div>', unsafe_allow_html=True)

    fotos_ropa = st.file_uploader(
        "Sube tus prendas (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="ropa_upload",
    )

    if fotos_ropa:
        # Previsualización en grid
        st.markdown(f"**{len(fotos_ropa)} prenda(s) seleccionada(s)**")
        cols = st.columns(min(len(fotos_ropa), 4))
        for i, f in enumerate(fotos_ropa):
            with cols[i % 4]:
                st.image(f, caption=f.name, use_container_width=True)

        if st.button("🔍 Analizar mi ropa"):
            with st.spinner("Procesando prendas... esto puede tardar unos minutos"):
                try:
                    from modules.background_remover import BackgroundRemover
                    from modules.cloth_analyzer import ClothAnalyzer

                    raw_dir  = Path("data/clothes_raw")
                    proc_dir = Path("data/clothes_processed")
                    raw_dir.mkdir(parents=True, exist_ok=True)

                    # Guardar fotos subidas
                    for f in fotos_ropa:
                        (raw_dir / f.name).write_bytes(f.read())

                    # Módulo 1 + 2
                    BackgroundRemover(padding=20, canvas_size=512).process_folder(raw_dir, proc_dir)
                    analyses = ClothAnalyzer().analyze_folder(
                        proc_dir,
                        output_json="data/clothes_processed/analysis.json",
                    )
                    st.session_state.analyses = analyses
                    st.success(f"✅ {len(analyses)} prendas analizadas correctamente")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # Mostrar prendas analizadas
    if st.session_state.analyses:
        st.markdown("---")
        st.markdown("#### 👗 Prendas en tu armario")
        analyses = st.session_state.analyses
        cols = st.columns(min(len(analyses), 4))
        for i, a in enumerate(analyses):
            with cols[i % 4]:
                img_path = Path(a.image_path)
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                st.markdown(f"**{a.cloth_type.capitalize()}**")
                st.markdown(f"🎨 {a.color.name}  `{a.color.hex}`")
                st.markdown(f"✨ {a.style}")


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Conjuntos
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown('<span class="step-number">PASO 3</span>', unsafe_allow_html=True)
    st.markdown("### Descubre tus mejores conjuntos")
    st.markdown("Analizamos compatibilidad de colores y estilos para recomendarte los mejores looks.")
    st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.analyses:
        st.info("👆 Primero analiza tu ropa en la pestaña **Mi Ropa**")
    else:
        if st.button("✨ Generar recomendaciones"):
            with st.spinner("Calculando los mejores conjuntos..."):
                try:
                    from modules.outfit_recommender import OutfitRecommender
                    top = OutfitRecommender(top_n=10, min_score=50).recommend(
                        analyses=st.session_state.analyses,
                        output_dir="outputs/outfits",
                    )
                    st.session_state.top_outfits = top
                    st.success(f"✅ {len(top)} conjuntos encontrados")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        if st.session_state.top_outfits:
            top = st.session_state.top_outfits

            # Informe visual
            report_path = Path("outputs/outfits/recommendations_report.jpg")
            if report_path.exists():
                st.markdown("#### 🖼️ Informe visual")
                st.image(str(report_path), use_container_width=True)

            st.markdown("---")
            st.markdown("#### 🏆 Top conjuntos")

            for i, outfit in enumerate(top, 1):
                score_pct = int(outfit.total_score)
                bar_color = "#48c9b0" if score_pct >= 80 else ("#f0c040" if score_pct >= 60 else "#e05050")

                with st.container():
                    st.markdown(f'<div class="outfit-card">', unsafe_allow_html=True)

                    col_rank, col_info, col_tryon = st.columns([1, 5, 2])

                    with col_rank:
                        st.markdown(f'<div class="outfit-score">#{i}<br>{score_pct}/100</div>', unsafe_allow_html=True)

                    with col_info:
                        st.markdown(f"**{outfit.outfit_label}**")
                        st.markdown(f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{score_pct}%;background:{bar_color}"></div></div>', unsafe_allow_html=True)
                        st.markdown(f"🎨 {outfit.color_explanation}")
                        st.markdown(f"✨ {outfit.style_explanation}")

                        # Miniaturas de prendas
                        thumb_cols = st.columns(len(outfit.garments))
                        for j, g in enumerate(outfit.garments):
                            with thumb_cols[j]:
                                p = Path(g.image_path)
                                if p.exists():
                                    st.image(str(p), width=80)
                                st.caption(g.cloth_type)

                    with col_tryon:
                        if st.session_state.avatar_result:
                            if st.button(f"👗 Probar", key=f"tryon_{i}"):
                                with st.spinner("Probando conjunto..."):
                                    try:
                                        from modules.virtual_tryon import VirtualTryOn
                                        result = VirtualTryOn().try_on(
                                            avatar_result=st.session_state.avatar_result,
                                            garments=[{"path": g.image_path, "type": g.cloth_type} for g in outfit.garments],
                                            output_dir="outputs/outfits",
                                            output_name=f"tryon_{i}.jpg",
                                        )
                                        tryon_path = Path(result.output_path)
                                        if tryon_path.exists():
                                            st.image(str(tryon_path), caption="Resultado", use_container_width=True)
                                    except Exception as e:
                                        st.error(f"❌ {e}")
                        else:
                            st.caption("Genera tu avatar primero")

                    st.markdown('</div>', unsafe_allow_html=True)
