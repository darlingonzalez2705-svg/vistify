"""
Módulo: avatar_generator.py
Genera un avatar digital del usuario a partir de una foto de cuerpo completo.

Flujo:
  1. PoseDetector      → detecta 33 landmarks corporales con MediaPipe Pose
  2. MeasureEstimator  → convierte distancias en píxeles a centímetros reales
                         usando la altura real del usuario como referencia
  3. AvatarGenerator   → dibuja el avatar anotado y guarda medidas en JSON
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Landmark:
    x: float   # coordenada normalizada [0, 1]
    y: float
    z: float
    visibility: float


@dataclass
class BodyMeasurements:
    height_cm: float          # altura real introducida por el usuario
    shoulder_width_cm: float  # ancho de hombros
    chest_width_cm: float     # ancho del pecho (aprox. axila-axila)
    waist_width_cm: float     # ancho de cintura (aprox. cadera alta)
    hip_width_cm: float       # ancho de cadera
    left_arm_cm: float        # longitud brazo izquierdo (hombro → muñeca)
    right_arm_cm: float       # longitud brazo derecho
    left_leg_cm: float        # longitud pierna izquierda (cadera → tobillo)
    right_leg_cm: float       # longitud pierna derecha
    torso_cm: float           # longitud torso (hombro → cadera)
    px_per_cm: float          # escala calculada (píxeles por cm)


@dataclass
class AvatarResult:
    photo_path: str
    avatar_path: str
    measurements: BodyMeasurements
    landmarks: dict[str, Landmark] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Índices de landmarks de MediaPipe Pose que usamos
# ──────────────────────────────────────────────────────────────────────
LM = mp.solutions.pose.PoseLandmark

LANDMARK_NAMES = {
    "nose":            LM.NOSE,
    "left_shoulder":   LM.LEFT_SHOULDER,
    "right_shoulder":  LM.RIGHT_SHOULDER,
    "left_elbow":      LM.LEFT_ELBOW,
    "right_elbow":     LM.RIGHT_ELBOW,
    "left_wrist":      LM.LEFT_WRIST,
    "right_wrist":     LM.RIGHT_WRIST,
    "left_hip":        LM.LEFT_HIP,
    "right_hip":       LM.RIGHT_HIP,
    "left_knee":       LM.LEFT_KNEE,
    "right_knee":      LM.RIGHT_KNEE,
    "left_ankle":      LM.LEFT_ANKLE,
    "right_ankle":     LM.RIGHT_ANKLE,
    "left_heel":       LM.LEFT_HEEL,
    "right_heel":      LM.RIGHT_HEEL,
}

# Conexiones para dibujar el esqueleto del avatar
SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("nose",           "left_shoulder"),
    ("nose",           "right_shoulder"),
]


# ──────────────────────────────────────────────────────────────────────
# 1. PoseDetector
# ──────────────────────────────────────────────────────────────────────
class PoseDetector:
    def __init__(self, min_detection_confidence: float = 0.6):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, image_path: str | Path) -> tuple[np.ndarray, dict[str, Landmark]]:
        """
        Detecta los landmarks de pose en una imagen.

        Args:
            image_path: Foto de cuerpo completo (frontal, fondo liso recomendado).

        Returns:
            (imagen BGR como ndarray, dict nombre→Landmark)

        Raises:
            ValueError: Si MediaPipe no detecta ninguna persona.
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(img_rgb)

        if not results.pose_landmarks:
            raise ValueError(
                "MediaPipe no detectó ninguna persona en la imagen. "
                "Asegúrate de que la foto sea de cuerpo completo y bien iluminada."
            )

        h, w = img_bgr.shape[:2]
        landmarks: dict[str, Landmark] = {}
        for name, idx in LANDMARK_NAMES.items():
            lm = results.pose_landmarks.landmark[idx]
            landmarks[name] = Landmark(
                x=lm.x * w,
                y=lm.y * h,
                z=lm.z,
                visibility=lm.visibility,
            )

        return img_bgr, landmarks


# ──────────────────────────────────────────────────────────────────────
# 2. MeasureEstimator
# ──────────────────────────────────────────────────────────────────────
class MeasureEstimator:
    def estimate(
        self,
        landmarks: dict[str, Landmark],
        height_cm: float,
        image_shape: tuple[int, int],
    ) -> BodyMeasurements:
        """
        Convierte distancias en píxeles a centímetros reales.

        La escala se calcula dividiendo la altura real del usuario entre
        la distancia en píxeles desde la nariz hasta los tobillos.

        Args:
            landmarks:   Landmarks detectados por PoseDetector.
            height_cm:   Altura real del usuario en centímetros.
            image_shape: (alto, ancho) de la imagen en píxeles.

        Returns:
            BodyMeasurements con todas las medidas en cm.
        """
        px_per_cm = self._compute_scale(landmarks, height_cm)

        def dist(a: str, b: str) -> float:
            return self._px_dist(landmarks[a], landmarks[b]) / px_per_cm

        def mid(a: str, b: str) -> Landmark:
            la, lb = landmarks[a], landmarks[b]
            return Landmark(
                x=(la.x + lb.x) / 2,
                y=(la.y + lb.y) / 2,
                z=(la.z + lb.z) / 2,
                visibility=min(la.visibility, lb.visibility),
            )

        # Puntos medios para medidas simétricas
        mid_shoulder = mid("left_shoulder", "right_shoulder")
        mid_hip      = mid("left_hip",      "right_hip")

        # Ancho de pecho: 80% del ancho de hombros (aproximación anatómica)
        shoulder_w = dist("left_shoulder", "right_shoulder")
        chest_w    = shoulder_w * 0.80

        # Cintura: punto medio entre hombros y cadera, ancho estimado al 65%
        waist_w = dist("left_hip", "right_hip") * 0.85

        return BodyMeasurements(
            height_cm=round(height_cm, 1),
            shoulder_width_cm=round(shoulder_w, 1),
            chest_width_cm=round(chest_w, 1),
            waist_width_cm=round(waist_w, 1),
            hip_width_cm=round(dist("left_hip", "right_hip"), 1),
            left_arm_cm=round(
                dist("left_shoulder", "left_elbow") + dist("left_elbow", "left_wrist"), 1
            ),
            right_arm_cm=round(
                dist("right_shoulder", "right_elbow") + dist("right_elbow", "right_wrist"), 1
            ),
            left_leg_cm=round(
                dist("left_hip", "left_knee") + dist("left_knee", "left_ankle"), 1
            ),
            right_leg_cm=round(
                dist("right_hip", "right_knee") + dist("right_knee", "right_ankle"), 1
            ),
            torso_cm=round(
                self._px_dist(mid_shoulder, mid_hip) / px_per_cm, 1
            ),
            px_per_cm=round(px_per_cm, 4),
        )

    def _compute_scale(self, landmarks: dict[str, Landmark], height_cm: float) -> float:
        """
        Calcula píxeles/cm usando la distancia nariz→tobillo como proxy
        de la altura visible en la imagen.
        """
        nose = landmarks["nose"]

        # Tobillo más bajo (mayor Y) como referencia del suelo
        ankle_y = max(landmarks["left_ankle"].y, landmarks["right_ankle"].y)
        ankle_x = (landmarks["left_ankle"].x + landmarks["right_ankle"].x) / 2
        ankle = Landmark(x=ankle_x, y=ankle_y, z=0, visibility=1)

        height_px = self._px_dist(nose, ankle)

        # La nariz está aprox. al 95% de la altura total del cuerpo
        full_height_px = height_px / 0.95
        return full_height_px / height_cm

    @staticmethod
    def _px_dist(a: Landmark, b: Landmark) -> float:
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


# ──────────────────────────────────────────────────────────────────────
# 3. AvatarGenerator
# ──────────────────────────────────────────────────────────────────────
class AvatarGenerator:
    # Colores del avatar (BGR para OpenCV)
    _COLOR_SKELETON  = (72, 201, 176)   # verde agua
    _COLOR_JOINT     = (255, 255, 255)  # blanco
    _COLOR_MEASURE   = (255, 200, 0)    # amarillo
    _COLOR_TEXT_BG   = (30, 30, 30)     # fondo oscuro etiquetas
    _COLOR_TEXT      = (255, 255, 255)

    def __init__(self, real_height_cm: float):
        """
        Args:
            real_height_cm: Altura real del usuario en centímetros.
                            Es el único dato que el usuario debe introducir manualmente.
        """
        self.height_cm = real_height_cm
        self._detector  = PoseDetector()
        self._estimator = MeasureEstimator()

    # ── API pública ────────────────────────────────────────────────────

    def generate(
        self,
        photo_path: str | Path,
        output_dir: str | Path = "data/avatar",
    ) -> AvatarResult:
        """
        Genera el avatar y las medidas a partir de una foto de cuerpo completo.

        Args:
            photo_path: Foto frontal del usuario (JPG/PNG).
            output_dir: Carpeta donde guardar el avatar y el JSON de medidas.

        Returns:
            AvatarResult con medidas y rutas de archivos generados.
        """
        photo_path = Path(photo_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📸 Procesando foto: {photo_path.name}")
        img_bgr, landmarks = self._detector.detect(photo_path)

        h, w = img_bgr.shape[:2]
        measurements = self._estimator.estimate(landmarks, self.height_cm, (h, w))

        avatar_img = self._draw_avatar(img_bgr.copy(), landmarks, measurements)

        avatar_path = output_dir / (photo_path.stem + "_avatar.jpg")
        cv2.imwrite(str(avatar_path), avatar_img)

        json_path = output_dir / (photo_path.stem + "_measurements.json")
        self._save_json(measurements, landmarks, json_path)

        result = AvatarResult(
            photo_path=str(photo_path),
            avatar_path=str(avatar_path),
            measurements=measurements,
            landmarks=landmarks,
        )
        self._print_summary(measurements)
        return result

    # ── Dibujo del avatar ──────────────────────────────────────────────

    def _draw_avatar(
        self,
        img: np.ndarray,
        landmarks: dict[str, Landmark],
        m: BodyMeasurements,
    ) -> np.ndarray:
        """Dibuja esqueleto, articulaciones y etiquetas de medidas sobre la imagen."""
        img = self._draw_skeleton(img, landmarks)
        img = self._draw_joints(img, landmarks)
        img = self._draw_measurements(img, landmarks, m)
        img = self._draw_legend(img, m)
        return img

    def _draw_skeleton(self, img: np.ndarray, landmarks: dict[str, Landmark]) -> np.ndarray:
        for a, b in SKELETON_CONNECTIONS:
            if a not in landmarks or b not in landmarks:
                continue
            la, lb = landmarks[a], landmarks[b]
            if la.visibility < 0.4 or lb.visibility < 0.4:
                continue
            pt1 = (int(la.x), int(la.y))
            pt2 = (int(lb.x), int(lb.y))
            cv2.line(img, pt1, pt2, self._COLOR_SKELETON, 3, cv2.LINE_AA)
        return img

    def _draw_joints(self, img: np.ndarray, landmarks: dict[str, Landmark]) -> np.ndarray:
        for lm in landmarks.values():
            if lm.visibility < 0.4:
                continue
            cv2.circle(img, (int(lm.x), int(lm.y)), 6, self._COLOR_JOINT, -1, cv2.LINE_AA)
            cv2.circle(img, (int(lm.x), int(lm.y)), 6, self._COLOR_SKELETON, 2, cv2.LINE_AA)
        return img

    def _draw_measurements(
        self,
        img: np.ndarray,
        lm: dict[str, Landmark],
        m: BodyMeasurements,
    ) -> np.ndarray:
        """Dibuja líneas horizontales con las medidas clave."""
        measures_to_draw = [
            ("left_shoulder", "right_shoulder", f"Hombros {m.shoulder_width_cm} cm"),
            ("left_hip",      "right_hip",      f"Cadera  {m.hip_width_cm} cm"),
        ]
        for a, b, label in measures_to_draw:
            if a not in lm or b not in lm:
                continue
            la, lb = lm[a], lm[b]
            pt1 = (int(min(la.x, lb.x)) - 10, int((la.y + lb.y) / 2))
            pt2 = (int(max(la.x, lb.x)) + 10, int((la.y + lb.y) / 2))
            cv2.line(img, pt1, pt2, self._COLOR_MEASURE, 2, cv2.LINE_AA)
            cv2.circle(img, pt1, 4, self._COLOR_MEASURE, -1)
            cv2.circle(img, pt2, 4, self._COLOR_MEASURE, -1)
            self._put_label(img, label, ((pt1[0] + pt2[0]) // 2, pt1[1] - 10))
        return img

    def _draw_legend(self, img: np.ndarray, m: BodyMeasurements) -> np.ndarray:
        """Panel lateral derecho con todas las medidas."""
        h, w = img.shape[:2]
        panel_w = 230
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)

        lines = [
            "📏 MEDIDAS",
            "",
            f"Altura       {m.height_cm} cm",
            f"Hombros      {m.shoulder_width_cm} cm",
            f"Pecho        {m.chest_width_cm} cm",
            f"Cintura      {m.waist_width_cm} cm",
            f"Cadera       {m.hip_width_cm} cm",
            f"Torso        {m.torso_cm} cm",
            "",
            f"Brazo izq.   {m.left_arm_cm} cm",
            f"Brazo der.   {m.right_arm_cm} cm",
            "",
            f"Pierna izq.  {m.left_leg_cm} cm",
            f"Pierna der.  {m.right_leg_cm} cm",
        ]

        y = 30
        for line in lines:
            color = (72, 201, 176) if "MEDIDAS" in line else (220, 220, 220)
            thickness = 2 if "MEDIDAS" in line else 1
            cv2.putText(panel, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, color, thickness, cv2.LINE_AA)
            y += 28

        # Unir panel a la derecha de la imagen
        if h > panel.shape[0]:
            pad = np.zeros((h - panel.shape[0], panel_w, 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        return np.hstack([img, panel])

    def _put_label(self, img: np.ndarray, text: str, pos: tuple[int, int]) -> None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = pos
        cv2.rectangle(img, (x - tw // 2 - 4, y - th - 4),
                      (x + tw // 2 + 4, y + 4), self._COLOR_TEXT_BG, -1)
        cv2.putText(img, text, (x - tw // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._COLOR_TEXT, 1, cv2.LINE_AA)

    # ── Persistencia ───────────────────────────────────────────────────

    @staticmethod
    def _save_json(
        m: BodyMeasurements,
        landmarks: dict[str, Landmark],
        path: Path,
    ) -> None:
        data = {
            "measurements": asdict(m),
            "landmarks": {
                name: {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for name, lm in landmarks.items()
            },
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 Medidas guardadas en: {path}")

    @staticmethod
    def _print_summary(m: BodyMeasurements) -> None:
        print(
            f"\n✅ Avatar generado\n"
            f"   📐 Escala         : {m.px_per_cm:.2f} px/cm\n"
            f"   📏 Altura         : {m.height_cm} cm\n"
            f"   👤 Hombros        : {m.shoulder_width_cm} cm\n"
            f"   💪 Pecho          : {m.chest_width_cm} cm\n"
            f"   ⬛ Cintura        : {m.waist_width_cm} cm\n"
            f"   🍑 Cadera         : {m.hip_width_cm} cm\n"
            f"   📏 Torso          : {m.torso_cm} cm\n"
            f"   💪 Brazo izq/der  : {m.left_arm_cm} / {m.right_arm_cm} cm\n"
            f"   🦵 Pierna izq/der : {m.left_leg_cm} / {m.right_leg_cm} cm"
        )


# ──────────────────────────────────────────────────────────────────────
# Ejecución directa
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generator = AvatarGenerator(real_height_cm=170)
    generator.generate(
        photo_path="data/user_photos/foto_frontal.jpg",
        output_dir="data/avatar",
    )
