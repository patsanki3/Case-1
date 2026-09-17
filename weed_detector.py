"""
Кейс 1: Сорняки на фотографиях с дрона
======================================

Пайплайн:
  1) Детекция растительности на фото поля (OpenCV, по цвету/контурам).
  2) Классификация вида сорняка по эталонным фото (CLIP-эмбеддинги, similarity search,
     без обучения своей модели — работает "из коробки").
  3) Оценка стадии вегетации (эвристика по размеру/форме контура; легко заменить
     на similarity по эталонам, если стадии размечены в подпапках).
  4) Экспорт результатов в CSV и JSON + сохранение аннотированных фото.

Запуск:
    pip install -r requirements.txt
    python weed_detector.py \
        --ref_dir /data/Сорняки \
        --field_dir /data/ФотоПолей \
        --out_dir ./results

Структура эталонов ожидается такой (папка = класс сорняка):
    /data/Сорняки/
        Осот/
            img1.jpg
            img2.jpg
        Пырей/
            img1.jpg
        ...

Если внутри есть подпапки по стадиям вегетации - скрипт будет определять
стадию точнее (через сравнение с эталонами, а не грубой эвристикой):
    /data/Сорняки/Осот/всходы/*.jpg
    /data/Сорняки/Осот/розетка/*.jpg
    /data/Сорняки/Осот/цветение/*.jpg

Чтобы отфильтровать культуру (пшеницу и т.п.) из подсчета сорняков, добавьте
для нее папку с "сорняковым" названием, содержащим одно из слов из
CROP_KEYWORDS (пшениц/культур/урожай/рожь/ячмен/овес/crop), например:
    /data/Сорняки/Пшеница/img1.jpg
Объекты, похожие на эти фото, будут распознаны, но НЕ попадут в результаты.
"""

import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np


def imread_unicode(path: str) -> Optional[np.ndarray]:
    """
    cv2.imread на Windows часто не может открыть файл, если путь содержит
    не-ASCII символы (кириллицу и т.п.) - молча возвращает None.
    Этот вариант читает файл через numpy + cv2.imdecode, что работает
    с любыми путями независимо от языка.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    """Аналогично imread_unicode, но для записи (нужно для путей с кириллицей)."""
    ext = os.path.splitext(path)[1] or ".jpg"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        return False
    encoded.tofile(path)
    return True
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Минимальная площадь контура (в пикселях) - возвращено к значению из
# первой рабочей версии.
MIN_CONTOUR_AREA = 400

# Порог косинусного сходства для классификации: если сходство с ЛУЧШИМ эталоном
# ниже этого значения — объект помечается как "unknown" (не уверены, что это
# именно этот вид сорняка).
SIMILARITY_THRESHOLD = 0.55

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Структура одной находки
# ---------------------------------------------------------------------------

@dataclass
class WeedDetection:
    photo: str
    weed_id: int
    species: str
    species_confidence: float
    weed_class: str          # "Двудольные" / "Злаковые" / "неизвестно"
    weed_subclass: str       # "малолетние" / "многолетние" / "неизвестно"
    stage: str                # исходная стадия (как есть, из эмбеддингов/эвристики)
    phase: str                 # нормализованная фаза по шпаргалке (для дозировки)
    dosage_note: str           # рекомендация ИИ по дозировке для этой фазы
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int


# ---------------------------------------------------------------------------
# Агрономическая логика (по шпаргалке ментора)
# ---------------------------------------------------------------------------

# КЛАССИФИКАЦИЯ: класс (двудольные/злаковые) x подкласс (малолетние/многолетние).
# Сопоставление по названию вида (без учета регистра, частичное совпадение).
# Дополните этот словарь под ваш реальный список видов из /data/Сорняки.
WEED_TAXONOMY: Dict[str, Tuple[str, str]] = {
    # Класс A: Двудольные (Широколистные)
    "бодяк": ("Двудольные", "многолетние"),
    "осот": ("Двудольные", "многолетние"),
    "вьюнок": ("Двудольные", "многолетние"),
    "ширица": ("Двудольные", "малолетние"),
    "марь": ("Двудольные", "малолетние"),
    # Класс B: Злаковые (Узколистные)
    "пырей": ("Злаковые", "многолетние"),
    "овсюг": ("Злаковые", "малолетние"),
    "куриное просо": ("Злаковые", "малолетние"),
    "просо": ("Злаковые", "малолетние"),
}


def get_taxonomy(species: str) -> Tuple[str, str]:
    """Возвращает (класс, подкласс) по названию вида. 'неизвестно', если вид
    не найден в WEED_TAXONOMY - дополните словарь под свои виды."""
    name_lower = species.lower()
    for keyword, (weed_class, subclass) in WEED_TAXONOMY.items():
        if keyword in name_lower:
            return weed_class, subclass
    return "неизвестно", "неизвестно"


# АНАЛИЗ ФАЗ РАЗВИТИЯ: нормализуем произвольную метку стадии (из эталонов
# или из эвристики estimate_stage) в одну из 3 канонических фаз шпаргалки.
# Дополните PHASE_KEYWORDS, если ваши эталоны используют другие названия
# стадий (например "2 листа", "4 листа" и т.п.).
PHASE_SEEDLING = "Семядоли — 2 листа"
PHASE_MID = "4-6 листьев"
PHASE_LATE = "Более 6 листьев / цветение"

PHASE_KEYWORDS: Dict[str, str] = {
    "семядол": PHASE_SEEDLING,
    "всход": PHASE_SEEDLING,
    "2 лист": PHASE_SEEDLING,
    "розетк": PHASE_MID,
    "4 лист": PHASE_MID,
    "5 лист": PHASE_MID,
    "6 лист": PHASE_MID,
    "цветени": PHASE_LATE,
    "вегетативный рост": PHASE_LATE,
    "более 6": PHASE_LATE,
}

DOSAGE_NOTES: Dict[str, str] = {
    PHASE_SEEDLING: "Идеальное окно для обработки. Базовая/минимальная дозировка.",
    PHASE_MID: "Сорняк грубеет. Увеличить дозировку на 15-20%.",
    PHASE_LATE: "Упущенное окно, сорняк устойчив. Риск сжечь культуру - "
    "предупреждение агроному о неэффективности обработки.",
}


def normalize_phase(stage_label: str) -> Tuple[str, str]:
    """Приводит произвольную метку стадии к одной из 3 фаз шпаргалки и
    возвращает (фаза, рекомендация по дозировке). Если метка не распознана -
    возвращает ('неопределено', '') и не мешает работе остального пайплайна."""
    label_lower = stage_label.lower()
    for keyword, phase in PHASE_KEYWORDS.items():
        if keyword in label_lower:
            return phase, DOSAGE_NOTES[phase]
    return "неопределено", ""


# ЛОГИКА ПРИНЯТИЯ РЕШЕНИЙ: экономический порог вредоносности на 1 кв.м.
# Работает на уровне ОДНОГО ФОТО - т.к. порог задан "на м2", для перевода
# количества найденных сорняков в плотность (шт/м2) нужно знать реальную
# площадь, которую покрывает одно фото поля. Передайте ее через
# --photo_area_m2 (если не передано - решение по опрыскиванию не считается,
# выводятся только количества).
def decide_treatment(annual_count: int, perennial_count: int, area_m2: Optional[float]) -> Dict[str, str]:
    if area_m2 is None or area_m2 <= 0:
        return {
            "annual_count": annual_count,
            "perennial_count": perennial_count,
            "annual_density_per_m2": "н/д (не указана площадь фото)",
            "perennial_density_per_m2": "н/д (не указана площадь фото)",
            "decision": "н/д - укажите --photo_area_m2 для расчета плотности",
        }

    annual_density = annual_count / area_m2
    perennial_density = perennial_count / area_m2

    # Многолетники - приоритетный сектор, критическая угроза важнее.
    if perennial_density >= 2:
        decision = "КРИТИЧЕСКАЯ УГРОЗА -> СРОЧНО ОБРАБОТАТЬ (многолетники >= 2 шт/м2)"
    elif annual_density > 15:
        decision = "Сильная засоренность -> Включить опрыскивание на максимуме"
    elif annual_density >= 6:
        decision = "Средняя засоренность -> Стандартная норма гербицида"
    else:
        decision = "Слабая засоренность -> НЕ ОПРЫСКИВАТЬ (экономически невыгодно)"

    return {
        "annual_count": annual_count,
        "perennial_count": perennial_count,
        "annual_density_per_m2": round(annual_density, 2),
        "perennial_density_per_m2": round(perennial_density, 2),
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Шаг 1. Детекция растительности (OpenCV)
# ---------------------------------------------------------------------------

# Ядро для склейки соседних листьев одного растения в единый контур.
# ВОЗВРАЩЕНО к небольшому значению из первой версии: она была надежнее
# (ничего не пропускала), пусть один сорняк иногда и распадается на
# несколько боксов - это лучше, чем пропущенный сорняк или слипшийся
# в один "ком" целый ряд культуры.
MERGE_KERNEL_SIZE = 5
MERGE_ITERATIONS = 2

# Дополнительное объединение боксов ОТКЛЮЧЕНО (было источником "цепной
# реакции" на густых участках). 0 = объединяются только уже пересекающиеся
# прямоугольники, без искусственного "притягивания" соседей.
MERGE_DISTANCE_PX = 0

# Если получившийся бокс покрывает больше этой доли площади всего кадра —
# это, скорее всего, артефакт склейки, а не одно растение. Оставлен как
# защитная сетка на случай нештатно большого пятна зелени.
MAX_BOX_AREA_RATIO = 0.5


def detect_plant_regions(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Детекция растительности по цвету (HSV) + контурам, с усиленной склейкой
    отдельных листьев в один объект (одно растение).

    Идея: сорняки и культурные растения - зеленые объекты на фоне почвы/стерни
    (коричневый/желтый). Маска по зеленому + сильная морфологическая склейка
    (иначе один сорняк с несколькими листьями режется на много мелких боксов)
    + поиск внешних контуров + финальное объединение близких боксов.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Диапазон зеленого. При необходимости подправьте под освещение ваших фото.
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Шаг A: убираем мелкий шум (отдельные пиксели, мусор)
    open_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # Шаг B: сильная склейка соседних листьев одного растения в единый блок.
    # Эллиптическое ядро лучше повторяет форму растительных скоплений, чем
    # прямоугольное.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MERGE_KERNEL_SIZE, MERGE_KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=MERGE_ITERATIONS)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = image_bgr.shape[0] * image_bgr.shape[1]
    max_area = frame_area * MAX_BOX_AREA_RATIO

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))

    boxes = merge_close_boxes(boxes, MERGE_DISTANCE_PX)

    # Повторно отфильтровываем после объединения - вдруг слияние близких
    # боксов само создало аномально большой прямоугольник.
    boxes = [
        (x, y, w, h) for (x, y, w, h) in boxes if w * h <= max_area
    ]

    return boxes


def _boxes_close(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], gap: int) -> bool:
    """True, если прямоугольники a и b пересекаются или расположены ближе gap пикселей."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax - gap, ay - gap, ax + aw + gap, ay + ah + gap
    bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def merge_close_boxes(
    boxes: List[Tuple[int, int, int, int]], gap: int
) -> List[Tuple[int, int, int, int]]:
    """
    Объединяет боксы, которые пересекаются или находятся ближе `gap` пикселей
    друг от друга, в один общий bbox. Устраняет случаи, когда одно растение
    все равно распалось на 2-3 отдельных бокса после морфологии.
    """
    if not boxes:
        return []

    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_close(boxes[i], boxes[j], gap):
                union(i, j)

    groups: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(boxes[i])

    merged = []
    for group in groups.values():
        xs1 = [x for x, y, w, h in group]
        ys1 = [y for x, y, w, h in group]
        xs2 = [x + w for x, y, w, h in group]
        ys2 = [y + h for x, y, w, h in group]
        x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
        merged.append((x1, y1, x2 - x1, y2 - y1))

    return merged


# ---------------------------------------------------------------------------
# Шаг 2. Классификация вида сорняка через CLIP-эмбеддинги (few-shot)
# ---------------------------------------------------------------------------

class ClipClassifier:
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        print(f"[CLIP] Загружаю модель {model_name} на {DEVICE} ...")
        self.model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @staticmethod
    def _to_tensor(feats):
        """
        В разных версиях `transformers` get_image_features() возвращает либо
        готовый тензор, либо объект-обёртку (например BaseModelOutputWithPooling).
        Эта функция достает сам тензор эмбеддингов независимо от версии.
        """
        if torch.is_tensor(feats):
            return feats
        for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
            if hasattr(feats, attr):
                value = getattr(feats, attr)
                if torch.is_tensor(value):
                    return value
        raise TypeError(
            f"Не удалось извлечь тензор эмбеддингов из объекта типа {type(feats)}. "
            f"Проверьте версию transformers."
        )

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        feats = self._to_tensor(self.model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0]

    @torch.no_grad()
    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.model.config.projection_dim))
        inputs = self.processor(images=images, return_tensors="pt").to(DEVICE)
        feats = self._to_tensor(self.model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()


# Названия папок-эталонов, которые считаются КУЛЬТУРОЙ (урожаем), а не
# сорняком - если папка с эталонными фото содержит любое из этих слов в
# названии (без учета регистра), объекты, классифицированные как эта культура,
# НЕ попадут в финальный список сорняков (results.csv/json), даже если
# детектор их нашел на фото поля.
#
# Как использовать: добавьте в /data/Сорняки папку с фото самой культуры,
# например /data/Сорняки/Пшеница/*.jpg (несколько характерных фото ростков
# пшеницы) - и она будет автоматически исключена из подсчета сорняков.
CROP_KEYWORDS = ["пшениц", "культур", "урожай", "рожь", "ячмен", "овес", "овёс", "crop"]


def is_crop_class(species_name: str) -> bool:
    name_lower = species_name.lower()
    return any(keyword in name_lower for keyword in CROP_KEYWORDS)


def build_reference_embeddings(
    ref_dir: str, clf: ClipClassifier
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """
    Проходит по /data/Сорняки/<Вид>/*.jpg и строит:
      1) species_embeddings: усредненный эмбеддинг по ВСЕМ фото вида (для
         классификации вида сорняка/культуры, как раньше).
      2) stage_embeddings: {вид: {стадия: эмбеддинг}} - ТОЛЬКО если у вида
         есть подпапки по стадиям вегетации, например:
             /data/Сорняки/Осот/всходы/*.jpg
             /data/Сорняки/Осот/розетка/*.jpg
             /data/Сорняки/Осот/цветение/*.jpg
         Если таких подпапок нет - для этого вида stage_embeddings будет
         пустым, и стадия определится эвристикой (estimate_stage) как раньше.

    Определение стадии через эталоны точнее эвристики, поэтому используйте
    его, если у вас есть фото сорняков, размеченные по стадиям.
    """
    ref_path = Path(ref_dir)
    species_embeddings: Dict[str, np.ndarray] = {}
    stage_embeddings: Dict[str, Dict[str, np.ndarray]] = {}

    species_dirs = [d for d in ref_path.iterdir() if d.is_dir()]
    if not species_dirs:
        raise ValueError(f"В {ref_dir} не найдено подпапок с видами сорняков.")

    for species_dir in species_dirs:
        image_paths = [
            p
            for p in species_dir.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        ]
        if not image_paths:
            print(f"[WARN] Нет фото для класса '{species_dir.name}', пропускаю.")
            continue

        images = []
        for p in image_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"[WARN] Не смог открыть {p}: {e}")

        if not images:
            print(f"[WARN] Все фото для класса '{species_dir.name}' не удалось открыть, пропускаю.")
            continue

        embeddings = clf.embed_batch(images)
        mean_emb = embeddings.mean(axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        species_embeddings[species_dir.name] = mean_emb

        tag = " [КУЛЬТУРА - будет исключена из подсчета сорняков]" if is_crop_class(species_dir.name) else ""
        print(f"[REF] {species_dir.name}: {len(images)} эталонных фото{tag}")

        # Проверяем, есть ли подпапки по стадиям вегетации внутри вида.
        stage_dirs = [d for d in species_dir.iterdir() if d.is_dir()]
        if stage_dirs:
            stage_embeddings[species_dir.name] = {}
            for stage_dir in stage_dirs:
                stage_image_paths = [
                    p
                    for p in stage_dir.rglob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
                ]
                stage_images = []
                for p in stage_image_paths:
                    try:
                        stage_images.append(Image.open(p).convert("RGB"))
                    except Exception:
                        pass
                if not stage_images:
                    continue
                stage_emb = clf.embed_batch(stage_images).mean(axis=0)
                stage_emb = stage_emb / np.linalg.norm(stage_emb)
                stage_embeddings[species_dir.name][stage_dir.name] = stage_emb
                print(f"       -> стадия '{stage_dir.name}': {len(stage_images)} фото")

    return species_embeddings, stage_embeddings


def classify_crop(
    crop_embedding: np.ndarray, species_embeddings: Dict[str, np.ndarray]
) -> Tuple[str, float]:
    """Возвращает (лучший_вид, sim) либо ('unknown', sim) если ниже порога."""
    best_species, best_sim = "unknown", -1.0
    for species, emb in species_embeddings.items():
        sim = float(np.dot(crop_embedding, emb))
        if sim > best_sim:
            best_species, best_sim = species, sim

    if best_sim < SIMILARITY_THRESHOLD:
        return "unknown", best_sim
    return best_species, best_sim


# ---------------------------------------------------------------------------
# Шаг 3. Оценка стадии вегетации (эвристика)
# ---------------------------------------------------------------------------

def estimate_stage(crop_bgr: np.ndarray) -> str:
    """
    Грубая эвристика по размеру и "плотности" зелени в кропе:
      - маленький, редкая зелень      -> "всходы"
      - средний, компактная розетка   -> "розетка"
      - крупный, разветвленный        -> "вегетативный рост"
      - заметная доля ярких/светлых
        нЕзеленых пятен (цветки)      -> "цветение"

    ВАЖНО: доли считаются относительно площади самого растения (зеленых
    пикселей внутри бокса), а не всего bbox - иначе светлая сухая почва
    внутри бокса ложно засчитывается как "цветение".

    Это заготовка для демонстрации логики. Если в эталонах есть фото,
    размеченные по стадиям (подпапки), лучше заменить на такой же
    similarity-подход, как для вида (см. build_reference_embeddings).
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_area = cv2.countNonZero(green_mask)

    if green_area == 0:
        return "неопределено"

    # Яркие/светлые пятна, которые при этом НЕ зеленые (белые/розовые/желтые
    # лепестки цветов) - считаем только внутри области растения, а не по
    # всему боксу, чтобы не путать с фоновой почвой.
    bright_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
    bright_non_green = cv2.bitwise_and(bright_mask, cv2.bitwise_not(green_mask))
    bright_area = cv2.countNonZero(bright_non_green)
    bright_ratio = bright_area / green_area

    if bright_ratio > 0.20:
        return "цветение"
    if green_area < 3000:
        return "всходы"
    if green_area < 12000:
        return "розетка"
    return "вегетативный рост"


def classify_stage(
    crop_embedding: np.ndarray,
    species: str,
    stage_embeddings: Dict[str, Dict[str, np.ndarray]],
    crop_bgr: np.ndarray,
) -> str:
    """
    Определяет стадию вегетации: если для данного вида есть эталонные фото
    по стадиям (stage_embeddings[species] не пусто) - используем similarity
    по эмбеддингам (точнее, т.к. видит форму листа/цвет так же, как для
    определения вида). Иначе - откатываемся на эвристику по размеру/цвету
    (estimate_stage).
    """
    species_stages = stage_embeddings.get(species)
    if species_stages:
        best_stage, best_sim = "неопределено", -1.0
        for stage_name, emb in species_stages.items():
            sim = float(np.dot(crop_embedding, emb))
            if sim > best_sim:
                best_stage, best_sim = stage_name, sim
        return best_stage

    return estimate_stage(crop_bgr)


def save_debug_mask(image_bgr: np.ndarray, out_path: Path):
    """
    Сохраняет визуализацию того, что "видит" детектор на этапе цветовой
    маски + склейки, ДО поиска контуров и классификации. Если реальный
    сорняк слился с соседним растением - будет видно белое пятно на месте
    обоих. Если сорняк раздробился на кусочки - будет видно несколько
    отдельных белых пятен вместо одного. Это позволяет подобрать
    MERGE_KERNEL_SIZE / MIN_CONTOUR_AREA "по картинке", а не наугад.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    open_kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MERGE_KERNEL_SIZE, MERGE_KERNEL_SIZE)
    )
    mask_closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, close_kernel, iterations=MERGE_ITERATIONS
    )

    # Собираем сравнение рядом: исходное фото | маска до склейки | маска после склейки
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_closed_bgr = cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([image_bgr, mask_bgr, mask_closed_bgr])
    imwrite_unicode(str(out_path), combined)


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def process_field_photos(
    field_dir: str,
    out_dir: str,
    species_embeddings: Dict[str, np.ndarray],
    stage_embeddings: Dict[str, Dict[str, np.ndarray]],
    clf: ClipClassifier,
    photo_area_m2: Optional[float] = None,
) -> Tuple[List[WeedDetection], List[Dict]]:
    """
    Пайплайн из ДВУХ явных шагов на каждое фото (по договоренности):
      ШАГ 1 - находим ВСЕ растения на фото и убираем культуру из рассмотрения
              (is_crop_class) - остаются только кандидаты в сорняки.
      ШАГ 2 - для оставшихся объектов определяем вид, класс/подкласс
              (двудольные/злаковые x малолетние/многолетние) и фазу развития,
              на основе которых формируется агрономическая рекомендация.

    Возвращает (все_находки, отчет_по_фото) - отчет_по_фото используется для
    решения "опрыскивать / не опрыскивать" по каждому фото (см. decide_treatment).
    """
    field_path = Path(field_dir)
    out_path = Path(out_dir)
    annotated_dir = out_path / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p
        for p in field_path.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    if not image_paths:
        raise ValueError(f"В {field_dir} не найдено изображений.")

    print(f"[INFO] Найдено {len(image_paths)} фото поля для обработки.")

    all_detections: List[WeedDetection] = []
    photo_reports: List[Dict] = []
    weed_id_counter = 0
    crop_excluded_count = 0

    for img_path in image_paths:
        t_start = time.perf_counter()

        image_bgr = imread_unicode(str(img_path))
        if image_bgr is None:
            print(f"[WARN] Не удалось прочитать {img_path}, пропускаю.")
            continue

        # --- ШАГ 1: находим все растения на фото ---
        boxes = detect_plant_regions(image_bgr)
        annotated = image_bgr.copy()
        photo_crop_count = 0
        weed_candidates = []  # (x, y, w, h, crop_bgr, embedding, species, sim)

        for (x, y, w, h) in boxes:
            crop_bgr = image_bgr[y : y + h, x : x + w]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            emb = clf.embed_image(crop_pil)
            species, sim = classify_crop(emb, species_embeddings)

            if is_crop_class(species):
                # Это культура - убираем из дальнейшего рассмотрения (ШАГ 1).
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 1)
                photo_crop_count += 1
                crop_excluded_count += 1
                continue

            weed_candidates.append((x, y, w, h, crop_bgr, emb, species, sim))

        # --- ШАГ 2: только среди того, что осталось после фильтра культуры,
        # определяем вид/класс/подкласс/фазу и формируем рекомендацию ---
        annual_count = 0
        perennial_count = 0

        for (x, y, w, h, crop_bgr, emb, species, sim) in weed_candidates:
            if species == "unknown":
                color = (0, 165, 255)  # оранжевый - "неопознано"
            else:
                color = (0, 0, 255)  # красный - опознанный сорняк

            stage = classify_stage(emb, species, stage_embeddings, crop_bgr)
            weed_class, weed_subclass = get_taxonomy(species)
            phase, dosage_note = normalize_phase(stage)

            if weed_subclass == "многолетние":
                perennial_count += 1
            elif weed_subclass == "малолетние":
                annual_count += 1
            # "неизвестно" не учитывается в подсчете плотности - дополните
            # WEED_TAXONOMY, чтобы вид попадал в малолетние/многолетние.

            weed_id_counter += 1
            all_detections.append(
                WeedDetection(
                    photo=img_path.name,
                    weed_id=weed_id_counter,
                    species=species,
                    species_confidence=round(sim, 3),
                    weed_class=weed_class,
                    weed_subclass=weed_subclass,
                    stage=stage,
                    phase=phase,
                    dosage_note=dosage_note,
                    bbox_x=x,
                    bbox_y=y,
                    bbox_w=w,
                    bbox_h=h,
                )
            )

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = f"#{weed_id_counter} {species} [{weed_subclass}] ({stage})"
            cv2.putText(
                annotated,
                label,
                (x, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        out_img_path = annotated_dir / img_path.name
        imwrite_unicode(str(out_img_path), annotated)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        print(
            f"[OK] {img_path.name}: найдено {len(boxes)} объект(ов), "
            f"из них {photo_crop_count} - культура; обработка фото: {elapsed_ms:.0f} мс"
        )

        report = decide_treatment(annual_count, perennial_count, photo_area_m2)
        report["photo"] = img_path.name
        photo_reports.append(report)

    if crop_excluded_count > 0:
        print(f"\n[INFO] Всего исключено как культура (не сорняк): {crop_excluded_count}")

    return all_detections, photo_reports


def save_results(detections: List[WeedDetection], photo_reports: List[Dict], out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # CSV
    df = pd.DataFrame([asdict(d) for d in detections])
    csv_path = out_path / "results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # JSON
    json_path = out_path / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in detections], f, ensure_ascii=False, indent=2)

    # Сводка по количеству сорняков каждого вида
    if not df.empty:
        summary = (
            df.groupby(["species", "weed_class", "weed_subclass"])
            .agg(count=("weed_id", "count"))
            .reset_index()
            .sort_values("count", ascending=False)
        )
        summary_path = out_path / "summary_by_species.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print("\n=== Сводка по видам ===")
        print(summary.to_string(index=False))

    # Решение по опрыскиванию для каждого фото (экономический порог по шпаргалке)
    decision_path = out_path / "decision_report.csv"
    decision_df = pd.DataFrame(photo_reports)
    decision_df.to_csv(decision_path, index=False, encoding="utf-8-sig")
    print("\n=== Решение по опрыскиванию (по фото) ===")
    print(decision_df.to_string(index=False))

    print(f"\nГотово. Сохранено:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {decision_path}")
    print(f"  аннотированные фото -> {out_path / 'annotated'}")


def main():
    parser = argparse.ArgumentParser(description="Детекция и классификация сорняков на фото поля")
    parser.add_argument("--ref_dir", help="Путь к эталонным фото сорняков (по папкам-видам)")
    parser.add_argument("--field_dir", required=True, help="Путь к фотографиям полей")
    parser.add_argument("--out_dir", default="./results", help="Куда сохранить результаты")
    parser.add_argument(
        "--debug_mask",
        action="store_true",
        help=(
            "Режим быстрой отладки детекции (без CLIP и классификации). "
            "Сохраняет для каждого фото поля картинку 'исходник | маска до склейки | "
            "маска после склейки' в out_dir/debug_masks/. Используйте это, чтобы "
            "подобрать MERGE_KERNEL_SIZE и MIN_CONTOUR_AREA в начале файла под свои "
            "фото за секунды, без долгой загрузки CLIP и построения эталонов."
        ),
    )
    parser.add_argument(
        "--photo_area_m2",
        type=float,
        default=None,
        help=(
            "Реальная площадь земли (в кв.м), которую покрывает ОДНО фото поля. "
            "Нужна для перевода количества найденных сорняков в плотность "
            "(шт/м2) и принятия решения по опрыскиванию согласно "
            "экономическому порогу вредоносности. Без этого параметра "
            "решение по опрыскиванию не считается, выводятся только "
            "абсолютные количества."
        ),
    )
    args = parser.parse_args()

    if args.debug_mask:
        field_path = Path(args.field_dir)
        out_path = Path(args.out_dir) / "debug_masks"
        out_path.mkdir(parents=True, exist_ok=True)
        image_paths = sorted(
            p for p in field_path.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
        if not image_paths:
            raise ValueError(f"В {args.field_dir} не найдено изображений.")
        for img_path in image_paths:
            image_bgr = imread_unicode(str(img_path))
            if image_bgr is None:
                print(f"[WARN] Не удалось прочитать {img_path}, пропускаю.")
                continue
            out_img_path = out_path / img_path.name
            save_debug_mask(image_bgr, out_img_path)
            print(f"[OK] {img_path.name} -> {out_img_path}")
        print(f"\nГотово. Сравнение (исходник | маска до склейки | маска после склейки) в {out_path}")
        print(
            "Подберите MERGE_KERNEL_SIZE / MIN_CONTOUR_AREA в начале файла так, чтобы "
            "на 3-й картинке (после склейки) каждый сорняк был одним цельным белым "
            "пятном, а разные растения не сливались друг с другом. Затем запускайте "
            "обычный режим (без --debug_mask)."
        )
        return

    if not args.ref_dir:
        parser.error("--ref_dir обязателен в обычном режиме (без --debug_mask)")

    clf = ClipClassifier()

    print("\n=== Строю эталонные эмбеддинги ===")
    species_embeddings, stage_embeddings = build_reference_embeddings(args.ref_dir, clf)

    print("\n=== Обрабатываю фото полей ===")
    detections, photo_reports = process_field_photos(
        args.field_dir,
        args.out_dir,
        species_embeddings,
        stage_embeddings,
        clf,
        photo_area_m2=args.photo_area_m2,
    )

    save_results(detections, photo_reports, args.out_dir)


if __name__ == "__main__":
    main()
