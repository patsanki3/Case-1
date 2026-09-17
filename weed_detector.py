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

Если внутри есть еще и подпапки по стадиям вегетации, например:
    /data/Сорняки/Осот/всходы/*.jpg
    /data/Сорняки/Осот/розетка/*.jpg
скрипт это тоже подхватит (см. build_reference_embeddings).
"""

import os
import json
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

# Минимальная площадь контура (в пикселях), чтобы считать его отдельным растением,
# а не шумом/мелким мусором. Подбирается под разрешение ваших фото.
MIN_CONTOUR_AREA = 800

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
    stage: str
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int


# ---------------------------------------------------------------------------
# Шаг 1. Детекция растительности (OpenCV)
# ---------------------------------------------------------------------------

def detect_plant_regions(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Простая, но рабочая детекция растительности по цвету (HSV) + контурам.
    Возвращает список bbox (x, y, w, h).

    Идея: сорняки и культурные растения - зеленые объекты на фоне почвы/стерни
    (коричневый/желтый). Маска по зеленому + морфология для склейки листьев
    одного растения + поиск внешних контуров.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Диапазон зеленого. При необходимости подправьте под освещение ваших фото.
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Морфология: убираем шум, склеиваем листья одного растения
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))

    return boxes


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


def build_reference_embeddings(
    ref_dir: str, clf: ClipClassifier
) -> Dict[str, np.ndarray]:
    """
    Проходит по /data/Сорняки/<Вид>/*.jpg (и рекурсивно глубже, если есть
    подпапки по стадиям) и строит усредненный эмбеддинг для каждого класса
    первого уровня (вида сорняка).

    Возвращает {species_name: mean_embedding}.
    """
    ref_path = Path(ref_dir)
    species_embeddings: Dict[str, np.ndarray] = {}

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
        print(f"[REF] {species_dir.name}: {len(images)} эталонных фото")

    return species_embeddings


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
      - присутствуют яркие/светлые
        пятна (цветки)                -> "цветение"

    Это заготовка для демонстрации логики. Если в эталонах есть фото,
    размеченные по стадиям (подпапки), лучше заменить на такой же
    similarity-подход, как для вида (см. build_reference_embeddings).
    """
    h, w = crop_bgr.shape[:2]
    area = h * w

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    # Доля "цветочных" ярких/светлых тонов (не зеленых) — грубый признак цветения
    bright_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
    bright_ratio = cv2.countNonZero(bright_mask) / area

    if bright_ratio > 0.05:
        return "цветение"
    if area < 4000:
        return "всходы"
    if area < 15000:
        return "розетка"
    return "вегетативный рост"


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def process_field_photos(
    field_dir: str,
    out_dir: str,
    species_embeddings: Dict[str, np.ndarray],
    clf: ClipClassifier,
) -> List[WeedDetection]:
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
    weed_id_counter = 0

    for img_path in image_paths:
        image_bgr = imread_unicode(str(img_path))
        if image_bgr is None:
            print(f"[WARN] Не удалось прочитать {img_path}, пропускаю.")
            continue

        boxes = detect_plant_regions(image_bgr)
        annotated = image_bgr.copy()

        for (x, y, w, h) in boxes:
            crop_bgr = image_bgr[y : y + h, x : x + w]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            emb = clf.embed_image(crop_pil)
            species, sim = classify_crop(emb, species_embeddings)

            if species == "unknown":
                # Не уверены, что это сорняк из нашего списка эталонов —
                # можно пропускать или помечать отдельным цветом.
                color = (0, 165, 255)  # оранжевый - "неопознано"
            else:
                color = (0, 0, 255)  # красный - опознанный сорняк

            stage = estimate_stage(crop_bgr)

            weed_id_counter += 1
            all_detections.append(
                WeedDetection(
                    photo=img_path.name,
                    weed_id=weed_id_counter,
                    species=species,
                    species_confidence=round(sim, 3),
                    stage=stage,
                    bbox_x=x,
                    bbox_y=y,
                    bbox_w=w,
                    bbox_h=h,
                )
            )

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            label = f"#{weed_id_counter} {species} ({stage})"
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
        print(f"[OK] {img_path.name}: найдено {len(boxes)} объект(ов)")

    return all_detections


def save_results(detections: List[WeedDetection], out_dir: str):
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
            df.groupby("species")
            .agg(count=("weed_id", "count"))
            .reset_index()
            .sort_values("count", ascending=False)
        )
        summary_path = out_path / "summary_by_species.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print("\n=== Сводка по видам ===")
        print(summary.to_string(index=False))

    print(f"\nГотово. Сохранено:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  аннотированные фото -> {out_path / 'annotated'}")


def main():
    parser = argparse.ArgumentParser(description="Детекция и классификация сорняков на фото поля")
    parser.add_argument("--ref_dir", required=True, help="Путь к эталонным фото сорняков (по папкам-видам)")
    parser.add_argument("--field_dir", required=True, help="Путь к фотографиям полей")
    parser.add_argument("--out_dir", default="./results", help="Куда сохранить результаты")
    args = parser.parse_args()

    clf = ClipClassifier()

    print("\n=== Строю эталонные эмбеддинги ===")
    species_embeddings = build_reference_embeddings(args.ref_dir, clf)

    print("\n=== Обрабатываю фото полей ===")
    detections = process_field_photos(args.field_dir, args.out_dir, species_embeddings, clf)

    save_results(detections, args.out_dir)


if __name__ == "__main__":
    main()
