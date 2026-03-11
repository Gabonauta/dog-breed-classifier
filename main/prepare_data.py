import os
import random
import shutil
from pathlib import Path

import kagglehub

SEED = 42
random.seed(SEED)

# Configuración
OUTPUT_DIR = Path("data")
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Pon None para usar todas las razas
MAX_BREEDS = 10

# Imágenes máximas por raza para balancear y acelerar pruebas
# Pon None si quieres usar todas las imágenes disponibles por raza
MAX_IMAGES_PER_BREED = 250


def find_image_root(dataset_path: Path) -> Path:
    """
    Busca automáticamente la carpeta que contiene subcarpetas por raza con imágenes dentro.
    """
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}

    candidates = []
    for root, dirs, files in os.walk(dataset_path):
        root_path = Path(root)

        # Si este directorio tiene subdirectorios y dentro de ellos hay imágenes, puede ser el root correcto
        if dirs:
            score = 0
            for d in dirs[:10]:
                subdir = root_path / d
                if subdir.is_dir():
                    img_count = sum(
                        1 for f in subdir.iterdir()
                        if f.is_file() and f.suffix.lower() in image_exts
                    )
                    if img_count > 0:
                        score += 1
            if score > 0:
                candidates.append((score, root_path))

    if not candidates:
        raise FileNotFoundError("No se encontró una carpeta raíz válida con imágenes por clase.")

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}


def safe_clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def split_list(items, train_ratio=0.7, val_ratio=0.15):
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    return train_items, val_items, test_items


def copy_files(files, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        shutil.copy2(file_path, dst_dir / file_path.name)


def main():
    print("Descargando dataset desde KaggleHub...")
    dataset_path = Path(kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset"))
    print(f"Dataset descargado en: {dataset_path}")

    print("Buscando carpeta raíz de imágenes...")
    image_root = find_image_root(dataset_path)
    print(f"Carpeta detectada: {image_root}")

    breed_dirs = [d for d in image_root.iterdir() if d.is_dir()]
    breed_dirs = sorted(breed_dirs, key=lambda x: x.name.lower())

    if MAX_BREEDS is not None:
        breed_dirs = breed_dirs[:MAX_BREEDS]

    print(f"Razas seleccionadas: {len(breed_dirs)}")

    # Reiniciar estructura de salida
    for split in ["train", "val", "test"]:
        safe_clear_dir(OUTPUT_DIR / split)

    summary = []

    for breed_dir in breed_dirs:
        breed_name = breed_dir.name.replace(" ", "_")
        images = [p for p in breed_dir.iterdir() if p.is_file() and is_image_file(p)]

        if len(images) < 10:
            print(f"Saltando {breed_name}: muy pocas imágenes ({len(images)})")
            continue

        if MAX_IMAGES_PER_BREED is not None:
            images = images[:MAX_IMAGES_PER_BREED]

        train_imgs, val_imgs, test_imgs = split_list(
            images,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO
        )

        copy_files(train_imgs, OUTPUT_DIR / "train" / breed_name)
        copy_files(val_imgs, OUTPUT_DIR / "val" / breed_name)
        copy_files(test_imgs, OUTPUT_DIR / "test" / breed_name)

        summary.append({
            "breed": breed_name,
            "total": len(images),
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
        })

    print("\nResumen:")
    for row in summary:
        print(
            f"{row['breed']}: total={row['total']}, "
            f"train={row['train']}, val={row['val']}, test={row['test']}"
        )

    print("\nDataset preparado en ./data")


if __name__ == "__main__":
    main()