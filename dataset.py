"""
================================================================================
Dataset: Cargador de datos para imágenes multi-iluminación
================================================================================

Este módulo proporciona clases para cargar datasets de imágenes capturadas
con múltiples luces.

Formato esperado de archivos:
    datasets/VAE/images/
        capture_001_light0.jpg  # Luz 0 (anillo/normal)
        capture_001_light1.jpg  # Luz 1 (tangencial 1)
        capture_001_light2.jpg  # Luz 2 (tangencial 2)
        capture_001_light3.jpg  # Luz 3 (tangencial 3)
        capture_001_light4.jpg  # Luz 4 (tangencial 4)
        capture_002_light0.jpg
        ...

Cada "muestra" consiste en 5 imágenes (una por cada luz) del mismo objeto/posición.

Autor: [Tu nombre]
Fecha: 2024
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Callable, List
import re


# ==============================================================================
# BLOQUE 1: Dataset para múltiples iluminaciones
# ==============================================================================
#
# Este dataset carga grupos de imágenes donde cada grupo tiene N imágenes
# correspondientes a N condiciones de iluminación diferentes.
#
# El nombre de archivo debe seguir un patrón que permita identificar:
#   1. El identificador de la muestra (ej: "capture_001")
#   2. El índice de la luz (ej: "light0", "light1", etc.)
#
# Puedes adaptar el patrón de nombres modificando el método _extract_prefix
# ==============================================================================

class MultiLightDataset(Dataset):
    """
    Dataset para imágenes capturadas con múltiples luces.

    Cada muestra contiene N imágenes (una por cada luz) que se apilan
    en un tensor de forma (N, H, W) para escala de grises o (N, H, W) si
    se convierten a un solo canal.
    """

    def __init__(
        self,
        root_dir: str,
        num_lights: int = 5,
        transform: Optional[Callable] = None,
        file_pattern: str = "light",
        image_extension: str = ".jpg"
    ):
        """
        Args:
            root_dir: directorio raíz con las imágenes
            num_lights: número de luces/canales (5 por defecto)
            transform: transformaciones a aplicar (ej: resize, crop)
            file_pattern: patrón para identificar el índice de luz (ej: "light", "_")
            image_extension: extensión de las imágenes (ej: ".jpg", ".png")
        """
        self.root_dir = root_dir
        self.num_lights = num_lights
        self.transform = transform
        self.file_pattern = file_pattern
        self.image_extension = image_extension

        # Encontrar todas las muestras únicas (prefijos)
        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No se encontraron muestras en {root_dir}")

        print(f"Dataset cargado: {len(self.samples)} muestras con {num_lights} luces cada una")

    def _find_samples(self) -> List[str]:
        """
        Encuentra todos los prefijos únicos de muestras en el directorio.

        Returns:
            Lista de prefijos únicos (ej: ["capture_001", "capture_002", ...])
        """
        if not os.path.exists(self.root_dir):
            raise ValueError(f"El directorio no existe: {self.root_dir}")

        files = os.listdir(self.root_dir)
        prefixes = set()

        for filename in files:
            if not filename.endswith(self.image_extension):
                continue

            # Extraer el prefijo (todo antes del patrón de luz)
            prefix = self._extract_prefix(filename)
            if prefix:
                prefixes.add(prefix)

        return sorted(list(prefixes))

    def _extract_prefix(self, filename: str) -> Optional[str]:
        """
        Extrae el prefijo de una muestra del nombre de archivo.

        Ejemplos:
            "capture_001_light0.jpg" → "capture_001"
            "sample_abc_light2.png" → "sample_abc"

        Modifica este método si tu formato de nombres es diferente.
        """
        # Patrón: todo antes de "_light" o similar
        pattern = f"(.+)_{self.file_pattern}\\d+{self.image_extension}$"
        match = re.match(pattern, filename)

        if match:
            return match.group(1)

        # Patrón alternativo: todo antes del último "_X" donde X es un número
        alt_pattern = f"(.+)_\\d+{self.image_extension}$"
        match = re.match(alt_pattern, filename)

        if match:
            return match.group(1)

        return None

    def _get_image_path(self, prefix: str, light_index: int) -> str:
        """
        Construye la ruta completa de una imagen.

        Args:
            prefix: prefijo de la muestra (ej: "capture_001")
            light_index: índice de la luz (0, 1, 2, 3, 4)

        Returns:
            Ruta completa al archivo de imagen
        """
        # Intenta varios formatos de nombre
        possible_names = [
            f"{prefix}_{self.file_pattern}{light_index}{self.image_extension}",
            f"{prefix}_light{light_index}{self.image_extension}",
            f"{prefix}_{light_index}{self.image_extension}",
        ]

        for name in possible_names:
            path = os.path.join(self.root_dir, name)
            if os.path.exists(path):
                return path

        # Si no encuentra ninguno, usa el primer formato y dejará un error claro
        return os.path.join(self.root_dir, possible_names[0])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Carga una muestra completa (todas las luces).

        Args:
            idx: índice de la muestra

        Returns:
            Tensor de forma (num_lights, H, W) con todas las imágenes
        """
        prefix = self.samples[idx]
        channels = []

        for light_idx in range(self.num_lights):
            # Construir ruta de la imagen
            img_path = self._get_image_path(prefix, light_idx)

            # Cargar imagen en escala de grises
            try:
                image = Image.open(img_path).convert('L')
            except FileNotFoundError:
                raise FileNotFoundError(f"No se encontró: {img_path}")

            # Convertir a tensor [0, 1]
            image = transforms.ToTensor()(image)  # (1, H, W)
            channels.append(image)

        # Apilar todos los canales: (num_lights, H, W)
        multi_channel = torch.cat(channels, dim=0)

        # Aplicar transformaciones si existen
        if self.transform:
            multi_channel = self.transform(multi_channel)

        return multi_channel


# ==============================================================================
# BLOQUE 2: Transformaciones de datos
# ==============================================================================
#
# Definimos transformaciones comunes para entrenamiento y validación.
# Las transformaciones de entrenamiento incluyen augmentación de datos
# para mejorar la generalización.
# ==============================================================================

def get_train_transforms(image_size: int = 512):
    """
    Transformaciones para entrenamiento (con augmentación).

    Args:
        image_size: tamaño de la imagen de salida (cuadrada)

    Returns:
        Composición de transformaciones
    """
    return transforms.Compose([
        # Recorte aleatorio con redimensionado
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.8, 1.0),     # Escala del recorte (80-100% del original)
            ratio=(0.9, 1.1)     # Relación de aspecto casi cuadrada
        ),
        # Rotación aleatoria pequeña
        transforms.RandomRotation(degrees=10),
        # Flip horizontal con 50% de probabilidad
        transforms.RandomHorizontalFlip(p=0.5),
    ])


def get_val_transforms(image_size: int = 512):
    """
    Transformaciones para validación (sin augmentación).

    Args:
        image_size: tamaño de la imagen de salida (cuadrada)

    Returns:
        Composición de transformaciones
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])


# ==============================================================================
# BLOQUE 3: Funciones auxiliares
# ==============================================================================

def create_dataloaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    num_lights: int = 5,
    batch_size: int = 4,
    image_size: int = 512,
    num_workers: int = 0
):
    """
    Crea DataLoaders para entrenamiento y validación.

    Args:
        train_dir: directorio con imágenes de entrenamiento
        val_dir: directorio con imágenes de validación (opcional)
        num_lights: número de luces
        batch_size: tamaño del batch
        image_size: tamaño de las imágenes
        num_workers: workers para carga paralela (0 para Windows)

    Returns:
        train_loader, val_loader (val_loader puede ser None)
    """
    # Dataset de entrenamiento
    train_dataset = MultiLightDataset(
        root_dir=train_dir,
        num_lights=num_lights,
        transform=get_train_transforms(image_size)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Dataset de validación (opcional)
    val_loader = None
    if val_dir and os.path.exists(val_dir):
        val_dataset = MultiLightDataset(
            root_dir=val_dir,
            num_lights=num_lights,
            transform=get_val_transforms(image_size)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


# ==============================================================================
# BLOQUE 4: Dataset compatible con el formato original (16 canales)
# ==============================================================================
#
# Este dataset es compatible con el formato del repositorio original
# para poder entrenar con sus datos y comparar resultados.
# ==============================================================================

class OriginalFormatDataset(Dataset):
    """
    Dataset compatible con el formato original del repositorio.

    El formato original usa:
        prefix_0.jpg, prefix_1.jpg, ..., prefix_15.jpg
    donde prefix tiene exactamente 23 caracteres.
    """

    def __init__(
        self,
        root_dir: str,
        num_channels: int = 16,
        transform: Optional[Callable] = None,
        prefix_length: int = 23
    ):
        """
        Args:
            root_dir: directorio con las imágenes
            num_channels: número de canales (16 para original, 5 para tu caso)
            transform: transformaciones a aplicar
            prefix_length: longitud del prefijo en el nombre de archivo
        """
        self.root_dir = root_dir
        self.num_channels = num_channels
        self.transform = transform
        self.prefix_length = prefix_length

        # Encontrar prefijos únicos
        self.prefixes = self._extract_prefixes()

        print(f"Dataset cargado: {len(self.prefixes)} muestras")

    def _extract_prefixes(self) -> List[str]:
        """Extrae prefijos únicos de los archivos."""
        files = os.listdir(self.root_dir)
        prefixes = set()

        for f in files:
            if f.endswith('.jpg') and len(f) > self.prefix_length:
                prefix = f[:self.prefix_length]
                prefixes.add(prefix)

        return sorted(list(prefixes))

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        prefix = self.prefixes[idx]
        channels = []

        for i in range(self.num_channels):
            filename = os.path.join(self.root_dir, f"{prefix}_{i}.jpg")
            image = Image.open(filename).convert('L')
            image = transforms.ToTensor()(image)
            channels.append(image)

        multi_channel = torch.cat(channels, dim=0)

        if self.transform:
            multi_channel = self.transform(multi_channel)

        return multi_channel


# ==============================================================================
# TEST: Verificar que el dataset funciona
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST DATASET")
    print("=" * 60)

    # Crear directorio de prueba con imágenes sintéticas
    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        # Crear imágenes de prueba
        for sample_idx in range(3):
            for light_idx in range(5):
                img = Image.fromarray(
                    np.random.randint(0, 255, (256, 256), dtype=np.uint8)
                )
                img.save(os.path.join(tmpdir, f"sample_{sample_idx:03d}_light{light_idx}.jpg"))

        # Probar el dataset
        dataset = MultiLightDataset(
            root_dir=tmpdir,
            num_lights=5,
            transform=get_train_transforms(128)
        )

        print(f"Número de muestras: {len(dataset)}")

        # Cargar una muestra
        sample = dataset[0]
        print(f"Forma de una muestra: {sample.shape}")
        print(f"Rango de valores: [{sample.min():.3f}, {sample.max():.3f}]")

        # Probar DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=True)
        batch = next(iter(loader))
        print(f"Forma de un batch: {batch.shape}")

    print("=" * 60)
    print("Test PASSED!")
