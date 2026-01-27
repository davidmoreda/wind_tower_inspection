"""
================================================================================
Entrenamiento: AE-TSCM + VQVAE para fusión de múltiples iluminaciones
================================================================================

Este script entrena el sistema completo:
    1. AE-TSCM: Fusiona 5 imágenes (luces) en 1 imagen RGB
    2. VQVAE: Reconstruye las 5 imágenes originales desde el RGB

Pipeline:
    5 luces (5, H, W) → AE-TSCM → RGB (3, H, W) → VQVAE → 5 luces reconstruidas

Pérdidas:
    - MSE: Error cuadrático medio entre original y reconstruido
    - SSIM: Similitud estructural (preserva bordes y texturas)
    - VQ: Pérdida de cuantización vectorial

Uso:
    python train.py --data_dir ./datasets/VAE/images --epochs 100

Autor: [Tu nombre]
Fecha: 2024
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Importar nuestros módulos
from models.ae_tscm import AE_TSCM
from models.vqvae import VQVAE
from dataset import MultiLightDataset, get_train_transforms, get_val_transforms

# Para SSIM loss (instalar con: pip install pytorch-msssim)
try:
    from pytorch_msssim import ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("Warning: pytorch-msssim no instalado. SSIM loss no disponible.")
    print("Instalar con: pip install pytorch-msssim")


# ==============================================================================
# BLOQUE 1: Configuración por defecto
# ==============================================================================

DEFAULT_CONFIG = {
    # Datos
    'num_lights': 5,           # Número de luces (canales de entrada)
    'image_size': 512,         # Tamaño de imagen (cuadrada)
    'batch_size': 4,           # Tamaño del batch

    # Modelo AE-TSCM
    'use_spatial_attention': True,  # Usar atención espacial

    # Modelo VQVAE
    'hidden_channels': 64,     # Canales ocultos en VQVAE
    'num_embeddings': 512,     # Tamaño del codebook
    'embedding_dim': 64,       # Dimensión de embeddings

    # Entrenamiento
    'epochs': 100,             # Número de épocas
    'learning_rate': 1e-3,     # Learning rate inicial
    'weight_decay': 1e-5,      # Regularización L2

    # Pesos de las pérdidas
    'lambda_mse': 1.0,         # Peso de MSE loss
    'lambda_ssim': 0.1,        # Peso de SSIM loss
    'lambda_vq': 0.05,         # Peso de VQ loss

    # Guardado
    'save_interval': 20,       # Guardar cada N épocas
    'log_interval': 10,        # Mostrar imágenes cada N épocas
}


# ==============================================================================
# BLOQUE 2: Funciones de pérdida
# ==============================================================================
#
# Usamos tres pérdidas complementarias:
#   - MSE: penaliza diferencias pixel a pixel
#   - SSIM: preserva estructura y bordes (importante para defectos)
#   - VQ: regulariza el espacio latente
# ==============================================================================

def compute_loss(original, reconstructed, vq_loss, config):
    """
    Calcula la pérdida total combinando MSE, SSIM y VQ.

    Args:
        original: imágenes originales (B, 5, H, W)
        reconstructed: imágenes reconstruidas (B, 5, H, W)
        vq_loss: pérdida de cuantización vectorial
        config: diccionario de configuración

    Returns:
        total_loss: pérdida total
        loss_dict: diccionario con cada componente de la pérdida
    """
    # MSE Loss: diferencia cuadrática media
    mse_loss = F.mse_loss(reconstructed, original)

    # SSIM Loss: similitud estructural (1 - ssim porque queremos minimizar)
    if SSIM_AVAILABLE:
        ssim_loss = 1 - ssim(reconstructed, original, data_range=1.0, size_average=True)
    else:
        ssim_loss = torch.tensor(0.0, device=original.device)

    # Pérdida total ponderada
    total_loss = (
        config['lambda_mse'] * mse_loss +
        config['lambda_ssim'] * ssim_loss +
        config['lambda_vq'] * vq_loss
    )

    loss_dict = {
        'total': total_loss.item(),
        'mse': mse_loss.item(),
        'ssim': ssim_loss.item() if SSIM_AVAILABLE else 0.0,
        'vq': vq_loss.item()
    }

    return total_loss, loss_dict


# ==============================================================================
# BLOQUE 3: Clase del modelo completo
# ==============================================================================
#
# Combina AE-TSCM y VQVAE en un solo módulo para facilitar el entrenamiento.
# ==============================================================================

class FullModel(nn.Module):
    """
    Modelo completo: AE-TSCM + VQVAE

    Pipeline:
        5 luces → AE-TSCM → RGB → VQVAE → 5 luces reconstruidas
    """

    def __init__(self, config):
        super().__init__()

        # AE-TSCM: 5 canales → 3 canales (RGB)
        self.ae_tscm = AE_TSCM(
            in_channels=config['num_lights'],
            out_channels=3,
            use_spatial_attention=config['use_spatial_attention']
        )

        # VQVAE: 3 canales → 5 canales (reconstrucción)
        self.vqvae = VQVAE(
            in_channels=3,
            out_channels=config['num_lights'],
            hidden_channels=config['hidden_channels'],
            num_embeddings=config['num_embeddings'],
            embedding_dim=config['embedding_dim']
        )

    def forward(self, x, return_intermediate=False):
        """
        Args:
            x: 5 imágenes de entrada (B, 5, H, W)
            return_intermediate: si devolver valores intermedios para visualización

        Returns:
            reconstructed: 5 imágenes reconstruidas (B, 5, H, W)
            vq_loss: pérdida de cuantización
            perplexity: perplexity del codebook
            (opcional) rgb: imagen RGB intermedia
            (opcional) attention: mapas de atención
        """
        # Paso 1: AE-TSCM fusiona las 5 luces en RGB
        if return_intermediate:
            rgb, ch_attention, sp_attention = self.ae_tscm(x, return_attention=True)
        else:
            rgb = self.ae_tscm(x, return_attention=False)

        # Paso 2: VQVAE reconstruye las 5 luces desde RGB
        reconstructed, vq_loss, perplexity = self.vqvae(rgb)

        if return_intermediate:
            return reconstructed, vq_loss, perplexity, rgb, ch_attention, sp_attention

        return reconstructed, vq_loss, perplexity


# ==============================================================================
# BLOQUE 4: Funciones de visualización
# ==============================================================================

def save_visualization(original, reconstructed, rgb, epoch, save_dir):
    """
    Guarda una visualización comparando original vs reconstruido.

    Args:
        original: tensor (B, 5, H, W)
        reconstructed: tensor (B, 5, H, W)
        rgb: tensor (B, 3, H, W) imagen RGB intermedia
        epoch: número de época
        save_dir: directorio donde guardar
    """
    # Tomar solo el primer elemento del batch
    original = original[0].detach().cpu()
    reconstructed = reconstructed[0].detach().cpu()
    rgb = rgb[0].detach().cpu()

    num_lights = original.shape[0]

    # Crear figura con 3 filas: original, reconstruido, RGB
    fig, axes = plt.subplots(3, max(num_lights, 3), figsize=(3 * num_lights, 9))

    # Fila 1: Original
    for i in range(num_lights):
        axes[0, i].imshow(original[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original L{i}')
        axes[0, i].axis('off')

    # Fila 2: Reconstruido
    for i in range(num_lights):
        axes[1, i].imshow(reconstructed[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Recon L{i}')
        axes[1, i].axis('off')

    # Fila 3: RGB fusionado (solo 3 canales)
    rgb_img = rgb.permute(1, 2, 0).numpy()  # (H, W, 3)
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    axes[2, 0].imshow(rgb_img)
    axes[2, 0].set_title('RGB Fusionado')
    axes[2, 0].axis('off')

    # Ocultar axes vacíos
    for i in range(1, num_lights):
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:04d}.png'), dpi=100)
    plt.close()


def plot_losses(history, save_dir):
    """
    Guarda gráfico de las pérdidas durante el entrenamiento.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs = range(1, len(history['total']) + 1)

    axes[0, 0].plot(epochs, history['total'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')

    axes[0, 1].plot(epochs, history['mse'])
    axes[0, 1].set_title('MSE Loss')
    axes[0, 1].set_xlabel('Epoch')

    axes[1, 0].plot(epochs, history['ssim'])
    axes[1, 0].set_title('SSIM Loss')
    axes[1, 0].set_xlabel('Epoch')

    axes[1, 1].plot(epochs, history['vq'])
    axes[1, 1].set_title('VQ Loss')
    axes[1, 1].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_history.png'), dpi=150)
    plt.close()


# ==============================================================================
# BLOQUE 5: Loop de entrenamiento
# ==============================================================================

def train_epoch(model, dataloader, optimizer, config, device):
    """
    Entrena una época completa.

    Returns:
        dict con las pérdidas promedio
    """
    model.train()
    total_losses = {'total': 0, 'mse': 0, 'ssim': 0, 'vq': 0}
    num_batches = 0

    progress_bar = tqdm(dataloader, desc='Training')

    for batch in progress_bar:
        batch = batch.to(device)

        # Forward pass
        reconstructed, vq_loss, perplexity = model(batch)

        # Calcular pérdida
        loss, loss_dict = compute_loss(batch, reconstructed, vq_loss, config)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Acumular pérdidas
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        num_batches += 1

        # Actualizar barra de progreso
        progress_bar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'mse': f"{loss_dict['mse']:.4f}",
            'perp': f"{perplexity.item():.1f}"
        })

    # Promediar pérdidas
    for key in total_losses:
        total_losses[key] /= num_batches

    return total_losses


# ==============================================================================
# BLOQUE 6: Función principal de entrenamiento
# ==============================================================================

def train(config):
    """
    Función principal de entrenamiento.

    Args:
        config: diccionario de configuración
    """
    # Crear directorio de salida
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('runs', f'experiment_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)

    print(f"Guardando resultados en: {save_dir}")

    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Dataset y DataLoader
    print(f"Cargando datos de: {config['data_dir']}")

    train_dataset = MultiLightDataset(
        root_dir=config['data_dir'],
        num_lights=config['num_lights'],
        transform=get_train_transforms(config['image_size'])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # 0 para evitar problemas en Windows
        pin_memory=True
    )

    # Modelo
    print("Creando modelo...")
    model = FullModel(config).to(device)

    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")

    # Optimizador
    optimizer = Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Scheduler (reduce LR gradualmente)
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Historial de pérdidas
    history = {'total': [], 'mse': [], 'ssim': [], 'vq': []}

    # Guardar configuración
    with open(os.path.join(save_dir, 'config.txt'), 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    # Loop de entrenamiento
    print(f"\nIniciando entrenamiento por {config['epochs']} épocas...")
    print("=" * 60)

    best_loss = float('inf')

    for epoch in range(1, config['epochs'] + 1):
        print(f"\nÉpoca {epoch}/{config['epochs']}")

        # Entrenar una época
        epoch_losses = train_epoch(model, train_loader, optimizer, config, device)

        # Actualizar scheduler
        scheduler.step()

        # Guardar historial
        for key in history:
            history[key].append(epoch_losses[key])

        # Mostrar resumen
        print(f"  Loss: {epoch_losses['total']:.4f} | "
              f"MSE: {epoch_losses['mse']:.4f} | "
              f"SSIM: {epoch_losses['ssim']:.4f} | "
              f"VQ: {epoch_losses['vq']:.4f}")

        # Guardar visualización
        if epoch % config['log_interval'] == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_loader)).to(device)
                recon, _, _, rgb, _, _ = model(sample_batch, return_intermediate=True)
                save_visualization(
                    sample_batch, recon, rgb, epoch,
                    os.path.join(save_dir, 'visualizations')
                )
            model.train()

        # Guardar checkpoint
        if epoch % config['save_interval'] == 0:
            checkpoint_path = os.path.join(
                save_dir, 'checkpoints', f'checkpoint_epoch_{epoch:04d}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses['total'],
                'config': config
            }, checkpoint_path)
            print(f"  Checkpoint guardado: {checkpoint_path}")

        # Guardar mejor modelo
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

    # Guardar modelo final
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))

    # Guardar gráfico de pérdidas
    plot_losses(history, save_dir)

    print("\n" + "=" * 60)
    print("Entrenamiento completado!")
    print(f"Mejor loss: {best_loss:.4f}")
    print(f"Resultados guardados en: {save_dir}")


# ==============================================================================
# BLOQUE 7: Punto de entrada
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar AE-TSCM + VQVAE')

    # Argumentos principales
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directorio con las imágenes de entrenamiento')
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                        help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'],
                        help='Tamaño del batch')
    parser.add_argument('--image_size', type=int, default=DEFAULT_CONFIG['image_size'],
                        help='Tamaño de imagen')
    parser.add_argument('--num_lights', type=int, default=DEFAULT_CONFIG['num_lights'],
                        help='Número de luces/canales')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'],
                        help='Learning rate')

    args = parser.parse_args()

    # Crear configuración
    config = DEFAULT_CONFIG.copy()
    config['data_dir'] = args.data_dir
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['image_size'] = args.image_size
    config['num_lights'] = args.num_lights
    config['learning_rate'] = args.lr

    # Entrenar
    train(config)
