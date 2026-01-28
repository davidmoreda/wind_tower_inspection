#!/usr/bin/env python3
"""
AE-TSCM + VQVAE v7.0 - Fix Codebook Training (Perplexity=0)

Pipeline de fusión multi-iluminación para detección de defectos superficiales.

FIXES CRÍTICOS v7.0 (v6.x producía NaN en todos los batches):
1. Resolución reducida a 1280×720 - mantiene aspect ratio 16:9
2. Codebook init N(0, 1) - matching exacto con BatchNorm del encoder
3. 512 codebook entries - vs 128 anterior, más capacidad
4. 3 downsamplings (8×) - vs 4 downsamplings (16×) anterior
5. Sin Sigmoid en decoder - alineado con paper
6. Output clamping [0,1] - en compute_loss
7. FP32 forzado en VQ - para operaciones EMA
8. Gradient clipping - max_norm=1.0

Configuración:
- Resolución: 1280×720 (16:9, desde 5472×3072 original)
- Encoder/Decoder: 3 etapas downsampling/upsampling (8× factor)
- Codebook: 512 entradas × 128d
- Latent map: 160×90 = 14,400 posiciones (cabe en FP16)
"""

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from pytorch_msssim import ssim
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo: {device}')
if device.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_DIR = 'dataset/dataset/MT001'

CONFIG = {
    # Datos
    'data_dir': DATASET_DIR,
    'num_lights': 5,              # SUP_IZQ, SUP_DER, INF_DER, INF_IZQ, ALL
    'target_resolution': (1280, 720),  # FIX: Reducir de 5472×3072, mantiene aspect ratio 16:9
    'batch_size': 2,              # FIX: Aumentado a 2 con resolución reducida
    'gradient_accumulation_steps': 2,  # Simula batch efectivo de 4

    # Modelo AE-TSCM
    'use_spatial_attention': True,

    # Modelo VQVAE — ajustado para 8× downsampling
    'hidden_channels': 128,       # Mantenido en 128 para capacidad
    'num_residual': 4,            # 4 bloques residuales
    'num_embeddings': 512,        # FIX: 128→512 (como paper) más capacidad
    'embedding_dim': 128,         # Mantenido en 128 para latent expresivo
    'num_downsamplings': 3,       # FIX: 4→3 etapas (8× vs 16×)

    # Codebook EMA
    'ema_decay': 0.99,            # Factor de decaimiento para EMA
    'commitment_cost': 0.25,      # Peso del commitment loss
    'codebook_reset_interval': 10,  # Reset entradas no usadas cada N épocas
    'codebook_usage_threshold': 1,  # Mínimo de usos para no ser reseteada

    # Entrenamiento
    'epochs': 150,
    'learning_rate': 1e-3,        # LR constante, sin scheduler
    'weight_decay': 1e-5,

    # Pesos de pérdidas
    'lambda_mse': 1.0,
    'lambda_ssim': 0.05,
    'lambda_vq': 0.02,

    # Guardado
    'save_interval': 25,
    'log_interval': 10,
}

# Directorio de salida
SAVE_DIR = 'runs/ae_tscm_v7_' + datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'visualizations'), exist_ok=True)
print(f'Resultados se guardarán en: {SAVE_DIR}')

# ============================================================
# DATASET
# ============================================================

LIGHT_NAMES = ['SUP_IZQ', 'SUP_DER', 'INF_DER', 'INF_IZQ', 'ALL']


class CropToDivisible:
    """Recorta la imagen para que H y W sean divisibles por un factor."""
    def __init__(self, factor=8):  # FIX: Cambiado de 16 a 8
        self.factor = factor

    def __call__(self, img):
        _, h, w = img.shape
        new_h = (h // self.factor) * self.factor
        new_w = (w // self.factor) * self.factor
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return img[:, top:top + new_h, left:left + new_w]


class MT001Dataset(Dataset):
    """
    Dataset para imágenes MT001 con 5 iluminaciones.
    Agrupa por num_captura y devuelve (5, H, W) en escala de grises.
    FIX: Ahora redimensiona a target_resolution manteniendo aspect ratio.
    """

    def __init__(self, root_dir, light_names=LIGHT_NAMES, transform=None, 
                 captures=None, target_resolution=None):
        self.root_dir = root_dir
        self.light_names = light_names
        self.num_lights = len(light_names)
        self.transform = transform
        self.target_resolution = target_resolution
        self.groups = self._parse_metadata(captures)
        print(f'  Dataset: {len(self.groups)} capturas, {self.num_lights} luces cada una')
        if target_resolution:
            print(f'  Resolución objetivo: {target_resolution[0]}×{target_resolution[1]}')

    def _parse_metadata(self, captures):
        metadata_path = os.path.join(self.root_dir, 'metadata.csv')
        groups = defaultdict(dict)

        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cap = int(row['num_captura'])
                luz = row['luz']
                if luz in self.light_names:
                    groups[cap][luz] = row['imagen']

        valid = []
        for cap in sorted(groups.keys()):
            if captures is not None and cap not in captures:
                continue
            if len(groups[cap]) == self.num_lights:
                valid.append((cap, groups[cap]))

        return valid

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        cap_num, filenames = self.groups[idx]
        channels = []

        for light_name in self.light_names:
            img_path = os.path.join(self.root_dir, filenames[light_name])
            image = Image.open(img_path).convert('L')  # Escala de grises
            
            # FIX: Resize a target_resolution si está definido
            if self.target_resolution is not None:
                image = image.resize(
                    (self.target_resolution[0], self.target_resolution[1]), 
                    Image.BILINEAR
                )
            
            image = transforms.ToTensor()(image)  # (1, H, W), rango [0,1]
            channels.append(image)

        multi_channel = torch.cat(channels, dim=0)  # (5, H, W)

        if self.transform:
            multi_channel = self.transform(multi_channel)

        return multi_channel


def get_train_transforms():
    """Transforms para entrenamiento con resolución reducida."""
    return transforms.Compose([
        CropToDivisible(8),  # FIX: factor=8 para 3 downsamplings
        transforms.RandomHorizontalFlip(p=0.5),
    ])


def get_val_transforms():
    """Transforms para validación con resolución reducida."""
    return transforms.Compose([
        CropToDivisible(8),  # FIX: factor=8 para 3 downsamplings
    ])


# ============================================================
# MODELOS - AE-TSCM
# ============================================================

class ChannelAttention(nn.Module):
    """SE-Net: aprende qué luces son más importantes."""
    def __init__(self, num_channels, reduction=2):
        super().__init__()
        hidden = max(num_channels // reduction, 4)
        self.attention = nn.Sequential(
            nn.Linear(num_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        gap = x.mean(dim=[2, 3])
        weights = self.attention(gap)
        return weights.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """CBAM: aprende qué regiones espaciales son más importantes."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        return self.conv(torch.cat([avg_out, max_out], dim=1))


class TaylorTransform(nn.Module):
    """Transformación no lineal de intensidad con coeficientes aprendibles."""
    def __init__(self, num_coefficients=5):
        super().__init__()
        initial = torch.zeros(num_coefficients)
        initial[1] = 1.0  # f(x) = x inicialmente
        self.coefficients = nn.Parameter(initial)

    def forward(self, x):
        c = self.coefficients
        return (c[0] + c[1]*x + c[2]*(x**2)/2 + c[3]*(x**3)/6 + c[4]*(x**4)/24)


class ChannelMixer(nn.Module):
    """Mezcla N canales a M canales con conv 1x1."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.mixer(x)


class ChannelNormalize(nn.Module):
    """Normaliza cada canal a [0, 1]."""
    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        mn = x_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
        mx = x_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
        # FIX: epsilon 1e-4 en vez de 1e-8 (1e-8 es cero en FP16)
        return (x - mn) / (mx - mn + 1e-4)


class AE_TSCM(nn.Module):
    """5 luces → 3 canales RGB."""
    def __init__(self, in_channels=5, out_channels=3, use_spatial_attention=True):
        super().__init__()
        self.use_spatial_attention = use_spatial_attention
        self.taylor = TaylorTransform(5)
        self.channel_attention = ChannelAttention(in_channels)
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(7)
        self.mixer = ChannelMixer(in_channels, out_channels)
        self.normalize = ChannelNormalize()

    def forward(self, x, return_attention=False):
        x = self.taylor(x)
        ch_w = self.channel_attention(x)
        x = x * ch_w
        sp_w = None
        if self.use_spatial_attention:
            sp_w = self.spatial_attention(x)
            x = x * sp_w
        x = self.mixer(x)
        x = self.normalize(x)
        if return_attention:
            return x, ch_w.squeeze(), sp_w
        return x


# ============================================================
# MODELOS - VQVAE con 3 downsamplings (8× factor)
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    """
    FIX: 3 etapas de downsampling (8×) para 1280×720 → 160×90 latent map.
    BatchNorm al final para normalizar salida antes del VQ.
    """
    def __init__(self, in_channels=3, hidden_channels=128, embedding_dim=128, num_residual=4):
        super().__init__()
        hc = hidden_channels
        # FIX: Solo 3 downsamplings (vs 4 anterior)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hc, 4, stride=2, padding=1), nn.ReLU())       # /2
        self.conv2 = nn.Sequential(
            nn.Conv2d(hc, hc, 4, stride=2, padding=1), nn.ReLU())                # /4
        self.conv3 = nn.Sequential(
            nn.Conv2d(hc, hc * 2, 4, stride=2, padding=1), nn.ReLU())            # /8
        # Ya no hay conv4
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hc * 2) for _ in range(num_residual)])
        # FIX: BatchNorm normaliza la salida del encoder (~N(0,1))
        self.to_latent = nn.Sequential(
            nn.Conv2d(hc * 2, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.residual_blocks(x)
        return self.to_latent(x)


class VectorQuantizerEMA(nn.Module):
    """
    VQ con actualización EMA del codebook.

    FIX CRITICO: Todas las operaciones internas se fuerzan a FP32.
    FIX: Codebook init N(0, 1) para matching con BatchNorm del encoder.
    """
    def __init__(self, num_embeddings=512, embedding_dim=128,
                 commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # FIX: Codebook — Normal(0, 1) para matching con BatchNorm del encoder
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.normal_(0, 1.0)  # FIX: std=1.0 (antes era 0.5)
        self.codebook.weight.requires_grad = False  # EMA actualiza, no gradientes

        # Buffers EMA (no son parámetros del optimizador)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_dw', self.codebook.weight.data.clone())
        self.register_buffer('usage_count', torch.zeros(num_embeddings))

    def forward(self, z):
        # FIX CRITICO: Deshabilitar autocast y forzar FP32
        with torch.amp.autocast('cuda', enabled=False):
            z = z.float()

            z = z.permute(0, 2, 3, 1).contiguous()
            z_shape = z.shape
            z_flat = z.view(-1, self.embedding_dim)

            # Distancias al codebook (FP32 seguro)
            distances = (
                torch.sum(z_flat**2, dim=1, keepdim=True) +
                torch.sum(self.codebook.weight**2, dim=1) -
                2 * torch.matmul(z_flat, self.codebook.weight.t())
            )

            encoding_indices = torch.argmin(distances, dim=1)
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            quantized = self.codebook(encoding_indices).view(z_shape)

            # EMA update del codebook (solo en training, todo en FP32)
            if self.training:
                encodings_sum = encodings.sum(0)
                dw = encodings.t() @ z_flat  # FP32: seguro para >65k posiciones

                self.ema_cluster_size.data.mul_(self.decay).add_(
                    encodings_sum, alpha=1 - self.decay)
                self.ema_dw.data.mul_(self.decay).add_(
                    dw, alpha=1 - self.decay)

                # Laplace smoothing para evitar divisiones por cero
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon) /
                    (n + self.num_embeddings * self.epsilon) * n
                )

                self.codebook.weight.data.copy_(
                    self.ema_dw / cluster_size.unsqueeze(1))

                # Acumular uso para reset
                self.usage_count.add_(encodings_sum)

            # Solo commitment loss (codebook se actualiza por EMA)
            commitment_loss = F.mse_loss(z, quantized.detach())
            loss = self.commitment_cost * commitment_loss

            # Straight-through estimator
            quantized = z + (quantized - z).detach()

            # Perplexity (mide utilización del codebook)
            avg_probs = encodings.mean(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            return quantized, loss, perplexity

    def reset_unused_codes(self, z_flat_samples, threshold=2):
        """
        Reemplaza entradas del codebook con uso < threshold
        por vectores aleatorios del encoder (z_flat_samples).
        """
        z_flat_samples = z_flat_samples.float()  # FIX: asegurar FP32
        unused_mask = self.usage_count < threshold
        num_unused = unused_mask.sum().item()

        if num_unused > 0 and z_flat_samples.shape[0] > 0:
            n_replace = min(int(num_unused), z_flat_samples.shape[0])
            perm = torch.randperm(z_flat_samples.shape[0], device=z_flat_samples.device)
            replace_vectors = z_flat_samples[perm[:n_replace]].detach()

            unused_indices = torch.where(unused_mask)[0][:n_replace]
            self.codebook.weight.data[unused_indices] = replace_vectors
            self.ema_dw.data[unused_indices] = replace_vectors
            self.ema_cluster_size.data[unused_indices] = 1.0

        # Reset contador de uso para el siguiente periodo
        self.usage_count.zero_()
        return num_unused


class Decoder(nn.Module):
    """
    FIX: 3 etapas de upsampling (8×) para simetría con encoder.
    FIX: SIN Sigmoid final (alineado con paper).
    """
    def __init__(self, out_channels=5, hidden_channels=128, embedding_dim=128, num_residual=4):
        super().__init__()
        hc = hidden_channels
        self.from_latent = nn.Conv2d(embedding_dim, hc * 2, 1)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hc * 2) for _ in range(num_residual)])
        # FIX: Solo 3 upsamplings (vs 4 anterior)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(hc * 2, hc * 2, 4, stride=2, padding=1), nn.ReLU())   # x2
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(hc * 2, hc, 4, stride=2, padding=1), nn.ReLU())       # x4
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(hc, out_channels, 4, stride=2, padding=1))            # x8
            # FIX: SIN Sigmoid - alineado con paper

    def forward(self, z):
        x = self.from_latent(z)
        x = self.residual_blocks(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return self.deconv3(x)


class VQVAE(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, hidden_channels=128,
                 num_residual=4, num_embeddings=512, embedding_dim=128,
                 commitment_cost=0.25, ema_decay=0.99):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim, num_residual)
        self.vq = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, ema_decay)
        self.decoder = Decoder(out_channels, hidden_channels, embedding_dim, num_residual)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, perplexity = self.vq(z)
        return self.decoder(z_q), vq_loss, perplexity

    def encode(self, x):
        z = self.encoder(x)
        z_q, _, _ = self.vq(z)
        return z_q


class FullModel(nn.Module):
    """5 luces → AE-TSCM → RGB → VQVAE → 5 luces reconstruidas"""

    def __init__(self, config):
        super().__init__()
        self.ae_tscm = AE_TSCM(
            in_channels=config['num_lights'],
            out_channels=3,
            use_spatial_attention=config['use_spatial_attention']
        )
        self.vqvae = VQVAE(
            in_channels=3,
            out_channels=config['num_lights'],
            hidden_channels=config['hidden_channels'],
            num_residual=config['num_residual'],
            num_embeddings=config['num_embeddings'],
            embedding_dim=config['embedding_dim'],
            commitment_cost=config['commitment_cost'],
            ema_decay=config['ema_decay']
        )

    def forward(self, x, return_intermediate=False):
        if return_intermediate:
            rgb, ch_att, sp_att = self.ae_tscm(x, return_attention=True)
        else:
            rgb = self.ae_tscm(x, return_attention=False)
        reconstructed, vq_loss, perplexity = self.vqvae(rgb)
        if return_intermediate:
            return reconstructed, vq_loss, perplexity, rgb, ch_att, sp_att
        return reconstructed, vq_loss, perplexity


# ============================================================
# FUNCIONES DE PÉRDIDA Y VISUALIZACIÓN
# ============================================================

def compute_loss(original, reconstructed, vq_loss, config):
    """
    FIX: SSIM removido de backprop (solo métrica), MSE + VQ para gradientes.
    """
    # Para métricas: clamp
    recon_clamped = reconstructed.clamp(0, 1)
    
    # MSE para backprop (sin clamp)
    mse_loss = F.mse_loss(reconstructed, original)
    
    # SSIM solo como métrica (NO gradientes)
    with torch.no_grad():
        ssim_val = ssim(recon_clamped, original, data_range=1.0, size_average=True)
        ssim_loss = 1 - ssim_val
    
    # Total: solo MSE + VQ contribuyen gradientes
    total = (config['lambda_mse'] * mse_loss +
             config['lambda_vq'] * vq_loss)
    
    return total, {
        'total': total.item(),
        'mse': mse_loss.item(),
        'ssim': ssim_loss.item(),
        'vq': vq_loss.item()
    }



def save_visualization(original, reconstructed, rgb, epoch, save_dir):
    original = original[0].detach().cpu()
    reconstructed = reconstructed[0].detach().cpu()
    rgb = rgb[0].detach().cpu()
    num_lights = original.shape[0]

    fig, axes = plt.subplots(3, max(num_lights, 3), figsize=(3*num_lights, 9))

    for i in range(num_lights):
        axes[0, i].imshow(original[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Orig {LIGHT_NAMES[i]}', fontsize=8)
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructed[i].clamp(0, 1), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Recon {LIGHT_NAMES[i]}', fontsize=8)
        axes[1, i].axis('off')

    rgb_img = rgb.permute(1, 2, 0).numpy()
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    axes[2, 0].imshow(rgb_img)
    axes[2, 0].set_title('RGB Fusionado')
    axes[2, 0].axis('off')
    for i in range(1, max(num_lights, 3)):
        axes[2, i].axis('off')

    plt.suptitle(f'Época {epoch}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch:04d}.png'), dpi=100)
    plt.close()


def plot_losses(history, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    epochs = range(1, len(history['total']) + 1)

    for ax, key, title in zip(
        axes.flat[:4],
        ['total', 'mse', 'ssim', 'vq'],
        ['Total Loss', 'MSE Loss', 'SSIM Loss', 'VQ Loss']
    ):
        ax.plot(epochs, history[key], label='train')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()

    # Perplexity (utilización del codebook)
    ax_perp = axes[1, 1]
    ax_perp.plot(epochs, history['perplexity'], label='perplexity', color='green')
    ax_perp.axhline(y=CONFIG['num_embeddings'], color='r', linestyle='--', alpha=0.5,
                    label=f'max ({CONFIG["num_embeddings"]})')
    ax_perp.set_title('Codebook Perplexity')
    ax_perp.set_xlabel('Epoch')
    ax_perp.legend()

    # SSIM value (1 - ssim_loss)
    ax_ssim_val = axes[1, 2]
    ssim_values = [1 - s for s in history['ssim']]
    ax_ssim_val.plot(epochs, ssim_values, label='SSIM', color='purple')
    ax_ssim_val.set_title('SSIM (calidad reconstrucción)')
    ax_ssim_val.set_xlabel('Epoch')
    ax_ssim_val.set_ylim(0, 1)
    ax_ssim_val.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_history.png'), dpi=150)
    plt.close()


# ============================================================
# ENTRENAMIENTO
# ============================================================

def run_epoch(model, dataloader, optimizer, config, scaler=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_losses = {'total': 0, 'mse': 0, 'ssim': 0, 'vq': 0}
    total_perplexity = 0
    num_batches = 0
    nan_batches = 0
    accum_steps = config.get('gradient_accumulation_steps', 1)
    use_amp = scaler is not None

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        if train:
            optimizer.zero_grad()
        for i, batch in enumerate(tqdm(dataloader, desc='Train' if train else 'Val', leave=False)):
            batch = batch.to(device)

            with autocast('cuda', enabled=use_amp):
                reconstructed, vq_loss, perplexity = model(batch)
                loss, loss_dict = compute_loss(batch, reconstructed, vq_loss, config)
                if train:
                    loss = loss / accum_steps

            # FIX: Detectar NaN/Inf y saltar el batch
            if not torch.isfinite(loss):
                nan_batches += 1
                if train:
                    optimizer.zero_grad()  # Limpiar gradientes corruptos
                continue

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    if (i + 1) % accum_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (i + 1) % accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()

            for k in total_losses:
                total_losses[k] += loss_dict[k]
            total_perplexity += perplexity.item()
            num_batches += 1

        # Flush gradientes restantes
        if train and num_batches % accum_steps != 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

    for k in total_losses:
        total_losses[k] /= max(num_batches, 1)
    avg_perplexity = total_perplexity / max(num_batches, 1)

    if nan_batches > 0:
        print(f'  >> AVISO: {nan_batches} batches con NaN/Inf saltados')

    return total_losses, avg_perplexity


def collect_encoder_outputs(model, dataloader, max_samples=8):
    """Recoge outputs del encoder para usar como reemplazos en codebook reset."""
    model.eval()
    z_samples = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_samples:
                break
            batch = batch.to(device)
            with autocast('cuda'):
                rgb = model.ae_tscm(batch)
                z = model.vqvae.encoder(rgb)
            # Flatten spatial dims: (B, D, H, W) → (B*H*W, D)
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, z.shape[1])
            z_samples.append(z_flat.float())
    if z_samples:
        return torch.cat(z_samples, dim=0)
    return torch.empty(0)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Crear dataset
    train_dataset = MT001Dataset(
        CONFIG['data_dir'],
        transform=get_train_transforms(),
        target_resolution=CONFIG['target_resolution'],  # FIX: Resize a 1280×720
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,          # FIX: 0 workers para evitar OOM (vs 2)
        pin_memory=False,       # FIX: Deshabilitado para evitar OOM en pin memory thread
        persistent_workers=False  # FIX: No persistent workers
    )

    # Verificar una muestra
    sample = train_dataset[0]
    print(f'\nForma muestra: {sample.shape}')
    print(f'Rango valores: [{sample.min():.3f}, {sample.max():.3f}]')
    print(f'Total capturas para entrenamiento: {len(train_dataset)}')

    # Verificar memoria disponible
    if torch.cuda.is_available():
        print(f'\nMemoria GPU:')
        print(f'  Asignada: {torch.cuda.memory_allocated(0)/1e9:.2f} GB')
        print(f'  Reservada: {torch.cuda.memory_reserved(0)/1e9:.2f} GB')
        total_mem = torch.cuda.get_device_properties(0).total_memory
        print(f'  Disponible: {(total_mem - torch.cuda.memory_allocated(0))/1e9:.2f} GB')

    # Crear modelo
    model = FullModel(CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nParámetros totales: {total_params:,}')
    print(f'Parámetros entrenables: {trainable_params:,}')
    print(f'Codebook: {CONFIG["num_embeddings"]} entradas × {CONFIG["embedding_dim"]}d')

    # Optimizador y scaler
    optimizer = Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scaler = GradScaler('cuda')

    # Historial
    history = {k: [] for k in ['total', 'mse', 'ssim', 'vq', 'perplexity']}
    best_loss = float('inf')

    print(f'\nIniciando entrenamiento: {CONFIG["epochs"]} épocas')
    print(f'Muestras: {len(train_dataset)} (todas, sin split)')
    print(f'Batch size: {CONFIG["batch_size"]} | Gradient accumulation: {CONFIG["gradient_accumulation_steps"]} (batch efectivo: {CONFIG["batch_size"] * CONFIG["gradient_accumulation_steps"]})')
    print(f'Learning rate: {CONFIG["learning_rate"]} (CONSTANTE, sin scheduler)')
    print(f'Mixed precision (AMP): activado | VQ forzado a FP32')
    print(f'Codebook: {CONFIG["num_embeddings"]} entradas, EMA decay={CONFIG["ema_decay"]}, reset cada {CONFIG["codebook_reset_interval"]} épocas')
    print(f'Gradient clipping: max_norm=1.0')
    print('=' * 70)

    nan_streak = 0
    MAX_NAN_STREAK = 5

    for epoch in range(1, CONFIG['epochs'] + 1):
        # Limpiar caché CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Train
        train_losses, avg_perplexity = run_epoch(model, train_loader, optimizer, CONFIG, scaler=scaler, train=True)

        # Detección temprana de NaN
        if math.isnan(train_losses['total']):
            nan_streak += 1
            print(f'Ep {epoch:3d}/{CONFIG["epochs"]} | Loss: NaN (racha: {nan_streak}/{MAX_NAN_STREAK})')
            if nan_streak >= MAX_NAN_STREAK:
                print(f'\n  ABORT: {MAX_NAN_STREAK} épocas consecutivas con NaN. Revisar modelo/hiperparámetros.')
                break
            continue
        else:
            nan_streak = 0

        # Guardar historial
        for k in ['total', 'mse', 'ssim', 'vq']:
            history[k].append(train_losses[k])
        history['perplexity'].append(avg_perplexity)

        # Guardar mejor modelo
        if train_losses['total'] < best_loss:
            best_loss = train_losses['total']
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pt'))

        # Log
        print(f'Ep {epoch:3d}/{CONFIG["epochs"]} | '
              f'Loss: {train_losses["total"]:.4f} (mse:{train_losses["mse"]:.4f} ssim:{train_losses["ssim"]:.4f} vq:{train_losses["vq"]:.4f}) | '
              f'Perp: {avg_perplexity:.1f}/{CONFIG["num_embeddings"]}')

        # Mostrar memoria cada 10 épocas
        if epoch % 10 == 0 and torch.cuda.is_available():
            print(f'  >> Mem: {torch.cuda.memory_allocated(0)/1e9:.2f} GB asignada, '
                  f'{torch.cuda.max_memory_allocated(0)/1e9:.2f} GB máx')

        # Codebook reset
        if epoch % CONFIG['codebook_reset_interval'] == 0:
            z_samples = collect_encoder_outputs(model, train_loader, max_samples=8)
            num_reset = model.vqvae.vq.reset_unused_codes(
                z_samples, threshold=CONFIG['codebook_usage_threshold'])
            usage_pct = (CONFIG['num_embeddings'] - num_reset) / CONFIG['num_embeddings'] * 100
            print(f'  >> Codebook reset: {num_reset} entradas reemplazadas, '
                  f'utilización: {usage_pct:.0f}%')

        # Visualización
        if epoch % CONFIG['log_interval'] == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                sample = next(iter(train_loader)).to(device)
                with autocast('cuda'):
                    recon, _, _, rgb, _, _ = model(sample, return_intermediate=True)
                save_visualization(sample, recon.float(), rgb.float(), epoch,
                                   os.path.join(SAVE_DIR, 'visualizations'))

        # Checkpoint
        if epoch % CONFIG['save_interval'] == 0:
            ckpt_path = os.path.join(SAVE_DIR, 'checkpoints', f'ckpt_epoch_{epoch:04d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': train_losses['total'],
                'perplexity': avg_perplexity,
                'config': CONFIG
            }, ckpt_path)
            print(f'  >> Checkpoint guardado: epoch {epoch}')

    # Guardar modelo final y gráficas
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'final_model.pt'))
    if history['total']:
        plot_losses(history, SAVE_DIR)

    print('\n' + '=' * 70)
    if history['total']:
        print(f'Entrenamiento completado. Loss final: {train_losses["total"]:.4f} | Mejor: {best_loss:.4f}')
        print(f'Perplexity final: {avg_perplexity:.1f}/{CONFIG["num_embeddings"]}')
    else:
        print('Entrenamiento fallido: ninguna época produjo loss válido.')
    print(f'Resultados en: {SAVE_DIR}')
