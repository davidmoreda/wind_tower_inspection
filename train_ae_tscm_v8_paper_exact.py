#!/usr/bin/env python3
"""
VQ-VAE v8.0 - ARQUITECTURA EXACTA DEL PAPER

FIXES CRÍTICOS vs v7:
1. Encoder: 2 downsamplings (4×) NO 3 (8×)
   - Conv1: k=4, s=2 → /2
   - Conv2: k=4, s=2 → /4
   - Conv3: k=3, s=1 → /4 (feature extraction, NO downsampling)
   
2. Codebook init: uniform(-1/K, 1/K) NO normal(0,1)
   
3. SIN BatchNorm en encoder (solo ResidualStack)

4. Decoder: 2 upsamplings (4×) matching encoder

RESULTADO ESPERADO:
- 1280×720 → 320×180 latent (57,600 posiciones)
- ENTRA en FP16 sin overflow
- Codebook converge desde época 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from torchvision import transforms
from pytorch_msssim import ssim
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import csv
import os
import math
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ============================================================
# CONFIGURACIÓN
# ============================================================

CONFIG = {
    # Dataset
    'data_dir': 'dataset/dataset/MT001',
    'num_lights': 5,
    'target_resolution': (1280, 720),  # FIX: 16:9 aspect ratio
    
    # Training
    'batch_size': 2,
    'gradient_accumulation_steps': 2,
    'epochs': 150,
    'learning_rate': 1e-3,
    'weight_decay': 0,
    
    # AE-TSCM
    'use_spatial_attention': True,
    
    # VQVAE - EXACTO COMO PAPER
    'hidden_channels': 128,        # h_dim del paper
    'num_residual': 4,             # n_res_layers
    'res_hidden': 64,              # res_h_dim
    'num_embeddings': 512,         # K
    'embedding_dim': 128,          # D
    'commitment_cost': 0.25,       # beta
    'ema_decay': 0.99,
    
    # Codebook management
    'codebook_reset_interval': 10,
    'codebook_usage_threshold': 2,
    
    # Loss weights
    'lambda_mse': 1.0,
    'lambda_ssim': 0.0,  # FIX: SSIM deshabilitado para evitar NaN
    'lambda_vq': 0.02,
    
    # Logging
    'save_interval': 25,
    'log_interval': 10,
}

# Directorio de salida
SAVE_DIR = 'runs/ae_tscm_v8_' + datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, 'visualizations'), exist_ok=True)
print(f'Resultados se guardarán en: {SAVE_DIR}')

# ============================================================
# DATASET
# ============================================================

LIGHT_NAMES = ['SUP_IZQ', 'SUP_DER', 'INF_DER', 'INF_IZQ', 'ALL']


class CropToDivisible:
    """Recorta para que H y W sean divisibles por factor."""
    def __init__(self, factor=4):  # FIX: factor=4 para 2 downsamplings
        self.factor = factor

    def __call__(self, img):
        _, h, w = img.shape
        new_h = (h // self.factor) * self.factor
        new_w = (w // self.factor) * self.factor
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        return img[:, top:top + new_h, left:left + new_w]


class MT001Dataset(Dataset):
    """Dataset MT001 con resize a target_resolution."""

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
            image = Image.open(img_path).convert('L')
            
            if self.target_resolution is not None:
                image = image.resize(
                    (self.target_resolution[0], self.target_resolution[1]), 
                    Image.BILINEAR
                )
            
            image = transforms.ToTensor()(image)
            channels.append(image)

        multi_channel = torch.cat(channels, dim=0)

        if self.transform:
            multi_channel = self.transform(multi_channel)

        return multi_channel


def get_train_transforms():
    return transforms.Compose([
        CropToDivisible(4),  # FIX: factor=4 para 2 downsamplings
        transforms.RandomHorizontalFlip(p=0.5),
    ])


def get_val_transforms():
    return transforms.Compose([
        CropToDivisible(4),
    ])


# ============================================================
# MODELOS - AE-TSCM (sin cambios)
# ============================================================

class ChannelAttention(nn.Module):
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
    def __init__(self, num_coefficients=5):
        super().__init__()
        initial = torch.zeros(num_coefficients)
        initial[1] = 1.0
        self.coefficients = nn.Parameter(initial)

    def forward(self, x):
        c = self.coefficients
        return (c[0] + c[1]*x + c[2]*(x**2)/2 + c[3]*(x**3)/6 + c[4]*(x**4)/24)


class ChannelMixer(nn.Module):
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
    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        mn = x_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
        mx = x_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
        return (x - mn) / (mx - mn + 1e-4)


class AE_TSCM(nn.Module):
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
# MODELOS - VQVAE EXACTO COMO PAPER
# ============================================================

class ResidualLayer(nn.Module):
    """Residual layer como en el paper."""
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualStack(nn.Module):
    """Stack de residual layers."""
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualLayer(in_channels, hidden_channels)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.relu(x)


class Encoder(nn.Module):
    """
    Encoder EXACTO del paper:
    - 2 downsamplings (4×): conv1 (s=2), conv2 (s=2)
    - 1 feature layer (s=1): conv3
    - ResidualStack
    - Projection a embedding_dim
    - SIN BatchNorm
    """
    def __init__(self, in_channels, h_dim, res_h_dim, n_res_layers, embedding_dim):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # Downsample 1: /2
            nn.Conv2d(in_channels, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Downsample 2: /4
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Feature extraction (NO downsampling)
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
            # ResidualStack
            ResidualStack(h_dim, res_h_dim, n_res_layers),
            # Project to embedding_dim (SIN BatchNorm)
            nn.Conv2d(h_dim, embedding_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.conv_stack(x)


class VectorQuantizerEMA(nn.Module):
    """
    VQ-EMA EXACTO del paper:
    - Init: uniform(-1/K, 1/K) NO normal(0,1)
    - Todo en FP32
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost,
                 decay, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # FIX CRÍTICO: Uniform init como en paper
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.codebook.weight.requires_grad = False

        # EMA buffers
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_dw', self.codebook.weight.data.clone())
        self.register_buffer('usage_count', torch.zeros(num_embeddings))

    def forward(self, z):
        # FIX: Forzar FP32
        with torch.amp.autocast('cuda', enabled=False):
            z = z.float()

            z = z.permute(0, 2, 3, 1).contiguous()
            z_shape = z.shape
            z_flat = z.view(-1, self.embedding_dim)

            # Distances
            distances = (
                torch.sum(z_flat**2, dim=1, keepdim=True) +
                torch.sum(self.codebook.weight**2, dim=1) -
                2 * torch.matmul(z_flat, self.codebook.weight.t())
            )

            encoding_indices = torch.argmin(distances, dim=1)
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            quantized = self.codebook(encoding_indices).view(z_shape)

            # EMA update
            if self.training:
                encodings_sum = encodings.sum(0)
                dw = encodings.t() @ z_flat

                self.ema_cluster_size.data.mul_(self.decay).add_(
                    encodings_sum, alpha=1 - self.decay)
                self.ema_dw.data.mul_(self.decay).add_(
                    dw, alpha=1 - self.decay)

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon) /
                    (n + self.num_embeddings * self.epsilon) * n
                )

                self.codebook.weight.data.copy_(
                    self.ema_dw / cluster_size.unsqueeze(1))

                self.usage_count.add_(encodings_sum)

            # Commitment loss
            commitment_loss = F.mse_loss(z, quantized.detach())
            loss = self.commitment_cost * commitment_loss

            # Straight-through
            quantized = z + (quantized - z).detach()

            # Perplexity
            avg_probs = encodings.mean(0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            return quantized, loss, perplexity

    def reset_unused_codes(self, z_flat_samples, threshold=2):
        z_flat_samples = z_flat_samples.float()
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

        self.usage_count.zero_()
        return num_unused


class Decoder(nn.Module):
    """
    Decoder EXACTO del paper:
    - Projection from embedding_dim
    - ResidualStack
    - 2 upsamplings (4×): deconv1 (s=2), deconv2 (s=2)
    - SIN Sigmoid final
    """
    def __init__(self, out_channels, h_dim, res_h_dim, n_res_layers, embedding_dim):
        super().__init__()
        self.inverse_conv_stack = nn.Sequential(
            # Project from embedding_dim (NO upsampling)
            nn.ConvTranspose2d(embedding_dim, h_dim, kernel_size=3, stride=1, padding=1),
            # ResidualStack
            ResidualStack(h_dim, res_h_dim, n_res_layers),
            # Upsample 1: x2
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Upsample 2: x4
            nn.ConvTranspose2d(h_dim // 2, out_channels, kernel_size=4, stride=2, padding=1),
            # SIN Sigmoid
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


class VQVAE(nn.Module):
    def __init__(self, in_channels, out_channels, h_dim, res_h_dim, n_res_layers,
                 num_embeddings, embedding_dim, commitment_cost, ema_decay):
        super().__init__()
        self.encoder = Encoder(in_channels, h_dim, res_h_dim, n_res_layers, embedding_dim)
        self.vq = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, ema_decay)
        self.decoder = Decoder(out_channels, h_dim, res_h_dim, n_res_layers, embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, perplexity = self.vq(z)
        return self.decoder(z_q), vq_loss, perplexity

    def encode(self, x):
        z = self.encoder(x)
        z_q, _, _ = self.vq(z)
        return z_q


class FullModel(nn.Module):
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
            h_dim=config['hidden_channels'],
            res_h_dim=config['res_hidden'],
            n_res_layers=config['num_residual'],
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
# LOSS Y VISUALIZACIÓN
# ============================================================

def compute_loss(original, reconstructed, vq_loss, config):
    """
    FIX: Solo MSE + VQ (SSIM deshabilitado para evitar NaN).
    """
    # MSE para backprop
    mse_loss = F.mse_loss(reconstructed, original)
    
    # Total: MSE + VQ
    total = (config['lambda_mse'] * mse_loss +
             config['lambda_vq'] * vq_loss)
    
    # SSIM solo para logging (NO gradientes)
    with torch.no_grad():
        recon_clamped = reconstructed.clamp(0, 1)
        ssim_val = ssim(recon_clamped, original, data_range=1.0, size_average=True)
        ssim_loss = 1 - ssim_val
    
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

    ax_perp = axes[1, 1]
    ax_perp.plot(epochs, history['perplexity'], label='perplexity', color='green')
    ax_perp.axhline(y=CONFIG['num_embeddings'], color='r', linestyle='--', alpha=0.5,
                    label=f'max ({CONFIG["num_embeddings"]})')
    ax_perp.set_title('Codebook Perplexity')
    ax_perp.set_xlabel('Epoch')
    ax_perp.legend()

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
# TRAINING
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

            # Check NaN
            if not torch.isfinite(loss):
                nan_batches += 1
                if train:
                    optimizer.zero_grad()
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

        # Flush
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
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, z.shape[1])
            z_samples.append(z_flat.float())
    if z_samples:
        return torch.cat(z_samples, dim=0)
    return torch.empty(0)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Dataset
    train_dataset = MT001Dataset(
        CONFIG['data_dir'],
        transform=get_train_transforms(),
        target_resolution=CONFIG['target_resolution'],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )

    sample = train_dataset[0]
    print(f'\nForma muestra: {sample.shape}')
    print(f'Rango valores: [{sample.min():.3f}, {sample.max():.3f}]')
    print(f'Total capturas para entrenamiento: {len(train_dataset)}')

    if torch.cuda.is_available():
        print(f'\nMemoria GPU:')
        print(f'  Asignada: {torch.cuda.memory_allocated(0)/1e9:.2f} GB')
        print(f'  Reservada: {torch.cuda.memory_reserved(0)/1e9:.2f} GB')
        total_mem = torch.cuda.get_device_properties(0).total_memory
        print(f'  Disponible: {(total_mem - torch.cuda.memory_allocated(0))/1e9:.2f} GB')

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Model
    model = FullModel(CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nParámetros totales: {total_params:,}')
    print(f'Parámetros entrenables: {trainable_params:,}')
    print(f'Codebook: {CONFIG["num_embeddings"]} entradas × {CONFIG["embedding_dim"]}d')
    
    # Verificar latent map size
    with torch.no_grad():
        dummy = torch.randn(1, 3, 720, 1280).to(device)
        z = model.vqvae.encoder(dummy)
        print(f'Latent map: {z.shape} → {z.shape[2] * z.shape[3]} posiciones')

    optimizer = Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scaler = GradScaler('cuda')

    history = {k: [] for k in ['total', 'mse', 'ssim', 'vq', 'perplexity']}
    best_loss = float('inf')

    print(f'\n🚀 ENTRENAMIENTO v8.0 - ARQUITECTURA EXACTA DEL PAPER')
    print(f'Downsampling: 2 etapas (4×) vs v7 3 etapas (8×)')
    print(f'Codebook init: uniform(-{1/CONFIG["num_embeddings"]:.5f}, {1/CONFIG["num_embeddings"]:.5f})')
    print(f'Batch size: {CONFIG["batch_size"]} | Accum: {CONFIG["gradient_accumulation_steps"]}')
    print(f'Learning rate: {CONFIG["learning_rate"]}')
    print('=' * 70)

    nan_streak = 0
    MAX_NAN_STREAK = 5

    for epoch in range(1, CONFIG['epochs'] + 1):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        train_losses, avg_perplexity = run_epoch(model, train_loader, optimizer, CONFIG, scaler=scaler, train=True)

        if math.isnan(train_losses['total']):
            nan_streak += 1
            print(f'Ep {epoch:3d}/{CONFIG["epochs"]} | Loss: NaN (racha: {nan_streak}/{MAX_NAN_STREAK})')
            if nan_streak >= MAX_NAN_STREAK:
                print(f'\n  ABORT: {MAX_NAN_STREAK} épocas consecutivas con NaN.')
                break
            continue
        else:
            nan_streak = 0

        for k in ['total', 'mse', 'ssim', 'vq']:
            history[k].append(train_losses[k])
        history['perplexity'].append(avg_perplexity)

        if train_losses['total'] < best_loss:
            best_loss = train_losses['total']
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pt'))

        print(f'Ep {epoch:3d}/{CONFIG["epochs"]} | '
              f'Loss: {train_losses["total"]:.4f} (mse:{train_losses["mse"]:.4f} ssim:{train_losses["ssim"]:.4f} vq:{train_losses["vq"]:.4f}) | '
              f'Perp: {avg_perplexity:.1f}/{CONFIG["num_embeddings"]}')

        if epoch % 10 == 0 and torch.cuda.is_available():
            print(f'  >> Mem: {torch.cuda.memory_allocated(0)/1e9:.2f} GB asignada, '
                  f'{torch.cuda.max_memory_allocated(0)/1e9:.2f} GB máx')

        if epoch % CONFIG['codebook_reset_interval'] == 0:
            z_samples = collect_encoder_outputs(model, train_loader, max_samples=8)
            num_reset = model.vqvae.vq.reset_unused_codes(
                z_samples, threshold=CONFIG['codebook_usage_threshold'])
            usage_pct = (CONFIG['num_embeddings'] - num_reset) / CONFIG['num_embeddings'] * 100
            print(f'  >> Codebook reset: {num_reset} entradas reemplazadas, '
                  f'utilización: {usage_pct:.0f}%')

        if epoch % CONFIG['log_interval'] == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                sample = next(iter(train_loader)).to(device)
                with autocast('cuda'):
                    recon, _, _, rgb, _, _ = model(sample, return_intermediate=True)
                save_visualization(sample, recon.float(), rgb.float(), epoch,
                                   os.path.join(SAVE_DIR, 'visualizations'))

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

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'final_model.pt'))
    if history['total']:
        plot_losses(history, SAVE_DIR)

    print('\n' + '=' * 70)
    if history['total']:
        print(f'✅ Entrenamiento completado. Loss final: {train_losses["total"]:.4f} | Mejor: {best_loss:.4f}')
        print(f'Perplexity final: {avg_perplexity:.1f}/{CONFIG["num_embeddings"]}')
    else:
        print('❌ Entrenamiento fallido: ninguna época produjo loss válido.')
    print(f'Resultados en: {SAVE_DIR}')
