#!/usr/bin/env python3
"""
Test rápido para verificar que v7 NO produce NaN en forward pass.
"""

import torch
import sys

# Importar modelo del script v7
sys.path.insert(0, '.')
from train_ae_tscm_v7_fixed import (
    FullModel, CONFIG, device,
    MT001Dataset, get_train_transforms
)

print("=" * 70)
print("TEST: Verificación de NaN en v7.0")
print("=" * 70)

# Crear dataset con 1 muestra
print("\n1. Cargando dataset...")
train_dataset = MT001Dataset(
    CONFIG['data_dir'],
    transform=get_train_transforms(),
    target_resolution=CONFIG['target_resolution'],
)

# Tomar una muestra
sample = train_dataset[0].unsqueeze(0).to(device)  # (1, 5, 720, 1280)
print(f"   Forma muestra: {sample.shape}")
print(f"   Rango valores: [{sample.min():.3f}, {sample.max():.3f}]")

# Crear modelo
print("\n2. Creando modelo...")
model = FullModel(CONFIG).to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"   Parámetros: {total_params:,}")
print(f"   Codebook: {CONFIG['num_embeddings']} embeddings × {CONFIG['embedding_dim']}d")

# Test forward pass
print("\n3. Test forward pass (sin AMP)...")
with torch.no_grad():
    reconstructed, vq_loss, perplexity = model(sample)

# Verificar resultados
print("\n4. Verificación de resultados:")
print(f"   Reconstructed shape: {reconstructed.shape}")
print(f"   Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
print(f"   VQ loss: {vq_loss.item():.4f}")
print(f"   Perplexity: {perplexity.item():.1f}/{CONFIG['num_embeddings']}")

# Checks
has_nan_recon = not torch.isfinite(reconstructed).all()
has_nan_loss = not torch.isfinite(vq_loss)
perp_is_zero = perplexity.item() == 0

print("\n5. Checks:")
print(f"   ✗ Reconstructed tiene NaN/Inf: {has_nan_recon}" if has_nan_recon else "   ✓ Reconstructed sin NaN/Inf")
print(f"   ✗ VQ loss es NaN/Inf: {has_nan_loss}" if has_nan_loss else "   ✓ VQ loss finito")
print(f"   ✗ Perplexity es 0: {perp_is_zero}" if perp_is_zero else "   ✓ Perplexity > 0")

# Test con AMP
print("\n6. Test forward pass (con AMP)...")
with torch.no_grad():
    with torch.amp.autocast('cuda'):
        reconstructed_amp, vq_loss_amp, perplexity_amp = model(sample)

has_nan_amp = not torch.isfinite(reconstructed_amp).all()
print(f"   ✗ Con AMP tiene NaN/Inf: {has_nan_amp}" if has_nan_amp else "   ✓ Con AMP sin NaN/Inf")
print(f"   Perplexity (AMP): {perplexity_amp.item():.1f}")

# Resultado final
print("\n" + "=" * 70)
if has_nan_recon or has_nan_loss or perp_is_zero or has_nan_amp:
    print("❌ TEST FALLIDO - Todavía hay problemas con NaN/perplexity")
    sys.exit(1)
else:
    print("✅ TEST EXITOSO - Modelo funciona correctamente sin NaN")
    print(f"   Perplexity inicial: {perplexity.item():.1f} (debe aumentar durante training)")
    print("\n   Puedes proceder al entrenamiento con:")
    print("   python train_ae_tscm_v7_fixed.py")
    sys.exit(0)
