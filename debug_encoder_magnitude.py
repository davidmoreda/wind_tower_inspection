#!/usr/bin/env python3
"""
Debug script para ver la magnitud de embeddings del encoder.
"""
import torch
import sys
sys.path.insert(0, '.')
from train_ae_tscm_v8_paper_exact import FullModel, CONFIG, device, MT001Dataset, get_train_transforms

print("Verificando magnitud de embeddings del encoder...")

# Dataset
train_dataset = MT001Dataset(
    CONFIG['data_dir'],
    transform=get_train_transforms(),
    target_resolution=CONFIG['target_resolution'],
)

# Model
model = FullModel(CONFIG).to(device)
model.eval()

# Sample
sample = train_dataset[0].unsqueeze(0).to(device)

with torch.no_grad():
    # Get encoder output
    rgb = model.ae_tscm(sample)
    z = model.vqvae.encoder(rgb)
    
    print(f"\n📊 Encoder output (z):")
    print(f"  Shape: {z.shape}")
    print(f"  Min: {z.min().item():.6f}")
    print(f"  Max: {z.max().item():.6f}")
    print(f"  Mean: {z.mean().item():.6f}")
    print(f"  Std: {z.std().item():.6f}")
    
    # Codebook range
    codebook_weight = model.vqvae.vq.codebook.weight.data
    print(f"\n📊 Codebook:")
    print(f"  Shape: {codebook_weight.shape}")
    print(f"  Min: {codebook_weight.min().item():.6f}")
    print(f"  Max: {codebook_weight.max().item():.6f}")
    print(f"  Mean: {codebook_weight.mean().item():.6f}")
    print(f"  Std: {codebook_weight.std().item():.6f}")
    
    # Magnitud comparison
    z_magnitude = z.abs().mean().item()
    cb_magnitude = codebook_weight.abs().mean().item()
    
    print(f"\n⚠️  Magnitude comparison:")
    print(f"  Encoder output magnitude: {z_magnitude:.6f}")
    print(f"  Codebook magnitude: {cb_magnitude:.6f}")
    print(f"  Ratio: {z_magnitude / cb_magnitude:.1f}× (encoder vs codebook)")
    
    if z_magnitude > cb_magnitude * 10:
        print(f"\n❌ PROBLEMA: Encoder output es {z_magnitude / cb_magnitude:.0f}× más grande que codebook!")
        print("   Esto causa codebook collapse.")
        print("\n💡 SOLUCIÓN: Aumentar codebook init range o añadir normalización.")
