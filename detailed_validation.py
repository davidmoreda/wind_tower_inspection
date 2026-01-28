#!/usr/bin/env python3
"""
Validación visual DETALLADA por luz individual.
Muestra Original vs Reconstrucción para CADA luz por separado.
"""

import torch
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '.')
from train_ae_tscm_v8_paper_exact import (
    FullModel, CONFIG, device, MT001Dataset, 
    get_val_transforms, LIGHT_NAMES
)

print("=" * 80)
print("🔍 VALIDACIÓN DETALLADA POR LUZ INDIVIDUAL")
print("=" * 80)

# Encontrar modelo
runs_dir = Path('runs')
v8_runs = sorted(runs_dir.glob('ae_tscm_v8_*'))
if not v8_runs:
    print("❌ No se encontró run de v8")
    sys.exit(1)

latest_run = v8_runs[-1]
best_model_path = latest_run / 'best_model.pt'

print(f"\n📂 Run: {latest_run.name}")

# Cargar modelo
model = FullModel(CONFIG).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
print("✓ Modelo cargado")

# Dataset
val_dataset = MT001Dataset(
    CONFIG['data_dir'],
    transform=get_val_transforms(),
    target_resolution=CONFIG['target_resolution'],
)

# Output
output_dir = latest_run / 'detailed_validation'
output_dir.mkdir(exist_ok=True)

print(f"\n📸 Generando comparaciones detalladas...")

num_samples = 3
for sample_idx in range(num_samples):
    # Cargar muestra
    original = val_dataset[sample_idx].unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Forward pass completo
        reconstructed, vq_loss, perplexity, rgb, ch_att, sp_att = model(
            original, return_intermediate=True
        )
        reconstructed = reconstructed.clamp(0, 1)
    
    # CPU
    original = original[0].cpu().numpy()  # (5, H, W)
    reconstructed = reconstructed[0].cpu().numpy()
    rgb = rgb[0].cpu().numpy()  # (3, H, W)
    
    # Crear figura GRANDE: 5 luces × 3 columnas (Original, Recon, Diferencia)
    fig, axes = plt.subplots(6, 3, figsize=(15, 24))
    
    # Para cada luz
    for i, light_name in enumerate(LIGHT_NAMES):
        orig = original[i]
        recon = reconstructed[i]
        diff = np.abs(orig - recon)
        
        # Error relativo
        mse = np.mean((orig - recon)**2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # Columna 1: Original
        axes[i, 0].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'{light_name}\nOriginal', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Columna 2: Reconstrucción
        axes[i, 1].imshow(recon, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Reconstrucción\nPSNR: {psnr:.1f} dB', fontsize=12)
        axes[i, 1].axis('off')
        
        # Columna 3: Diferencia (error absoluto)
        im = axes[i, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[i, 2].set_title(f'Error absoluto\nMSE: {mse:.6f}', fontsize=12)
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)
    
    # Fila 6: RGB fusionado (solo para referencia)
    rgb_img = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
    # Normalizar para visualización
    rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
    
    axes[5, 0].imshow(rgb_img)
    axes[5, 0].set_title('RGB Fusionado\n(AE-TSCM output)', fontsize=12, fontweight='bold')
    axes[5, 0].axis('off')
    
    # Canal R
    axes[5, 1].imshow(rgb[0], cmap='Reds', vmin=0, vmax=1)
    axes[5, 1].set_title('Canal R\n(típ: luz superior)', fontsize=10)
    axes[5, 1].axis('off')
    
    # Canal G  
    axes[5, 2].imshow(rgb[1], cmap='Greens', vmin=0, vmax=1)
    axes[5, 2].set_title('Canal G\n(típ: luz inferior)', fontsize=10)
    axes[5, 2].axis('off')
    
    plt.suptitle(
        f'Captura {sample_idx+1}/{num_samples} - Análisis Detallado por Luz\n'
        f'Perplexity: {perplexity.item():.1f}/512',
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout()
    
    # Guardar
    output_path = output_dir / f'detalle_captura_{sample_idx+1:02d}.png'
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {output_path.name}")

print(f"\n✅ Análisis detallado guardado en: {output_dir}")

print("\n" + "=" * 80)
print("📊 CÓMO INTERPRETAR LAS IMÁGENES:")
print("=" * 80)
print("\n🔍 COLUMNAS:")
print("  1. Original: Imagen de entrada para cada luz")
print("  2. Reconstrucción: Lo que genera el modelo")
print("  3. Error absoluto: |Original - Recon| (rojo = más error)")
print("\n📈 MÉTRICAS:")
print("  • PSNR > 30 dB = Excelente")
print("  • PSNR 25-30 dB = Buena")
print("  • PSNR < 25 dB = Necesita mejorar")
print("  • MSE < 0.001 = Muy bajo error")
print("\n🎨 RGB FUSIONADO (fila inferior):")
print("  • NO son colores reales")
print("  • Es cómo el modelo representa las 5 luces en 3 canales")
print("  • Los 'colores' son una codificación interna aprendida")
print("\n💡 QUÉ BUSCAR:")
print("  ✓ Mapas de error (columna 3) mayormente oscuros/azules")
print("  ✓ Detalles visibles en reconstrucciones (bordes, texturas)")
print("  ✓ PSNR > 25 dB en la mayoría de luces")
print("\n⚠️  DIFERENCIAS CON EL PAPER:")
print("  • Paper: metal plano, 16 luces, defectos obvios")
print("  • Tu caso: superficies curvas, 5 luces, defectos sutiles")
print("  • NORMAL que sea más difícil: menos información de entrada")
print("=" * 80)
