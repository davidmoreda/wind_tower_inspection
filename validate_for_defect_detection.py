#!/usr/bin/env python3
"""
VALIDACIÓN PARA DETECCIÓN DE DEFECTOS

El VQ-VAE NO es el objetivo final. Es un PASO INTERMEDIO.

OBJETIVO REAL: Detectar defectos en torres eólicas
PREGUNTA CLAVE: ¿El codebook aprendido agrupa regiones similares?

PRUEBA:
1. Generar mapa de códigos del VQ para cada imagen
2. Ver si zonas diferentes (soldadura, defectos, metal normal) usan códigos diferentes
3. Si SÍ → El VQ-VAE es útil para detección
4. Si NO → Necesita mejorar
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from matplotlib.patches import Rectangle

sys.path.insert(0, '.')
from train_ae_tscm_v8_paper_exact import (
    FullModel, CONFIG, device, MT001Dataset, 
    get_val_transforms, LIGHT_NAMES
)

print("=" * 80)
print("🎯 VALIDACIÓN PARA DETECCIÓN DE DEFECTOS")
print("=" * 80)
print("\nPREGUNTA: ¿El codebook aprendido distingue zonas diferentes?")
print("RESPUESTA: Si zonas distintas (soldadura, defecto, normal) usan códigos")
print("           distintos → VQ-VAE es ÚTIL para detección")
print("=" * 80)

# Cargar modelo
runs_dir = Path('runs')
v8_runs = sorted(runs_dir.glob('ae_tscm_v8_*'))
if not v8_runs:
    print("❌ No se encontró run de v8")
    sys.exit(1)

latest_run = v8_runs[-1]
best_model_path = latest_run / 'best_model.pt'

model = FullModel(CONFIG).to(device)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

# Dataset
val_dataset = MT001Dataset(
    CONFIG['data_dir'],
    transform=get_val_transforms(),
    target_resolution=CONFIG['target_resolution'],
)

# Output
output_dir = latest_run / 'defect_validation'
output_dir.mkdir(exist_ok=True)

print(f"\n📊 Generando mapas de códigos VQ...")

num_samples = 3
for sample_idx in range(num_samples):
    original = val_dataset[sample_idx].unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get RGB fusion
        rgb = model.ae_tscm(original)
        
        # Get encoder output
        z = model.vqvae.encoder(rgb)
        
        # Get VQ codes (sin quantize, solo indices)
        z_flat = z.permute(0, 2, 3, 1).contiguous()
        z_shape = z_flat.shape
        z_flat = z_flat.view(-1, CONFIG['embedding_dim'])
        
        # Find nearest codes
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True) +
            torch.sum(model.vqvae.vq.codebook.weight**2, dim=1) -
            2 * torch.matmul(z_flat, model.vqvae.vq.codebook.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Reshape to spatial map
        code_map = encoding_indices.view(z_shape[0], z_shape[1], z_shape[2])  # (1, H_latent, W_latent)
        code_map = code_map[0].cpu().numpy()  # (H_latent, W_latent)
        
        # Reconstrucción
        reconstructed, vq_loss, perplexity = model(original)
    
    # CPU
    original_cpu = original[0].cpu().numpy()  # (5, H, W)
    reconstructed_cpu = reconstructed[0].clamp(0, 1).cpu().numpy()
    rgb_cpu = rgb[0].cpu().numpy()
    
    # Crear figura: 3 filas × 3 columnas
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Fila 1: 3 luces originales
    for i in range(3):
        axes[0, i].imshow(original_cpu[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'Original: {LIGHT_NAMES[i]}', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
    
    # Fila 2: RGB fusionado + mapa de códigos VQ
    rgb_viz = np.transpose(rgb_cpu, (1, 2, 0))
    rgb_viz = (rgb_viz - rgb_viz.min()) / (rgb_viz.max() - rgb_viz.min() + 1e-8)
    
    axes[1, 0].imshow(rgb_viz)
    axes[1, 0].set_title('RGB Fusionado (AE-TSCM)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # MAPA DE CÓDIGOS VQ (CLAVE)
    im = axes[1, 1].imshow(code_map, cmap='tab20b', interpolation='nearest')
    axes[1, 1].set_title(
        f'MAPA DE CÓDIGOS VQ\n{code_map.shape[0]}×{code_map.shape[1]} = {code_map.size} posiciones',
        fontsize=12, fontweight='bold'
    )
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
    
    # Histograma de códigos usados
    unique_codes, counts = np.unique(code_map, return_counts=True)
    axes[1, 2].bar(range(len(unique_codes)), counts, color='steelblue')
    axes[1, 2].set_title(f'Códigos usados: {len(unique_codes)}/{CONFIG["num_embeddings"]}', 
                         fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Código ID (ordenado)')
    axes[1, 2].set_ylabel('Frecuencia')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Fila 3: Reconstrucciones
    for i in range(3):
        axes[2, i].imshow(reconstructed_cpu[i], cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'Reconstrucción: {LIGHT_NAMES[i]}', fontsize=12)
        axes[2, i].axis('off')
    
    plt.suptitle(
        f'Captura {sample_idx+1} - Análisis de Códigos VQ\n'
        f'Perplexity: {perplexity.item():.1f}/512',
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout()
    
    # Guardar
    output_path = output_dir / f'codemap_captura_{sample_idx+1:02d}.png'
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {output_path.name} ({len(unique_codes)} códigos únicos)")

print(f"\n✅ Mapas guardados en: {output_dir}")

print("\n" + "=" * 80)
print("🔍 CÓMO INTERPRETAR EL MAPA DE CÓDIGOS VQ:")
print("=" * 80)
print("\n📊 MAPA CENTRAL (2da fila, 2da columna):")
print("  • Cada pixel es un código del VQ (0-511)")
print("  • Mismo color = mismo código = región 'similar' según el modelo")
print("  • Color diferente = código diferente = región 'distinta'")
print("\n✅ BUENA SEÑAL (modelo útil para detección):")
print("  • Zonas visualmente diferentes tienen códigos diferentes")
print("  • Soldaduras usan códigos distintos vs metal normal")
print("  • Defectos/grafitti usan códigos únicos")
print("  • Transiciones suaves donde la superficie es uniforme")
print("\n❌ MALA SEÑAL (modelo NO útil):")
print("  • Todo usa el mismo código (mapa uniforme)")
print("  • Códigos cambian aleatoriamente sin patrón")
print("  • Zonas visualmente iguales usan códigos muy diferentes")
print("\n📈 HISTOGRAMA (2da fila, 3ra columna):")
print("  • Muestra cuántas veces se usa cada código")
print("  • Distribución balanceada = buen uso del codebook")
print("  • Pocos códigos dominantes = colapso parcial")
print("\n💡 PRÓXIMO PASO:")
print("  Si el mapa muestra patrones coherentes → CONTINUAR con detección")
print("  Si el mapa es aleatorio/uniforme → MEJORAR VQ-VAE primero")
print("=" * 80)
