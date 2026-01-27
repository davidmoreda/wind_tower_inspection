# Wind Tower Inspection - AE-TSCM + VQVAE

Sistema de fusión multi-iluminación para detección de defectos superficiales en torres eólicas.

## Arquitectura

```
5 Imágenes (luces) → AE-TSCM → RGB Fusionado → VQVAE → 5 luces reconstruidas
                        ↓
         Channel Attention (SE-Net) + Spatial Attention (CBAM)
         + Taylor Series Transform
```

**Componentes clave:**
- **AE-TSCM**: Attention-Enhanced Taylor Series Channel Mixer — fusiona 5 iluminaciones en 3 canales RGB
- **VQVAE**: Vector Quantized VAE con codebook EMA — comprime y reconstruye las 5 luces originales
- **Codebook Reset**: Reemplaza entradas no usadas del codebook para evitar collapse

## Estructura

```
wind_tower_inspection/
├── train_ae_tscm_colab_v6.ipynb  # Notebook principal (Colab A100)
├── models/
│   ├── __init__.py
│   ├── ae_tscm.py                # AE-TSCM (módulo local)
│   └── vqvae.py                  # VQVAE (módulo local)
├── dataset.py                    # Cargador de datos MT001
├── train.py                      # Entrenamiento local (CLI)
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

Formato MT001: 62 capturas × 5 condiciones de iluminación = 310 imágenes.

```
dataset/dataset/MT001/
├── metadata.csv
├── captura_001_SUP_IZQ.jpg    # 5472×3072 px
├── captura_001_SUP_DER.jpg
├── captura_001_INF_DER.jpg
├── captura_001_INF_IZQ.jpg
├── captura_001_ALL.jpg
├── captura_002_SUP_IZQ.jpg
└── ...
```

Las 5 posiciones de luz: `SUP_IZQ`, `SUP_DER`, `INF_DER`, `INF_IZQ`, `ALL`

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Google Colab (recomendado — requiere A100)

1. Sube la carpeta `MT001/` a Google Drive
2. Abre `train_ae_tscm_colab_v6.ipynb` en Colab
3. Selecciona runtime A100
4. Ajusta `DATASET_DIR` a la ruta de tu Drive
5. Ejecuta todas las celdas

### Local (CLI)

```bash
python train.py --data_dir ./dataset/dataset/MT001 --epochs 150
```

## Configuración v6

| Parámetro | Valor |
|---|---|
| hidden_channels | 128 |
| num_residual | 4 |
| num_embeddings | 256 |
| embedding_dim | 128 |
| Codebook update | EMA (decay=0.99) |
| Codebook reset | Cada 20 épocas |
| Learning rate | 1e-3 (constante) |
| Batch size | 1 + grad accum 4 |
| Mixed precision | AMP (float16) |

## Resultados

Los resultados se guardan en Google Drive (`runs/ae_tscm_v6_TIMESTAMP/`):
- `checkpoints/` — modelos guardados cada 25 épocas
- `visualizations/` — comparaciones original vs reconstrucción
- `loss_history.png` — gráfico de losses + perplexity
- `attention_analysis.png` — mapas de atención
- `final_model.pt` — modelo final
