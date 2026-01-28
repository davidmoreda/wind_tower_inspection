#!/bin/bash
# Setup script para verificar e instalar dependencias para v7

echo "======================================================================="
echo "VQ-VAE v7.0 - Verificación de Dependencias"
echo "======================================================================="

# Verificar que estamos en el directorio correcto
if [ ! -f "train_ae_tscm_v7_fixed.py" ]; then
    echo "❌ Error: No se encuentra train_ae_tscm_v7_fixed.py"
    echo "   Ejecuta este script desde: wind_tower_inspection/"
    exit 1
fi

echo -e "\n1. Verificando Python..."
python3 --version || { echo "❌ Python3 no encontrado"; exit 1; }

echo -e "\n2. Verificando PyTorch..."
python3 -c "import torch; print(f'   PyTorch {torch.__version__}')" || {
    echo "❌ PyTorch no instalado"
    exit 1
}

echo -e "\n3. Verificando CUDA..."
python3 -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo -e "\n4. Verificando dependencias..."

# Lista de paquetes requeridos
PACKAGES=(
    "pytorch-msssim"
    "tqdm"
    "Pillow"
    "matplotlib"
)

MISSING=()

for pkg in "${PACKAGES[@]}"; do
    python3 -c "import ${pkg//-/_}" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "   ✓ $pkg"
    else
        echo "   ✗ $pkg (falta)"
        MISSING+=("$pkg")
    fi
done

# Instalar faltantes
if [ ${#MISSING[@]} -gt 0 ]; then
    echo -e "\n5. Instalando paquetes faltantes..."
    for pkg in "${MISSING[@]}"; do
        echo "   Instalando $pkg..."
        pip install "$pkg" -q
    done
    echo "   ✓ Instalación completada"
else
    echo -e "\n   ✓ Todas las dependencias están instaladas"
fi

echo -e "\n6. Verificando dataset..."
DATASET_PATH="dataset/dataset/MT001"
if [ -d "$DATASET_PATH" ]; then
    echo "   ✓ Dataset encontrado en $DATASET_PATH"
    
    if [ -f "$DATASET_PATH/metadata.csv" ]; then
        NUM_LINES=$(wc -l < "$DATASET_PATH/metadata.csv")
        echo "   ✓ metadata.csv ($NUM_LINES líneas)"
    else
        echo "   ✗ metadata.csv no encontrado"
    fi
else
    echo "   ✗ Dataset no encontrado en $DATASET_PATH"
    echo "     Por favor, ajusta DATASET_DIR en train_ae_tscm_v7_fixed.py"
fi

echo -e "\n7. Verificando GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'   ✓ GPU: {torch.cuda.get_device_name(0)}')
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'   ✓ Memoria: {mem_gb:.1f} GB')
    if mem_gb < 8:
        print('   ⚠️  ADVERTENCIA: Memoria GPU < 8GB, puede haber OOM')
else:
    print('   ✗ No hay GPU disponible (CUDA)')
    print('   ⚠️  El entrenamiento será MUY lento en CPU')
"

echo -e "\n======================================================================="
echo "✅ Setup completado"
echo ""
echo "Próximos pasos:"
echo "  1. Test rápido:     python3 test_v7_no_nan.py"
echo "  2. Entrenamiento:   python3 train_ae_tscm_v7_fixed.py"
echo "======================================================================="
