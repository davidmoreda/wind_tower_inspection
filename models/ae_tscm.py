"""
================================================================================
AE-TSCM: Attention-Enhanced Taylor Series Channel Mixer
================================================================================

Este módulo combina dos técnicas probadas:
1. TSCM (Taylor Series Channel Mixer) - del paper original
2. Channel & Spatial Attention - de SE-Net y CBAM

La idea es mejorar el TSCM original añadiendo mecanismos de atención que
permitan a la red aprender qué luces son más importantes para cada tipo
de defecto y qué regiones espaciales son más relevantes.

Arquitectura:
    5 imágenes (luces) → Taylor Transform → Channel Attention → Spatial Attention → RGB

Autor: [Tu nombre]
Fecha: 2024
"""

import torch
import torch.nn as nn


# ==============================================================================
# BLOQUE 1: Channel Attention (Squeeze-and-Excitation)
# ==============================================================================
#
# Este bloque aprende a ponderar la importancia de cada canal (cada luz).
# Funciona así:
#   1. Global Average Pooling: reduce cada canal a un solo valor (su "resumen")
#   2. MLP: procesa estos valores para aprender relaciones entre canales
#   3. Sigmoid: genera pesos entre 0 y 1 para cada canal
#
# Ejemplo: si el scratch se ve mejor con luz tangencial, el attention
# aprenderá a dar más peso a ese canal cuando detecte un scratch.
#
# Referencia: "Squeeze-and-Excitation Networks" (Hu et al., 2018)
# ==============================================================================

class ChannelAttention(nn.Module):
    """
    Módulo de atención por canal.
    Aprende qué canales (luces) son más importantes para la tarea.
    """

    def __init__(self, num_channels, reduction=2):
        """
        Args:
            num_channels: número de canales de entrada (5 para tus 5 luces)
            reduction: factor de reducción en el MLP (menor = más parámetros)
        """
        super().__init__()

        hidden_dim = max(num_channels // reduction, 4)

        self.attention = nn.Sequential(
            # Paso 1: Global Average Pooling (se hace en forward)
            # Paso 2: MLP para aprender relaciones entre canales
            nn.Linear(num_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_channels),
            # Paso 3: Sigmoid para obtener pesos entre 0 y 1
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: tensor de forma (batch, canales, alto, ancho)

        Returns:
            weights: tensor de forma (batch, canales, 1, 1) con los pesos
        """
        batch_size, num_channels, _, _ = x.shape

        # Global Average Pooling: (B, C, H, W) → (B, C)
        gap = x.mean(dim=[2, 3])

        # Calcular pesos de atención: (B, C) → (B, C)
        weights = self.attention(gap)

        # Reshape para multiplicar: (B, C) → (B, C, 1, 1)
        weights = weights.view(batch_size, num_channels, 1, 1)

        return weights


# ==============================================================================
# BLOQUE 2: Spatial Attention
# ==============================================================================
#
# Este bloque aprende qué regiones espaciales de la imagen son más importantes.
# Funciona así:
#   1. Calcula el promedio y máximo a lo largo de los canales
#   2. Concatena ambos mapas (2 canales)
#   3. Convolución 7x7 para generar mapa de atención espacial
#   4. Sigmoid para obtener pesos entre 0 y 1
#
# Ejemplo: si hay un defecto en el centro de la imagen, el attention
# aprenderá a dar más peso a esa región.
#
# Referencia: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
# ==============================================================================

class SpatialAttention(nn.Module):
    """
    Módulo de atención espacial.
    Aprende qué regiones de la imagen son más importantes.
    """

    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size: tamaño del kernel de convolución (7 es el estándar)
        """
        super().__init__()

        # Padding para mantener el mismo tamaño espacial
        padding = kernel_size // 2

        self.conv = nn.Sequential(
            # Entrada: 2 canales (avg + max), Salida: 1 canal (mapa de atención)
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: tensor de forma (batch, canales, alto, ancho)

        Returns:
            attention_map: tensor de forma (batch, 1, alto, ancho)
        """
        # Promedio a lo largo de canales: (B, C, H, W) → (B, 1, H, W)
        avg_out = x.mean(dim=1, keepdim=True)

        # Máximo a lo largo de canales: (B, C, H, W) → (B, 1, H, W)
        max_out, _ = x.max(dim=1, keepdim=True)

        # Concatenar: (B, 2, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)

        # Convolución para generar mapa de atención: (B, 1, H, W)
        attention_map = self.conv(combined)

        return attention_map


# ==============================================================================
# BLOQUE 3: Taylor Series Transform
# ==============================================================================
#
# Aplica una transformación polinómica (serie de Taylor) a los valores de pixel.
# La serie de Taylor permite aproximar cualquier función suave:
#
#   f(x) = c₀ + c₁·x + c₂·x²/2! + c₃·x³/3! + c₄·x⁴/4!
#
# Los coeficientes c₀, c₁, c₂, c₃, c₄ son APRENDIBLES, lo que permite
# a la red ajustar la transformación de intensidad óptima.
#
# Inicialización: c = [0, 1, 0, 0, 0] → f(x) = x (identidad)
# La red aprende a modificar estos coeficientes durante el entrenamiento.
#
# Referencia: Paper original del repositorio Metal-Surface-Defect-Detection
# ==============================================================================

class TaylorTransform(nn.Module):
    """
    Transformación de Taylor con coeficientes aprendibles.
    Permite transformaciones no lineales de intensidad.
    """

    def __init__(self, num_coefficients=5):
        """
        Args:
            num_coefficients: número de términos de Taylor (5 es suficiente)
        """
        super().__init__()

        # Coeficientes aprendibles, inicializados como identidad
        # [0, 1, 0, 0, 0] significa f(x) = x al inicio
        initial_coeffs = torch.zeros(num_coefficients)
        initial_coeffs[1] = 1.0  # c₁ = 1 para que f(x) = x inicialmente

        self.coefficients = nn.Parameter(initial_coeffs)

    def forward(self, x):
        """
        Args:
            x: tensor de cualquier forma, valores típicamente en [0, 1]

        Returns:
            tensor transformado de la misma forma
        """
        c = self.coefficients

        # Serie de Taylor: f(x) = c₀ + c₁x + c₂x²/2! + c₃x³/3! + c₄x⁴/4!
        # Los factoriales (2!, 3!, 4!) = (2, 6, 24) ayudan a la estabilidad numérica

        output = (c[0] +
                  c[1] * x +
                  c[2] * (x ** 2) / 2 +
                  c[3] * (x ** 3) / 6 +
                  c[4] * (x ** 4) / 24)

        return output


# ==============================================================================
# BLOQUE 4: Channel Mixer
# ==============================================================================
#
# Combina los 5 canales (luces) en 3 canales (RGB) usando una convolución 1x1.
# Cada pixel de salida es una combinación lineal de los 5 canales de entrada:
#
#   out_r = w₀·in₀ + w₁·in₁ + w₂·in₂ + w₃·in₃ + w₄·in₄ + bias
#   out_g = ...
#   out_b = ...
#
# Los pesos w se aprenden durante el entrenamiento.
# ==============================================================================

class ChannelMixer(nn.Module):
    """
    Mezcla N canales de entrada en M canales de salida.
    Usa convolución 1x1 (equivalente a combinación lineal por pixel).
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: número de canales de entrada (5)
            out_channels: número de canales de salida (3 para RGB)
        """
        super().__init__()

        self.mixer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.mixer(x)


# ==============================================================================
# BLOQUE 5: AE-TSCM Completo
# ==============================================================================
#
# Este es el módulo principal que combina todo:
#
#   1. Taylor Transform: transforma las intensidades de manera no lineal
#   2. Channel Attention: pondera la importancia de cada luz
#   3. Spatial Attention: pondera la importancia de cada región
#   4. Channel Mixer: combina 5 canales en 3 (RGB)
#
# El flujo es:
#   Input (B, 5, H, W)
#       → Taylor Transform
#       → Channel Attention (multiplicar por pesos de canal)
#       → Spatial Attention (multiplicar por mapa espacial)
#       → Channel Mixer
#   Output (B, 3, H, W)
# ==============================================================================

class AE_TSCM(nn.Module):
    """
    Attention-Enhanced Taylor Series Channel Mixer.

    Combina TSCM original con mecanismos de atención para mejorar
    la fusión de múltiples iluminaciones.
    """

    def __init__(self, in_channels=5, out_channels=3, use_spatial_attention=True):
        """
        Args:
            in_channels: número de canales de entrada (5 luces)
            out_channels: número de canales de salida (3 para RGB)
            use_spatial_attention: si usar atención espacial además de canal
        """
        super().__init__()

        self.use_spatial_attention = use_spatial_attention

        # Módulos del pipeline
        self.taylor = TaylorTransform(num_coefficients=5)
        self.channel_attention = ChannelAttention(in_channels)

        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(kernel_size=7)

        self.mixer = ChannelMixer(in_channels, out_channels)

        # Normalización final
        self.normalize = ChannelNormalize()

    def forward(self, x, return_attention=False):
        """
        Args:
            x: tensor de forma (batch, 5, alto, ancho) con las 5 imágenes
            return_attention: si devolver los mapas de atención para visualización

        Returns:
            output: tensor de forma (batch, 3, alto, ancho) - imagen RGB fusionada
            (opcional) channel_weights: pesos de atención por canal
            (opcional) spatial_weights: mapa de atención espacial
        """
        # Paso 1: Transformación de Taylor
        x = self.taylor(x)

        # Paso 2: Channel Attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights

        # Paso 3: Spatial Attention (opcional)
        spatial_weights = None
        if self.use_spatial_attention:
            spatial_weights = self.spatial_attention(x)
            x = x * spatial_weights

        # Paso 4: Mezclar canales (5 → 3)
        x = self.mixer(x)

        # Paso 5: Normalizar a rango [0, 1]
        x = self.normalize(x)

        if return_attention:
            return x, channel_weights.squeeze(), spatial_weights

        return x


# ==============================================================================
# BLOQUE 6: Normalización por Canal
# ==============================================================================
#
# Normaliza cada canal independientemente al rango [0, 1].
# Esto es importante para que la salida sea una imagen válida.
# ==============================================================================

class ChannelNormalize(nn.Module):
    """
    Normaliza cada canal al rango [0, 1].
    """

    def forward(self, x):
        """
        Args:
            x: tensor de forma (batch, canales, alto, ancho)

        Returns:
            tensor normalizado al rango [0, 1]
        """
        batch, channels, height, width = x.shape

        # Reshape para calcular min/max por canal: (B, C, H*W)
        x_flat = x.view(batch, channels, -1)

        # Mínimo y máximo por canal
        min_vals = x_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)  # (B, C, 1, 1)
        max_vals = x_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)  # (B, C, 1, 1)

        # Normalizar, evitando división por cero
        x = (x - min_vals) / (max_vals - min_vals + 1e-8)

        return x


# ==============================================================================
# TEST: Verificar que todo funciona
# ==============================================================================

if __name__ == "__main__":
    # Crear modelo
    model = AE_TSCM(in_channels=5, out_channels=3)

    # Input simulado: batch de 2 imágenes, 5 canales (luces), 256x256
    x = torch.rand(2, 5, 256, 256)

    # Forward pass
    output, ch_weights, sp_weights = model(x, return_attention=True)

    print("=" * 60)
    print("TEST AE-TSCM")
    print("=" * 60)
    print(f"Input shape:              {x.shape}")
    print(f"Output shape:             {output.shape}")
    print(f"Channel weights shape:    {ch_weights.shape}")
    print(f"Spatial weights shape:    {sp_weights.shape}")
    print(f"Output range:             [{output.min():.3f}, {output.max():.3f}]")
    print(f"Taylor coefficients:      {model.taylor.coefficients.data.numpy()}")
    print("=" * 60)
    print("Test PASSED!")
