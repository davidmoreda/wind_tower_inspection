"""
================================================================================
VQVAE: Vector Quantized Variational Autoencoder
================================================================================

Este módulo implementa un VQVAE simplificado para reconstruir las 5 imágenes
originales (5 luces) a partir de la representación RGB comprimida.

El VQVAE tiene tres componentes principales:
1. Encoder: comprime la imagen RGB a un espacio latente
2. Vector Quantizer: discretiza el espacio latente (como un "vocabulario" visual)
3. Decoder: reconstruye las 5 imágenes originales desde el latente

¿Por qué usar VQVAE?
- La cuantización vectorial actúa como regularización
- Aprende un "diccionario" de patrones visuales
- Mejora la generalización comparado con autoencoders simples

Referencia: "Neural Discrete Representation Learning" (van den Oord et al., 2017)

Autor: [Tu nombre]
Fecha: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# BLOQUE 1: Bloque Residual
# ==============================================================================
#
# Los bloques residuales permiten entrenar redes más profundas.
# La idea es simple: en lugar de aprender f(x), aprendemos f(x) + x
# Esto facilita el flujo del gradiente durante el entrenamiento.
#
# Referencia: "Deep Residual Learning" (He et al., 2015)
# ==============================================================================

class ResidualBlock(nn.Module):
    """
    Bloque residual básico: output = input + transform(input)
    """

    def __init__(self, channels):
        """
        Args:
            channels: número de canales (entrada = salida)
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


# ==============================================================================
# BLOQUE 2: Encoder
# ==============================================================================
#
# El encoder comprime la imagen de entrada a un espacio latente de menor
# resolución pero más canales.
#
# Arquitectura (para imagen 256x256):
#   Input: (3, 256, 256)
#   → Conv + downsample: (64, 128, 128)
#   → Conv + downsample: (128, 64, 64)
#   → ResBlocks: (128, 64, 64)
#   → Conv to latent: (embedding_dim, 64, 64)
#
# El factor de reducción es 4x en cada dimensión espacial.
# ==============================================================================

class Encoder(nn.Module):
    """
    Encoder del VQVAE.
    Comprime imagen RGB a espacio latente.
    """

    def __init__(self, in_channels=3, hidden_channels=64, embedding_dim=64, num_residual=2):
        """
        Args:
            in_channels: canales de entrada (3 para RGB)
            hidden_channels: canales en capas intermedias
            embedding_dim: dimensión del espacio latente
            num_residual: número de bloques residuales
        """
        super().__init__()

        # Capas de downsampling (reducen resolución a la mitad cada una)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Bloques residuales
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels * 2) for _ in range(num_residual)]
        )

        # Proyección al espacio latente
        self.to_latent = nn.Conv2d(hidden_channels * 2, embedding_dim, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: imagen de entrada (batch, 3, H, W)

        Returns:
            latent: representación latente (batch, embedding_dim, H/4, W/4)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual_blocks(x)
        x = self.to_latent(x)
        return x


# ==============================================================================
# BLOQUE 3: Vector Quantizer
# ==============================================================================
#
# Este es el corazón del VQVAE. Discretiza el espacio latente usando un
# "codebook" (diccionario) de vectores aprendidos.
#
# Funcionamiento:
#   1. Para cada vector latente, encuentra el vector más cercano en el codebook
#   2. Reemplaza el vector latente por el del codebook
#   3. Usa el "straight-through estimator" para permitir backpropagation
#
# El codebook actúa como un vocabulario visual: cada "palabra" representa
# un patrón visual que la red ha aprendido.
#
# Losses:
#   - VQ loss: acerca los vectores del codebook a los latentes
#   - Commitment loss: acerca los latentes a los vectores del codebook
# ==============================================================================

class VectorQuantizer(nn.Module):
    """
    Cuantizador vectorial con codebook aprendible.
    """

    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        """
        Args:
            num_embeddings: tamaño del codebook (número de "palabras" visuales)
            embedding_dim: dimensión de cada vector del codebook
            commitment_cost: peso del commitment loss
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook: matriz de num_embeddings vectores de dimensión embedding_dim
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)

        # Inicialización uniforme
        self.codebook.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        """
        Args:
            z: latente continuo (batch, embedding_dim, H, W)

        Returns:
            quantized: latente cuantizado (batch, embedding_dim, H, W)
            vq_loss: pérdida de cuantización
            perplexity: medida de uso del codebook (mayor = más diverso)
        """
        # Reorganizar: (B, D, H, W) → (B, H, W, D) → (B*H*W, D)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_shape = z.shape
        z_flat = z.view(-1, self.embedding_dim)

        # Calcular distancias a cada vector del codebook
        # Usando: ||z - e||² = ||z||² + ||e||² - 2*z·e
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) +
            torch.sum(self.codebook.weight ** 2, dim=1) -
            2 * torch.matmul(z_flat, self.codebook.weight.t())
        )

        # Encontrar el índice del vector más cercano
        encoding_indices = torch.argmin(distances, dim=1)

        # Obtener los vectores del codebook correspondientes
        quantized_flat = self.codebook(encoding_indices)
        quantized = quantized_flat.view(z_shape)

        # Calcular losses
        # VQ loss: acerca codebook a los latentes
        vq_loss = F.mse_loss(quantized, z.detach())
        # Commitment loss: acerca latentes al codebook
        commitment_loss = F.mse_loss(z, quantized.detach())
        # Loss total
        loss = vq_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copia el gradiente
        quantized = z + (quantized - z).detach()

        # Calcular perplexity (medida de diversidad del codebook)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Reorganizar de vuelta: (B, H, W, D) → (B, D, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, loss, perplexity


# ==============================================================================
# BLOQUE 4: Decoder
# ==============================================================================
#
# El decoder reconstruye las 5 imágenes originales (5 luces) a partir
# del espacio latente cuantizado.
#
# Arquitectura (para imagen 256x256):
#   Input: (embedding_dim, 64, 64)
#   → Conv from latent: (128, 64, 64)
#   → ResBlocks: (128, 64, 64)
#   → Upsample: (64, 128, 128)
#   → Upsample: (5, 256, 256)
#
# Usa ConvTranspose2d para hacer upsampling aprendible.
# ==============================================================================

class Decoder(nn.Module):
    """
    Decoder del VQVAE.
    Reconstruye las 5 imágenes de luz desde el espacio latente.
    """

    def __init__(self, out_channels=5, hidden_channels=64, embedding_dim=64, num_residual=2):
        """
        Args:
            out_channels: canales de salida (5 para las 5 luces)
            hidden_channels: canales en capas intermedias
            embedding_dim: dimensión del espacio latente
            num_residual: número de bloques residuales
        """
        super().__init__()

        # Proyección desde el espacio latente
        self.from_latent = nn.Conv2d(embedding_dim, hidden_channels * 2, kernel_size=1)

        # Bloques residuales
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels * 2) for _ in range(num_residual)]
        )

        # Capas de upsampling (aumentan resolución al doble cada una)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Salida en rango [0, 1]
        )

    def forward(self, z):
        """
        Args:
            z: latente cuantizado (batch, embedding_dim, H/4, W/4)

        Returns:
            reconstructed: imágenes reconstruidas (batch, 5, H, W)
        """
        x = self.from_latent(z)
        x = self.residual_blocks(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x


# ==============================================================================
# BLOQUE 5: VQVAE Completo
# ==============================================================================
#
# Combina Encoder, VectorQuantizer y Decoder en un solo módulo.
#
# Flujo:
#   RGB (3, H, W) → Encoder → Latente (D, H/4, W/4)
#                           → VQ → Latente cuantizado
#                                → Decoder → 5 luces (5, H, W)
# ==============================================================================

class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder.

    Comprime imagen RGB y reconstruye las 5 imágenes de luz originales.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=5,
        hidden_channels=64,
        num_residual=2,
        num_embeddings=512,
        embedding_dim=64,
        commitment_cost=0.25
    ):
        """
        Args:
            in_channels: canales de entrada (3 para RGB del AE-TSCM)
            out_channels: canales de salida (5 para las 5 luces)
            hidden_channels: canales en capas intermedias
            num_residual: número de bloques residuales
            num_embeddings: tamaño del codebook
            embedding_dim: dimensión de los vectores del codebook
            commitment_cost: peso del commitment loss
        """
        super().__init__()

        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim, num_residual)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(out_channels, hidden_channels, embedding_dim, num_residual)

    def forward(self, x):
        """
        Args:
            x: imagen RGB (batch, 3, H, W)

        Returns:
            reconstructed: 5 imágenes reconstruidas (batch, 5, H, W)
            vq_loss: pérdida de cuantización vectorial
            perplexity: medida de uso del codebook
        """
        # Encoder
        z = self.encoder(x)

        # Vector Quantization
        z_quantized, vq_loss, perplexity = self.vq(z)

        # Decoder
        reconstructed = self.decoder(z_quantized)

        return reconstructed, vq_loss, perplexity

    def encode(self, x):
        """Obtiene solo la representación latente cuantizada."""
        z = self.encoder(x)
        z_quantized, _, _ = self.vq(z)
        return z_quantized

    def decode(self, z):
        """Decodifica desde representación latente."""
        return self.decoder(z)


# ==============================================================================
# TEST: Verificar que todo funciona
# ==============================================================================

if __name__ == "__main__":
    # Crear modelo
    model = VQVAE(in_channels=3, out_channels=5)

    # Input simulado: batch de 2, RGB, 256x256
    x = torch.rand(2, 3, 256, 256)

    # Forward pass
    reconstructed, vq_loss, perplexity = model(x)

    print("=" * 60)
    print("TEST VQVAE")
    print("=" * 60)
    print(f"Input shape:         {x.shape}")
    print(f"Output shape:        {reconstructed.shape}")
    print(f"VQ Loss:             {vq_loss.item():.4f}")
    print(f"Perplexity:          {perplexity.item():.1f}")
    print(f"Codebook size:       {model.vq.num_embeddings}")
    print(f"Output range:        [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    print("=" * 60)
    print("Test PASSED!")
