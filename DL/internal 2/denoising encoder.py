import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = tf.keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train, epochs=10, batch_size=128, shuffle=True, validation_data=(x_test_noisy, x_test))
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
for i in range(n):
    plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")
    
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")
plt.show()

"""
================================================================================
DENOISING CONVOLUTIONAL AUTOENCODER ANALYSIS ON MNIST DATASET
================================================================================

1. ARCHITECTURE
================================================================================
Input Layer: 28×28×1 (Grayscale image with noise)
↓
===================== ENCODER =====================
Conv2D: 32 filters, 3×3 kernel, ReLU, padding='same' → Output: 28×28×32
↓
MaxPooling2D: 2×2 pool, padding='same' → Output: 14×14×32
↓
Conv2D: 32 filters, 3×3 kernel, ReLU, padding='same' → Output: 14×14×32
↓
MaxPooling2D: 2×2 pool, padding='same' → Output: 7×7×32 (Bottleneck)
↓
===================== DECODER =====================
Conv2D: 32 filters, 3×3 kernel, ReLU, padding='same' → Output: 7×7×32
↓
UpSampling2D: 2×2 upsampling → Output: 14×14×32
↓
Conv2D: 32 filters, 3×3 kernel, ReLU, padding='same' → Output: 14×14×32
↓
UpSampling2D: 2×2 upsampling → Output: 28×28×32
↓
Conv2D: 1 filter, 3×3 kernel, Sigmoid, padding='same' → Output: 28×28×1

Shape progression:
(28,28,1) → (28,28,32) → (14,14,32) → (14,14,32) → (7,7,32) [Bottleneck]
→ (7,7,32) → (14,14,32) → (14,14,32) → (28,28,32) → (28,28,1)

2. NUMBER OF LEARNING PARAMETERS (WEIGHTS & BIAS)
================================================================================
Layer                                    | Weights                    | Bias    | Total
-----------------------------------------|----------------------------|---------|------------
Conv2D (32×3×3×1)                        | 32×3×3×1 = 288             | 32      | 320
Conv2D (32×3×3×32)                       | 32×3×3×32 = 9,216          | 32      | 9,248
Conv2D (32×3×3×32) - Decoder             | 32×3×3×32 = 9,216          | 32      | 9,248
Conv2D (32×3×3×32) - Decoder             | 32×3×3×32 = 9,216          | 32      | 9,248
Conv2D (1×3×3×32) - Output               | 1×3×3×32 = 288             | 1       | 289
TOTAL                                    | 28,224                     | 129     | 28,353

3. LOSS CALCULATION FORMULA
================================================================================
Binary Cross-Entropy (Reconstruction Loss for denoising):

For batch: Loss = -1/N Σ Σ [x_ij × log(x̂_ij) + (1-x_ij) × log(1-x̂_ij)]
                    i=1..N j=1..784

where:
- N = batch size (128)
- x = clean original image (target)
- x̂ = denoised output image (prediction)
- 784 = total pixels (28×28)

Why Binary Cross-Entropy?
- Pixels normalized to [0,1]
- Treats each pixel as binary classification
- Sigmoid activation ensures output between 0 and 1

4. WEIGHT UPDATE FORMULA
================================================================================
Adam Optimizer:

θ(t+1) = θ(t) - (η / (√v̂(t) + ε)) × m̂(t)

where:
- θ(t) = weights/biases at time step t
- η = learning rate (default 0.001)
- m̂(t) = bias-corrected first moment (mean of gradients)
- v̂(t) = bias-corrected second moment (uncentered variance)
- ε = 10⁻⁷

5. SAMPLES USED FOR TRAINING, VALIDATION, TESTING
================================================================================
Training Samples: 60,000 (MNIST training images with added noise)
Validation Samples: 10,000 (MNIST test images with added noise)
Testing/Inference Samples: 10,000 (same test set for denoising prediction)
Visualization: 10 noisy + 10 denoised images displayed

Important distinction:
- Input (x_train_noisy): Images with Gaussian noise
- Target (x_train): Clean original images
- Model learns to map noisy → clean

6. LABELED OR UNLABELED DATA?
================================================================================
Unlabeled Data (Self-Supervised) ✓

Characteristics:
- No digit labels used (underscore _ ignores labels)
- Training pairs: (noisy_image, clean_image)
- Model learns denoising without knowing what digit it is
- This is self-supervised learning (pseudo-labels from data itself)

Contrast with Supervised:
- Supervised: requires class labels (0-9)
- Denoising Autoencoder: only needs clean targets (same image without noise)

================================================================================
NOISE ADDITION DETAILS
================================================================================
Noise Type: Gaussian Noise
Noise Factor: 0.5 (standard deviation multiplier)
Mean (loc): 0.0
Scale: 1.0

Formula: x_noisy = x_clean + 0.5 × N(0, 1)

Noise characteristics:
- Random noise added to every pixel independently
- Values clipped to [0,1] after addition
- Creates corrupted versions while preserving digit structure

================================================================================
ARCHITECTURE CHOICES EXPLAINED
================================================================================
1. Convolutional layers instead of Dense:
   - Preserves spatial structure of digits
   - Fewer parameters than Dense autoencoder
   - Better for image data

2. Padding='same':
   - Preserves spatial dimensions after convolution
   - Makes upsampling easier

3. MaxPooling for downsampling:
   - Reduces spatial dimensions
   - Creates bottleneck (7×7×32 = 1568 → compresses from 784)

4. UpSampling2D for upsampling:
   - Increases spatial dimensions
   - Simpler than transposed convolution

5. ReLU activation in hidden layers:
   - Non-linear transformations
   - Helps learn complex patterns

6. Sigmoid in output layer:
   - Restricts pixel values to [0,1]
   - Matches input normalization

================================================================================
WHAT THE MODEL LEARNS
================================================================================
- Removes Gaussian noise while preserving digit structure
- Understands what a "clean" digit looks like
- Learns common patterns in digits (strokes, curves, holes)
- Can reconstruct missing or corrupted pixels

================================================================================
DENOISING PROCESS
================================================================================
Input: Noisy image (barely recognizable digit)
↓
Encoder: Compresses to latent representation (7×7×32)
↓
Bottleneck: Forces learning of essential features
↓
Decoder: Reconstructs clean image from compressed representation
↓
Output: Denoised image (clear, recognizable digit)

================================================================================
COMPARISON WITH SIMPLE AUTOENCODER
================================================================================
Feature                 | Simple Autoencoder | Denoising Conv Autoencoder
------------------------|--------------------|----------------------------
Architecture            | Dense layers       | Convolutional layers
Input shape             | 784 (flattened)    | 28×28×1 (spatial)
Parameters              | 50,992             | 28,353 (fewer!)
Training data           | Clean → Clean      | Noisy → Clean
Purpose                 | Compression        | Denoising + Compression
Spatial structure       | Lost (flattened)   | Preserved

================================================================================
WHY DENOISING WORKS
================================================================================
1. Model cannot simply memorize noise (noise is random each epoch)
2. Forced to learn underlying data distribution
3. Learns to separate signal (digit) from noise
4. Generalizes to unseen noisy images

================================================================================
TRAINING DETAILS
================================================================================
- Epochs: 10
- Batch size: 128
- Loss: Binary Cross-Entropy
- Optimizer: Adam
- Shuffle: True (randomizes order)
- Validation: Monitors performance on unseen noisy test digits

================================================================================
EXPECTED RESULTS
================================================================================
- Noisy images: Grainy, hard to recognize digits
- Denoised images: Clean, smooth, recognizable digits
- Some fine details may be lost (trade-off)
- Background noise effectively removed

================================================================================
APPLICATIONS
================================================================================
1. Medical Imaging (CT scans, X-rays noise removal)
2. Photography (low-light image enhancement)
3. Document Processing (scanned text cleaning)
4. Video Denoising
5. Audio Denoising (adapted for 1D signals)
6. Preprocessing for other computer vision tasks
"""