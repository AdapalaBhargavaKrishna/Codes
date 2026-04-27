import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

encoding_dim = 32
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
encoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show()

"""
================================================================================
AUTOENCODER ANALYSIS ON MNIST DATASET (WITH VISUALIZATION)
================================================================================

1. ARCHITECTURE
================================================================================
Input Layer: 784 (flattened 28×28 image)
↓
Encoder: Dense(32, activation='relu') → Bottleneck/Code: 32
↓
Decoder: Dense(784, activation='sigmoid') → Output: 784 (reconstructed image)
↓
Full Autoencoder: Input(784) → Encoder(32) → Decoder(784) → Output(784)

Visual representation:
[784] → [32] → [784]
Input    Code   Output (Reconstruction)

2. NUMBER OF LEARNING PARAMETERS (WEIGHTS & BIAS)
================================================================================
Layer                    | Weights              | Bias    | Total
-------------------------|----------------------|---------|------------
Encoder (784→32)         | 784×32 = 25,088      | 32      | 25,120
Decoder (32→784)         | 32×784 = 25,088      | 784     | 25,872
TOTAL                    | 50,176               | 816     | 50,992

3. LOSS CALCULATION FORMULA
================================================================================
Binary Cross-Entropy (Reconstruction Loss):

For batch: Loss = -1/N Σ Σ [x_ij × log(x̂_ij) + (1-x_ij) × log(1-x̂_ij)]
                    i=1..N j=1..784

where:
- N = batch size (256)
- x = original input image
- x̂ = reconstructed output image
- 784 = number of pixels per image

Why Binary Cross-Entropy?
- MNIST pixels normalized to [0,1]
- Treats each pixel as independent binary classification
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
Training Samples: 60,000 (MNIST training images)
Validation Samples: 10,000 (MNIST test images used as validation)
Testing/Inference Samples: 10,000 (same test set for prediction)
Visualization Samples: 10 (first 10 test images shown in output)

Note: In autoencoders, both x_train and y_train are identical (input = target)
      Model learns to reconstruct input without using labels

6. LABELED OR UNLABELED DATA?
================================================================================
Unlabeled Data ✓

Key indicators:
- Labels are ignored (underscore _ in (x_train, _), (x_test, _))
- No y_train or y_test used in training
- Model learns from input data only (self-supervised)
- Target is same as input (reconstruction task)

Contrast with Supervised Learning:
- Supervised: input → label (e.g., image → digit 0-9)
- Autoencoder: input → reconstructed input (compression → decompression)

================================================================================
VISUALIZATION OUTPUT
================================================================================
The code displays 10 digits with:
- Top row: Original MNIST digits (28×28)
- Bottom row: Reconstructed digits from autoencoder

What to expect:
- Original: Sharp, clear digits
- Reconstructed: Slightly blurry but recognizable digits
- Quality depends on encoding_dim (32 = heavy compression)

================================================================================
AUTOENCODER COMPONENTS EXPLAINED
================================================================================
1. Encoder: Input → Code (compression)
   - Reduces 784 dimensions to 32 dimensions (24.5x compression)
   - Learns most important features of digits
   - ReLU activation for non-linearity

2. Bottleneck/Code: 32-dimensional vector
   - Compressed "essence" or "latent representation"
   - Forces model to learn only essential features
   - Cannot memorize all pixels due to bottleneck

3. Decoder: Code → Reconstruction
   - Expands 32 dimensions back to 784
   - Sigmoid activation (output between 0 and 1)
   - Reconstructs original image from compressed code

================================================================================
COMPRESSION RATIO
================================================================================
Input size: 784 (28×28)
Code size: 32
Compression ratio: 784/32 = 24.5x

Even with 24.5x compression, digits remain recognizable!

================================================================================
WHY AUTOENCODERS WORK
================================================================================
- MNIST digits have structure (not random pixels)
- Can be represented in lower-dimensional manifold
- Model learns this manifold through bottleneck
- Decoder learns to map from manifold back to image space

================================================================================
TRAINING DETAILS
================================================================================
- Epochs: 10 (passes through entire dataset)
- Batch size: 256 (images per gradient update)
- Loss: Binary Cross-Entropy (pixel-wise)
- Validation: Monitors reconstruction on unseen test digits
- Shuffle: Randomizes order each epoch

================================================================================
INTERPRETING RESULTS
================================================================================
Good reconstruction (what to look for):
- Digits clearly recognizable
- Correct shape preserved
- Background is black (0), digit is white (1)

Poor reconstruction indicators:
- Blurry or ghost-like digits
- Wrong digit shape
- Noisy background

================================================================================
USE CASES
================================================================================
1. Dimensionality Reduction (non-linear PCA)
2. Feature Extraction for downstream tasks
3. Anomaly Detection (high reconstruction error = anomaly)
4. Image Denoising (train noisy→clean)
5. Data Compression
6. Pretraining for deeper networks
"""