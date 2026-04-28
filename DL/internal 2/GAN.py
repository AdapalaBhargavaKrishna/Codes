import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam

(X_train, _), _ = mnist.load_data()
X_train = (X_train - 127.5) / 127.5
X_train = X_train.reshape(-1, 784)

opt = Adam(0.0002, 0.5)

def build_generator():
    model = Sequential([
        Dense(128, input_dim=100),
        LeakyReLU(0.2),
        Dense(784, activation='tanh'),
    ])
    return model

def build_discriminator():
    model = Sequential([
        Dense(128, input_shape=(784,)),
        LeakyReLU(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

generator = build_generator()
discriminator = build_discriminator()

noise = np.random.normal(0, 1, (128, 100))
fake_images = generator.predict(noise)
real_images = X_train[np.random.randint(0, X_train.shape[0], 128)]

real_labels = np.ones((128, 1))
fake_labels = np.zeros((128, 1))

d_loss_real = discriminator.train_on_batch(real_images, real_labels)
d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)

print("Discriminator accuracy on real:", d_loss_real[1]*100)
print("Discriminator accuracy on fake:", d_loss_fake[1]*100)

"""
================================================================================
GAN (GENERATIVE ADVERSARIAL NETWORK) ANALYSIS ON MNIST DATASET
================================================================================

1. ARCHITECTURE
================================================================================

===================== GENERATOR NETWORK =====================
Input: Random noise vector (100 dimensions)
↓
Dense: 128 neurons → Output: 128
↓
LeakyReLU: alpha=0.2 (allows small negative values)
↓
Dense: 784 neurons, activation='tanh' → Output: 784 (28×28 image)
↓
Output: Generated fake image (pixels in range [-1, 1])

===================== DISCRIMINATOR NETWORK =====================
Input: Image (784 flattened pixels)
↓
Dense: 128 neurons → Output: 128
↓
LeakyReLU: alpha=0.2
↓
Dense: 1 neuron, activation='sigmoid' → Output: Probability (0=fake, 1=real)

2. NUMBER OF LEARNING PARAMETERS (WEIGHTS & BIAS)
================================================================================

GENERATOR:
Layer                    | Weights              | Bias    | Total
-------------------------|----------------------|---------|------------
Dense (100→128)          | 100×128 = 12,800     | 128     | 12,928
Dense (128→784)          | 128×784 = 100,352    | 784     | 101,136
TOTAL GENERATOR          | 113,152              | 912     | 114,064

DISCRIMINATOR:
Layer                    | Weights              | Bias    | Total
-------------------------|----------------------|---------|------------
Dense (784→128)          | 784×128 = 100,352    | 128     | 100,480
Dense (128→1)            | 128×1 = 128          | 1       | 129
TOTAL DISCRIMINATOR      | 100,480              | 129     | 100,609

GRAND TOTAL PARAMETERS: 114,064 + 100,609 = 214,673

3. LOSS CALCULATION FORMULA
================================================================================

Binary Cross-Entropy (for both Generator and Discriminator):

Discriminator Loss:
Loss_D = -1/N Σ [log(D(real)) + log(1 - D(fake))]
           i=1..N

Generator Loss:
Loss_G = -1/N Σ [log(D(fake))]  (Wants D(fake) → 1)

where:
- N = batch size (128)
- D(real) = discriminator output on real images (target = 1)
- D(fake) = discriminator output on fake images (target = 0 for D, 1 for G)

Adversarial training: Two networks competing:
- Discriminator: Maximize log(D(real)) + log(1 - D(fake))
- Generator: Minimize log(1 - D(fake)) OR maximize log(D(fake))

4. WEIGHT UPDATE FORMULA
================================================================================

Adam Optimizer (β1=0.5, β2=0.999, learning_rate=0.0002):

θ(t+1) = θ(t) - (η / (√v̂(t) + ε)) × m̂(t)

where:
- θ(t) = weights/biases at time step t
- η = learning rate (0.0002 - smaller than usual for GAN stability)
- m̂(t) = bias-corrected first moment (mean of gradients)
- v̂(t) = bias-corrected second moment (uncentered variance)
- ε = 10⁻⁷

Special GAN considerations:
- β1=0.5 (instead of default 0.9) for better GAN training
- Lower learning rate (0.0002) for stability

5. SAMPLES USED FOR TRAINING, VALIDATION, TESTING
================================================================================

Training Samples: 60,000 (MNIST training images)
Validation Samples: None (GANs typically don't use validation)
Testing/Inference Samples:
- 128 random noise vectors for this forward pass
- Real images: 128 randomly sampled from training set
- Fake images: 128 generated from noise

Total batch size for this single step: 128 real + 128 fake = 256 images

NOTE: This is ONLY ONE TRAINING STEP (not full training loop)
- No epochs implemented
- Only single batch processed

6. LABELED OR UNLABELED DATA?
================================================================================

Unlabeled Data (for Generator) but Pseudo-labeled for Discriminator ✓

Explanation:
- Generator: Uses only random noise (no labels) → Unsupervised
- Discriminator: Uses pseudo-labels (1 for real, 0 for fake) → Self-supervised
- Real images: No digit labels needed (only "real" vs "fake")
- The GAN learns the data distribution without class labels

This is UNSUPERVISED GENERATIVE MODELING:
- No MNIST digit labels (0-9) are used
- Discriminator only learns to distinguish real vs generated
- Generator learns to create realistic digits without knowing what digit it is

================================================================================
GAN COMPONENTS EXPLAINED
================================================================================

1. GENERATOR (The Artist):
   - Input: Random noise (100 dims) from Normal distribution
   - Output: 28×28 grayscale image in range [-1, 1]
   - Goal: Fool discriminator into thinking fake images are real
   - Architecture: Simple feedforward (not convolutional for simplicity)

2. DISCRIMINATOR (The Critic):
   - Input: Image (784 dims)
   - Output: Probability (0 = fake, 1 = real)
   - Goal: Correctly classify real vs fake images
   - Architecture: Simple binary classifier

3. LATENT SPACE (Noise Vector):
   - 100-dimensional random vector
   - Different noise → Different generated digits
   - Continuous space allows interpolation between digits

================================================================================
DATA PREPROCESSING
================================================================================
Normalization: (x - 127.5) / 127.5 → Range: [-1, 1]

Why [-1, 1] instead of [0,1]?
- Generator uses tanh activation (output range [-1, 1])
- Matches output range of generator
- Better for GAN training stability

================================================================================
ADVERSARIAL TRAINING DYNAMICS
================================================================================

Training Game:
1. Discriminator tries to maximize: log(D(real)) + log(1 - D(fake))
2. Generator tries to minimize: log(1 - D(fake)) OR maximize: log(D(fake))

Equilibrium: Nash equilibrium
- Generator produces perfect fakes
- Discriminator guesses randomly (50% accuracy)

Current step shows:
- Real accuracy: How well D identifies real images
- Fake accuracy: How well D identifies fake images (lower = better generator)

================================================================================
OPTIMIZER CHOICES FOR GANs
================================================================================
- Adam optimizer (standard for GANs)
- β1=0.5 (instead of default 0.9) - recommended for GANs
- Learning rate=0.0002 (lower than typical 0.001)
- Helps prevent mode collapse and training instability

================================================================================
LIMITATIONS OF THIS CODE
================================================================================
1. Only ONE training step (not full training)
2. No training loop (no epochs)
3. Generator and Discriminator not trained adversarially together
4. Missing: Combined GAN model (Discriminator frozen during Generator training)
5. No visualization of generated images

Complete GAN training would require:
- Full training loop (many epochs)
- Alternating training of D and G
- Combined model for Generator training
- Monitoring generated image quality

================================================================================
GAN APPLICATIONS
================================================================================
1. Image Generation (faces, digits, objects)
2. Image-to-Image Translation
3. Super-Resolution
4. Style Transfer
5. Data Augmentation
6. Anomaly Detection
7. Text-to-Image Synthesis

================================================================================
COMMON GAN ISSUES
================================================================================
1. Mode Collapse: Generator produces same output repeatedly
2. Non-convergence: Loss oscillates without stabilizing
3. Vanishing Gradients: Discriminator becomes too good
4. Instability: Training diverges

Solutions used here:
- LeakyReLU (prevents dead neurons)
- Adam with β1=0.5 (stability)
- Moderate learning rate (0.0002)
"""