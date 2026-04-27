import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d salader/dogsvscats

import zipfile
zip_ref = zipfile.ZipFile('/content/dogsvscats.zip','r')
zip_ref.extractall('/content')
zip_ref.close()

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='/content/train', labels='inferred', label_mode='int',
    batch_size=32, image_size=(256,256))
test_ds = tf.keras.utils.image_dataset_from_directory(
    directory='/content/test', labels='inferred', label_mode='int',
    batch_size=32, image_size=(256,256))

def process(x, y):
    x = tf.cast(x / 255., tf.float32)
    return x, y
train_ds = train_ds.map(process)
test_ds = test_ds.map(process)

input_layer = Input(shape=(256,256,3))

vgg = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=input_layer)

for layer in vgg.layers:
    layer.trainable = False
for layer in resnet.layers:
    layer.trainable = False

vgg_features = GlobalAveragePooling2D()(vgg.output)
resnet_features = GlobalAveragePooling2D()(resnet.output)
merged = Concatenate()([vgg_features, resnet_features])

x = Dense(256, activation='relu')(merged)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, validation_data=test_ds, epochs=10, verbose=1)

print("Training Accuracy:", history.history['accuracy'][-1]*100)
print("Validation Accuracy:", history.history['val_accuracy'][-1]*100)

"""
================================================================================
ENSEMBLE MODEL (VGG16 + RESNET50) ANALYSIS ON DOGS VS CATS DATASET
================================================================================

1. ARCHITECTURE
================================================================================
Input Layer: 256×256×3 (RGB image)
↓
=================== TWO BRANCHES (PARALLEL) ===================
↓                              ↓
BRANCH 1: VGG16                BRANCH 2: ResNet50
- 13 Conv layers               - 50 layers with residual blocks
- 5 MaxPool layers             - BatchNorm after each conv
- include_top=False            - include_top=False
- Frozen (trainable=False)     - Frozen (trainable=False)
↓                              ↓
GlobalAveragePooling2D()       GlobalAveragePooling2D()
Output: 512                    Output: 2048
↓                              ↓
==================== MERGE LAYER ====================
Concatenate([vgg_features, resnet_features])
Output: 512 + 2048 = 2560
↓
Dense: 256 neurons, ReLU activation
↓
Dense: 128 neurons, ReLU activation
↓
Dense: 1 neuron, Sigmoid activation (Binary classification)

VGG16 Output Shape: 8×8×512 → GlobalAvgPool → 512
ResNet50 Output Shape: 8×8×2048 → GlobalAvgPool → 2048

2. NUMBER OF LEARNING PARAMETERS (WEIGHTS & BIAS)
================================================================================
VGG16 Base Model (Frozen): ~14,714,688 parameters (0 trainable)
ResNet50 Base Model (Frozen): ~23,587,712 parameters (0 trainable)

Custom Layers Added:
Layer                              | Weights                    | Bias    | Total
-----------------------------------|----------------------------|---------|------------
Dense (2560→256)                   | 2,560×256 = 655,360        | 256     | 655,616
Dense (256→128)                    | 256×128 = 32,768           | 128     | 32,896
Dense (128→1)                      | 128×1 = 128                | 1       | 129
TOTAL TRAINABLE                    | 688,256                    | 385     | 688,641

TOTAL FROZEN PARAMETERS: 38,302,400
TOTAL TRAINABLE PARAMETERS: 688,641
GRAND TOTAL: 38,991,041 parameters

3. LOSS CALCULATION FORMULA
================================================================================
Binary Cross-Entropy (Binary Classification):

For batch: Loss = -1/N Σ [y_i × log(p_i) + (1-y_i) × log(1-p_i)]

where:
- N = batch size (32)
- y_i = true label (0 for Cat, 1 for Dog)
- p_i = predicted probability from sigmoid

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

Note: Only custom Dense layers (256→128→1) are updated
      Both VGG16 and ResNet50 weights are completely frozen

5. SAMPLES USED FOR TRAINING, VALIDATION, TESTING
================================================================================
Training Samples: ~20,000 (dogs + cats from /content/train)
Validation Samples: ~5,000 (dogs + cats from /content/test)
Testing Samples: None explicitly (could use validation set for testing)

Note: validation_data=test_ds means test set is used as validation during training.

6. LABELED OR UNLABELED DATA?
================================================================================
Labeled Data ✓

Dataset has folder structure with labels:
- /content/train/dogs/ (labeled as Dog→1)
- /content/train/cats/ (labeled as Cat→0)
- /content/test/dogs/ (labeled as Dog→1)
- /content/test/cats/ (labeled as Cat→0)

image_dataset_from_directory infers labels from subfolder names.

================================================================================
ENSEMBLE ADVANTAGES
================================================================================
1. Feature Diversity: VGG16 (deep sequential) + ResNet50 (residual connections)
2. Complementary Features: VGG captures fine details, ResNet captures complex patterns
3. Robustness: Ensemble reduces overfitting compared to single model
4. Better Generalization: Combined features improve accuracy on unseen data

================================================================================
ENSEMBLE DISADVANTAGES
================================================================================
1. Computational Cost: 2x memory usage (both models loaded simultaneously)
2. Slower Training: Forward pass through both models
3. More Parameters: 38.9M total parameters (mostly frozen though)
4. Longer Inference Time: Must process image through both networks

================================================================================
COMPARISON WITH SINGLE MODELS
================================================================================
Metric              | VGG16 Only | ResNet50 Only | Ensemble (VGG+ResNet)
--------------------|------------|---------------|----------------------
Trainable Params    | ~8.3M      | ~33.5M        | ~0.68M
Frozen Params       | ~14.7M     | ~23.6M        | ~38.3M
Total Params        | ~23M       | ~57M          | ~39M
Feature Vector Size | 512        | 2048          | 2560
Inference Speed     | Fast       | Medium        | Slow
Memory Usage        | Low        | Medium        | High

================================================================================
NOTES ON THIS ARCHITECTURE
================================================================================
- This is a feature-level ensemble (concatenation, not averaging)
- Both pretrained models share the same input layer
- GlobalAveragePooling2D reduces spatial dimensions to 1×1 per channel
- Using GlobalAvgPool instead of Flatten reduces overfitting
- Only the classification head (Dense layers) is trained
- Both backbones are completely frozen (no fine-tuning)
"""