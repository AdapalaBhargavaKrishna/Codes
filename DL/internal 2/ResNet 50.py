import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow import keras
from tensorflow.keras.models import Model
from keras.layers import Dense,Flatten,Dropout

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!kaggle datasets download -d salader/dogsvscats

import zipfile
zip_ref = zipfile.ZipFile('/content/dogsvscats.zip','r')
zip_ref.extractall('/content')
zip_ref.close()

train_ds = keras.utils.image_dataset_from_directory(
    directory='/content/train', labels='inferred', label_mode='int',
    batch_size=32, image_size=(224,224))

test_ds = keras.utils.image_dataset_from_directory(
    directory='/content/test', labels='inferred', label_mode='int',
    batch_size=32, image_size=(224,224))

def process(image,label):
    image = tf.cast(image/255., tf.float32)
    return image,label
train_ds = train_ds.map(process)
test_ds = test_ds.map(process)

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(256,256,3))
base_model.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=x)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=test_ds)

test_img = cv2.imread('/content/test/dogs/dog.10006.jpg')
test_img = cv2.resize(test_img, (256,256))
test_input = test_img.reshape((1,256,256,3))
p = model.predict(test_input)
print("DOG" if p >= 0.5 else "CAT")

"""
================================================================================
RESNET50 TRANSFER LEARNING ANALYSIS ON DOGS VS CATS DATASET
================================================================================

1. ARCHITECTURE
================================================================================
Input Layer: 256×256×3 (RGB image)
↓
ResNet50 Base Model (Pre-trained on ImageNet):
- 50 layers deep with Residual blocks
- include_top=False → No Fully Connected Layers
- trainable=False → Frozen pre-trained weights
↓
Output from ResNet50: 8×8×2048 (after Global Average Pooling in original, but here Flatten is used)
↓
Flatten → Output: 131,072 (8×8×2048)
↓
Dense: 256 neurons, ReLU activation
↓
Dropout: 0.5
↓
Dense: 1 neuron, Sigmoid activation (Binary classification)

ResNet50 Architecture Summary:
- Stem: Conv(64) → MaxPool
- Stage 1: 3 Residual blocks (64, 64, 256)
- Stage 2: 4 Residual blocks (128, 128, 512)
- Stage 3: 6 Residual blocks (256, 256, 1024)
- Stage 4: 3 Residual blocks (512, 512, 2048)
- Each residual block has BatchNorm and ReLU

2. NUMBER OF LEARNING PARAMETERS (WEIGHTS & BIAS)
================================================================================
ResNet50 Base Model (Frozen - Not Trainable):
Total Parameters: ~23,587,712 (0 trainable because base_model.trainable = False)

Custom Layers Added:
Layer                    | Weights                    | Bias    | Total
-------------------------|----------------------------|---------|---------------
Dense (131072→256)       | 131,072×256 = 33,554,432   | 256     | 33,554,688
Dense (256→1)            | 256×1 = 256                | 1       | 257
TOTAL TRAINABLE          | 33,554,688                 | 257     | 33,554,945

GRAND TOTAL (Frozen + Trainable): ~57,142,657 parameters

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

Note: Only custom Dense layers are updated (ResNet50 weights are frozen)

5. SAMPLES USED FOR TRAINING, VALIDATION, TESTING
================================================================================
Training Samples: ~20,000 (dogs + cats from /content/train)
Validation Samples: ~5,000 (dogs + cats from /content/test)
Testing Samples: 1 (dog.10006.jpg image)

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
Transfer learning uses pre-trained labels from ImageNet (1,000 classes).

================================================================================
RESNET50 vs VGG16 COMPARISON
================================================================================
- ResNet50 has ~23.5M parameters vs VGG16's ~14.7M (ResNet deeper but more efficient)
- ResNet uses skip connections to avoid vanishing gradient
- ResNet50 has BatchNormalization after every convolution
- ResNet50 is faster to train due to residual learning
- Both achieve ~99% accuracy on ImageNet but ResNet uses fewer parameters per layer depth
"""