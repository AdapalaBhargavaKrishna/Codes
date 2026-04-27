import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Dropout
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

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(256,256,3))
base_model.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=x)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, validation_data=test_ds)
print("Training Accuracy:", history.history['accuracy'][-1]*100)
print("Validation Accuracy:", history.history['val_accuracy'][-1]*100)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.show()
plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='validation')
plt.legend()
plt.show()

import cv2
test_img = cv2.imread('/content/test/dogs/dog.10006.jpg')
plt.imshow(test_img)
test_img = cv2.resize(test_img, (256,256))
test_input = test_img.reshape((1,256,256,3))
p = model.predict(test_input)
print(p)
print("DOG" if p >= 0.5 else "CAT")

"""
================================================================================
VGG16 TRANSFER LEARNING ANALYSIS ON DOGS VS CATS DATASET
================================================================================

1. ARCHITECTURE
================================================================================
Input Layer: 256×256×3 (RGB image)
↓
VGG16 Base Model (Pre-trained on ImageNet):
- 13 Convolutional Layers + 5 MaxPooling Layers
- include_top=False → No Fully Connected Layers
- trainable=False → Frozen pre-trained weights
↓
Output from VGG16: 8×8×512 (after flattening)
↓
Flatten → Output: 32,768 (8×8×512)
↓
Dense: 256 neurons, ReLU activation
↓
Dropout: 0.5
↓
Dense: 1 neuron, Sigmoid activation (Binary classification)

VGG16 Base Architecture Summary:
Block 1: Conv(64) → Conv(64) → MaxPool
Block 2: Conv(128) → Conv(128) → MaxPool
Block 3: Conv(256) → Conv(256) → Conv(256) → MaxPool
Block 4: Conv(512) → Conv(512) → Conv(512) → MaxPool
Block 5: Conv(512) → Conv(512) → Conv(512) → MaxPool

2. NUMBER OF LEARNING PARAMETERS (WEIGHTS & BIAS)
================================================================================
VGG16 Base Model (Frozen - Not Trainable):
Total Parameters: ~14,714,688 (0 trainable because base_model.trainable = False)

Custom Layers Added:
Layer                    | Weights              | Bias    | Total
-------------------------|----------------------|---------|------------
Dense (32768→256)        | 32,768×256 = 8,388,608| 256     | 8,388,864
Dense (256→1)            | 256×1 = 256          | 1       | 257
TOTAL TRAINABLE          | 8,388,864            | 257     | 8,389,121

GRAND TOTAL (Frozen + Trainable): ~23,103,809 parameters

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

Note: Only custom Dense layers are updated (VGG16 weights are frozen)

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
TRANSFER LEARNING NOTES
================================================================================
- Base VGG16 frozen to retain ImageNet features
- Only custom Dense layers are trained on dogs vs cats
- Input size changed from 224×224 (original VGG16) to 256×256
- include_top=False removes VGG16's original classification layers
"""