import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

model = Sequential()
model.add(Conv2D(96, (11,11), strides=4, activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))
model.add(Conv2D(256, (5,5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))
model.add(Conv2D(384, (3,3), padding='same', activation='relu'))
model.add(Conv2D(384, (3,3), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_ds, epochs=10, validation_data=test_ds, verbose=1)
print("Last Epoch Training Accuracy:", history.history['accuracy'][-1])
print("Last Epoch Validation Accuracy:", history.history['val_accuracy'][-1])

import numpy as np
from tensorflow.keras.preprocessing import image
img = image.load_img('cat_or_dog.jpg', target_size=(224,224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
print("Dog 🐶" if prediction[0][0] > 0.5 else "Cat 🐱")

import cv2
import matplotlib.pyplot as plt
test_img = cv2.imread('/content/test/dogs/dog.10006.jpg')
plt.imshow(test_img)
print("test image shape:", test_img.shape)
test_img = cv2.resize(test_img, (224,224))
test_input = test_img.reshape((1,224,224,3))
p = model.predict(test_input)
print("DOG" if p >= 0.5 else "CAT")

test_img = cv2.imread('/content/test/cats/cat.10030.jpg')
plt.imshow(test_img)
print("test image shape:", test_img.shape)
test_img = cv2.resize(test_img, (224,224))
test_input = test_img.reshape((1,224,224,3))
p = model.predict(test_input)
print("DOG" if p >= 0.5 else "CAT")

"""
================================================================================
ALEXNET ARCHITECTURE ANALYSIS ON DOGS VS CATS DATASET
================================================================================

1. ARCHITECTURE
================================================================================
Input Layer: 224×224×3 (RGB image)
↓
Conv2D: 96 filters, 11×11 kernel, stride=4, ReLU → Output: 54×54×96
↓
MaxPooling2D: 3×3 pool, stride=2 → Output: 26×26×96
↓
Conv2D: 256 filters, 5×5 kernel, padding='same', ReLU → Output: 26×26×256
↓
MaxPooling2D: 3×3 pool, stride=2 → Output: 12×12×256
↓
Conv2D: 384 filters, 3×3 kernel, padding='same', ReLU → Output: 12×12×384
↓
Conv2D: 384 filters, 3×3 kernel, padding='same', ReLU → Output: 12×12×384
↓
Conv2D: 256 filters, 3×3 kernel, padding='same', ReLU → Output: 12×12×256
↓
MaxPooling2D: 3×3 pool, stride=2 → Output: 5×5×256
↓
Flatten → Output: 6400 (5×5×256)
↓
Dense: 4096 neurons, ReLU
↓
Dropout: 0.5
↓
Dense: 4096 neurons, ReLU
↓
Dropout: 0.5
↓
Dense: 1 neuron, Sigmoid (Binary classification)

2. NUMBER OF LEARNING PARAMETERS (WEIGHTS & BIAS)
================================================================================
Layer                    | Weights              | Bias    | Total
-------------------------|----------------------|---------|------------
Conv2D (96×11×11×3)      | 96×11×11×3 = 34,848  | 96      | 34,944
Conv2D (256×5×5×96)      | 256×5×5×96 = 614,400 | 256     | 614,656
Conv2D (384×3×3×256)     | 384×3×3×256 = 884,736| 384     | 885,120
Conv2D (384×3×3×384)     | 384×3×3×384 = 1,327,104| 384   | 1,327,488
Conv2D (256×3×3×384)     | 256×3×3×384 = 884,736| 256     | 884,992
Dense (6400→4096)        | 6400×4096 = 26,214,400| 4,096   | 26,218,496
Dense (4096→4096)        | 4096×4096 = 16,777,216| 4,096   | 16,781,312
Dense (4096→1)           | 4096×1 = 4,096       | 1       | 4,097
TOTAL                    | 46,741,536           | 9,569   | 46,751,105

3. LOSS CALCULATION FORMULA
================================================================================
Binary Cross-Entropy (Binary Classification):

For single sample: Loss = -[y × log(p) + (1-y) × log(1-p)]

For batch: Loss = -1/N Σ [y_i × log(p_i) + (1-y_i) × log(1-p_i)]

where:
- N = batch size (32)
- y = true label (0 for Cat, 1 for Dog)
- p = predicted probability (output of sigmoid)

4. WEIGHT UPDATE FORMULA
================================================================================
Adam Optimizer:

θ(t+1) = θ(t) - (η / (√v̂(t) + ε)) × m̂(t)

where:
- θ(t) = weights/biases at time step t
- η = learning rate (default 0.001)
- m̂(t) = bias-corrected first moment (mean of gradients)
- v̂(t) = bias-corrected second moment (uncentered variance)
- ε = 10⁻⁷ (small constant)

5. SAMPLES USED FOR TRAINING, VALIDATION, TESTING
================================================================================
Training Samples: ~20,000 (dogs + cats from /content/train)
Validation Samples: ~5,000 (dogs + cats from /content/test)
Testing Samples: 2 (individual images: dog.10006.jpg, cat.10030.jpg + cat_or_dog.jpg)

Note: validation_data=test_ds means test set is used as validation during training.

6. LABELED OR UNLABELED DATA?
================================================================================
Labeled Data ✓

Dataset has folder structure:
- /content/train/dogs/ (labeled as Dog→1)
- /content/train/cats/ (labeled as Cat→0)
- /content/test/dogs/ (labeled as Dog→1)
- /content/test/cats/ (labeled as Cat→0)

image_dataset_from_directory infers labels from subfolder names.

================================================================================
PREDICTION OUTPUT
================================================================================
- If prediction > 0.5 → DOG 🐶
- If prediction < 0.5 → CAT 🐱
"""