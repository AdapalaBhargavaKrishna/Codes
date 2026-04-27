import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

image_path = "/content/batman.jpg"
image = load_img(image_path)
image_array = img_to_array(image)
plt.imshow(image_array)
print(image_array.shape)

image_resized = load_img(image_path, target_size=(424, 424))
image_array = img_to_array(image_resized)
image_shape = image_array.shape
print(image_shape)

model = Sequential([
    Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), activation=None, input_shape=image_shape)
])
filtered_image = model.predict(image_array.reshape(1, 424, 424, 3))
plt.imshow(filtered_image[0])
plt.axis("off")
plt.show()

conv_layer = model.layers[0]
weights, biases = conv_layer.get_weights()
print("Kernel Weights Shape:", weights.shape)

gray_image = load_img(image_path, color_mode="grayscale")
plt.imshow(gray_image, cmap="gray")
gray_image = load_img(image_path, color_mode="grayscale", target_size=(224, 224))
image_array = img_to_array(gray_image)
image_shape = image_array.shape
print(image_shape)

kernel = np.array([[[-1, 1, 1],
                    [-1, 1, 1],
                    [-1, 1, 1]]], dtype=np.float32)
kernel = kernel.reshape((3, 3, 1, 1))

# Reshape to match Conv2D weight format (height, width, input_channels, output_channels)
model = Sequential([
    Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), activation=None, input_shape=image_shape)
])

bias = np.zeros(1)
model.layers[0].set_weights([kernel, bias])

filtered_image = model.predict(image_array.reshape(1, 224, 224, 1))

plt.imshow(filtered_image[0])
plt.axis("off")
plt.show()

"""
================================================================================
CONVOLUTIONAL FILTER ANALYSIS ON IMAGE
================================================================================

1. ARCHITECTURE
================================================================================

MODEL 1 (RGB Image):
Input Layer: 424×424×3 (RGB color image)
↓
Conv2D: 1 filter, 3×3 kernel, stride=(1,1), no activation → Output: 422×422×1

MODEL 2 (Grayscale Image with Custom Kernel):
Input Layer: 224×224×1 (Grayscale image)
↓
Conv2D: 1 filter, 3×3 kernel, stride=(1,1), no activation → Output: 222×222×1

Custom Kernel Used:
[[[-1, 1, 1],
  [-1, 1, 1],
  [-1, 1, 1]]]

2. NUMBER OF LEARNING PARAMETERS (WEIGHTS & BIAS)
================================================================================

MODEL 1 (RGB Image - Random initialized weights):
Layer                    | Weights              | Bias    | Total
-------------------------|----------------------|---------|-----------
Conv2D (1×3×3×3)         | 1×3×3×3 = 27         | 1       | 28
TOTAL                    | 27                   | 1       | 28

MODEL 2 (Grayscale - Custom kernel fixed weights):
Layer                    | Weights              | Bias    | Total
-------------------------|----------------------|---------|-----------
Conv2D (1×3×3×1)         | 1×3×3×1 = 9          | 1       | 10
TOTAL                    | 9                    | 1       | 10

Note: In MODEL 2, weights are manually set, not learned.

3. LOSS CALCULATION FORMULA
================================================================================
No loss function used in this code because:
- Model is NOT compiled or trained
- Only forward pass (prediction/inference) is performed
- No training, no loss calculation

If training were done with binary classification:
Binary Cross-Entropy = -[y log(p) + (1-y) log(1-p)]

If training were done with multi-class classification:
Categorical Cross-Entropy = - Σ y_c × log(p_c)

4. WEIGHT UPDATE FORMULA
================================================================================
No weight update in this code because:
- Model is NOT compiled (no optimizer specified)
- Model is NOT trained (no fit() called)
- Only model.predict() for forward pass/inference

If training were done with SGD:
θ(t+1) = θ(t) - η × ∇L(θ(t))

5. SAMPLES USED FOR TRAINING, VALIDATION, TESTING
================================================================================
Training Samples: 0 (No training performed)
Validation Samples: 0 (No validation split)
Testing Samples: 1 (Single batman.jpg image used for inference only)

This is inference/prediction only, not training.

6. LABELED OR UNLABELED DATA?
================================================================================
Unlabeled Data

The code loads a single image (batman.jpg) without any corresponding label.
No ground truth or target values are provided.
This is unsupervised inference/feature extraction using convolution filters.

================================================================================
ADDITIONAL NOTES
================================================================================
- The custom kernel acts as an edge detection filter (negative on left column, positive on right)
- Output shape reduction: (424-3+1) = 422 for first model
- Output shape reduction: (224-3+1) = 222 for second model
- Stride=(1,1) means filter moves 1 pixel at a time
- No activation function (activation=None) means linear activation (f(x)=x)
"""