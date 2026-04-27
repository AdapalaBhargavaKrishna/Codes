import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img,img_to_array
image_path="/content/batman.jpg"
image=load_img(image_path)
image_array=img_to_array(image)
plt.imshow(image_array)
print(image_array.shape)

import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
image_resized = load_img(image_path, target_size=(424, 424))
image_array = img_to_array(image_resized)
image_shape = image_array.shape
print(image_shape)

model=Sequential([
    Conv2D(filters=1,kernel_size=(3,3),strides=(1,1),activation=None,input_shape=image_shape)
])
filtered_image=model.predict(image_array.reshape(1,424,424,3))
plt.imshow(filtered_image[0])
plt.axis("off") 
plt.show()

conv_layer=model.layers[0]
weights,biases=conv_layer.get_weights()
print("Kernel Weights Shape:", weights.shape)  # Shape: (height, width, no of channels, no. of filters)

gray_image = load_img(image_path, color_mode="grayscale")
plt.imshow(gray_image, cmap="gray")
gray_image = load_img(image_path, color_mode="grayscale",target_size=(224, 224))
image_array = img_to_array(gray_image)
image_shape = image_array.shape
print(image_shape)

# Define a custom 3x3 kernel
kernel = np.array([[[-1, 1, 1],
                    [-1, 1, 1],
                    [-1, 1, 1]]], dtype=np.float32)
# Reshape to match Conv2D weight format (height, width, input_channels, output_channels)
kernel = kernel.reshape((3, 3, 1, 1))

model = Sequential([
    Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), activation=None, input_shape=image_shape)
])

# Set the weights of the convolutional layer to your custom kernel
bias=np.zeros(1)
model.layers[0].set_weights([kernel, bias])

# Apply the convolution filter
filtered_image = model.predict(image_array.reshape(1, 224, 224, 1))

plt.imshow(filtered_image[0])
plt.axis("off")
plt.show()