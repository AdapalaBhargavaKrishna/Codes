from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

(X_train , y_train) , (X_test , y_test) = keras.datasets.mnist.load_data()