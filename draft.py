import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import common_layer as cl

# input_shape = (1, 1, 2, 2)
# x = np.arange(np.prod(input_shape)).reshape(input_shape)
# print(x)
# y = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
# print(y)

# t1 = np.array([[[1, 2], [2, 3]], [[4, 4], [5, 3]]])
# t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
#
# print(t1.shape)
# t12 = tf.concat([t1, t2], -1)
# print(t12)

model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len,
                       input_shape=(max_len,), name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

from keras.utils import plot_model

plot_model(model, show_shapes=True, to_file='model.png')