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

input_data = tf.keras.layers.Input([608, 608, 3])
x = layers.Conv2D(filters=32,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_data)
x = layers.Conv2D(filters=64,
                  kernel_size=3,
                  strides=2,
                  padding='valid',
                  kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_data)
x = layers.

output_data = layers.Dense(4)(x)
model = keras.Model(input_data, output_data)
model.summary()