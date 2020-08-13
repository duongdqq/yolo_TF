import tensorflow as tf
import numpy as np

input_shape = (1, 1, 2, 2)
x = np.arange(np.prod(input_shape)).reshape(input_shape)
print(x)
y = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
print(y)
