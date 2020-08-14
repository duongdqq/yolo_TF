import tensorflow as tf
import yolo


input_layer = tf.keras.layers.Input([608, 608, 3])
output_layer = yolo.yolo_v4(input_layer, NUM_CLASS=4)
model = tf.keras.Model(input_layer, output_layer)
model.summary()

