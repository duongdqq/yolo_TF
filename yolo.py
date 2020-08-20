import tensorflow as tf
import backbone
import common_layer as cl
from tensorflow.keras import layers
from tensorflow import keras


def yolo_v3(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.darknet53(input_layer)

    # 1st detector
    conv = cl.convolution(conv, (1, 1, 1024, 512))
    conv = cl.convolution(conv, (3, 3, 512, 1024))
    conv = cl.convolution(conv, (1, 1, 1024, 512))
    conv = cl.convolution(conv, (3, 3, 512, 1024))
    conv = cl.convolution(conv, (1, 1, 1024, 512))

    conv_lobj_branch = cl.convolution(conv, (3, 3, 512, 1024))

    conv_lbbox = cl.convolution(conv_lobj_branch, (1, 1, 1024, 3 * (1 + 4 + NUM_CLASS)), activate=False, bn=False)

    conv = cl.convolution(conv, (1, 1, 512, 256))
    conv = cl.upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    # 2nd detector
    conv = cl.convolution(conv, (1, 1, 768, 256))
    conv = cl.convolution(conv, (3, 3, 256, 512))
    conv = cl.convolution(conv, (1, 1, 512, 256))
    conv = cl.convolution(conv, (3, 3, 256, 512))
    conv = cl.convolution(conv, (1, 1, 512, 256))

    conv_mobj_branch = cl.convolution(conv, (3, 3, 256, 512))

    conv_mbbox = cl.convolution(conv_mobj_branch, (1, 1, 512, 3 * (1 + 4 + NUM_CLASS)), activate=False, bn=False)

    conv = cl.convolution(conv, (1, 1, 256, 128))
    conv = cl.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    # 3rd detector
    conv = cl.convolution(conv, (1, 1, 384, 128))
    conv = cl.convolution(conv, (3, 3, 128, 256))
    conv = cl.convolution(conv, (1, 1, 256, 128))
    conv = cl.convolution(conv, (3, 3, 128, 256))
    conv = cl.convolution(conv, (1, 1, 256, 128))

    conv_sobj_branch = cl.convolution(conv, (3, 3, 128, 256))

    conv_sbbox = cl.convolution(conv_sobj_branch, (1, 1, 256, 3 * (1 + 4 + NUM_CLASS)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def yolo_v4(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.cspdarknet53(input_layer)

    route = conv
    conv = cl.convolution(conv, (1, 1, 512, 256))
    conv = cl.upsample(conv)
    route_2 = cl.convolution(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = cl.convolution(conv, (1, 1, 512, 256))
    conv = cl.convolution(conv, (3, 3, 256, 512))
    conv = cl.convolution(conv, (1, 1, 512, 256))
    conv = cl.convolution(conv, (3, 3, 256, 512))
    conv = cl.convolution(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = cl.convolution(conv, (1, 1, 256, 128))
    conv = cl.upsample(conv)
    route_1 = cl.convolution(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = cl.convolution(conv, (1, 1, 256, 128))
    conv = cl.convolution(conv, (3, 3, 128, 256))
    conv = cl.convolution(conv, (1, 1, 256, 128))
    conv = cl.convolution(conv, (3, 3, 128, 256))
    conv = cl.convolution(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = cl.convolution(conv, (3, 3, 128, 256))
    conv_sbbox = cl.convolution(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = cl.convolution(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = cl.convolution(conv, (1, 1, 512, 256))
    conv = cl.convolution(conv, (3, 3, 256, 512))
    conv = cl.convolution(conv, (1, 1, 512, 256))
    conv = cl.convolution(conv, (3, 3, 256, 512))
    conv = cl.convolution(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = cl.convolution(conv, (3, 3, 256, 512))
    conv_mbbox = cl.convolution(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = cl.convolution(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = cl.convolution(conv, (1, 1, 1024, 512))
    conv = cl.convolution(conv, (3, 3, 512, 1024))
    conv = cl.convolution(conv, (1, 1, 1024, 512))
    conv = cl.convolution(conv, (3, 3, 512, 1024))
    conv = cl.convolution(conv, (1, 1, 1024, 512))

    conv = cl.convolution(conv, (3, 3, 512, 1024))
    conv_lbbox = cl.convolution(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]


def yolo_v4_TF():
    x_input = tf.keras.Input(shape=(608, 608, 3))

    x = layers.Conv2D(filters=32,
                      kernel_size=3,
                      strides=1,
                      use_bias=False,
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x_input)
    x = layers.BatchNormalization()(x)
    x = x * tf.math.tanh(tf.math.softplus(x))

    x = layers.Conv2D(filters=64,
                      kernel_size=3,
                      strides=2)(x)
    x = x * tf.math.tanh(tf.math.softplus(x))

    x_output = layers.Dense(4)(x)
    model = tf.keras.Model(x_input, x_output)
    model.summary()
    tf.keras.utils.plot_model(model, "yolo_v4_TF.png", show_shapes=True)


yolo_v4_TF()
