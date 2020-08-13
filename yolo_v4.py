import tensorflow as tf
import backbone
import common_layer as cl


def YOLO_v3(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = cl.convolution(conv, (1, 1, 1024, 512))
    conv = cl.convolution(conv, (3, 3, 512, 1024))
    conv = cl.convolution(conv, (1, 1, 1024, 512))
    conv = cl.convolution(conv, (3, 3, 512, 1024))
    conv = cl.convolution(conv, (1, 1, 1024, 512))

    conv_lobj_branch = cl.convolution(conv, (3, 3, 512, 1024))
    conv_lbbox = cl.convolution(conv_lobj_branch, (1, 1, 1024, 3 * (1 + 4 + NUM_CLASS )), activate=False, bn=False)

    conv = cl.convolution(conv, (1, 1, 512, 256))
    conv = cl.upsample(conv)

    conv = tf.concat([])