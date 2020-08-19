import tensorflow as tf
import backbone
import common_layer as cl


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
#! /usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg

# NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
# STRIDES         = np.array(cfg.YOLO.STRIDES)
# IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
# XYSCALE = cfg.YOLO.XYSCALE
# ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)

def YOLO(input_layer, NUM_CLASS, model='yolov4', is_tiny=False):
    if is_tiny:
        if model == 'yolov4':
            return YOLOv4_tiny(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3_tiny(input_layer, NUM_CLASS)
    else:
        if model == 'yolov4':
            return YOLOv4(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3(input_layer, NUM_CLASS)

def YOLOv3(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 768, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 384, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.cspdarknet53(input_layer)

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = tf.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = tf.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = tf.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
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

