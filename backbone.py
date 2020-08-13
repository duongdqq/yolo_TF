import tensorflow as tf
import common_layer as cl


def darknet53(input_data):
    input_data = cl.convolution(input_data, (3, 3, 3, 32))
    input_data = cl.convolution(input_data, (3, 3, 32, 64), downsample=True)

    for i in range(1):
        input_data = cl.residual_block(input_data, 64, 32, 64)

    input_data = cl.convolution(input_data, (3, 3, 64, 128), downsample=True)

    for i in range(2):
        input_data = cl.residual_block(input_data, 128, 64, 128)

    input_data = cl.convolution(input_data, (3, 3, 128, 256), downsample=True)

    # 1st residual block for detection
    for i in range(8):
        input_data = cl.residual_block(input_data, 256, 128, 256)

    route_1 = input_data

    input_data = cl.convolution(input_data, (3, 3, 256, 512), downsample=True)

    # 2nd residual block for detection
    for i in range(8):
        input_data = cl.residual_block(input_data, 512, 256, 512)

    route_2 = input_data

    input_data = cl.convolution(input_data, (3, 3, 512, 1024), downsample=True)

    # 3rd residual block for detection
    for i in range(4):
        input_data = cl.residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data