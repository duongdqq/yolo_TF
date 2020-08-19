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

    route_1 = input_data  # output = img / 8 for small obj

    input_data = cl.convolution(input_data, (3, 3, 256, 512), downsample=True)

    # 2nd residual block for detection
    for i in range(8):
        input_data = cl.residual_block(input_data, 512, 256, 512)

    route_2 = input_data  # output = img / 16 for medium obj

    input_data = cl.convolution(input_data, (3, 3, 512, 1024), downsample=True)

    # 3rd residual block for detection
    for i in range(4):
        input_data = cl.residual_block(input_data, 1024, 512, 1024)  # output = img / 32 for large obj

    return route_1, route_2, input_data


def cspdarknet53(input_data):
    input_data = cl.convolution(input_data, (3, 3, 3, 32), activate_type='mish')
    input_data = cl.convolution(input_data, (3, 3, 32, 64), downsample=True, activate_type='mish')

    route = input_data
    route = cl.convolution(route, (1, 1, 64, 64), activate_type='mish')
    input_data = cl.convolution(input_data, (1, 1, 64, 64), activate_type='mish')
    for i in range(1):
        input_data = cl.residual_block(input_data, 64, 32, 64, activate_type='mish')
    input_data = cl.convolution(input_data, (1, 1, 64, 64), activate_type='mish')
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = cl.convolution(input_data, (1, 1, 128, 64), activate_type='mish')
    input_data = cl.convolution(input_data, (3, 3, 64, 128), downsample=True, activate_type='mish')
    route = input_data
    route = cl.convolution(route, (1, 1, 128, 64), activate_type='mish')
    input_data = cl.convolution(input_data, (1, 1, 128, 64), activate_type='mish')
    for i in range(2):
        input_data = cl.residual_block(input_data, 64, 64, 64, activate_type='mish')
    input_data = cl.convolution(input_data, (1, 1, 64, 64), activate_type='mish')
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = cl.convolution(input_data, (1, 1, 128, 128), activate_type='mish')
    input_data = cl.convolution(input_data, (3, 3, 128, 256), downsample=True, activate_type='mish')
    route = input_data
    route = cl.convolution(route, (1, 1, 256, 128), activate_type='mish')
    input_data = cl.convolution(input_data, (1, 1, 256, 128), activate_type='mish')
    for i in range(8):
        input_data = cl.residual_block(input_data, 128, 128, 128, activate_type='mish')
    input_data = cl.convolution(input_data, (1, 1, 128, 128), activate_type='mish')
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = cl.convolution(input_data, (1, 1, 256, 256), activate_type='mish')
    route_1 = input_data
    input_data = cl.convolution(input_data, (3, 3, 256, 512), downsample=True, activate_type='mish')
    route = input_data
    route = cl.convolution(route, (1, 1, 512, 256), activate_type='mish')
    input_data = cl.convolution(input_data, (1, 1, 512, 256), activate_type='mish')
    for i in range(8):
        input_data = cl.residual_block(input_data, 256, 256, 256, activate_type='mish')
    input_data = cl.convolution(input_data, (1, 1, 256, 256), activate_type='mish')
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = cl.convolution(input_data, (1, 1, 512, 512), activate_type='mish')
    route_2 = input_data
    input_data = cl.convolution(input_data, (3, 3, 512, 1024), downsample=True, activate_type='mish')
    route = input_data
    route = cl.convolution(route, (1, 1, 1024, 512), activate_type='mish')
    input_data = cl.convolution(input_data, (1, 1, 1024, 512), activate_type='mish')
    for i in range(4):
        input_data = cl.residual_block(input_data, 512, 512, 512, activate_type='mish')
    input_data = cl.convolution(input_data, (1, 1, 512, 512), activate_type='mish')
    input_data = tf.concat([input_data, route], axis=-1)

    input_data = cl.convolution(input_data, (1, 1, 1024, 1024), activate_type='mish')
    input_data = cl.convolution(input_data, (1, 1, 1024, 512))
    input_data = cl.convolution(input_data, (3, 3, 512, 1024))
    input_data = cl.convolution(input_data, (1, 1, 1024, 512))

    input_data = tf.concat([tf.nn.max_pool(input_data, ksize=13, padding='same', strides=1),
                           tf.nn.max_pool(input_data, ksize=13, padding='same', strides=1),
                           tf.nn.max_pool(input_data, ksize=13, padding='same', strides=1),
                           input_data],
                           axis=-1)
    input_data = cl.convolution(input_data, (1, 1, 2048, 512))
    input_data = cl.convolution(input_data, (3, 3, 512, 1024))
    input_data = cl.convolution(input_data, (1, 1, 1024, 512))

    return route_1, route_2, input_data
