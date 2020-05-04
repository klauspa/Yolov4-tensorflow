import tensorflow as tf
from conf import IMAGE_HEIGHT, IMAGE_WIDTH, CATEGORY_NUM
import numpy as np

#mish activation
def mish(x):
    return x*tf.tanh(tf.math.log(1+tf.exp(x)))

class Mish(tf.keras.layers.Layer):
    def __init__(self):
        super(Mish, self).__init__()
    def call(self, x):
        return mish(x)

#conv block with mish
def single_conv_mish(inputs, filters, kernel, strides):
    padding = 'valid' if strides == 2 else 'same'
    if strides == 2:
        inputs = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(inputs)
    out = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, use_bias=False,
                                                strides=strides, padding=padding)(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = Mish()(out)
    return out

#conv block with leaky
def single_conv_leaky(inputs, filters, kernel, strides):
    padding = 'valid' if strides == 2 else 'same'
    if strides == 2:
        inputs = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(inputs)
    out = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel, use_bias=False,
                                                strides=strides, padding=padding)(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU(0.1)(out)
    return out

#res connection
def res_conn_block(inputs, filters, is_half):
    out = single_conv_mish(inputs, filters//2, 1, 1)
    out = single_conv_mish(out, filters//2 if is_half else filters, 3, 1)
    return out

#single res conn block
def ResBlock(inputs, filters, res_num, is_half):
    downsample_out = single_conv_mish(inputs, filters, 3, 2)
    right_conv = single_conv_mish(downsample_out, filters//2 if is_half else filters, 1, 1)

    left_conv = single_conv_mish(downsample_out, filters//2 if is_half else filters, 1, 1)
    for i in range(res_num):
        res_intermidiate = res_conn_block(left_conv, filters, is_half)
        left_conv = left_conv + res_intermidiate
    left_conv = single_conv_mish(left_conv, filters//2 if is_half else filters, 1, 1)
    concat_out = tf.keras.layers.Concatenate()([left_conv, right_conv])
    out = single_conv_mish(concat_out, filters, 1, 1)

    return out

#conv leaky stacked layers
def make_leaky_convs(inputs, layer_num, filters, strides):
    if layer_num == 1:
        out = single_conv_leaky(inputs, filters, 1, strides)

    if layer_num == 3:
        out = single_conv_leaky(inputs, filters, 1, strides)
        out = single_conv_leaky(out, filters*2, 3, strides)
        out = single_conv_leaky(out, filters, 1, strides)

    if layer_num == 5:
        out = single_conv_leaky(inputs, filters, 1, strides)
        out = single_conv_leaky(out, filters*2, 3, strides)
        out = single_conv_leaky(out, filters, 1, strides)
        out = single_conv_leaky(out, filters*2, 3, strides)
        out = single_conv_leaky(out, filters, 1, strides)

    return out

#spp module
def spp_module(inputs):
    pool1 = tf.keras.layers.MaxPooling2D((13,13), strides=1, padding='same')(inputs)
    pool2 = tf.keras.layers.MaxPooling2D((9,9), strides=1, padding='same')(inputs)
    pool3 = tf.keras.layers.MaxPooling2D((5,5), strides=1, padding='same')(inputs)
    out = tf.keras.layers.Concatenate()([pool1, pool2, pool3, inputs])
    return out

#transorm yolo feature map
#reference: https://github.com/hunglc007/tensorflow-yolov4-tflite
def transform(conv_output, NUM_CLASS, i=0):
    """
    return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
            contains (x, y, w, h, score, probability)
    """
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
    conv_raw_xywh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (4, 1, NUM_CLASS), axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    return tf.concat([conv_raw_xywh, pred_conf, pred_prob], axis=-1)

#load weights
#reference: https://github.com/hunglc007/tensorflow-yolov4-tflite
def load_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(110):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [93, 101, 109]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [93, 101, 109]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    print("load OK")
    wf.close()


def yolo_body(inputs, classes):
    #cspdarknet53
    first_conv = single_conv_mish(inputs, 32, 3, 1)
    res_block_1 = ResBlock(first_conv, 64, 1, False)
    res_block_2 = ResBlock(res_block_1, 128, 2, True)
    res_block_3 = ResBlock(res_block_2, 256, 8, True)

    intermediate_1 = res_block_3

    res_block_4 = ResBlock(res_block_3, 512, 8, True)

    intermediate_2 = res_block_4

    res_block_5 = ResBlock(res_block_4, 1024, 4, True)

    pred_spp = make_leaky_convs(res_block_5, 3, 512, 1)
    spp_out = spp_module(pred_spp)
    succ_spp  = make_leaky_convs(spp_out, 3, 512, 1)

    intermediate_3 = succ_spp

    head2_right = make_leaky_convs(intermediate_3, 1, 256, 1)
    head2_right = tf.keras.layers.UpSampling2D()(head2_right)
    head2_left = make_leaky_convs(intermediate_2, 1, 256, 1)
    head2 = tf.keras.layers.Concatenate()([head2_left, head2_right])
    head2 = make_leaky_convs(head2, 5, 256, 1)

    intermediate_4 = head2

    head1_right = make_leaky_convs(intermediate_4, 1, 128, 1)
    head1_right = tf.keras.layers.UpSampling2D()(head1_right)
    head1_left = make_leaky_convs(intermediate_1, 1, 128, 1)
    head1 = tf.keras.layers.Concatenate()([head1_left, head1_right])
    head1 = make_leaky_convs(head1, 5, 128, 1) #conv92

    intermediate_5 = head1

    head1 = single_conv_leaky(head1, 256, 3, 1)
    head1_out = tf.keras.layers.Conv2D(3*(4+1+classes), kernel_size=1, padding='same')(head1)

    head2_side = single_conv_leaky(intermediate_5, 256, 3, 2)
    head2 = tf.keras.layers.Concatenate()([head2_side, intermediate_4])
    head2 = make_leaky_convs(head2, 5, 256, 1)  

    intermediate_6 = head2

    head2 = single_conv_leaky(head2, 512, 3, 1)
    head2_out = tf.keras.layers.Conv2D(3*(4+1+classes), kernel_size=1, padding='same')(head2)

    head3_right = single_conv_leaky(intermediate_6, 512, 3, 2)
    head3 = tf.keras.layers.Concatenate()([head3_right, intermediate_3])
    head3 = make_leaky_convs(head3, 5, 512, 1)
    head3 = single_conv_leaky(head3, 1024, 3, 1)
    head3_out = tf.keras.layers.Conv2D(3*(4+1+classes), kernel_size=1, padding='same')(head3)

    conv_out = [head1_out, head2_out, head3_out]

    return conv_out

def Yolo_Model():
    inputs = tf.keras.layers.Input(shape=[IMAGE_WIDTH, IMAGE_HEIGHT, 3])
    yolobody_out = yolo_body(inputs, CATEGORY_NUM)
    conv_outs = []
    for i, conv_out in enumerate(yolobody_out):
        transformed_out = transform(conv_out, CATEGORY_NUM, i)
        conv_outs.append(transformed_out)
    
    return tf.keras.Model(inputs=inputs, outputs=conv_outs)

#simple test
if __name__ == "__main__":
    inputs = tf.keras.layers.Input(shape=(608, 608, 3))
    outputs = Yolo_Model(inputs, 80)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    x = tf.random.normal(shape=(1, 608, 608, 3))
    out = model(x)
    head1, head2, head3 = out[0], out[1], out[2]

    print("head1 shape: ", head1.shape)
    print("head2 shape: ", head2.shape)
    print("head3 shape: ", head3.shape)