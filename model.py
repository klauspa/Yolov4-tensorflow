import tensorflow as tf
import tensorflow_addons as tfa

#hyper parameters
batch = 64
subdivisions=8
width=608
height=608
channels=3
momentum=0.949
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000
max_batches = 500500
#policy=steps
steps=400000,450000
scales=.1,.1

#cutmix=1
mosaic=1

weightfile = "../yolov4.weights"
import numpy as np

def load_weight():
    
    print('Loading weights.')
    weights_file = fp = open(weightfile, 'rb')
    major, minor, revision = np.ndarray(
        shape=(3, ), dtype='int32', buffer=weights_file.read(12))
    if (major*10+minor)>=2 and major<1000 and minor<1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)
    fp.close()

def mish(x):
    return tfa.activations.mish(x)

def leaky(x):
    return tf.nn.leaky_relu(x)

def res_conn_block(filters, is_half):
    res = tf.keras.Sequential()
    
    res.add(Conv2D_BN_Mish(filters//2, kernel=1, strides=1))
    res.add(Conv2D_BN_Mish(filters//2 if is_half else filters, kernel=3, strides=1))

    return res

class Conv2D_BN_Mish(tf.keras.Model):
    def __init__(self, filters, kernel, strides):
        super(Conv2D_BN_Mish, self).__init__()
        padding = 'valid' if strides == 2 else 'same'
        if strides == 2:
            self.conv2d_bn = tf.keras.Sequential([
                                    tf.keras.layers.ZeroPadding2D(((1,0),(1,0))),
                                    tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel, 
                                                                strides = strides, padding = padding),
                                    tf.keras.layers.BatchNormalization(),
                                    ])
        else:
            self.conv2d_bn = tf.keras.Sequential([
                                    tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel, 
                                                                strides = strides, padding = padding),
                                    tf.keras.layers.BatchNormalization(),
                                    ])

    def call(self, x):
        x = self.conv2d_bn(x)
        x = mish(x)
        return x

class Conv2D_BN_Leaky(tf.keras.Model):
    def __init__(self, filters, kernel, strides):
        super(Conv2D_BN_Leaky, self).__init__()

        if strides == 2:
            self.conv2d_bn = tf.keras.Sequential([
                                    tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0))),
                                    tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel, 
                                                                strides = strides, padding = 'valid'),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.LeakyReLU(),
                                    ])
        else:
            self.conv2d_bn = tf.keras.Sequential([
                                tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel, 
                                                            strides = strides, padding = 'same'),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.LeakyReLU(),
                                ])

    def call(self, x):
        x = self.conv2d_bn(x)
        return x

class ResBlock(tf.keras.Model):
    def __init__(self, filters, res_num, is_half):
        super(ResBlock, self).__init__()
        self.res_num = res_num

        self.pad_conv = Conv2D_BN_Mish(filters, kernel=3, strides=2)
        self.pred_block_conv = Conv2D_BN_Mish(filters//2 if is_half else filters, kernel=1, strides=1)
        self.res_conn_block = res_conn_block(filters, is_half)
        self.succ_block_conv = Conv2D_BN_Mish(filters//2 if is_half else filters, kernel=1, strides=1)
        self.right_conv = Conv2D_BN_Mish(filters//2 if is_half else filters, kernel=1, strides=1)
        self.after_concat_conv = Conv2D_BN_Mish(filters, kernel=1, strides=1)
    
    def call(self,x):
        pred_res = self.pad_conv(x)
        right_conv = self.right_conv(pred_res)
        left_conv = self.pred_block_conv(pred_res)
        for i in range(self.res_num):
            res_block_out = self.res_conn_block(left_conv)
            left_conv = left_conv + res_block_out
        left_conv = self.succ_block_conv(left_conv)

        concat_x = tf.concat([left_conv, right_conv], axis=-1)
        out = self.after_concat_conv(concat_x)
        return out

def make_leaky_convs(layer_num, filters, strides=1):

    layers = tf.keras.Sequential()
    if layer_num == 1:
        layers.add(Conv2D_BN_Leaky(filters, kernel=1, strides=strides))

    if layer_num == 3:
        layers.add(Conv2D_BN_Leaky(filters, kernel=1, strides=strides))
        layers.add(Conv2D_BN_Leaky(filters*2, kernel=3, strides=strides))
        layers.add(Conv2D_BN_Leaky(filters, kernel=1, strides=strides))
    
    if layer_num == 5:
        layers.add(Conv2D_BN_Leaky(filters, kernel=1, strides=strides))
        layers.add(Conv2D_BN_Leaky(filters*2, kernel=3, strides=strides))
        layers.add(Conv2D_BN_Leaky(filters, kernel=1, strides=strides))
        layers.add(Conv2D_BN_Leaky(filters*2, kernel=3, strides=strides))
        layers.add(Conv2D_BN_Leaky(filters, kernel=1, strides=strides))

    return layers

class spp(tf.keras.Model):
    def __init__(self):
        super(spp, self).__init__()
        self.pool1 = tf.keras.layers.MaxPooling2D((5,5), strides=1, padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D((9,9), strides=1, padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D((13,13), strides=1, padding='same')
    
    def call(self, x):
        return tf.concat([self.pool1(x), self.pool2(x), self.pool3(x), x], -1)
        

class Yolo_Model(tf.keras.Model):
    def __init__(self,):
        super(Yolo_Model, self).__init__()
        self.conv_last1 = tf.keras.layers.Conv2D(255, kernel_size=1, padding='same')
        self.conv_last2 = tf.keras.layers.Conv2D(255, kernel_size=1, padding='same')
        self.conv_last3 = tf.keras.layers.Conv2D(255, kernel_size=1, padding='same')
        self.pad = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
        self.first_conv = Conv2D_BN_Mish(filters=32, kernel=3, strides=1)
        self.res_block1 = ResBlock(64, 1, False)
        self.res_block2 = ResBlock(128, 2, True)
        self.res_block3 = ResBlock(256, 8, True)

        self.res_block4 = ResBlock(512, 8, False)
        self.res_block5 = ResBlock(1024, 4, False)

        self.conv_leaky3_1 = make_leaky_convs(3, 512)
        self.conv_leaky3_2 = make_leaky_convs(3, 512)
        self.conv_leaky1_1 = make_leaky_convs(1, 256)
        self.conv_leaky1_2 = make_leaky_convs(1, 128)
        self.conv_leaky1_3 = make_leaky_convs(1, 512)
        self.conv_leaky1_4 = make_leaky_convs(1, 1024)
        self.conv_leaky1_5 = make_leaky_convs(1, 256)
        self.conv_leaky1_6 = Conv2D_BN_Mish(256, 3, 2)
        self.conv_leaky1_7 = Conv2D_BN_Mish(256, 3, 2)
        self.conv_leaky5_1 = make_leaky_convs(5, 256)
        self.conv_leaky5_2= make_leaky_convs(5, 128)
        self.conv_leaky5_3= make_leaky_convs(5, 512)
        self.spp_layer = spp()
        self.upsampling = tf.keras.layers.UpSampling2D(2)


    def call(self, x):
        #cspdarknet53
        first_conv_out = self.first_conv(x)
        res_block1_out = self.res_block1(first_conv_out)
        res_block2_out = self.res_block2(res_block1_out)
        res_block3_out = self.res_block3(res_block2_out)

        intermediate_1 = res_block3_out

        res_block4_out = self.res_block4(res_block3_out)

        intermediate_2 = res_block4_out

        res_block5_out = self.res_block5(res_block4_out)

        #spp
        pred_spp = self.conv_leaky3_1(res_block5_out)
        spp_out = self.spp_layer(pred_spp)

        succ_spp = self.conv_leaky3_2(spp_out)

        intermediate_3 = succ_spp

        head2_1 = self.conv_leaky1_1(intermediate_2)
        head2_2 = self.conv_leaky1_1(intermediate_3)
        head2_2 = self.upsampling(head2_2)
        head2 = tf.concat([head2_1, head2_2], axis=-1)
        head2 = self.conv_leaky5_1(head2)

        intermediate_4 = head2

        head1_1 = self.conv_leaky1_2(intermediate_1)
        head1_2 = self.conv_leaky1_2(intermediate_4)
        head1_2 = self.upsampling(head1_2)
        head1 = tf.concat([head1_1, head1_2], axis=-1)
        head1 = self.conv_leaky5_2(head1)

        intermediate_5 = head1

        head1 = self.conv_leaky1_5(head1)

        head1_out = self.conv_last1(head1)

        head2_3 = self.conv_leaky1_6(intermediate_5)

        head2 = tf.concat([intermediate_4, head2_3], axis=-1)
        head2 = self.conv_leaky5_1(head2)

        intermediate_6 = head2

        head2 = self.conv_leaky1_3(head2)
        head2_out = self.conv_last2(head2)

        head3_2 = self.conv_leaky1_7(intermediate_6)
        head3 = tf.concat([intermediate_3, head3_2], axis=-1)
        head3 = self.conv_leaky5_3(head3)
        head3 = self.conv_leaky1_4(head3)
        head3_out = self.conv_last3(head3)

        return head1_out, head2_out, head3_out

if __name__ == "__main__":
    model = Yolo_Model()
    x = tf.random.normal(shape=(1, 608, 608, 3))
    head1, head2, head3 = model(x)

    print("head1 shape: ", head1.shape)
    print("head2 shape: ", head2.shape)
    print("head3 shape: ", head3.shape)












        


















        


 
