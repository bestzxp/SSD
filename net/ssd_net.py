import tensorflow as tf
from net.vgg16 import VGG16
class SSD(object):
    def __init__(self):
        pass

    def conv_bn_layer(self, inputs, output_channel, kernel_size, stride=1, padding='SAME', name=None):
        output = tf.layers.conv2d(inputs, filters=output_channel,
                                  kernel_size=(kernel_size, kernel_size),
                                  strides=(1, 1), padding=padding, name=name)
        output = tf.layers.batch_normalization(output)
        return output

    def build_model(self):
        # 对于输入为300， conv5_3输出为19*19*1024
        base_net = VGG16(input_shape=(300, 300), class_num=20)
        conv4_3, conv5_3 = base_net.build_model()
        # 19*19*1024
        conv6 = self.conv_bn_layer(conv5_3, 1024, 3)
        # 19*19*1024
        conv7 = self.conv_bn_layer(conv6, 1024, 1)

        # 10*10*512
        conv8_1 = self.conv_bn_layer(conv7, 256, 1, 1)
        conv8_2 = self.conv_bn_layer(conv8_1, 512, 3, 2)
        # 5*5*256
        conv9_1 = self.conv_bn_layer(conv8_2, 128, 1, 1)
        conv9_2 = self.conv_bn_layer(conv9_1, 256, 3, 2)
        # 3*3*256
        conv10_1 = self.conv_bn_layer(conv9_2, 128, 1, 1)
        conv10_2 = self.conv_bn_layer(conv10_1, 256, 3, 1, padding='VALID')

        # 1*1*256
        conv11_1 = self.conv_bn_layer(conv10_2, 128, 1, 1)
        conv11_2 = self.conv_bn_layer(conv11_1, 256, 3, 1, padding='VALID')

        out1 = self.conv_bn_layer(conv4_3, 4*(21 + 4), 3, 1)
        out2 = self.conv_bn_layer(conv7, 6 * (21 + 4), 3, 1)
        out3 = self.conv_bn_layer(conv8_2, 6 * (21 + 4), 3, 1)
        out4 = self.conv_bn_layer(conv9_2, 6 * (21 + 4), 3, 1)
        out5 = self.conv_bn_layer(conv10_2, 6 * (21 + 4), 3, 1)
        out6 = self.conv_bn_layer(conv11_2, 6 * (21 + 4), 3, 1)

        out1 = tf.reshape(out1, [-1, 38 * 38 * 4, 25])
        out2 = tf.reshape(out2, [-1, 19 * 19 * 6, 25])
        out3 = tf.reshape(out3, [-1, 10 * 10 * 6, 25])
        out4 = tf.reshape(out4, [-1, 5 * 5 * 6, 25])
        out5 = tf.reshape(out5, [-1, 3 * 3 * 6, 25])
        out6 = tf.reshape(out6, [-1, 1 * 1 * 4, 25])

        outputs = tf.concat([out1, out2, out3, out4, out5, out6], axis=1)

        cls_predict = outputs[:, :, :21]
        reg_predict = outputs[:, :, 21:]

        return cls_predict, reg_predict

net = SSD()
net.build_model()