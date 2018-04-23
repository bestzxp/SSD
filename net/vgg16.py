import tensorflow as tf
RGB_MEAN = [123.68, 116.78, 103.94]
class VGG16(object):
    def __init__(self, input_shape=(224, 224), weights_path=None, class_num=1000, train=True):
        self.weights_path = weights_path
        self.class_num = class_num
        self.height, self.width = input_shape
        self.train = train

    def build_model(self):
        self.input_holder = tf.placeholder(tf.float32, shape=[None, self.height, self.width, 3], name='input_holder')
        input_holder = self.input_holder -  RGB_MEAN
        self.conv1_1 = tf.layers.conv2d(input_holder, filters=64, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME')
        self.conv1_1 = tf.layers.batch_normalization(self.conv1_1, training=self.train, momentum=0.9)
        self.conv1_2 = tf.layers.conv2d(self.conv1_1, filters=64, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME')
        self.conv1_2 = tf.layers.batch_normalization(self.conv1_2, training=self.train, momentum=0.9)
        self.pool1 = tf.layers.max_pooling2d(self.conv1_2, pool_size=(2,2), strides=(2, 2))
        # ===============
        self.conv2_1 = tf.layers.conv2d(self.pool1, filters=128, kernel_size=(3, 3), activation=tf.nn.relu,
                                        padding='SAME')
        self.conv2_1 = tf.layers.batch_normalization(self.conv2_1, training=self.train, momentum=0.9)
        self.conv2_2 = tf.layers.conv2d(self.conv2_1, filters=128, kernel_size=(3, 3), activation=tf.nn.relu,
                                        padding='SAME')
        self.conv2_2 = tf.layers.batch_normalization(self.conv2_2, training=self.train, momentum=0.9)
        self.pool2 = tf.layers.max_pooling2d(self.conv2_2, pool_size=(2, 2), strides=(2, 2))
        # ===============
        self.conv3_1 = tf.layers.conv2d(self.pool2, filters=256, kernel_size=(3, 3), activation=tf.nn.relu,
                                        padding='SAME')
        self.conv3_1 = tf.layers.batch_normalization(self.conv3_1, training=self.train, momentum=0.9)
        self.conv3_2 = tf.layers.conv2d(self.conv3_1, filters=256, kernel_size=(3, 3), activation=tf.nn.relu,
                                        padding='SAME')
        self.conv3_2 = tf.layers.batch_normalization(self.conv3_2, training=self.train, momentum=0.9)
        self.conv3_3 = tf.layers.conv2d(self.conv3_2, filters=256, kernel_size=(3, 3), activation=tf.nn.relu,
                                        padding='SAME')
        self.conv3_3 = tf.layers.batch_normalization(self.conv3_3, training=self.train, momentum=0.9)
        self.pool3 = tf.layers.max_pooling2d(self.conv3_3, pool_size=(2, 2), strides=(2, 2))
        # ===============
        self.conv4_1 = tf.layers.conv2d(self.pool3, filters=512, kernel_size=(3, 3), activation=tf.nn.relu,
                                        padding='SAME')
        self.conv4_1 = tf.layers.batch_normalization(self.conv4_1, training=self.train, momentum=0.9)
        self.conv4_2 = tf.layers.conv2d(self.conv4_1, filters=512, kernel_size=(3, 3), activation=tf.nn.relu,
                                        padding='SAME')
        self.conv4_2 = tf.layers.batch_normalization(self.conv4_2, training=self.train, momentum=0.9)
        self.conv4_3 = tf.layers.conv2d(self.conv4_2, filters=512, kernel_size=(3, 3), activation=tf.nn.relu,
                                        padding='SAME')
        self.conv4_3 = tf.layers.batch_normalization(self.conv4_3, training=self.train, momentum=0.9)
        self.pool4 = tf.layers.max_pooling2d(self.conv4_3, pool_size=(2, 2), strides=(2, 2))
        # =========================
        self.conv5_1 = tf.layers.conv2d(self.pool4, filters=512, kernel_size=(3, 3), activation=tf.nn.relu,
                                        padding='SAME')
        self.conv5_1 = tf.layers.batch_normalization(self.conv5_1, training=self.train, momentum=0.9)
        self.conv5_2 = tf.layers.conv2d(self.conv5_1, filters=512, kernel_size=(3, 3), activation=tf.nn.relu,
                                        padding='SAME')
        self.conv5_2 = tf.layers.batch_normalization(self.conv5_2, training=self.train, momentum=0.9)
        self.conv5_3 = tf.layers.conv2d(self.conv5_2, filters=512, kernel_size=(3, 3), activation=tf.nn.relu,
                                        padding='SAME')
        self.conv5_3 = tf.layers.batch_normalization(self.conv5_3, training=self.train, momentum=0.9)
        return self.conv4_3, self.conv5_3


    def loss(self):
        return tf.nn.softmax_cross_entropy_with_logits(logits=self.fc8, labels=self.label)

