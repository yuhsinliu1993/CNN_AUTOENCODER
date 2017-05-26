import tensorflow as tf


class CNNAutoencoder(object):

    def __init__(self, img_height, img_width, img_depth, mode):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.mode = mode

    def build_graph(self):
        self.global_step = tf.contrib.framework.get_or_create_global_step()

        self._build_model()

        if self.mode == 'train':
            self._build_train_op()

    def _build_train_op(self):
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars,
                                                  global_step=self.global_step,
                                                  name="train_step")

        # self.grad_summaries = []
        # for grad, var in grads_and_vars:
        #     if grad is not None:
        #         name = re.sub(r':', '_', var.name)
        #         grad_hist_summary = tf.summary.histogram("{}/grad/histogram".format(name), grad)
        #         sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(name), tf.nn.zero_fraction(grad))
        #         self.grad_summaries.append(grad_hist_summary)
        #         self.grad_summaries.append(sparsity_summary)

    def _build_model(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.img_height, self.img_width, self.img_depth), name='X')

        # Encoder
        self.x = tf.layers.conv2d(self.X, filters=16, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='conv1')
        self.x = tf.layers.max_pooling2d(self.x, pool_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        self.x = tf.layers.conv2d(self.x, filters=8, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='conv2')
        self.x = tf.layers.max_pooling2d(self.x, pool_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        self.x = tf.layers.conv2d(self.x, filters=8, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='conv3')
        self.encoded = tf.layers.max_pooling2d(self.x, pool_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')

        # Decoder
        self.x = tf.layers.conv2d(self.encoded, filters=8, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='conv4')
        self.x = tf.layers.conv2d_transpose(self.x, filters=8, kernel_size=(2, 2), strides=(2, 2), name='decon4')
        self.x = tf.layers.conv2d(self.x, filters=8, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='conv5')
        self.x = tf.layers.conv2d_transpose(self.x, filters=8, kernel_size=(2, 2), strides=(2, 2), name='decon5')
        self.x = tf.layers.conv2d(self.x, filters=16, kernel_size=(3, 3), activation=tf.nn.relu, name='conv6')
        self.x = tf.layers.conv2d_transpose(self.x, filters=16, kernel_size=(2, 2), strides=(2, 2), name='decon6')
        self.decoded = tf.layers.conv2d(self.x, filters=3, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='conv7')

        # Use mean-square-error as loss
        self.loss = tf.reduce_mean(tf.square(self.X - self.decoded))
