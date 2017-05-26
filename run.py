import sys
import argparse
import tensorflow as tf

from model import CNNAutoencoder
from input_handler import load_data, batch_generator

tf.logging.set_verbosity(tf.logging.INFO)
# Basic model parameters as external flags.
FLAGS = None


def _train_step(X_batch, session, model, print_every=100):
    feed_dict = {
        model.X: X_batch
    }

    _, step, loss = session.run([model.train_op, model.global_step, model.loss], feed_dict)

    if step % print_every == 0:
        print("step: %d\tloss: %.8f" % (step, loss))


def train(img_height=28, img_width=28, img_depth=3, n_epochs=10, batch_size=128, mode='train'):
    # Load Data
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_data('../data/MNIST_M.npy')
    X_train = X_train.astype('float32') / 255.
    X_dev = X_dev.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    # num_training_data = X_train.shape[0]

    with tf.Graph().as_default():
        sess = tf.Session()

        with sess.as_default():
            autoencoder = CNNAutoencoder(img_height, img_width, img_depth, mode)
            autoencoder.build_graph()

            # train_summary_op = tf.summary.merge([autoencoder.grad_summaries, autoencoder.acc_summary, autoencoder.loss_summary])
            # train_summary_dir = os.path.join(FLAGS.output_dir, "sentiment", "summaries", "train")
            # train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # dev_summary_op = tf.summary.merge([autoencoder.loss_summary, autoencoder.acc_summary])
            # dev_summary_dir = os.path.join(FLAGS.output_dir, "sentiment", "summaries", "dev")
            # dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # checkpoint_dir = os.path.join(".", "checkpoints")
            # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            # if not os.path.exists(checkpoint_dir):
            #     os.makedirs(checkpoint_dir)

            # Initialize all variables
            tf.logging.info('Create new session')
            sess.run(tf.global_variables_initializer())

            # num_batch_per_epoch = num_training_data // batch_size
            batches = batch_generator(X_train, y_train, batch_size, n_epochs, shuffle=True)

            for batch in batches:
                X_batch, _ = zip(*batch)
                _train_step(X_batch, sess, autoencoder)


def run(_):
    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'eval':
        pass
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        help='Specify mode: `train` or `eval`',
        required=True
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
