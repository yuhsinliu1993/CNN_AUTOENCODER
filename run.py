import sys
import os
import argparse

import numpy as np
import tensorflow as tf

from cnn_autoencoder import CNNAutoencoder
from input_handler import load_data, batch_generator

tf.logging.set_verbosity(tf.logging.INFO)
# Basic model parameters as external flags.
FLAGS = None


def train(num_epochs=10, batch_size=200, mode='train', plot=False):

    # Load Data
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_data('./data/MNIST_M.npy')
    X_train = X_train.astype('float32') / 255.
    X_dev = X_dev.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    # Split the training data by depth. each has shape = (59000, 28, 28, 1)
    X_train_1, X_train_2, X_train_3 = np.split(X_train, 3, axis=3)

    num_samples, img_height, img_width, img_depth = X_train_1.shape[0], X_train_1.shape[1], X_train_1.shape[2], X_train_1.shape[3]

    with tf.Graph().as_default():
        sess = tf.Session()

        with sess.as_default():

            train_summary_writer_1 = tf.summary.FileWriter(os.path.join('summary', 'depth_1'), sess.graph)
            train_summary_writer_2 = tf.summary.FileWriter(os.path.join('summary', 'depth_2'), sess.graph)
            train_summary_writer_3 = tf.summary.FileWriter(os.path.join('summary', 'depth_3'), sess.graph)

            autoencoder_1 = CNNAutoencoder(img_height, img_width, img_depth, mode='train', name='autoencoder1')
            autoencoder_1.build_graph()

            autoencoder_2 = CNNAutoencoder(img_height, img_width, img_depth, mode='train', name='autoencoder2')
            autoencoder_2.build_graph()

            autoencoder_3 = CNNAutoencoder(img_height, img_width, img_depth, mode='train', name='autoencoder3')
            autoencoder_3.build_graph()

            # Initialize all variables
            tf.logging.info('Create new session')
            sess.run(tf.global_variables_initializer())

            batches_1 = batch_generator(X_train_1, y_train, batch_size, num_epochs, shuffle=True)
            batches_2 = batch_generator(X_train_2, y_train, batch_size, num_epochs, shuffle=True)
            batches_3 = batch_generator(X_train_3, y_train, batch_size, num_epochs, shuffle=True)

            print("\n[*] Training depth 1 ...")
            for batch in batches_1:
                X_batch, _ = zip(*batch)
                _train_step(X_batch, sess, autoencoder_1, train_summary_writer_1)

            print("\n[*] Training depth 2 ...")
            for batch in batches_2:
                X_batch, _ = zip(*batch)
                _train_step(X_batch, sess, autoencoder_2, train_summary_writer_2)

            print("\n[*] Training depth 3 ...")
            for batch in batches_3:
                X_batch, _ = zip(*batch)
                _train_step(X_batch, sess, autoencoder_3, train_summary_writer_3)

            if plot:
                x_sample_1 = X_train_1[:batch_size]
                x_sample_2 = X_train_2[:batch_size]
                x_sample_3 = X_train_3[:batch_size]
                x_sample = np.concatenate([x_sample_1, x_sample_2, x_sample_3], -1)

                x_recon_1 = autoencoder_1.get_reconstructed_images(sess, x_sample_1)
                x_recon_2 = autoencoder_2.get_reconstructed_images(sess, x_sample_2)
                x_recon_3 = autoencoder_3.get_reconstructed_images(sess, x_sample_3)

                x_recon = np.concatenate([x_recon_1, x_recon_2, x_recon_3], -1)

                _plot(sess, x_sample, x_recon)


def _train_step(X_batch, session, model, writer=None, print_every=50):

    feed_dict = {
        model.X: X_batch
    }

    _, loss, summary = session.run([model.train_op, model.loss, model.loss_summary], feed_dict)

    if writer is not None:
        writer.add_summary(summary, model.step)

    if model.step % print_every == 0 or model.step == 0:
        print("step: %4d\tloss: %.8f" % (model.step, loss))

    model.step += 1


def _plot(sess, figs, recons):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 12))
    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(figs[i], vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(recons[i], vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.show()


def evaluation():
    pass


def run(_):
    if FLAGS.mode == 'train':
        train(num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, mode=FLAGS.mode, plot=FLAGS.plot)
    elif FLAGS.mode == 'eval':
        evaluation()
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Specify the number of epoches',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Specify batch size',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='Specify mode: `train` or `eval`',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        default=False,
        help='Whether plot the reconstructed images or not',
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=run, argv=[sys.argv[0]] + unparsed)
