import numpy as np


def load_data(data_path):
    MNIST_M = np.load(data_path)
    train_data, train_label = MNIST_M[0]
    valid_data, valid_label = MNIST_M[1]
    test_data, test_label = MNIST_M[2]

    return train_data, train_label, valid_data, valid_label, test_data, test_label


def batch_generator(X, y, batch_size, num_epochs, shuffle=True):
    data_size = X.shape[0]
    num_batches_per_epoch = data_size // batch_size + 1

    for epoch in range(num_epochs):
        # print("In epoch >> " + str(epoch + 1))
        # print("num batches per epoch is: " + str(num_batches_per_epoch))

        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            X_shuffled = X[shuffle_indices]
            y_shuffled = y[shuffle_indices]
        else:
            X_shuffled = X
            y_shuffled = y

        for batch_num in range(num_batches_per_epoch - 1):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            X_batch = X_shuffled[start_index:end_index]
            y_batch = y_shuffled[start_index:end_index]
            batch = list(zip(X_batch, y_batch))

            yield batch
