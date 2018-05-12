import click

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@click.command()
@click.option('-e', '--epochs', default=1000, help='epochs')
@click.option('-b', '--batch-size', default=128, help='batch size')
def main(epochs, batch_size):
    raw_data = np.load('datasets/cullpdb+profile_6133.npy')
    raw_data = np.reshape(raw_data, (raw_data.shape[0], -1, 57))
    data = raw_data[:,:,:21]
    data = np.reshape(data, (-1, 21))

    n_batches = int(np.floor(data.shape[0] / batch_size))

    X = tf.placeholder(tf.float32, shape=[None, 21])
    Wh = tf.get_variable('wh', shape=(21, 2), initializer=tf.random_normal_initializer())
    e_0 = tf.nn.tanh(tf.matmul(X, Wh))
    Wo = tf.get_variable('wo', shape=(2, 21), initializer=tf.random_normal_initializer())
    y = tf.nn.tanh(tf.nn.tanh(tf.matmul(e_0, Wo)))

    loss = tf.reduce_mean(tf.subtract(y, X))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    train_op = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            for i in range(n_batches):
                batch_data = data[i*batch_data:(i+1)*batch_size]
                loss_ = sess.run([train_op], feed_dict={X: batch_data})

            if epoch % 100 == 0:
                print('Loss: {:2f}'.format(loss_))

if __name__ == '__main__':
    main()