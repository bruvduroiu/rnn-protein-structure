import os

import click
import numpy as np
import tensorflow as tf

from model.protein_rnn import ProteinRNN

@click.command()
@click.option('-d', '--data',
               help='Dataset for training. Should have format \
              (X_train, y_train, X_test, y_test, X_valid, y_valid).')
@click.option('--use-gpu/--no-use-gpu', default=False,
              help='Set this flag to enable GPU support')
@click.option('--cuda-devices', default='0, 1',
              help='CUDA Device indexes to expose to Tensorflow')
@click.option('--log-device-placement/--no-log-device-placement',
              default=False)
def main(data, use_gpu, cuda_devices, log_device_placement):
    data_raw = np.load(data)
    dataset_dict = dict(data_raw.items())

    config = tf.ConfigProto()
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices 
        config.gpu_options.allow_growth = True
        config.log_device_placement = log_device_placement
        config.allow_soft_placement = True

    with tf.Session(config=config) as sess:

        # Form Model
        protein_rnn = ProteinRNN(session=sess)

        # Train model
        protein_rnn.train(data=dataset_dict)

if __name__ == '__main__':
    main()
