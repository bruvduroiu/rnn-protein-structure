import os

import numpy as np
import tensorflow as tf

from model.protein_rnn import ProteinRNN

def main():
    data_raw = np.load('datasets/cb513.npz')
    dataset_dict = dict(data_raw.items())

    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:

        # Form Model
        protein_rnn = ProteinRNN(session=sess)

        # Train model
        protein_rnn.train(data=dataset_dict)

if __name__ == '__main__':
    main()
