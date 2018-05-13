import os

import numpy as np
import tensorflow as tf

from model.protein_rnn import ProteinRNN

def main():
    data_raw = np.load('datasets/cb513.npz')
    dataset_dict = dict(data_raw.items())

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        # Form Model
        protein_rnn = ProteinRNN(session=sess, param_file='model/params_test.yml')

        # Train model
        protein_rnn.train(data=dataset_dict)

if __name__ == '__main__':
    main()
