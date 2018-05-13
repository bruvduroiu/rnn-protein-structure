import yaml

import numpy as np
import tensorflow as tf

class ProteinRNN:
    def __init__(self, sess=None, name='ProteinRNN', param_file='model/params.yml'):
        self.name = name
        self.params = self._load_params(param_file=param_file)

        self.seq_length = self.params['seq_length']
        self.n_amino = self.params['n_amino']
        self.n_structures = self.params['n_structures']

        self._build_graph()
        self._build_optimiser()

        self.sess = sess or tf.Session()
        self.saver = tf.train.Saver()

    def _load_params(self, param_file='model/params.yml'):
        '''Loads the parameters for every layer from a YAML file defined
        in `model/params.yml`
        Returns:
            params: Dictionary representing layer -> parameter mapping
        '''
        stream = open('model/params.yml', 'r')
        param_strem = yaml.load_all(stream)
        params = {}
        # load_all returns a generator
        for param in param_strem:
            for k in param.keys():
                params[k] = param[k]

        return params

    def _build_graph(self):
        with tf.variable_scope(self.name) as scope:
            self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.n_amino])
            self.y = tf.placeholder(tf.int32, [None, self.n_structures])

            X_embed_reshape = tf.reshape(self.X, [-1, 22])
            self.W_embedding = tf.layers.dense(X_embed_reshape, **self.params['embedding_layer'])

            X_in = tf.reshape(self.W_embedding, [-1, 1, 700, 50])

            with tf.name_scope('CNN_Cascade'):
                # Conv Block 1: [3 x 50] kernel size
                self.conv1 = tf.layers.conv2d(X_in, **self.params['conv1'])
                self.conv1_transpose = tf.transpose(self.conv1, perm=[0,2,1,3])
                self.conv1_reshape = tf.reshape(self.conv1_transpose, [-1, 700, 64])
                self.conv1_bn = tf.layers.batch_normalization(self.conv1_reshape)

                # Conv Block 2: [7 x 50 kernel size]
                self.conv2 = tf.layers.conv2d(X_in, **self.params['conv2'])
                self.conv2_transpose = tf.transpose(self.conv2, perm=[0,2,1,3])
                self.conv2_reshape = tf.reshape(self.conv2_transpose, [-1, 700, 64])
                self.conv2_bn = tf.layers.batch_normalization(self.conv2_reshape)

                # Conv Block 3: [11 x 50 kernel size]
                self.conv3 = tf.layers.conv2d(X_in, **self.params['conv3'])
                self.conv3_transpose = tf.transpose(self.conv3, perm=[0,2,1,3])
                self.conv3_reshape = tf.reshape(self.conv2_transpose, [-1, 700, 64])
                self.conv3_bn = tf.layers.batch_normalization(self.conv3_reshape)

                # Concatenate CNNs & apply batch norm
                self.concat_cnn = tf.concat([self.conv1_bn, self.conv2_bn, self.conv3_bn], axis=2)
                self.concat_cnn_bn = tf.layers.batch_normalization(self.concat_cnn)

            # Placeholder for dynamic Dropout (on during training, off during testing)
            self.gru_keep_prob = tf.placeholder_with_default(1.0, shape=())

            # Bi-directional GRU Block1
            with tf.name_scope('GRU_Block1'):
                gru_layer_1 = tf.contrib.rnn.GRUCell(**self.params['gru_layer1'])
                gru_layer_1_drop = tf.contrib.rnn.DropoutWrapper(gru_layer_1, 
                                                                input_keep_prob=self.gru_keep_prob)
                with tf.name_scope('fwd') as fwd_scope:
                    self.fwd1_out, _ = tf.nn.dynamic_rnn(gru_layer_1_drop, self.concat_cnn_bn,
                                                        dtype=tf.float32, scope=fwd_scope)
                # Feed in output backward
                with tf.name_scope('bwd') as bwd_scope:
                    concat_cnn_bn_r = tf.reverse(self.concat_cnn_bn, axis=[1])
                    bwd1_out_r, _ = tf.nn.dynamic_rnn(gru_layer_1_drop, concat_cnn_bn_r, dtype=tf.float32,
                                                    scope=bwd_scope)
                    # Reverse to match fwd1_out
                    self.bwd1_out = tf.reverse(bwd1_out_r, axis=[1])
                self.bgru1 = tf.concat([self.fwd1_out, self.bwd1_out], axis=2)

            # Bi-directional GRU Block2
            with tf.name_scope('GRU_Block2'):
                gru_layer_2 = tf.contrib.rnn.GRUCell(**self.params['gru_layer2'])
                gru_layer_2_drop = tf.contrib.rnn.DropoutWrapper(gru_layer_2, 
                                                                input_keep_prob=self.gru_keep_prob)
                with tf.name_scope('fwd') as fwd_scope:
                    self.fwd2_out, _ = tf.nn.dynamic_rnn(gru_layer_2_drop, self.bgru1,
                                                        dtype=tf.float32, scope=fwd_scope)
                # Feed in output backward
                with tf.name_scope('bwd') as bwd_scope:
                    bgru1_r = tf.reverse(self.bgru1, axis=[1])
                    bwd2_out_r, _ = tf.nn.dynamic_rnn(gru_layer_2_drop, bgru1_r, dtype=tf.float32,
                                                    scope=bwd_scope)
                    # Reverse to match fwd1_out
                    self.bwd2_out = tf.reverse(bwd2_out_r, axis=[1])
                self.bgru2 = tf.concat([self.fwd2_out, self.bwd2_out], axis=2)

            # Bi-directional GRU Block3
            with tf.name_scope('GRU_Block3'):
                gru_layer_3 = tf.contrib.rnn.GRUCell(**self.params['gru_layer3'])
                gru_layer_3_drop = tf.contrib.rnn.DropoutWrapper(gru_layer_3, 
                                                                input_keep_prob=self.gru_keep_prob)
                with tf.name_scope('fwd') as fwd_scope:
                    self.fwd3_out, _ = tf.nn.dynamic_rnn(gru_layer_3_drop, self.bgru2,
                                                         dtype=tf.float32, scope=fwd_scope)
                # Feed in output backward
                with tf.name_scope('bwd') as bwd_scope:
                    bgru2_r = tf.reverse(self.bgru2, axis=[1])
                    bwd3_out_r, _ = tf.nn.dynamic_rnn(gru_layer_3_drop, bgru2_r, dtype=tf.float32,
                                                    scope=bwd_scope)
                    # Reverse to match fwd1_out
                    self.bwd3_out = tf.reverse(bwd3_out_r, axis=[1])

            # Concatenation of GRU Block 3 FWD + BWD + Cascaded CNNs
            self.local_global_context = tf.concat([self.fwd3_out, self.bwd3_out, self.concat_cnn], axis=2)

            # FWD(300) + BWD(300) + Conv1(64) + Conv2(64) + Conv3(64)
            self.reshape_context = tf.reshape(self.local_global_context, [-1, 300+300+64+64+64])

            self.fc_keep_prob = tf.placeholder_with_default(1.0, shape=())

            self.fc_1 = tf.layers.dense(self.reshape_context, **self.params['fc_1'])
            self.drop_1 = tf.layers.dropout(self.fc_1, rate=self.fc_keep_prob)
            self.fc_2 = tf.layers.dense(self.drop_1, **self.params['fc_2'])
            self.drop_2 = tf.layers.dropout(self.fc_2, rate=self.fc_keep_prob)
            
            self.logits = tf.layers.dense(self.drop_2, **self.params['out'])
            self.y_proba = tf.nn.softmax(self.logits)

            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                                logits=self.y_proba)
            self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    def _build_optimiser(self):
        with tf.name_scope('optimiser'):
            optimiser = tf.train.AdamOptimizer()
            self.training_op = optimiser.minimize(self.loss, var_list=self.train_vars, name='training_op')

    def train(self):
        pass

    def predict(self):
        pass

if __name__ == '__main__':
    p = ProteinRNN(param_file='params.yml')