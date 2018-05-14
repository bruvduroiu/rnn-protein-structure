import yaml

import numpy as np
import tensorflow as tf

from sklearn import metrics

from model.rnn_utils import DeviceCellWrapper

class ProteinRNN:
    def __init__(self, session=None, name='ProteinRNN', param_file='model/params.yml'):
        self.name = name
        self.params = self._load_params(param_file=param_file)

        self.seq_length = self.params['seq_length']
        self.n_amino = self.params['n_amino']
        self.n_structures = self.params['n_structures']

        self._build_graph()
        self._build_optimiser()

        self.sess = session or tf.Session()
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
            with tf.device('/cpu:0'):
                self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.n_amino])
                self.y = tf.placeholder(tf.int32, [None, self.seq_length, self.n_structures])

                X_embed_reshape = tf.reshape(self.X, [-1, self.n_amino])
                self.W_embedding = tf.layers.dense(X_embed_reshape, **self.params['embedding_layer'])

                X_in = tf.reshape(self.W_embedding, [-1, 1, 700, 50])

            with tf.name_scope('CNN_Cascade'):
                # Conv Block 1: [3 x 50] kernel size 
                with tf.device('/cpu:0'):
                    self.conv1 = tf.layers.conv2d(X_in, **self.params['conv1'])
                self.conv1_transpose = tf.transpose(self.conv1, perm=[0,2,1,3])
                self.conv1_reshape = tf.reshape(self.conv1_transpose, [-1, 700, 64])
                self.conv1_bn = tf.layers.batch_normalization(self.conv1_reshape)

                # Conv Block 2: [7 x 50 kernel size]
                with tf.device('/cpu:0'):
                    self.conv2 = tf.layers.conv2d(X_in, **self.params['conv2'])
                self.conv2_transpose = tf.transpose(self.conv2, perm=[0,2,1,3])
                self.conv2_reshape = tf.reshape(self.conv2_transpose, [-1, 700, 64])
                self.conv2_bn = tf.layers.batch_normalization(self.conv2_reshape)

                # Conv Block 3: [11 x 50 kernel size]
                with tf.device('/cpu:0'):
                    self.conv3 = tf.layers.conv2d(X_in, **self.params['conv3'])
                self.conv3_transpose = tf.transpose(self.conv3, perm=[0,2,1,3])
                self.conv3_reshape = tf.reshape(self.conv2_transpose, [-1, 700, 64])
                self.conv3_bn = tf.layers.batch_normalization(self.conv3_reshape)

                # Concatenate CNNs & apply batch norm
                self.concat_cnn = tf.concat([self.conv1_bn, self.conv2_bn, self.conv3_bn], axis=2)
                self.concat_cnn_bn = tf.layers.batch_normalization(self.concat_cnn)
                self.concat_cnn_bn_r = tf.reverse(self.concat_cnn_bn, axis=[1])

            # Placeholder for dynamic Dropout (on during training, off during testing)
            self.gru_keep_prob = tf.placeholder_with_default(1.0, shape=())

            # Bi-directional GRU Block1
            with tf.name_scope('GRU_Block1'):
                gru_layer_1 = tf.contrib.rnn.GRUCell(**self.params['gru_layer1'])

                # Map forward and backward GRUs onto different devices
                with tf.name_scope('fwd') as fwd_scope:
                    with tf.device('/cpu:0') as device:
                        gru_fwd_1 = DeviceCellWrapper(device, gru_layer_1, scope=fwd_scope)
                    gru_fwd_1_drop = tf.contrib.rnn.DropoutWrapper(gru_fwd_1, 
                                                                   input_keep_prob=self.gru_keep_prob)
                    self.fwd1_out, _ = tf.nn.dynamic_rnn(gru_fwd_1_drop, self.concat_cnn_bn,
                                                         dtype=tf.float32, scope=fwd_scope)
                # Feed in output backward
                with tf.name_scope('bwd') as bwd_scope:
                    with tf.device('/cpu:0') as device:
                        gru_bwd_1 = DeviceCellWrapper(device, gru_layer_1, scope=bwd_scope)
                    gru_bwd_1_drop = tf.contrib.rnn.DropoutWrapper(gru_bwd_1,
                                                                   input_keep_prob=self.gru_keep_prob)
                    bwd1_out_r, _ = tf.nn.dynamic_rnn(gru_bwd_1_drop, self.concat_cnn_bn_r,
                                                      dtype=tf.float32, scope=bwd_scope)
                    # Reverse to match fwd1_out
                    self.bwd1_out = tf.reverse(bwd1_out_r, axis=[1])
                self.bgru1 = tf.concat([self.fwd1_out, self.bwd1_out], axis=2)
                self.bgru1_r = tf.reverse(self.bgru1, axis=[1])

            # Bi-directional GRU Block2
            with tf.name_scope('GRU_Block2'):
                gru_layer_2 = tf.contrib.rnn.GRUCell(**self.params['gru_layer2'])

                with tf.name_scope('fwd') as fwd_scope:
                    with tf.device('/cpu:0') as device:
                        gru_fwd_2 = DeviceCellWrapper(device, gru_layer_2, scope=fwd_scope)
                    gru_fwd_2_drop = tf.contrib.rnn.DropoutWrapper(gru_fwd_2,
                                                                   input_keep_prob=self.gru_keep_prob)
                    self.fwd2_out, _ = tf.nn.dynamic_rnn(gru_fwd_2_drop, self.bgru1,
                                                         dtype=tf.float32, scope=fwd_scope)
                # Feed in output backward
                with tf.name_scope('bwd') as bwd_scope:
                    with tf.device('/cpu:0') as device:
                        gru_bwd_2 = DeviceCellWrapper(device, gru_layer_2, scope=bwd_scope)
                    gru_bwd_2_drop = tf.contrib.rnn.DropoutWrapper(gru_bwd_2,
                                                                   input_keep_prob=self.gru_keep_prob)
                    bwd2_out_r, _ = tf.nn.dynamic_rnn(gru_bwd_2_drop, self.bgru1_r,
                                                      dtype=tf.float32, scope=bwd_scope)
                    # Reverse to match fwd2_out
                    self.bwd2_out = tf.reverse(bwd2_out_r, axis=[1])
                self.bgru2 = tf.concat([self.fwd2_out, self.bwd2_out], axis=2)
                self.bgru2_r = tf.reverse(self.bgru2, axis=[1])

            # Bi-directional GRU Block3
            with tf.name_scope('GRU_Block3'):
                gru_layer_3 = tf.contrib.rnn.GRUCell(**self.params['gru_layer3'])

                with tf.name_scope('fwd') as fwd_scope:
                    with tf.device('/cpu:0') as device:
                        gru_fwd_3 = DeviceCellWrapper(device, gru_layer_3, scope=fwd_scope)
                    gru_fwd_3_drop = tf.contrib.rnn.DropoutWrapper(gru_fwd_3,
                                                                   input_keep_prob=self.gru_keep_prob)
                    self.fwd3_out, _ = tf.nn.dynamic_rnn(gru_fwd_3_drop, self.bgru2,
                                                         dtype=tf.float32, scope=fwd_scope)
                # Feed in output backward
                with tf.name_scope('bwd') as bwd_scope:
                    with tf.device('/cpu:0') as device:
                        gru_bwd_3 = DeviceCellWrapper(device, gru_layer_3, scope=bwd_scope)
                    gru_bwd_3_drop = tf.contrib.rnn.DropoutWrapper(gru_bwd_3,
                                                                   input_keep_prob=self.gru_keep_prob)
                    bwd3_out_r, _ = tf.nn.dynamic_rnn(gru_bwd_3_drop, self.bgru2_r, dtype=tf.float32,
                                                      scope=bwd_scope)
                    # Reverse to match fwd1_out
                    self.bwd3_out = tf.reverse(bwd3_out_r, axis=[1])

            # Concatenation of GRU Block 3 FWD + BWD + Cascaded CNNs
            self.local_global_context = tf.concat([self.fwd3_out, self.bwd3_out, self.concat_cnn], axis=2)

            # FWD(300) + BWD(300) + Conv1(64) + Conv2(64) + Conv3(64)
            self.reshape_context = tf.reshape(self.local_global_context, [-1, 300+300+64+64+64])

            self.fc_keep_prob = tf.placeholder_with_default(1.0, shape=())

            with tf.device('/cpu:0') as device:
                self.fc_1 = tf.layers.dense(self.reshape_context, **self.params['fc_1'])
                self.drop_1 = tf.layers.dropout(self.fc_1, rate=self.fc_keep_prob)
                self.fc_2 = tf.layers.dense(self.drop_1, **self.params['fc_2'])
                self.drop_2 = tf.layers.dropout(self.fc_2, rate=self.fc_keep_prob)
                
            self.logits = tf.layers.dense(self.drop_2, **self.params['out'])
            self.y_proba_flat = tf.nn.softmax(self.logits)
            self.y_proba = tf.reshape(self.y_proba_flat, [-1, self.seq_length, self.n_structures])
            self.y_pred = tf.argmax(self.y_proba, axis=2, name='y_pred')
        
            self.y_flat = tf.reshape(self.y, [-1, self.n_structures])
            self.y_pred_flat = tf.argmax(self.y_proba_flat, axis=1)

            y_label = tf.argmax(self.y_flat, axis=1)
            correct = tf.equal(y_label, self.y_pred_flat)

            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

            self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_flat,
                                                                logits=self.logits,
                                                                name='loss')
            self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)

    def _build_optimiser(self):
        with tf.name_scope('optimiser'):
            optimiser = tf.train.AdamOptimizer()
            self.training_op = optimiser.minimize(self.loss, var_list=self.train_vars, name='training_op')

    def train(self, data, restore_checkpoint=False, checkpoint_path='./checkpoints'):
        self.n_epochs = self.params['train']['epochs']
        self.batch_size = self.params['train']['batch_size']
        self.valid_batch_size = self.params['valid']['batch_size']
        n_iterations_per_epoch = data['X_train'].shape[0] // self.batch_size
        n_iterations_validation = data['y_valid'].shape[0] // self.valid_batch_size
        best_loss_valid = np.infty

        gru_dropout = self.params['gru_dropout']
        fc_dropout = self.params['fc_dropout']

        if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
            self.saver.restore(self.sess, checkpoint_path)
        else:
            tf.global_variables_initializer().run(session=self.sess)

        # Assumes a dictionary will be passed to the train function
        X_train, y_train, X_test, y_test, X_valid, y_valid = data.values()

        epochs_train = []
        epochs_valid = []
        losses_train = []
        losses_valid = []
        accuracy_valid = []
        n_since_save = 0
        EARLY_STOPPING_N = 10

        for epoch in range(self.n_epochs):
            for iteration in range(n_iterations_per_epoch):
                start = iteration * self.batch_size
                end = min((iteration + 1) * self.batch_size, y_train.shape[0])
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                feed_dict = {
                    self.X: X_batch,
                    self.y: y_batch,
                    self.gru_keep_prob: gru_dropout,
                    self.fc_keep_prob: fc_dropout
                }

                _, loss_train = self.sess.run([self.training_op, self.loss],
                                              feed_dict=feed_dict)
                
                loss_train = np.squeeze(loss_train)
                loss_train = loss_train[0]
                epochs_train.append(epoch)
                losses_train.append(loss_train)
            
            train_loss_str = '\rIteration: {}/{} ({:.1f}%)  Loss: {:.3f}\n'.format(
                    iteration, n_iterations_per_epoch,
                    iteration * 100 / n_iterations_per_epoch,
                    loss_train)
            with open('train_progress.txt', 'a') as file:
                file.write(train_loss_str)
            print(train_loss_str)

            loss_valid_ = []
            acc_valid_ = []
            for iteration in range(n_iterations_validation):
                start = iteration * self.valid_batch_size
                end = min((iteration + 1) * self.valid_batch_size, y_valid.shape[0])

                X_batch = X_valid[start:end]
                y_batch = y_valid[start:end]

                feed_dict = {self.X: X_batch, self.y: y_batch}
                loss_valid, acc_valid = self.sess.run([self.loss, self.accuracy],
                                                      feed_dict=feed_dict)

                loss_valid = np.squeeze(loss_valid)
                acc_valid = np.squeeze(acc_valid)

                loss_valid = loss_valid[0]

                loss_valid_.append(loss_valid)
                acc_valid_.append(acc_valid)

            loss_valid = np.mean(loss_valid_)
            acc_valid = np.mean(acc_valid_)

            epochs_valid.append(epoch)
            losses_valid.append(loss_valid)
            accuracy_valid.append(acc_valid)

            valid_loss_str = '\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}\n'.format(
                epoch + 1, acc_valid * 100, loss_valid,
                ' (improved)' if loss_valid < best_loss_valid else '')
            with open('valid_progress.txt', 'a') as file:
                file.write(valid_loss_str)
            print(valid_loss_str)

            if loss_valid < best_loss_valid:
                save_path = self.saver.save(self.sess, checkpoint_path)
                best_loss_valid = loss_valid
                n_since_save = 0
            else:
                n_since_save += 1

            if n_since_save >= EARLY_STOPPING_N:
                break
        
        train_loss_data = np.array([epochs_train, losses_train])
        np.save('train_loss_data', train_loss_data)

        valid_loss_data = np.array([epochs_train, losses_valid, accuracy_valid])
        np.save('valid_loss_data', valid_loss_data)

    def load(self, checkpoint_path='./checkpoints'):
        # Restores session from last checkpoint
        if tf.train.checkpoint_exists(checkpoint_path):
            self.saver.restore(self.sess, checkpoint_path)
            print('Checkpoint Restored')
        else:
            tf.global_variables_initializer().run(session=self.sess)
            print('Could not restore. Variables re-initialized.')

    def predict(self):
        pass

    def eval(self, X_test, y_test):
        target_names = self.params['labels']

        feed_dict = {self.X: X_test, self.y: y_test}
        predictions = self.sess.run([self.y_proba], feed_dict=feed_dict)

        predictions = predictions[0]

        return self.eval_classifier(y_test, predictions, target_names=target_names)

    def eval_classifier(self, y_true, y_pred, target_names=None):
        '''y_true and y_pred should be encoded as probailities eg one-hot for labels'''
        y_true = np.argmax(y_true, axis=2).reshape((-1,))
        y_pred = np.argmax(y_pred, axis=2).reshape((-1,))
        print('{:=^80}'.format('Evaluation'))
        accuracy = metrics.accuracy_score(y_true, y_pred)
        print('Accuracy = %.3f' % accuracy)
        print('{:-^80}'.format('Classification report'))
        report = metrics.classification_report(y_true, y_pred, target_names=target_names)
        print(report)
        cm = metrics.confusion_matrix(y_true, y_pred)
        print('{:-^80}'.format('Confusion Matrix'))
        print(cm)
        return accuracy, cm, report

if __name__ == '__main__':
    p = ProteinRNN(param_file='params.yml')