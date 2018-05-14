import tensorflow as tf

class DeviceCellWrapper(tf.contrib.rnn.RNNCell):
    '''Wrapper that allows pinning a tf.contrib.rnn.RNNCell to a specific device.
    Also, allows for specifying the scope of the cell.

    Adapted implementation from Aurélien Geron's work in 
    'Hands-On Machine Learning with Scikit-Learn & Tensorflow'
    '''
    def __init__(self, device, cell=None, scope=None):
        self._cell = cell
        self._device = device
        self._scope = scope

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        with tf.device(self._device):
            return self._cell(inputs, state, scope=self._scope)
