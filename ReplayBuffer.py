from collections import deque
import random
from collections import namedtuple
import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self, buffer_size, odim, adim, batch_size=32):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        # self.buffer = deque(maxlen=self.buffer_size)
        # self.buffer = deque(maxlen=self.buffer_size)
        # self.buffer = deque(maxlen=self.buffer_size)
        # self.buffer = deque(maxlen=self.buffer_size)
        self.batch_size = batch_size
        self.odim = odim
        self.adim = adim

    def append(self, o, a, r, o1, d):
        # o, a, r, o1, d = [tf.convert_to_tensor(p) for p in [o, a, r, o1, d]]
        self.buffer.append([o, a, r, o1, d])

    def sample(self):
        size = self.batch_size if len(self.buffer) > self.batch_size else len(self.buffer)
        minibatch = np.transpose(np.array(random.sample(self.buffer, size),dtype='object'))

        # a = tf.constant(value=[minibatch[0][0]])
        b = []
        for a in range(5):
            if type(minibatch[a][0]) == np.ndarray:
                b.append(np.concatenate(minibatch[a]).reshape(size, *self.odim))
            else:
                # b.append(minibatch[a].reshape(size, -1))
                b.append(minibatch[a])

        return b[0], b[1], b[2], b[3], b[4]