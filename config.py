# Configuration, set model parameter
class Config:
    def __init__(self):
        #Buffer
        self.steps_per_epoch = 5000
        self.gamma = 0.99
        self.buffer_size = 10000
        self.mini_batch_size = 32
        #Update
        self.epochs = 1000000
        self.evaluate_every = 10
