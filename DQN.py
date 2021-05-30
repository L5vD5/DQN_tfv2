import tensorflow as tf

class DQNNetwork(tf.keras.Model):
    def __init__(self, odim, adim):
        super(DQNNetwork, self).__init__()
        self.normalize = tf.keras.layers.Lambda(lambda x: x / 255.0)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0), activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))
        self.output_layer = tf.keras.layers.Dense(adim, activation='linear', kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0))

    def call(self, x):
        x = self.normalize(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.output_layer(x)
        return x
