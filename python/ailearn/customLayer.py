import tensorflow as tf
from tensorflow import keras
import numpy as np

class RandomCutout(Layer):
    def __init__(self, mask_size=(50, 50), **kwargs):
        super(RandomCutout, self).__init__(**kwargs)
        self.mask_size = mask_size

    def call(self, inputs, training=True):
        if training:
            input_shape = tf.shape(inputs)
            batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

            mask_width, mask_height = self.mask_size
            
            # Randomly choose the top-left point of the mask
            pad_h = tf.maximum(0, height - mask_height)
            pad_w = tf.maximum(0, width - mask_width)
            top = tf.random.uniform([], 0, pad_h, dtype=tf.int32)
            left = tf.random.uniform([], 0, pad_w, dtype=tf.int32)

            mask = np.ones((batch_size, height, width, channels), dtype=np.float32)
            mask[:, top:top+mask_height, left:left+mask_width, :] = 0

            return inputs * mask
        else:
            return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "mask_size": self.mask_size,
        })
        return config

# Now, include it in your data_augmentation Sequential model
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
        RandomCutout(mask_size=(50, 50))
    ]
)