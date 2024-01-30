import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
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

class RadialBlur(Layer):
    def __init__(self, blur_strength=0.5, **kwargs):
        super(RadialBlur, self).__init__(**kwargs)
        self.blur_strength = blur_strength

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        # Generate the x and y grids
        xs = tf.linspace(-1.0, 1.0, width)
        ys = tf.linspace(-1.0, 1.0, height)
        X, Y = tf.meshgrid(xs, ys)

        # Calculate the distance of each pixel from the center
        distance = tf.sqrt(X**2 + Y**2)

        # Calculate the blur mask
        blur_mask = 1.0 - (distance * self.blur_strength)
        blur_mask = tf.clip_by_value(blur_mask, 0.0, 1.0)

        # Reshape the mask
        blur_mask = blur_mask[:, :, tf.newaxis]
        blur_mask = tf.tile(blur_mask, [1, 1, channels])

        blur_mask = tf.expand_dims(blur_mask, 0)  # Add a batch dimension
        blur_mask = tf.tile(blur_mask, [batch_size, 1, 1, 1])  # Match the batch size of the inputs

        return inputs * blur_mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "blur_strength": self.blur_strength,
        })
        return config
    

import cv2

class RadialBlurLayer(Layer):
    def __init__(self, **kwargs):
        super(RadialBlurLayer, self).__init__(**kwargs)
    
    # def build(self, input_shape):
    #     super(RadialBlurLayer, self).build(input_shape)
    
    def call(self, inputs, training=True):
        if training:
            # Ensure the lambda function returns a tensor with the same dtype as the input
            blurred_images = tf.numpy_function(self.radial_blur_func, [inputs], tf.float32)
            # Adjust the shape of the output tensor to match the inputs
            blurred_images.set_shape(inputs.shape)
            return blurred_images
        else:
            return inputs
    
    def radial_blur_func(self, images):
        # images is already a numpy array here

        # Iterate over the batch of images to apply radial blur
        for i in range(images.shape[0]):
            image_np = images[i]

            # Randomly select a blur center
            height, width = image_np.shape[:2]
            center_x = np.random.randint(width*0.3, width*0.8)
            center_y = np.random.randint(height*0.3, height*0.8)

            # Apply the radial blur
            blurred_image_np = self.apply_radial_blur(image_np, center_x, center_y)

            # Update the batch of images with the blurred image
            images[i] = blurred_image_np
        
        return images

    def apply_radial_blur(self, img, center_x, center_y):
        # Applying radial blur effect using OpenCV
        
        ksize = 31  # Example kernel size for GaussianBlur, should be replaced with radial blur logic
        blur_img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        # blur_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
        
        # Create the radial mask centered at (center_x, center_y)
        y, x = np.indices((img.shape[0], img.shape[1]))
        mask = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask = np.exp(-(mask ** 2) / (2.0 * ksize ** 2))
        
        # Ensure mask values ​​lie between 0 and 1 and are of type float32
        mask = np.clip(mask, 0, 1).astype(np.float32)

        diff = img - blur_img
        for i in range(3):
            diff[:, :, i] = diff[:, :, i] * mask
        blended = diff + blur_img

        return blended
        
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(RadialBlurLayer, self).get_config()
        return base_config



# Now, include it in your data_augmentation Sequential model
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.2),
        RadialBlurLayer(),
    ]
)