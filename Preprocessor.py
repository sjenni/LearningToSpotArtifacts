from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


class Preprocessor:
    def __init__(self, target_shape, augment_color=False, aspect_ratio_range=(0.75, 1.33), area_range=(0.333, 1.0)):
        self.target_shape = target_shape
        self.augment_color = augment_color
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range

    def central_crop(self, image):
        # Crop the central region of the image with an area containing 85% of the original image.
        image = tf.image.central_crop(image, central_fraction=0.85)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [self.target_shape[0], self.target_shape[1]], align_corners=False)
        image = tf.squeeze(image, [0])

        # Resize to output size
        image.set_shape([self.target_shape[0], self.target_shape[1], 3])
        return image

    def extract_random_patch(self, image):
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            [[[0, 0, 1, 1]]],
            aspect_ratio_range=self.aspect_ratio_range,
            area_range=self.area_range,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        image = tf.slice(image, bbox_begin, bbox_size)
        image = tf.expand_dims(image, 0)
        resized_image = tf.cond(
            tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
            true_fn=lambda: tf.image.resize_bilinear(image, self.target_shape[:2], align_corners=False),
            false_fn=lambda: tf.image.resize_bicubic(image, self.target_shape[:2], align_corners=False))
        image = tf.squeeze(resized_image)
        image.set_shape(self.target_shape)
        return image

    def color_and_scale(self, image, bright_max_delta=32. / 255., lower_sat=0.5, upper_sat=1.5):
        image = tf.to_float(image)/255.

        if self.augment_color:
            image = tf.cond(tf.random_uniform(shape=(), minval=0.0, maxval=1.0) > 0.5,
                            true_fn=lambda: tf.image.random_saturation(
                                tf.image.random_brightness(image, max_delta=bright_max_delta),
                                lower=lower_sat, upper=upper_sat),
                            false_fn=lambda: tf.image.random_brightness(
                                tf.image.random_saturation(image, lower=lower_sat, upper=upper_sat),
                                max_delta=bright_max_delta))
        # Scale to [-1, 1]
        image = tf.to_float(image) * 2. - 1.
        image = tf.clip_by_value(image, -1., 1.)
        return image

    def process_train(self, image):
        image = self.extract_random_patch(image)
        image = tf.image.random_flip_left_right(image)
        image = self.color_and_scale(image)
        return image

    def process_test(self, image):
        image = self.central_crop(image)
        image = self.color_and_scale(image)
        return image
