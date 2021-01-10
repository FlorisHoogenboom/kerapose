import math

import cv2
import numpy as np


def get_new_dims_from_max(orig_w, orig_h, max_size):
    """
    Utility to get the new height and width when a maximum size for either dimension
    is specified.
    """
    if orig_w > orig_h:
        new_w = max_size
        new_h = math.ceil((max_size / orig_w) * orig_h)
    else:
        new_w = math.ceil((max_size / orig_h) * orig_w)
        new_h = max_size

    return new_w, new_h


def resize_batch(max_size, images):
    """
    Resize a batch of images to a specified maximum size.

    Args:
        max_size (tuple or int): The maximal dimensions of the new result. If a tuple is
            passed this specifies the new width and height. Otherwise it specifies the
            size of the largest dimension of the resized image.
        images: (4d Numpy Array): A batch of images structured in a Numpy array
            (batch_size, width, height, channels)

    Returns:
        4d Numpy Array: The resized batch of images.
    """
    orig_w, orig_h = images.shape[1:3]

    if type(max_size) == tuple:
        new_w, new_h = max_size
    else:
        new_w, new_h = get_new_dims_from_max(orig_w, orig_h, max_size)

    resized_images = np.zeros((images.shape[0], new_h, new_w, images.shape[3]))

    for i in range(images.shape[0]):
        resized_images[i] = cv2.resize(images[i], (new_w, new_h))

    return resized_images


def rescale_joint_predictions(joint_predictions, w, h):
    """
    This function rescales the joint predictions to the size of the original input image.
    This is a convenience function for plotting
    Args:
        joint_predictions (list): Predictions per joint location
        w (float): The desired output width
        h (float): The desired output height

    Returns:
        list: A list of joint predictions (i.e. coordinates) that match the desired
            height and width.
    """
    rescaled_predictions = []

    for orig_joint_coord in joint_predictions:
        rescaled_predictions.append(
            [[h], [w]] * orig_joint_coord
        )

    return rescaled_predictions
