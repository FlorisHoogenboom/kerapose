import math

import cv2
import numpy as np


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
    elif orig_w > orig_h:
        new_w = max_size
        new_h = math.ceil((max_size / orig_w) * orig_h)
    else:
        new_w = math.ceil((max_size / orig_h) * orig_w)
        new_h = max_size

    resized_images = np.zeros((images.shape[0], new_w, new_h, images.shape[3]))

    for i in range(images.shape[0]):
        resized_images[i] = cv2.resize(images[i], (new_h, new_w))

    return resized_images