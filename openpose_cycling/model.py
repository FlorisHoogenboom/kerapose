import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate

from . import utils
from . import config as c


def create_vgg_feature_extractor(x):
    """
    Create a base feature extractor network that uses the VGG architecture to
    derive relevant visual features.

    Args:
        x (Keras Tensor): Tensor that contains the input images.

    Returns:
        Keras Tensor: Tensor of the features extracted by VGG.

    """
    # Block 1
    x = Conv2D(64, (3, 3), padding="same", activation="relu", name="conv1_1")(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu", name="conv1_2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool1_1")(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2_2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool2_1")(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3_1")(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3_2")(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3_3")(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3_4")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="pool3_1")(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv4_1")(x)
    x = Conv2D(512, (3, 3), padding="same", activation="relu", name="conv4_2")(x)

    # Additional non vgg layers
    x = Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_3_CPM")(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu", name="conv4_4_CPM")(x)

    return x


def create_first_stage_block(x, num_p, branch):
    """
    Creates a block uses for the first stage of the pose estimation network.
    This stage directly follows the VGG base network.

    Args:
        x (Keras Tensor): The ouput of the VGG network
        num_p (int): The number of layers the output should have
        branch (str): The number of the networks branch this block is on.

    Returns:
        Keras Tensor: The output of this stage, a tensor of shape
            (?, w, h, num_p)

    """
    x = Conv2D(128, (3, 3), padding="same", activation="relu",
               name="conv5_1_CPM_L%d" % branch)(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu",
               name="conv5_2_CPM_L%d" % branch)(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu",
               name="conv5_3_CPM_L%d" % branch)(x)
    x = Conv2D(512, (1, 1), padding="same", activation="relu",
               name="conv5_4_CPM_L%d" % branch)(x)
    x = Conv2D(num_p, (1, 1), padding="same", activation="relu",
               name="conv5_5_CPM_L%d" % branch)(x)

    return x


def create_later_stage_block(x, num_p, stage, branch):
    """
    Create a block that can be used as a block following an earlier stage of the pose
    estimation network (e.g. either the result of this function or of
    `create_first_stage_block`.

    Args:
        x (Keras Tensor): The result of an earlier stage
        num_p (int): The desired number of output layers for this branch
        stage (str): The number of the stage this block is used
        branch (str): The branch this block is on

    Returns:
        Keras Tensor: The output of this stage, a tensor of shape (?, w, h, num_p)

    """
    x = Conv2D(128, (7, 7), padding="same", activation="relu",
               name="Mconv1_stage%d_L%d" % (stage, branch))(x)
    x = Conv2D(128, (7, 7), padding="same", activation="relu",
               name="Mconv2_stage%d_L%d" % (stage, branch))(x)
    x = Conv2D(128, (7, 7), padding="same", activation="relu",
               name="Mconv3_stage%d_L%d" % (stage, branch))(x)
    x = Conv2D(128, (7, 7), padding="same", activation="relu",
               name="Mconv4_stage%d_L%d" % (stage, branch))(x)
    x = Conv2D(128, (7, 7), padding="same", activation="relu",
               name="Mconv5_stage%d_L%d" % (stage, branch))(x)
    x = Conv2D(128, (1, 1), padding="same", activation="relu",
               name="Mconv6_stage%d_L%d" % (stage, branch))(x)
    x = Conv2D(num_p, (1, 1), padding="same", activation="relu",
               name="Mconv7_stage%d_L%d" % (stage, branch))(x)

    return x


class PoseEstimator(object):
    """
    Class that wraps a Keras model that implements the multi person pose
    estimation model.

    Attributes:
        model (Keras Model): The underlying network that is used to infer
        the heatmaps and PAF predictions.

    """

    def __init__(self):
        self.model = self._construct_model()

    def _construct_model(self):
        """
        Creates an instance of the pose estimation network that infers the heatmaps and
        the PAF predictions that are used to determine the locations of joints and
        there connections.

        Returns:
            keras.models.Model: The pose estimation network
        """

        img_input = Input(shape=(None, None, 3))
        normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)

        # Start with a VGG base that extracts visual features where needed
        vgg_features = create_vgg_feature_extractor(normalized)

        # Compute the PAF and Heatmap flows for the first stage
        stage1_paf = create_first_stage_block(
            vgg_features, c.NETWORK_N_OUTPUT_PAF_BRANCH, branch=1
        )

        stage1_hm = create_first_stage_block(
            vgg_features, c.NETWORK_N_OUTPUT_HM_BRANCH, branch=2
        )

        # The input for any subsequent stage are these PAF and HM feature ouputs
        # concatenated with the VGG features.
        next_stage_input = Concatenate()([stage1_paf, stage1_hm, vgg_features])

        # Apply the next stage blocks sequentially, until we have reached the
        # number of stages specified in the configuration.
        for stage_nr in range(2, c.NETWORK_N_STAGES + 1):
            stage_paf = create_later_stage_block(
                next_stage_input, c.NETWORK_N_OUTPUT_PAF_BRANCH, stage_nr, 1
            )

            stage_hm = create_later_stage_block(
                next_stage_input, c.NETWORK_N_OUTPUT_HM_BRANCH, stage_nr, 2
            )

            next_stage_input = Concatenate()([stage1_paf, stage1_hm, vgg_features])

        return Model(img_input, [stage_paf, stage_hm])

    def predict_from_image(self, images):
        """
        Creates a PAF and Heatmap prediction from a single image or a batch of images
        by applying the pose estimation network on multiple scales and averaging the
        results.

        Args:
            images (3 or 4d - Numpy Array ): The image to predict on. This can be either
                a single image (3d Array) or a batch of images (4d Array). The channels
                should be ordered RGB.

        Returns:
            4d Numpy Array: The predicted PAF field with the before last dimension
                corresponding to the joints.
            3d Numpy Array: The predicted Heatmap with the last dimension corresponding
                to the joint that was searched for
        """
        return_batch = images.ndim == 4

        if not return_batch:
            # If not a batch of images has been passed, add the batch dimension to the
            # input because the model still expects this.
            images = images[None, ...]

        # The model makes prediction on multiple sizes of the same image. The results
        # of those predictions need to be averaged into a single PAF and heatmap
        # prediction. We create two placeholders where we can add all predictions to and
        # later use these to average the results.
        added_heatmaps = np.zeros(images.shape[0:3] + (c.NETWORK_N_OUTPUT_HM_BRANCH,))
        added_pafs = np.zeros((images.shape[0:3] + (c.NETWORK_N_OUTPUT_PAF_BRANCH,)))

        for max_size in c.PREDICT_STACK_SIZES:
            resized_images = utils.resize_batch(max_size, images)

            # We feed the images into the model in BGR orientation
            paf, heatmap = (
                self.model.predict(resized_images[..., [2, 1, 0]])
            )

            heatmap_upsampled = utils.resize_batch(images.shape[1:3], heatmap)
            paf_upsampled = utils.resize_batch(images.shape[1:3], paf)

            added_heatmaps = heatmap_upsampled + added_heatmaps
            added_pafs = paf_upsampled + added_pafs

        heatmaps_averaged = added_heatmaps / len(c.PREDICT_STACK_SIZES)
        pafs_averaged = added_pafs / len(c.PREDICT_STACK_SIZES)

        # Reshape the PAFs to let them better depict a vector field.
        pafs_averaged_resh = np.reshape(
            pafs_averaged,
            pafs_averaged.shape[0:-1] + (int(c.NETWORK_N_OUTPUT_PAF_BRANCH / 2), 2)
        )

        if return_batch:
            return pafs_averaged_resh, heatmaps_averaged
        else:
            return pafs_averaged_resh[0], heatmaps_averaged[0]
