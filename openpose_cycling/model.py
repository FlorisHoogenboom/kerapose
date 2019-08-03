from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate

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
