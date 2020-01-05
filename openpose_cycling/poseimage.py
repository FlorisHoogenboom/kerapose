import cv2
import numpy as np
import os.path
import matplotlib.pyplot as plt


from . import config as c


class PoseImage(object):
    """
    Image with Pose annotations, supplies methods for preparation

    Attributes:
        img: The wrapped image
        shape: The shape of the wrapped image
    """

    __slots__ = (
        "imgpath",
        "img",
        "annotations",
        "predictions",
        "heatmap",
        "paf"
    )

    def __init__(self, img):
        """
        Initialize PoseImage object

        Args:
            img: Either image path or image itself (ideally opencv).

        Returns:
            PoseImage object which wraps provided image
        """
        if isinstance(img, str):  # assume filepath
            if not os.path.exists(img):
                raise FileNotFoundError
            else:
                self.imgpath = img
                self.img = cv2.imread(self.imgpath)
        elif isinstance(img, np.ndarray):  # assume np.ndarray (cv2.imread output)
            if not (len(img.shape) == 3 and img.shape[2] == 3):
                raise Exception("cv2 image does not have the right shape (h, w, 3)")
            self.img = img

    @property
    def shape(self):
        return self.img.shape

    def add_annotations(self, annotations):
        """ Add annotations to a PoseImage class

        Args:
            annotations: Annotations for the image. Formatted
                [[x11, y11, label11, x12, y12, label12, ...],[x21, y21, label21], ...]
        """  # TODO: Implement for training purposes
        raise NotImplementedError()

    def add_heatmap(self, heatmap):
        """
        Append heatmap from model

        Args:
            heatmap: Heatmap as provided by model
        """
        if heatmap.shape[0:2] != self.shape[0:2] and heatmap.shape[2] != 19:
            raise Exception(
                ("Heatmap shape {} does not correspond with "
                 "image shape {}").format(heatmap.shape, self.shape)
            )

        self.heatmap = heatmap

    def add_paf(self, paf):
        """
        Append PAF from model

        Args:
            paf: paf array as provided by model
        """
        if paf.shape[0:2] != self.shape[0:2] and paf.shape[2] != 38:
            raise Exception(
                ("Paf shape {} does not correspond with "
                 "image shape {}").format(paf.shape, self.shape)
            )

        self.paf = paf

    def generate_heatmap(self):
        """Make heatmap"""  # TODO: Implement for training purposes
        raise NotImplementedError()

    def imshow(self, rgbcode="BGR"):
        """
        Call plt.imshow

        Args:
            rgbcode: Color code of image (defaults to BGR, as in cv2 and OpenPose)

        Returns:
            Pyplot object
        """

        rgbcode.upper()
        if any(letter not in rgbcode for letter in "RGB") or len(rgbcode) != 3:
            raise Exception("RGB code is incorrect, should only contain R, G and B")
        colour_recode = [rgbcode.index("R"), rgbcode.index("G"), rgbcode.index("B")]

        return plt.imshow(self.img[:, :, colour_recode])

    def plot_heatmap(self, joint, rgbcode="BGR", alpha=.5):
        """
        Plot the overlapping heatmap of a joint

        Args:
            joint: joint index (see openpose_cycling.ELEMENTS)
            rgbcode: RGB variant, accepts both uppercase and lowercase
                (default: BGR, as in cv2 and OpenPose)
            alpha: alpha parameter for overlapping heatmap
        """
        rgbcode.upper()
        if any(letter not in rgbcode for letter in "RGB") or len(rgbcode) != 3:
            raise Exception("RGB code is incorrect, should only contain R, G and B")
        colour_recode = [rgbcode.index("R"), rgbcode.index("G"), rgbcode.index("B")]

        if not self.heatmap:
            raise Exception("Add a heatmap to the object before plotting it")

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot image
        ax.imshow(self.img[:, :, colour_recode])

        # Overlay heatmap
        ax.imshow(self.heatmap[:, :, joint], alpha=alpha)

        return fig

    def plot_paf(self, joints, rgbcode="BGR", paf_threshold=.25, spacing=5, img_alpha=0.5,
                 quiver_alpha=0.5):
        """
        Plot the paf between two joints

        Args:
            joints: joint indexes, arrows go from joints[0] to joints[1]
                (see openpose_cycling.ELEMENTS)
            rgbcode: RGB variant, accepts both uppercase and lowercase
                (default: BGR, as in cv2 and OpenPose)
            paf_threshold: Threshold for paf to be displayed
                (defaults to .25, as in OpenPose demo)
            spacing: Space between arrows in quiver plot (defaults to 5)
            img_alpha: alpha for image (default 0.5)
            quiver_alpha: alpha for quiverplot (default 0.5)
        """
        rgbcode.upper()
        if any(letter not in rgbcode for letter in "RGB") or len(rgbcode) != 3:
            raise Exception("RGB code is incorrect, should only contain R, G and B")
        colour_recode = [rgbcode.index("R"), rgbcode.index("G"), rgbcode.index("B")]

        if not self.paf:
            raise Exception("Add paf fields to the object before plotting them")

        if (joints[0], joints[1]) in c.LINKS:
            paf_index = c.LINKS.index((joints[0], joints[1]))
        elif (joints[1], joints[0]) in c.LINKS:
            paf_index = c.LINKS.index((joints[1], joints[0]))
        else:
            raise Exception("The link between those joints does not exist")

        # Set up quiver plot elements
        X = np.arange(self.shape[1])
        Y = np.arange(self.shape[0])

        U = self.paf[:, :, paf_index, 0] * -1
        V = self.paf[:, :, paf_index, 1]

        M = np.zeros(U.shape, dtype="bool")
        M[U**2 + V**2 < paf_threshold] = True  # Mask arrows that are not long enough

        U = np.ma.masked_array(U, M)
        V = np.ma.masked_array(V, M)

        # Set up fig, ax
        fig = plt.figure(figsize=[20, 10])
        ax = fig.add_subplot(111)

        # Plot image
        ax.imshow(self.img[:, :, colour_recode], alpha=.5)

        # Plot quiver
        q = ax.quiver(X[::spacing], Y[::spacing],
                      U[::spacing, ::spacing], V[::spacing, ::spacing],
                      scale=50, headaxislength=4, alpha=.5, width=0.001, color="r")

        plt.quiverkey(q, )

        return fig

    def plot_pose(self, connected=False):
        pass
