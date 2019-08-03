import cv2
import numpy as np
import os.path
import matplotlib.pyplot as plt


class PoseImage:
    """ Image with Pose annotations, supplies methods for preparation """

    __slots__ = (
        "imgpath",
        "img",
        "annotations",
        "heatmap",
        "paf"
    )

    def __init__(self, img):
        """ Initialize PoseImage object

        Keyword args:
        img -- Either image path or image itself.
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

    def add_annotations(self, annotations):
        """ Add annotations to a PoseImage class"""
        pass

    def generate_heatmap(self):
        """ Make heatmap """
        self.heatmap = None
        pass

    def add_heatmap(self, heatmap):
        """ Append heatmap from model """
        if heatmap.shape[0:2] != self.shape[0:2] and heatmap.shape[2] != 19:
            raise Exception(
                ("Heatmap shape {} does not correspond with "
                 "image shape {}").format(heatmap.shape, self.shape)
            )

        self.heatmap = heatmap

    def add_paf(self, paf):
        """ Append paf from model """
        if paf.shape[0:2] != self.shape[0:2] and paf.shape[2] != 38:
            raise Exception(
                ("Paf shape {} does not correspond with "
                 "image shape {}").format(paf.shape, self.shape)
            )

        self.paf = paf

    def imshow(self, rgbcode="BGR"):
        """ Call plt.imshow, with BGR colors, set rgbcode parameter to change."""

        rgbcode.upper()
        if any(letter not in rgbcode for letter in "RGB") or len(rgbcode) != 3:
            raise Exception("RGB code is incorrect, should only contain R, G and B")
        colour_recode = [rgbcode.index("R"), rgbcode.index("G"), rgbcode.index("B")]

        return plt.imshow(self.img[:, :, colour_recode])

    def plot_heatmap(self, joint, rgbcode="BGR", alpha=.5):
        """ Plot the overlapping heatmap of a joint

        Keyword args:
        joint -- joint index (see openpose_cycling.ELEMENTS)
        rgbcode -- RGB variant, accepts both uppercase and lowercase
            (default: BGR, as in cv2 and OpenPose)
        alpha -- alpha parameter for overlapping heatmap
        """
        rgbcode.upper()
        if any(letter not in rgbcode for letter in "RGB") or len(rgbcode) != 3:
            raise Exception("RGB code is incorrect, should only contain R, G and B")
        colour_recode = [rgbcode.index("R"), rgbcode.index("G"), rgbcode.index("B")]

        if not self.heatmap:
            raise Exception("No heatmap added to this object!")

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot image
        ax.imshow(self.img[:, :, colour_recode])

        # Overlay heatmap
        ax.imshow(self.heatmap, alpha=alpha)

        return fig, ax

    def plot_paf(self, joints, rgbcode="BGR", paf_threshold=.25,
                 spacing=5, img_alpha=.5, quiver_alpha=.5):
        """ Plot the paf between two joints

        Keyword args:
        joints -- joint indexes, arrows go from joints[0] to joints[1]
            (see openpose_cycling.ELEMENTS)
        rgbcode -- RGB variant, accepts both uppercase and lowercase
            (default: BGR, as in cv2 and OpenPose)
        paf_threshold -- Threshold for paf to be displayed
            (defaults to .25, as in OpenPose demo)
        spacing -- Space between arrows in quiver plot (defaults to 5)
        img_alpha -- alpha for image
        quiver_alpha -- alpha for quiverplot
        """
        rgbcode.upper()
        if any(letter not in rgbcode for letter in "RGB") or len(rgbcode) != 3:
            raise Exception("RGB code is incorrect, should only contain R, G and B")
        colour_recode = [rgbcode.index("R"), rgbcode.index("G"), rgbcode.index("B")]

        if not self.paf:
            raise Exception("No paf fields added to this object!")

        # Set up quiver plot elements
        X = np.arange(self.shape[1])
        Y = np.arange(self.shape[0])

        U = self.paf[:, :, joints[0]] * -1  # Reverse arrows to point away from the joint
        V = self.paf[:, :, joints[1]]

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
        ax.quiver(X[::spacing], Y[::spacing],
                  U[::spacing, ::spacing], V[::spacing, ::spacing],
                  scale=50, headaxislength=4, alpha=.5, width=0.001, color="r")

        return fig, ax

    def plot_pose(self, connected=False):
        pass

    @property
    def shape(self):
        return self.img.shape
