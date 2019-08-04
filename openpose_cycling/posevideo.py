import numpy as np
import os.path
from tqdm import tqdm
import skvideo.io

from . import poseimage
from . import config as c


def video_batch_gen(video_path, batch_size=3, downsample=5):
    """ Generate batches from a video as numpy arrays

    Args:
        video_path (str): path to video file
        batch_size (int): batch size (default 3)
        downsample (int): sample 1 frame for every n. The procedure
            then takes the first frame of every sequence of length n
            (default 5).
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError()

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise Exception("Please specify a non-null positive integer for batch_size")

    if not isinstance(downsample, int) or downsample <= 0:
        raise Exception("Please specify a non-null positive integer for downsample")

    videoreader = skvideo.io.vreader(video_path)

    try:
        first = next(videoreader)
    except StopIteration:
        return

    batch_shape = (batch_size, *first.shape)

    batch = np.zeros(batch_shape, dtype="uint8")
    batch[0] = first
    position_in_batch = 1

    while True:
        # If downsampling: we skip n-1 frames after the first. If the reader
        # is empty somewhere here, we simply yield the batch and then return,
        # which then terminates the iteration in the generator statement.
        if downsample > 1:
            for i in range(downsample-1):
                try:
                    next(videoreader)
                except StopIteration:
                    if position_in_batch > 0:
                        yield batch[:position_in_batch, ...]
                    return

        try:
            batch[position_in_batch] = next(videoreader)

            # Evaluate if batch is complete (10th position filled)
            if position_in_batch == batch_size - 1:
                yield batch
                batch = np.empty(batch_shape)
                position_in_batch = 0
            else:
                position_in_batch += 1

        except StopIteration:
            # No data left: Yield the constructed part of the last batch
            # if there is such a part and then finished.
            if position_in_batch > 0:
                yield batch[:position_in_batch, ...]

            break


class PoseVideo(object):
    """ Wraps a video for pose tagging """

    __slots__ = (
        "frames"
    )

    def __init__(self, video, downsample=5):
        """ Initialize PoseVideo object

        Args:
            video: Either video path or video itself.
            downsample (int): Use only every nth frame. Set to 1 to use all

        Returns:
            PoseVideo object which wraps provided video
        """

        self.frames = []

        if isinstance(video, np.ndarray):
            # Take frames and initialise a PoseImage for each
            for i in range(video.shape[0]):
                self.frames.append(poseimage.PoseImage(video[i, ...]))

        elif isinstance(video, str):
            if not os.path.exists(video):
                raise FileNotFoundError()

            if not isinstance(downsample, int) or downsample <= 0:
                raise Exception(("Please specify a non-null positive"
                                 "integer for downsample"))

            videoreader = skvideo.io.vreader(video)

            for frame in videoreader:
                self.frames.append(poseimage.PoseImage(next(videoreader)))

    def write(self, filepath, fps, outputdict=None, overlay=False):
        """
        Write the video to file

        Args:
            filepath (str): Filepath to write video to
            fps (int): Frames per second to put in output video
            outputdict: ffmpeg command line options, passed to skvideo.io.FFmpegWriter
                Default only includes pixel format({"pix_fmt": "yuv420p"}) through
                config.py. Framerate does not need specification (drawn from fps)
            overlay: Overlay pose in the video (Not implemented)
        """

        if not isinstance(fps, int) or fps <= 0:
            raise Exception("Please pick a non-null integer value for fps")

        # Set input framerate, this is set to output framerate, because ffmpeg keeps the
        # video length identical. Hence, if it falls back to default input framerate it
        # changes our video
        inputdict = {"-r": str(fps)}

        # Set output dict, default with just -pix_fmt, otherwise check if it is specified
        # and if not add it.
        if not outputdict:
            outputdict = {"-pix_fmt": c.FFMPEG_PIX_FMT}
        elif not isinstance(outputdict, dict):
            raise TypeError("outputdict should be of type dict")
        else:
            if "-pix_fmt" not in outputdict:
                outputdict.update({"-pix_fmt": c.FFMPEG_PIX_FMT})
        # Add framerate (or override, since it has to be identical to inputdict)
        outputdict.update({"-r": str(fps)})

        writer = skvideo.io.FFmpegWriter(filepath,
                                         inputdict=inputdict,
                                         outputdict=outputdict)

        for frame in self.frames:
            writer.writeFrame(frame.img)

        writer.close()

    def predict(self, model, batch_size=10):
        """ Carry out pose estimation using an instance of PoseEstimator

        args:
            model (PoseEstimator): A PoseEstimator instance
            batch_size (int): Batch size to use in prediction (default 10)
        """

        for i in tqdm(range(0, len(self.frames), batch_size)):
            paf, heatmap = model.predict_from_image(
                np.array([frame.img for frame in self.frames[i: i + batch_size]])
            )

            for j in range(paf.shape[0]):
                self.frames[j].add_paf(paf[j])
                self.frames[j].add_heatmap(heatmap[j])
