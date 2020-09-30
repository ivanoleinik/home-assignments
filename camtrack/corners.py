#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'filter_frame_corners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import create_cli, filter_frame_corners


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    def find_new_corners(image, maxLevel=2, blockSize=5, frame_corners=None):
        feature_params = dict(maxCorners=0,
                              qualityLevel=0.1,
                              blockSize=blockSize,
                              useHarrisDetector=False)

        id_cnt = 0 if frame_corners is None \
            else (frame_corners.ids.ravel()[-1] + 1).astype(int)
        images = [image]
        for _ in range(maxLevel):
            images.append(cv2.pyrDown(images[-1]))
        ids, corners, sizes = np.empty((0,)), np.empty((0, 1, 2)), np.empty((0,))

        for level, image in enumerate(images):
            scale = 2 ** level
            size = scale * blockSize

            mask = np.full_like(image, 255, dtype=np.uint8)
            if frame_corners is not None:
                cond = (frame_corners.sizes == size).ravel()
                good_corners = frame_corners.points[cond]
                for x, y in good_corners:
                    rx = np.round(x / scale).astype(int)
                    ry = np.round(y / scale).astype(int)
                    cv2.circle(mask, (rx, ry), size, thickness=-1, color=0)

            corner_group = cv2.goodFeaturesToTrack(image,
                                                   minDistance=size,
                                                   mask=mask,
                                                   **feature_params)
            if corner_group is not None:
                corner_group_size = corner_group.shape[0]
                sizes = np.append(sizes, np.full(corner_group_size, size))
                corners = np.append(corners, corner_group * scale, axis=0)
        ids = np.array(list(range(id_cnt, id_cnt + corners.shape[0])))
        return ids, corners, sizes

    def _to_8bit(image):
        return (image * 255).astype(np.uint8)

    lk_params = dict(winSize=(11, 11),
                     maxLevel=7,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    ids, corners, sizes = find_new_corners(frame_sequence[0])
    old = FrameCorners(ids, corners, sizes)
    for frame, image in enumerate(frame_sequence[1:], 1):
        new, st, _ = cv2.calcOpticalFlowPyrLK(_to_8bit(frame_sequence[frame - 1]),
                                              _to_8bit(image),
                                              old.points.astype('float32'),
                                              None,
                                              **lk_params)
        st = st.ravel()
        good_old = filter_frame_corners(old, st == 1)
        builder.set_corners_at_frame(frame - 1, good_old)

        good_new = FrameCorners(good_old.ids, new[st == 1], good_old.sizes)
        ids, corners, sizes = find_new_corners(image, frame_corners=good_new)

        ids = np.append(good_new.ids.ravel(), ids, axis=0)
        corners = np.append(good_new.points.ravel(), corners.ravel(), axis=0)
        sizes = np.append(good_new.sizes.ravel(), sizes, axis=0)
        old = FrameCorners(ids, corners.reshape((-1, 1, 2)), sizes)
    builder.set_corners_at_frame(frame, old)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
