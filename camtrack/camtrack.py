#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import frameseq
import numpy as np
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    triangulate_correspondences,
    TriangulationParameters,
)
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    triangulation_parameters = TriangulationParameters(max_reprojection_error=1.0,
                                                       min_triangulation_angle_deg=1.15,
                                                       min_depth=0.1)
    correspondences = build_correspondences(corner_storage[known_view_1[0]],
                                            corner_storage[known_view_2[0]])
    points, ids, _ = triangulate_correspondences(correspondences,
                                                 pose_to_view_mat3x4(known_view_1[1]),
                                                 pose_to_view_mat3x4(known_view_2[1]),
                                                 intrinsic_mat,
                                                 triangulation_parameters)
    point_cloud_builder = PointCloudBuilder(ids, points)
    seq_size = len(rgb_sequence)
    view_mats = [None] * seq_size
    changed = True

    while changed:
        changed = False
        for frame in range(seq_size):
            corners = corner_storage[frame]
            if view_mats[frame] is None:
                intersection = [corner in corners.ids
                                for idx, corner in enumerate(point_cloud_builder.ids)]
                points = np.array([p for idx, p
                                   in zip(corners.ids, corners.points)
                                   if idx in point_cloud_builder.ids])
                if points.shape[0] > 5:
                    retval, rvec, tvec, inliers = cv2.solvePnPRansac(point_cloud_builder.points[intersection],
                                                                     points,
                                                                     intrinsic_mat,
                                                                     None)
                    if retval:
                        print(f'Frame {frame} is processing, {len(inliers)} inliers were found')
                        view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
                        changed = True
            if view_mats[frame] is not None:
                for i in range(frame):
                    correspondences = build_correspondences(corner_storage[i],
                                                            corners,
                                                            point_cloud_builder.ids)
                    if view_mats[i] is not None and len(correspondences.ids) > 0:
                        new_points, new_ids, _ = triangulate_correspondences(correspondences,
                                                                             view_mats[i],
                                                                             view_mats[frame],
                                                                             intrinsic_mat,
                                                                             triangulation_parameters)
                        if len(new_points) > 0:
                            print(f'Frame {frame}... frame {i}: {len(new_points)} points were triangulated')
                        point_cloud_builder.add_points(new_ids, new_points)
                print(f'Frame {frame}... current point cloud size is {point_cloud_builder.points.size}')

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
