#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import frameseq
import numpy as np
import sortednp as snp
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    project_points,
    rodrigues_and_translation_to_view_mat3x4,
    triangulate_correspondences,
    TriangulationParameters,
)
from _corners import filter_frame_corners
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose

# !!! DON'T TOUCH, GOOD CONSTANTS !!!
MAX_REPROJECTION_ERROR = 8.0
MIN_TRIANGULATION_ANGLE_DEG = 1.0
MIN_DEPTH = 0.1
MIN_RETRIANGULATION_FRAME_COUNT = 10
MIN_RETRIANGULATION_INLIERS = 6
RETRIANGULATION_INLIERS_ITERATIONS = 25
SEED = 42


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

    triangulation_parameters = TriangulationParameters(max_reprojection_error=MAX_REPROJECTION_ERROR,
                                                       min_triangulation_angle_deg=MIN_TRIANGULATION_ANGLE_DEG,
                                                       min_depth=MIN_DEPTH)
    frame1, pose1 = known_view_1
    frame2, pose2 = known_view_2
    correspondences = build_correspondences(corner_storage[frame1],
                                            corner_storage[frame2])
    view_mat1 = pose_to_view_mat3x4(pose1)
    view_mat2 = pose_to_view_mat3x4(pose2)
    points3d, ids, _ = triangulate_correspondences(correspondences,
                                                   view_mat1,
                                                   view_mat2,
                                                   intrinsic_mat,
                                                   triangulation_parameters)
    point_cloud_builder = PointCloudBuilder(ids, points3d)

    seq_size = len(rgb_sequence)
    view_mats = [None] * seq_size
    view_mats[frame1] = view_mat1
    view_mats[frame2] = view_mat2

    changed = True
    while changed:
        changed = False
        retr_frames = []
        for frame in range(seq_size):
            corners = corner_storage[frame]
            if view_mats[frame] is None:
                intersection = [corner in corners.ids
                                for idx, corner in enumerate(point_cloud_builder.ids)]
                points = np.array([p for idx, p
                                   in zip(corners.ids, corners.points)
                                   if idx in point_cloud_builder.ids])
                if len(points) >= 3:
                    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                        objectPoints=point_cloud_builder.points[intersection],
                        imagePoints=points,
                        cameraMatrix=intrinsic_mat,
                        distCoeffs=None,
                        flags=cv2.SOLVEPNP_EPNP,
                        confidence=0.99,
                        reprojectionError=MAX_REPROJECTION_ERROR
                    )
                    if retval:
                        _, rvec, tvec = cv2.solvePnP(objectPoints=point_cloud_builder.points[intersection][inliers],
                                                     imagePoints=points[inliers],
                                                     cameraMatrix=intrinsic_mat,
                                                     distCoeffs=None,
                                                     flags=cv2.SOLVEPNP_ITERATIVE,
                                                     useExtrinsicGuess=True,
                                                     rvec=rvec,
                                                     tvec=tvec)
                        view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
                        print(f'Frame {frame} is processing, {len(inliers)} inliers were found')
                        changed = True

                        for i in range(seq_size):
                            if view_mats[i] is not None:
                                correspondences = build_correspondences(corner_storage[i],
                                                                        filter_frame_corners(corners, inliers),
                                                                        point_cloud_builder.ids)
                                if len(correspondences.ids):
                                    new_points, new_ids, _ = triangulate_correspondences(correspondences,
                                                                                         view_mats[i],
                                                                                         view_mats[frame],
                                                                                         intrinsic_mat,
                                                                                         triangulation_parameters)
                                    if len(new_points):
                                        print(f'Frame {frame}... frame {i}: {len(new_points)} points were triangulated')
                                    point_cloud_builder.add_points(new_ids, new_points)

                        retr_frames.append(frame)
                        if len(retr_frames) >= MIN_RETRIANGULATION_FRAME_COUNT:
                            proj = np.array([intrinsic_mat @ view_mats[frame]
                                             for frame in retr_frames])
                            intersection = snp.intersect(corner_storage[retr_frames[0]].ids.flatten(),  # first frame
                                                         corner_storage[retr_frames[-1]].ids.flatten())  # last frame
                            if len(intersection):
                                points = []
                                for retr_frame in retr_frames:  # choose corners
                                    _, (_, ids) = snp.intersect(intersection,
                                                                corner_storage[retr_frame].ids.flatten(),
                                                                indices=True)
                                    points.append(corner_storage[retr_frame].points[ids])
                                points = np.array(points)
                                rows, cols = points.shape[:2]
                                np.random.seed(SEED)
                                for iteration in range(RETRIANGULATION_INLIERS_ITERATIONS):  # search inliers
                                    ids1, ids2 = np.argsort(np.random.rand(rows, cols), axis=0)[:2]
                                    points1 = np.dstack((np.choose(ids1, points[:, :, 0]),
                                                         np.choose(ids1, points[:, :, 1])))[0]
                                    points2 = np.dstack((np.choose(ids2, points[:, :, 0]),
                                                         np.choose(ids2, points[:, :, 1])))[0]
                                    _, _, vh = np.linalg.svd(np.stack([  # solve equation system using svd
                                        proj[ids1][:, 2] * points1[:, [0]] - proj[ids1][:, 0],
                                        proj[ids1][:, 2] * points1[:, [1]] - proj[ids1][:, 1],
                                        proj[ids2][:, 2] * points2[:, [0]] - proj[ids2][:, 0],
                                        proj[ids2][:, 2] * points2[:, [1]] - proj[ids2][:, 1]], axis=1))
                                    guess = vh[:, -1, :][:, :3] / vh[:, -1, :][:, [-1]]
                                    err = np.array([np.linalg.norm(  # reprojection error
                                        points[i] - project_points(guess, proj[i]), axis=1)
                                        for i in range(rows)])
                                    mean = np.mean(err, axis=0)
                                    inliers_mask = err < MAX_REPROJECTION_ERROR
                                    inliers_cnt = np.count_nonzero(inliers_mask, axis=0)
                                    if iteration:
                                        mask = np.logical_and(err_min < mean,
                                                              inliers_max <= inliers_cnt)
                                        err_min = np.where(mask, mean, err_min)
                                        inliers_max = np.where(mask, inliers_cnt, inliers_max)
                                        inliers[:, mask] = inliers_mask[:, mask]
                                    else:
                                        err_min = mean
                                        inliers_max = inliers_cnt
                                        inliers = inliers_mask
                                projs = np.repeat(proj[:, np.newaxis], cols, axis=1)
                                projs[~inliers] = np.zeros(proj.shape[1:])
                                ok = np.count_nonzero(inliers, axis=0) > MIN_RETRIANGULATION_INLIERS  # retriangulated
                                _, _, vh = np.linalg.svd(np.swapaxes(np.concatenate(np.stack([
                                    projs[:, ok][:, :, 2] * points[:, ok][:, :, [0]] - projs[:, ok][:, :, 0],
                                    projs[:, ok][:, :, 2] * points[:, ok][:, :, [1]] - projs[:, ok][:, :, 1]], axis=1),
                                    axis=0), 0, 1))  # solve equation system
                                point_cloud_builder.add_points(intersection[ok],
                                                               vh[:, -1, :][:, :3] / vh[:, -1, :][:, [-1]])
                                print(f'Frame {frame}... {sum(ok)} points were retriangulated')
                            retr_frames.clear()
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
