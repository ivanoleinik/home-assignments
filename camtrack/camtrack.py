#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np
import sortednp as snp
from scipy.optimize import least_squares, approx_fprime

import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    eye3x4,
    project_points,
    remove_correspondences_with_ids,
    rodrigues_and_translation_to_view_mat3x4,
    triangulate_correspondences,
    Correspondences,
    TriangulationParameters,
    calc_inlier_indices,
    compute_reprojection_errors
)
from _corners import filter_frame_corners
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose

# !!! DON'T TOUCH, GOOD CONSTANTS !!!
MIN_INITIAL_FRAMES_DISTANCE = 6
MAX_INITIAL_FRAMES_DISTANCE = 60
MAX_REPROJECTION_ERROR = 8.0
MIN_TRIANGULATION_ANGLE_DEG = 1.0
MIN_DEPTH = 0.1
MIN_RETRIANGULATION_FRAME_COUNT = 10
MIN_RETRIANGULATION_INLIERS = 6
RETRIANGULATION_INLIERS_ITERATIONS = 25
CONFIDENCE = 0.99
THRESHOLD = 1
HOMOGRAPHY_THRESHOLD = 0.4
BUNDLE_ADJUSTMENT_ITERATIONS = 5
EPSILON = 1e-7
SEED = 42


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    triangulation_parameters = TriangulationParameters(max_reprojection_error=MAX_REPROJECTION_ERROR,
                                                       min_triangulation_angle_deg=MIN_TRIANGULATION_ANGLE_DEG,
                                                       min_depth=MIN_DEPTH)
    seq_size = len(rgb_sequence)

    frame1, frame2 = None, None
    if known_view_1 is None or known_view_2 is None:
        view_mat1 = eye3x4()
        frame_pairs = np.array([(f, s) for f in range(seq_size - MIN_INITIAL_FRAMES_DISTANCE)
                                for s in range(MIN_INITIAL_FRAMES_DISTANCE + f,
                                               min(MAX_INITIAL_FRAMES_DISTANCE + f + 1, seq_size))])
        np.random.seed(SEED)
        frame_pairs = frame_pairs[np.random.choice(
            len(frame_pairs),
            min(len(frame_pairs), 3 * seq_size),
            replace=False
        )]

        best_cnt = -1
        for frame_1, frame_2 in frame_pairs:
            ids_1 = corner_storage[frame_1].ids.flatten().astype(np.int64)
            _, (idx1, idx2) = snp.intersect(ids_1,
                                            corner_storage[frame_2].ids.flatten().astype(np.int64),
                                            indices=True)
            correspondences = Correspondences(
                ids_1[idx1],
                corner_storage[frame_1].points[idx1],
                corner_storage[frame_2].points[idx2]
            )
            cnt = -1
            if len(correspondences.ids) >= 7:
                mat, mat_mask = cv2.findEssentialMat(
                    correspondences.points_1,
                    correspondences.points_2,
                    intrinsic_mat,
                    method=cv2.RANSAC,
                    prob=CONFIDENCE,
                    threshold=THRESHOLD
                )

                if mat is not None and mat_mask is not None:
                    correspondences = remove_correspondences_with_ids(
                        correspondences, np.argwhere(mat_mask.flatten() == 0).astype(np.int64))

                    _, homography_mask = cv2.findHomography(correspondences[1],
                                                            correspondences[2],
                                                            method=cv2.RANSAC,
                                                            ransacReprojThreshold=THRESHOLD,
                                                            confidence=CONFIDENCE)

                    if homography_mask is not None:
                        if np.count_nonzero(homography_mask) / np.count_nonzero(mat_mask) < HOMOGRAPHY_THRESHOLD:
                            rot_1, rot_2, translation = cv2.decomposeEssentialMat(mat)
                            for rot in (rot_1.T, rot_2.T):
                                for tran in (translation, -translation):
                                    view = pose_to_view_mat3x4(Pose(rot, rot @ tran))
                                    pt_cnt = len(triangulate_correspondences(
                                        correspondences,
                                        view_mat1,
                                        view,
                                        intrinsic_mat,
                                        triangulation_parameters
                                    )[1])
                                    if cnt < pt_cnt:
                                        view_2 = view
                                        cnt = pt_cnt

            if best_cnt < cnt:
                frame1 = frame_1
                frame2 = frame_2
                view_mat2 = view_2
                best_cnt = cnt
    else:
        if known_view_1[0] > known_view_2[0]:
            known_view_1, known_view_2 = known_view_2, known_view_1
        frame1, pose1 = known_view_1
        frame2, pose2 = known_view_2
        view_mat1 = pose_to_view_mat3x4(pose1)
        view_mat2 = pose_to_view_mat3x4(pose2)

    if frame1 is None or frame2 is None:
        print("Initialization failed. No good pair of frames was found.")
        exit(0)

    correspondences = build_correspondences(corner_storage[frame1],
                                            corner_storage[frame2])
    points3d, ids, _ = triangulate_correspondences(correspondences,
                                                   view_mat1,
                                                   view_mat2,
                                                   intrinsic_mat,
                                                   triangulation_parameters)
    point_cloud_builder = PointCloudBuilder(ids, points3d)

    view_mats = [None] * seq_size
    view_mats[frame1] = view_mat1
    view_mats[frame2] = view_mat2

    def retriangulate(to_retr):
        ok = None
        if len(to_retr.shape) > 2:
            rows, cols = to_retr.shape[:2]
            np.random.seed(SEED)
            for iteration in range(RETRIANGULATION_INLIERS_ITERATIONS):  # search inliers
                ids1, ids2 = np.argsort(np.random.rand(rows, cols), axis=0)[:2]
                points1 = np.dstack((np.choose(ids1, to_retr[:, :, 0]),
                                     np.choose(ids1, to_retr[:, :, 1])))[0]
                points2 = np.dstack((np.choose(ids2, to_retr[:, :, 0]),
                                     np.choose(ids2, to_retr[:, :, 1])))[0]
                _, _, vh = np.linalg.svd(np.stack([  # solve equation system using svd
                    proj[ids1][:, 2] * points1[:, [0]] - proj[ids1][:, 0],
                    proj[ids1][:, 2] * points1[:, [1]] - proj[ids1][:, 1],
                    proj[ids2][:, 2] * points2[:, [0]] - proj[ids2][:, 0],
                    proj[ids2][:, 2] * points2[:, [1]] - proj[ids2][:, 1]], axis=1))
                guess = vh[:, -1, :][:, :3] / vh[:, -1, :][:, [-1]]
                err = np.array([np.linalg.norm(  # reprojection error
                    to_retr[i] - project_points(guess, proj[i]), axis=1)
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
            ok = np.count_nonzero(inliers,
                                  axis=0) > MIN_RETRIANGULATION_INLIERS  # retriangulated
            _, _, vh = np.linalg.svd(np.swapaxes(np.concatenate(np.stack([
                projs[:, ok][:, :, 2] * to_retr[:, ok][:, :, [0]] - projs[:, ok][:, :, 0],
                projs[:, ok][:, :, 2] * to_retr[:, ok][:, :, [1]] - projs[:, ok][:, :, 1]],
                axis=1),
                axis=0), 0, 1))  # solve equation system
            point_cloud_builder.add_points(intersection[ok],
                                           vh[:, -1, :][:, :3] / vh[:, -1, :][:, [-1]])
        return ok

    changed = True
    while changed:
        changed = False
        retr_frames = []
        for frame in range(seq_size):
            corners = corner_storage[frame]
            if view_mats[frame] is None:
                intersection = [corner.astype(np.int64) in corners.ids
                                for idx, corner in enumerate(point_cloud_builder.ids)]
                points = np.array([p for idx, p
                                   in zip(corners.ids.astype(np.int64), corners.points)
                                   if idx in point_cloud_builder.ids.astype(np.int64)])
                if len(points) >= 3:
                    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                        objectPoints=point_cloud_builder.points[intersection],
                        imagePoints=points,
                        cameraMatrix=intrinsic_mat,
                        distCoeffs=None,
                        flags=cv2.SOLVEPNP_EPNP,
                        confidence=CONFIDENCE,
                        reprojectionError=MAX_REPROJECTION_ERROR
                    )
                    if retval:
                        # _, rvec, tvec = cv2.solvePnP(objectPoints=point_cloud_builder.points[intersection][inliers],
                        #                              imagePoints=points[inliers],
                        #                              cameraMatrix=intrinsic_mat,
                        #                              distCoeffs=None,
                        #                              flags=cv2.SOLVEPNP_ITERATIVE,
                        #                              useExtrinsicGuess=True,
                        #                              rvec=rvec,
                        #                              tvec=tvec)

                        # solve PnP iterative using M-estimators
                        def residuals(vec):
                            r = vec[:3, np.newaxis]
                            t = vec[3:]
                            mat = np.eye(4)[:3]
                            mat[:3, :3] = cv2.Rodrigues(r)[0]
                            mat[:3, 3] = t
                            view_proj = intrinsic_mat @ mat
                            return (project_points(
                                point_cloud_builder.points[intersection][inliers.flatten()],
                                view_proj) - points[inliers.flatten()]).flatten()

                        vec6 = least_squares(fun=residuals,
                                             x0=np.zeros(6, dtype=np.float64),  # prevent huge max rotation error
                                             loss='huber',
                                             method='trf').x
                        view_mats[frame] = rodrigues_and_translation_to_view_mat3x4(vec6[:3, np.newaxis],
                                                                                    vec6[3:, np.newaxis])
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
                            intersection = snp.intersect(corner_storage[retr_frames[0]].ids.flatten().astype(np.int64),
                                                         # first frame
                                                         corner_storage[retr_frames[-1]].ids.flatten().astype(
                                                             np.int64))  # last frame
                            if len(intersection):
                                points = []
                                for retr_frame in retr_frames:  # choose corners
                                    _, (_, ids) = snp.intersect(intersection,
                                                                corner_storage[retr_frame].ids.flatten().astype(
                                                                    np.int64),
                                                                indices=True)
                                    points.append(corner_storage[retr_frame].points[ids])
                                retriangulated = retriangulate(np.array(points))
                                if retriangulated is not None:
                                    print(f'Frame {frame}... {np.sum(retriangulated)} points were retriangulated')
                            retr_frames.clear()
                        print(f'Frame {frame}... current point cloud size is {point_cloud_builder.points.size}')

    if seq_size < 100:
        frames, corners_ids, points_ids = [], [], []
        for frame in range(seq_size):
            c_ids, pt_ids = snp.intersect(corner_storage[frame].ids.flatten(),
                                          point_cloud_builder.ids.flatten(),
                                          indices=True)[1]
            inliers = calc_inlier_indices(point_cloud_builder.points[pt_ids],
                                          corner_storage[frame].points[c_ids],
                                          intrinsic_mat @ view_mats[frame], 1.0)
            frames.extend([frame] * len(inliers))
            corners_ids.extend(c_ids[inliers])
            points_ids.extend(pt_ids[inliers])

        mats = np.array([np.concatenate([cv2.Rodrigues(view_mat[:, :3])[0].squeeze(),
                                         view_mat[:, 3]])
                         for view_mat in view_mats])
        unique = list(set(points_ids))
        v = np.array(point_cloud_builder.points[unique])
        points_ids = [unique.index(point_id) for point_id in points_ids]
        sz = len(mats.reshape(-1))

        def verr(xk, corner):
            return compute_reprojection_errors(xk[6:].reshape(1, -1),
                                               corner.reshape(1, -1),
                                               intrinsic_mat @ rodrigues_and_translation_to_view_mat3x4(
                                                   xk[:3].reshape(3, 1),
                                                   xk[3:6].reshape(3, 1)))[0]

        def err():
            return np.array([verr(np.concatenate([mats[frame],
                                                  v[point_id]]),
                                  corner_storage[frame].points[corner_id])
                             for frame, corner_id, point_id in zip(frames, corners_ids, points_ids)])

        prev_err = err().mean()
        for _ in range(BUNDLE_ADJUSTMENT_ITERATIONS):
            A = np.zeros((len(frames), sz + len(v.reshape(-1))))
            for idx, (frame, corner_id, point_id) in enumerate(zip(frames, corners_ids, points_ids)):
                xk = np.concatenate([mats[frame], v[point_id]])
                derivative = approx_fprime(xk=xk,
                                           f=lambda xk: verr(xk, corner_storage[frame].points[corner_id]),
                                           epsilon=np.full_like(xk, EPSILON))
                A[idx, (sz + point_id * 3): (sz + (point_id + 1) * 3)] = derivative[6:]
                A[idx, (frame * 6): ((frame + 1) * 6)] = derivative[:6]
            grad = A.T @ err()
            d = np.diag(np.diag(A.T @ A)) * 10 + A.T @ A
            V, Wi = d[:sz, sz:], np.linalg.inv(d[sz:, sz:])
            Vt = V.T
            V_Wi = V @ Wi
            diff = np.linalg.solve(d[:sz, :sz] - V_Wi @ Vt, V_Wi @ grad[sz:] - grad[:sz])
            mats = (diff + mats.reshape(-1)).reshape((-1, 6))
            v = (v.reshape(-1) - Wi @ (grad[sz:] + Vt @ diff)).reshape((-1, 3))

        if np.mean(err()) < prev_err:
            view_mats = [rodrigues_and_translation_to_view_mat3x4(mat[:3].reshape(3, 1),
                                                                  mat[3:6].reshape(3, 1)) for mat in mats]
            point_cloud_builder.update_points(point_cloud_builder.ids[unique], v)

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
