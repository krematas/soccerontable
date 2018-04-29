from scipy.spatial import Delaunay
import numpy as np


def triangulate_depthmap_points(points2d, depth, pix_thresh=2, depth_thresh=0.2):
    tri = Delaunay(points2d)
    faces_ = tri.simplices.copy()
    faces = []

    for j in range(faces_.shape[0]):
        if np.linalg.norm(points2d[faces_[j, 0], :] - points2d[faces_[j, 1], :]) <= pix_thresh and \
                        np.linalg.norm(points2d[faces_[j, 2], :] - points2d[faces_[j, 1], :]) <= pix_thresh and \
                        np.linalg.norm(points2d[faces_[j, 0], :] - points2d[faces_[j, 2], :]) <= pix_thresh and \
                        np.linalg.norm(depth[faces_[j, 0]] - depth[faces_[j, 1]]) <= depth_thresh and \
                        np.linalg.norm(depth[faces_[j, 2]] - depth[faces_[j, 1]]) <= depth_thresh and \
                        np.linalg.norm(depth[faces_[j, 0]] - depth[faces_[j, 2]]) <= depth_thresh:
            # faces.append(faces_[j, :])
            faces.append(faces_[j, (2, 1, 0)])
    # faces = np.array(faces)
    return faces


def pack_textures():
    return 0

