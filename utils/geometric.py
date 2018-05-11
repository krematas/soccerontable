import numpy as np


def ray_plane_intersection(ray_origin, ray_dir, plane_origin, plane_normal):
    n_rays = ray_dir.shape[0]
    denom = np.inner(plane_normal, ray_dir)
    p0l0 = plane_origin - ray_origin
    t = np.divide(np.inner(p0l0, plane_normal), denom)
    point3d = np.tile(ray_origin, (n_rays, 1)) + np.multiply(np.tile(t, (3, 1)).T, ray_dir)
    return point3d


def ray_triangle_intersection_vec(ray_origin, ray_dir, v1, v2, v3):
    eps = 0.000001
    edge1 = v2 - v1  # 3 x ,
    edge2 = v3 - v1  # 3 x ,
    pvec = np.cross(ray_dir, edge2)  # N x 3
    det = edge1[:, np.newaxis].T.dot(pvec.T)  # 1 x N

    good_index1 = (abs(det) >= eps).nonzero()[1]

    inv_det = 1. / det[0, good_index1]
    tvec = ray_origin - v1
    u = tvec[:, np.newaxis].T.dot(pvec[good_index1, :].T) * inv_det
    good_index2_ = (u >= 0.).nonzero()[1]
    good_index2__ = (u <= 1.).nonzero()[1]
    good_index2 = np.intersect1d(good_index2_, good_index2__)

    qvec = np.cross(tvec, edge1)
    v = ray_dir[good_index1[good_index2], :].dot(qvec) * inv_det[good_index2]
    good_index3_ = (v >= 0.).nonzero()[0]
    good_index3__ = (u[0, good_index2] + v <= 1.).nonzero()[0]
    good_index3 = np.intersect1d(good_index3_, good_index3__)

    t = edge2.dot(qvec) * inv_det[good_index2[good_index3]]
    good_index4 = (t >= eps).nonzero()[0]

    valid_index = good_index1[good_index2[good_index3[good_index4]]]
    return valid_index, t[good_index4]


# def mesh_from_mask():