import numpy as np
import math


def Rx(theta):
    theta = np.deg2rad(theta)
    rcos = math.cos(theta)
    rsin = math.sin(theta)
    A = np.array([[1, 0, 0],
                  [0, rcos, -rsin],
                  [0, rsin, rcos]])
    return A


def Ry(theta):
    theta = np.deg2rad(theta)
    rcos = math.cos(theta)
    rsin = math.sin(theta)
    A = np.array([[rcos, 0, rsin],
                  [0, 1, 0],
                  [-rsin, 0, rcos]])
    return A


def Rz(theta):
    theta = np.deg2rad(theta)
    rcos = math.cos(theta)
    rsin = math.sin(theta)
    A = np.array([[rcos, -rsin, 0],
                  [rsin, rcos, 0],
                  [0, 0, 1]])
    return A


def R_from_vectors(a, b):

    # Normalize the vectors
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    aa = a/na
    bb = b/nb

    cross_prod = np.cross(aa.T, bb.T)
    cross_mag = np.linalg.norm(cross_prod)

    dot_prod = np.dot(aa.T, bb)
    dot_prod = min(dot_prod[0][0], 1)

    if cross_mag > 0:
        cross_prod = cross_prod[0]/np.linalg.norm(cross_prod[0])
    else:
        cross_prod = cross_prod[0]

    theta = math.acos(dot_prod)
    t = cross_prod

    u = t[0]
    v = t[1]
    w = t[2]

    matrix = np.zeros((3, 3))
    rcos = math.cos(theta)
    rsin = math.sin(theta)
    matrix[0][0] = rcos + u * u * (1 - rcos)
    matrix[1][0] = w * rsin + v * u * (1 - rcos)
    matrix[2][0] = -v * rsin + w * u * (1 - rcos)
    matrix[0][1] = -w * rsin + u * v * (1 - rcos)
    matrix[1][1] = rcos + v * v * (1 - rcos)
    matrix[2][1] = u * rsin + w * v * (1 - rcos)
    matrix[0][2] = v * rsin + u * w * (1 - rcos)
    matrix[1][2] = -u * rsin + v * w * (1 - rcos)
    matrix[2][2] = rcos + w * w * (1 - rcos)

    return matrix


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis /= math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def get_angle_from_rotation(R):

    M = np.asarray(R)
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat

    if abs(r31) != 1:
        y = -np.arcsin(r31)
        y2 = np.pi-y
        # x = math.atan2(r32 / math.cos(y), r33 / math.cos(y))
        x2 = math.atan2(r32 / math.cos(y2), r33 / math.cos(y2))
        # z = math.atan2(r21 / math.cos(y), r11 / math.cos(y))
        z2 = math.atan2(r21 / math.cos(y2), r11 / math.cos(y2))

        y = y2
        x = x2
        z = z2
    else:
        z = 0
        if r31 == -1:
            y = np.pi/2.
            x = math.atan2(r12, r13)
        else:
            y = -np.pi / 2.
            x = math.atan2(-r12, -r13)

    return np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)


def rigid_transform_3d(A, B):

    assert len(A) == len(B)

    N = A.shape[0]  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = AA.T.dot(BB)

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T

    return R, t
