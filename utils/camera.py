import numpy as np
import cv2
import os
from . import transform
from . import geometric


def grid_search_focal_length(points3d, points2d, h, w, same_f=False, fx_step=20):
    """ Finds the focal length that minimizes the reprojection error between a set of 3D points and its corresponding
    2D location. It searchs over the predefined interval 0.5, 1.5 x width
    :param points3d: set of 3D points
    :param points2d: its corresponding 2d points in the image
    :param h: height of the image
    :param w: width of the image
    :param same_f: if focal length in x and y direction are the same
    :param fx_step: step size of focal length search
    :return: the best focal length
    """

    best_score = 10e10
    best_fx, best_fy = -1, -1
    min_fx, max_fx = w/2., 10.*w
    min_fy, max_fy, fy_step = h / 2., 10. * h, fx_step
    if same_f:
        min_fy, max_fy, fy_step = 0, 1, 1

    for i in np.arange(min_fx, max_fx, fx_step):
        fx = i

        for j in np.arange(min_fy, max_fy, fy_step):
            if same_f:
                fy = fx
            else:
                fy = j

            A = intrinsic_matrix_from_focal_length(fx, fy, h, w)
            _, rvec, tvec = cv2.solvePnP(points3d, points2d, A, None)
            reproj, _ = cv2.projectPoints(points3d, rvec, tvec, A, None)
            reproj = np.squeeze(reproj)

            score = np.sum(np.linalg.norm(points2d - reproj, axis=1))

            # print(fx, fy, score)
            if score < best_score:
                best_score = score
                best_fx = fx
                best_fy = fy

    return best_fx, best_fy


def intrinsic_matrix_from_focal_length(fx, fy, h, w):
    return np.array([[fx, 0, w/2.], [0, fy, h/2.], [0, 0, 1]])


def project(points3d, A, R, T, scale_factor=1.0, dtype=np.int32):
    """ Project a set of 3D points (Nx3 or 3XN) to a camera with parameters A, R T.
    Return the pixel coordinates and its corresponding depth
    """

    if points3d.shape[0] != 3:
        points3d = points3d.T

    assert(T.shape == (3, 1))

    n_points = points3d.shape[1]

    projected_points_ = A.dot(R.dot(points3d) + np.tile(T, (1, n_points)))
    depth = projected_points_[2, :]
    pixels = projected_points_[0:2, :] / projected_points_[2, :] / scale_factor

    if issubclass(dtype, np.integer):
        pixels = np.round(pixels)

    pixels = np.array(pixels.T, dtype=dtype)

    return pixels, depth


def inside_frame(points2d, height, width, margin=0):
    valid = np.logical_and(np.logical_and(points2d[:, 0] >= 0+margin, points2d[:, 0] < width-margin),
                           np.logical_and(points2d[:, 1] >= 0+margin, points2d[:, 1] < height-margin))
    points2d = points2d[valid, :]
    return points2d, valid


def look_at(eye=np.array([[0, 0, -1]]).T, target=np.array([[0, 0, 0]]).T, up=np.array([[0, 1, 0]]).T):
    delta = eye - target
    z = np.divide(delta, np.linalg.norm(delta))
    x = np.divide(np.cross(up, z, axis=0), np.linalg.norm(np.cross(up, z, axis=0)))
    y = np.divide(np.cross(z, x, axis=0), np.linalg.norm(np.cross(z, x, axis=0)))

    view_matrix = np.array([[x[0][0], x[1][0], x[2], np.dot(x.T, (-eye))[0]],
                 [y[0][0], y[1][0], y[2][0], np.dot(y.T, (-eye))[0]],
                 [z[0][0], z[1][0], z[2][0], np.dot(z.T, (-eye))[0]],
                 [0,    0,   0,    1]])

    return view_matrix.T


def perspective(fov=60, aspect=1, zNear=0.01, zFar=1000):

    fov = np.pi*fov/180
    deltaZ = zFar-zNear
    cotangent = np.cos(fov*0.5)/np.sin(fov*0.5)

    projection_matrix = np.array([[cotangent/aspect, 0, 0, 0],
                        [0, cotangent, 0, 0],
                        [0, 0, (zFar+zNear)/deltaZ, -2*zNear*zFar/deltaZ],
                        [0, 0, -1, 0]])

    return projection_matrix.T


def d3d_perspective_fov_rh(fovy=60, near=1, far=1000, aspect_ratio=1.0):

    # fovx = np.pi*fovx/180.
    fovy = np.pi * fovy/180.
    delta_z = near-far
    # cotangentx = np.cos(fovx*0.5)/np.sin(fovx*0.5)
    cotangenty = np.cos(fovy * 0.5) / np.sin(fovy * 0.5)
    yScale = cotangenty
    xScale = yScale / aspect_ratio
    projection_matrix = np.array([[xScale, 0, 0, 0],
                        [0, yScale, 0, 0],
                        [0, 0, far/delta_z, near*far/delta_z],
                        [0, 0, -1, 0]])

    return projection_matrix


def d3d_perspective_rh(h, w, near=1, far=100):
    projection_matrix = np.array([[2*(near/w), 0, 0, 0],
                        [0, 2*(near/h), 0, 0],
                        [0, 0, far/(near - far), far*near/(near-far)],
                        [0, 0, -1, 0]])

    return projection_matrix


def opencv_to_opengl(A, R, T, h, w, near=1, far=1000):

    fx, fy, cx, cy = A[0, 0], A[1, 1], A[0, 2], A[1, 2]

    # fov_rad_x = 2*np.arctan(0.5 * w / fx)
    # fov_x = np.degrees(fov_rad_x)
    #
    # fov_rad_y = 2 * np.arctan(0.5 * h / fy)
    # fov_y = np.degrees(fov_rad_y)
    #
    # F = np.array([[mp.cot(fov_x / 2 / 180 * np.pi), 0, 0, 0],
    #               [0, mp.cot(fov_y / 2 / 180 * np.pi), 0, 0],
    #               [0, 0, -(far+near)/(far-near), -2*far*near/(far-near)],
    #               [0, 0, -1, 0]])

    F = np.array([[fx/cx, 0, 0, 0],
                  [0, fy/cy, 0, 0],
                  [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                  [0, 0, -1, 0]])

    projection_matrix = F.T

    deg = 180
    t=deg*np.pi/180.
    Rz = np.array([[np.cos(t), -np.sin(t), 0],
                    [np.sin(t), np.cos(t), 0],
                    [0,0,1]])

    Ry = np.array([[np.cos(t), 0, np.sin(t)],
                    [0,1,0],
                    [-np.sin(t), 0, np.cos(t)]])

    R_gl = Rz.dot(Ry.dot(R))

    view_matrix=np.zeros((4,4))
    view_matrix[0:3,0:3]=R_gl.T
    view_matrix[0][3] = 0.0
    view_matrix[1][3] = 0.0
    view_matrix[2][3] = 0.0
    view_matrix[3][0] = T[0]
    # 	also invert Y and Z of translation
    view_matrix[3][1] = -T[1]
    view_matrix[3][2] = -T[2]
    view_matrix[3][3] = 1.0

    return np.array(view_matrix).astype(np.float32), np.array(projection_matrix).astype(np.float32)


def plane_points_to_3d(points2d, cam, plane_origin=np.array([0, 0, 0]), plane_direction=np.array([0, 1, 0])):
    p3 = cam.unproject(points2d, 0.5)
    origin = cam.get_position().T
    direction = p3.T - np.tile(origin, (p3.shape[1], 1))
    direction /= np.tile(np.linalg.norm(direction, axis=1)[:, np.newaxis], (1, 3))
    plane3d = geometric.ray_plane_intersection(origin, direction, plane_origin, plane_direction)
    return plane3d


class Camera:

    def __init__(self, name=None, A=None, R=None, T=None, h=None, w=None):

        self.name = name
        self.A = np.eye(3, 3)
        self.A_i = np.eye(3, 3)
        self.R = np.eye(3, 3)
        self.T = np.zeros((3, 1))
        self.dist_coeff = np.zeros((4, 1))

        self.view = []
        self.mask = []

        self.height = None
        self.width = None

        self.org_height = -1
        self.org_width = -1

        self.scale_factor = 1.0

        if A is not None:
            self.set_intrinsics(A)
        if R is not None and T is not None:
            self.set_extrinsics(R, T)
        if h is not None and w is not None:
            self.set_size(h, w)

    def set_all_parameters(self, np_struct, h, w):
        self.set_intrinsics(np_struct['A'])
        self.set_extrinsics(np_struct['R'], np_struct['T'])
        self.set_size(h, w)

    def set_intrinsics(self, A):
        self.A = A
        self.A_i = np.linalg.inv(A)

    def set_extrinsics(self, R, T):
        assert(T.shape == (3, 1))
        self.R = R
        self.T = T

    def set_parameters(self, A, R, T):
        assert (T.shape == (3, 1))
        self.A = A
        self.R = R
        self.T = T

    def set_size(self, h, w):
        self.width = w
        self.height = h

    def project(self, points3d, dtype=np.int32):

        pixels, depth = project(points3d, self.A, self.R, self.T, dtype=dtype)

        return pixels, depth

    def unproject(self, points2d, depth):

        if points2d.shape[0] != 2:
            points2d = points2d.T

        n_points = points2d.shape[1]

        points2d = np.vstack((points2d*self.scale_factor, np.ones(points2d.shape[1])))
        pixel_i = self.A_i.dot(points2d)
        pixel_world = self.R.T.dot(np.multiply(depth, pixel_i) - np.tile(self.T, (1, n_points)))

        return pixel_world

    def get_position(self):
        return -self.R.T.dot(self.T)

    def get_direction(self):
        direction = self.R.T.dot(np.array([[0, 0, 1]]).T)[:, 0]
        return direction/np.linalg.norm(direction)

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

    def get_euler_rotation(self):
        theta_x, theta_y, theta_z = transform.get_angle_from_rotation(self.R)
        return theta_x, theta_y, theta_z

    def to_opengl(self, znear=1, zfar=10):

        modelview = np.zeros((4, 4))
        projection = np.zeros((4, 4))

        fx, fy, cx, cy = self.A[0, 0], self.A[1, 1], self.A[0, 2], self.A[1, 2]
        h, w = self.height, self.width

        projection[0, 0], projection[1, 1] = fx / cx, fy / cy
        projection[2, 2], projection[2, 3] = -(znear + zfar) / (zfar - znear), (-2 * znear * zfar) / (zfar - znear)
        projection[3, 2] = -1

        modelview[0:3, 0:3] = self.R
        modelview[0:3, 3] = self.T[:, 0]
        modelview[2, 3] *= -1
        modelview[3, 3] = 1
        modelview[0, (1, 2)] *= 1
        modelview[1, (0, 3)] *= 1
        modelview[2, (0, 3)] *= 1

        return modelview, projection

    def set_view(self, img_name, undistort=True):
        img1 = cv2.imread(img_name)
        self.org_height, self.org_width = img1.shape[0:2]

        if undistort:
            img1 = cv2.undistort(img1, self.A, self.dist_coeff)
        img1 = cv2.resize(img1, None, fx=1.0 / self.scale_factor, fy=1.0 / self.scale_factor)

        self.view = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)/255.
        self.height = self.view.shape[0]
        self.width = self.view.shape[1]

    def set_mask(self, mask_name, undistort=True):
        if os.path.exists(mask_name):
            mask = cv2.imread(mask_name, 0)
            if undistort:
                mask = cv2.undistort(mask, self.A, self.dist_coeff)
            mask = cv2.resize(mask, None, fx=1.0 / self.scale_factor, fy=1.0 / self.scale_factor,
                              interpolation=cv2.INTER_NEAREST)
            self.mask = mask / 255
        else:
            self.mask = np.ones((self.height, self.width), dtype=int)

    def depthmap_to_pointcloud(self, depth_buffer, thresh=0.0):
        I, J = (depth_buffer > thresh).nonzero()
        points2d = np.array([J, I]).T
        depth = depth_buffer[I, J]
        points3d = self.unproject(points2d, depth)
        return points3d.T
