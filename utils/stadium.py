import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Delaunay
import utils.camera as cam_utils
import cv2


W, H = 52.365+2, 33.87+3


def get_bottom_params():
    vertex = np.array([[-W, 0., H],
                       [W, 0., H],
                       [W, 0., -H],
                       [-W, 0., -H]])

    plane_normal = np.array([0, 1, 0])
    plane_origin = np.array([0, 0, 0])

    return vertex, plane_origin, plane_normal


def get_back_params():
    vertex = np.array([[-W,   0., -H],
                       [W, 0., -H],
                       [W, 20., -H],
                       [-W, 20., -H]])

    plane_normal = np.array([0, 0, 1])
    plane_origin = np.array([0, 0, -H])

    return vertex, plane_origin, plane_normal


def get_right_params():
    vertex = np.array([[W,   0., -H],
                       [W, 0., H],
                       [W, 20., H],
                       [W, 20., -H]])

    plane_normal = np.array([-1, 0, 0])
    plane_origin = np.array([W, 0, 0])

    return vertex, plane_origin, plane_normal


def get_left_params():
    vertex = np.array([[-W,   0., -H],
                       [-W, 0., H],
                       [-W, 20., H],
                       [-W, 20., -H]])

    plane_normal = np.array([1, 0, 0])
    plane_origin = np.array([-W, 0, 0])

    return vertex, plane_origin, plane_normal


def project_plane_to_image(vertex, cam, plane_origin, plane_normal):
    p2, depth = cam.project(vertex, dtype=np.float32)
    behind_points = (depth < 0).nonzero()[0]
    p2[behind_points, :] *= -1

    # Find intersection between field lines and image borders
    poly1 = Polygon([p2[0, :], p2[1, :], p2[2, :], p2[3, :]])
    poly2 = Polygon([(0, 0), (cam.width, 0), (cam.width, cam.height), (0, cam.height)])
    if not poly1.intersects(poly2):
        return None, None, None
    inter_poly = poly1.intersection(poly2)
    xy = inter_poly.exterior.coords.xy

    points = np.array([xy[0], xy[1]]).T

    tri = Delaunay(points)
    faces = tri.simplices.copy()

    # Convert faces to 3D points
    points3d = cam_utils.plane_points_to_3d(points, cam, plane_origin, plane_normal)

    uv = points.copy()
    uv[:, 0] = uv[:, 0]/cam.width
    uv[:, 1] = 1-uv[:, 1]/cam.height

    return points3d, uv, faces


def rectify_image(img, cam, vertex, p2):
    # Wrap the image
    filled = (img * 255).astype(np.uint8)

    field_borders, _ = cam.project(vertex)
    margin = 400
    image_borders = np.array([[0 + margin, 0 + margin], [cam.width - margin, 0 + margin],
                              [cam.width - margin, cam.height - margin], [0 + margin, cam.height - margin]])

    M = cv2.getPerspectiveTransform(field_borders.astype(np.float32), image_borders.astype(np.float32))
    dst = cv2.warpPerspective(filled, M, (cam.width, cam.height), )

    transformed_p2 = cv2.perspectiveTransform(np.array([p2], dtype=np.float32), M)
    transformed_p2 = transformed_p2[0, :]

    return dst, transformed_p2
