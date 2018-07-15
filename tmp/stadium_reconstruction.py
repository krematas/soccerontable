import os
import argparse
import numpy as np
import soccer3d
import utils.files as file_utils
import utils.io as io
import utils.camera as cam_utils
from shapely.geometry import Polygon
from scipy.spatial import Delaunay
import cv2
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/Users/krematas/data/Singleview/Soccer/Japan-Something-0', help='path')
opt, _ = parser.parse_known_args()

db = soccer3d.YoutubeVideo(opt.path_to_data)
# db.gather_detectron()
db.digest_metadata()

vertex_data, _, _ = io.read_obj('/Users/krematas/data/Singleview/Soccer/field_simple.obj')
vertex, _, _ = io.ply_to_numpy(vertex_data)


img = db.get_frame(0)
calib_data = db.calib[db.frame_basenames[0]]

cam = cam_utils.Camera('_', calib_data['A'], calib_data['R'], calib_data['T'], db.shape[0], db.shape[1])

name = 'front'
side = np.zeros((4, 3))
side[0, :] = [-52.365,   0., -33.87]
side[1, :] = [52.365,   0., -33.87]
side[2, :] = [52.365,   20., -33.87]
side[3, :] = [-52.365,   20., -33.87]

# side[0, :] = [52.365,   0., -33.87]
# side[1, :] = [52.365,   0., 33.87]
# side[2, :] = [52.365,   20., 33.87]
# side[3, :] = [52.365,   20., -33.87]

vertex = side.copy()

plane_normal = np.array([0, 0, 1])
plane_origin = np.array([0, 0, -33.87])

# plane_normal = np.array([0, 0, 0])
# plane_origin = np.array([0, 0, 0])
#
# plane_normal = np.array([-1, 0, 0])
# plane_origin = np.array([52.365, 0, 0])

plane = vertex.copy()
# plane[0, 0] -= 2
# plane[1, 0] += 2
# plane[2, 0] += 2
# plane[3, 0] -= 2
#
# plane[0, 2] += 20
# plane[1, 2] += 20
# plane[2, 2] -= 2
# plane[3, 2] -= 2


def project_plane_to_image(vertex, cam, plane_origin, plane_normal):
    p2, depth = cam.project(vertex, dtype=np.float32)
    behind_points = (depth < 0).nonzero()[0]
    p2[behind_points, :] *= -1

    # Find intersection between field lines and image borders
    poly1 = Polygon([p2[0, :], p2[1, :], p2[2, :], p2[3, :]])
    poly2 = Polygon([(0, 0), (cam.width, 0), (cam.width, cam.height), (0, cam.height)])
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
    image_borders = np.array([[0 + margin, 0 + margin], [db.shape[1] - margin, 0 + margin],
                              [db.shape[1] - margin, db.shape[0] - margin], [0 + margin, db.shape[0] - margin]])

    M = cv2.getPerspectiveTransform(field_borders.astype(np.float32), image_borders.astype(np.float32))
    dst = cv2.warpPerspective(filled, M, (db.shape[1], db.shape[0]), )

    transformed_p2 = cv2.perspectiveTransform(np.array([p2], dtype=np.float32), M)
    transformed_p2 = transformed_p2[0, :]

    return dst, transformed_p2


points3d, uv, faces = project_plane_to_image(plane, cam, plane_origin, plane_normal)
p2, _ = cam.project(points3d)

dst, transformed_p2 = rectify_image(img, cam, vertex, p2)

plt.imshow(dst)
plt.triplot(transformed_p2[:, 0], transformed_p2[:,1], faces)
plt.show()


cv2.imwrite('/Users/krematas/data/Singleview/field3D/{0}.jpg'.format(name), dst[:, :, ::-1])

vertex_data_out = io.numpy_to_ply(points3d)
uv = transformed_p2.copy()
uv[:, 0] = uv[:, 0] / db.shape[1]
uv[:, 1] = 1 - uv[:, 1] / db.shape[0]

io.write_obj('/Users/krematas/data/Singleview/field3D/{0}.obj'.format(name), vertex_data_out, faces, uv, '{0}.jpg'.format(name))



# plt.imshow(img)
# # plt.plot(xy[0], xy[1])
# plt.plot(_p2[:, 0], _p2[:, 1], 'o')
# plt.triplot(_p2[:,0], _p2[:,1], faces)
# plt.show()

# io.imshow(img, points=p2)