import numpy as np
import cv2
from . import camera
from .import geometric
from . import io


def sample_spherical(npoints, ndim=3):
    np.random.seed(42)

    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def get_box_from_3d_shpere(cam, center3d):
    h, w = cam.height, cam.width
    sphere = sample_spherical(100) * 1.2

    bounding_shpere = sphere.T + center3d
    sphere2d, _ = cam.project(bounding_shpere, dtype=int)
    sphere2d, _ = camera.inside_frame(sphere2d, h, w)

    tmp_mask = np.zeros((h, w), dtype=np.float32)
    tmp_mask[sphere2d[:, 1], sphere2d[:, 0]] = 1

    # center2d, _ = cam.project(np.array([center3d]), dtype=int)
    # img[center2d[0, 1], center2d[:, 0], 0] = 1
    # io.imshow(img)

    contours, hierarchy, _ = cv2.findContours(tmp_mask.astype(np.uint8), 1, 2)
    x_, y_, ww_, hh_ = cv2.boundingRect(contours)
    box = np.array([x_, y_, x_ + ww_, y_ + hh_, 1])

    return box


def lift_keypoints_in_3d(cam, keypoints, pad=0):
    """
    cam: camera class
    points: Nx3 matrix, N number of keypoints and X, Y, score
    Assumes that the lowest of the point touces the ground
    """

    # Make a bounding box
    x1, y1, x2, y2 = min(keypoints[:, 0])-pad, min(keypoints[:, 1])-pad, max(keypoints[:, 0])+pad, max(keypoints[:, 1])+pad
    bbox = np.array([[x1, y2], [x2, y2], [x1, y1], [x2, y1]])

    bbox_camplane = cam.unproject(bbox, 0.5)
    origin = cam.get_position().T
    bbox_direction = bbox_camplane.T - np.tile(origin, (bbox_camplane.shape[1], 1))
    bbox_direction /= np.tile(np.linalg.norm(bbox_direction, axis=1)[:, np.newaxis], (1, 3))
    bbox_onground = geometric.ray_plane_intersection(origin, bbox_direction, np.array([0, 0, 0]), np.array([0, 1, 0]))

    # Find the billboard plane
    p0 = bbox_onground[0, :]
    p1 = bbox_onground[1, :]
    p3_ = bbox_onground[0, :].copy()
    p3_[1] = 1.0  # Just a bit lifted, since we do not know the original extend

    billboard_n = np.cross(p1 - p0, p3_ - p0)

    # Find the pixels that are masked and contained in the bbox
    keypoints_camplane = cam.unproject(keypoints[:, :2], 0.5)
    kp_direction = keypoints_camplane.T - np.tile(origin, (keypoints_camplane.shape[1], 1))
    kp_direction /= np.tile(np.linalg.norm(kp_direction, axis=1)[:, np.newaxis], (1, 3))
    kepoints_lifted = geometric.ray_plane_intersection(origin, kp_direction, p0, billboard_n)

    return kepoints_lifted


def lift_box_in_3d(cam, bbox):
    n_boxes = bbox.shape[0]
    bbox3d = []

    for i in range(n_boxes):
        # Intersect ground points with the field
        x1, y1, x2, y2 = bbox[i, 0:4]
        points2d = np.array([[x1, y2], [x2, y2], [x1, y1], [x2, y1]])

        p3 = cam.unproject(points2d, 0.5)
        origin = cam.get_position().T
        direction = p3.T - np.tile(origin, (p3.shape[1], 1))
        direction /= np.tile(np.linalg.norm(direction, axis=1)[:, np.newaxis], (1, 3))
        plane3d = geometric.ray_plane_intersection(origin, direction, np.array([0, 0, 0]), np.array([0, 1, 0]))

        # Find the billboard plane
        p0 = plane3d[0, :]
        p1 = plane3d[1, :]
        p3_ = plane3d[0, :].copy()
        p3_[1] = 1.0    # Just a bit lifted, since we do not know the original extend

        billboard_n = np.cross(p1 - p0, p3_ - p0)

        # Find the pixels that are masked and contained in the bbox
        p3 = cam.unproject(points2d, 0.5)
        direction = p3.T - np.tile(origin, (p3.shape[1], 1))
        direction /= np.tile(np.linalg.norm(direction, axis=1)[:, np.newaxis], (1, 3))
        billboard = geometric.ray_plane_intersection(origin, direction, p0, billboard_n)

        bbox3d.append(billboard)

    return np.array(bbox3d)


def putting_objects_in_perspective(camera, boxes, max_height=2.5, min_height=1.5, margin=0.):

    W, H = 104.73, 67.74

    keep = []
    billboards = []
    for i in range(len(boxes)):
        # Interset ground points with the field
        x1, y1, x2, y2 = boxes[i, 0:4]
        points2d = np.array([[x1, y2], [x2, y2], [x1, y1], [x2, y1]])

        p3 = camera.unproject(points2d, 0.5)
        origin = camera.get_position().T
        direction = p3.T - np.tile(origin, (p3.shape[1], 1))
        direction /= np.tile(np.linalg.norm(direction, axis=1)[:, np.newaxis], (1, 3))
        plane3d = geometric.ray_plane_intersection(origin, direction, np.array([0, 0, 0]), np.array([0, 1, 0]))

        # Find the billboard plane
        p0 = plane3d[0, :].copy()
        p1 = plane3d[1, :].copy()
        keypoints_img = plane3d[0, :].copy()
        keypoints_img[1] = 1.0

        billboard_n = np.cross(p1 - p0, keypoints_img - p0)

        # Find the pixels that are masked and contained in the bbox
        p3 = camera.unproject(points2d, 0.5)
        direction = p3.T - np.tile(origin, (p3.shape[1], 1))
        direction /= np.tile(np.linalg.norm(direction, axis=1)[:, np.newaxis], (1, 3))
        billboard = geometric.ray_plane_intersection(origin, direction, p0, billboard_n)

        player_height = billboard[3, 1] - billboard[0, 1]

        if min_height <= player_height <= max_height \
                and (-W / 2. - margin) <= billboard[0, 0] <= (W / 2. + margin) \
                and (-H / 2. - margin) <= billboard[0, 2] <= (H / 2. + margin):
            keep.append(i)
            billboards.append(billboard)

    return keep, billboards


def pack_textures(textures_atlas, uvs_atlas, n_rows=1):

    n_columns = len(textures_atlas)//n_rows+1
    row_images = []
    canvas_h, canvas_w = 0, 0
    for i in range(n_rows):
        max_h, total_w, total_col = 0, 0, 0
        for j in range(n_columns):
            if i*n_columns + j >= len(textures_atlas):
                break

            total_col = j

            h, w = textures_atlas[i*n_columns + j].shape[:2]
            if h > max_h:
                max_h = h
            total_w += w

        row_image = np.zeros((max_h, total_w, 3), dtype=np.float32)
        moving_w = 0
        for j in range(total_col+1):
            h, w = textures_atlas[i * n_columns + j].shape[:2]
            row_image[:h, moving_w:(moving_w+w), :] = textures_atlas[i * n_columns + j]
            uvs_atlas[i * n_columns + j][:, 0] += moving_w
            moving_w += w

        if row_image.shape[1] > canvas_w:
            canvas_w = row_image.shape[1]

        canvas_h += row_image.shape[0]
        row_images.append(row_image)

    atlas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    moving_h = 0
    for i in range(n_rows):
        h, w = row_images[i].shape[:2]
        atlas[moving_h:(moving_h+h), :w, :] = row_images[i]

        for j in range(n_columns):
            if i*n_columns + j >= len(textures_atlas):
                break

            uvs_atlas[i * n_columns + j][:, 1] += moving_h
        moving_h += h

    final_uvs = np.zeros((0, 2))
    for i in range(len(textures_atlas)):
        final_uvs = np.vstack((final_uvs, uvs_atlas[i]))

    # io.imshow(atlas, points=final_uvs)

    final_uvs[:, 0] /= canvas_w
    final_uvs[:, 1] = 1. - final_uvs[:, 1]/canvas_h

    return atlas, final_uvs


class Detection:

    def __init__(self, frame, player, pos2d, pos3d, keypoints, bbox, center3d):
        self.pos2d = pos2d
        self.pos3d = pos3d
        self.frame = frame
        self.player = player
        self.velocity = -1
        self.in_image_border = False
        self.keypoints = keypoints
        self.bbox = bbox
        self.center3d = center3d


class Tracklet:

    def __init__(self, index, detection_list):
        self.id = index
        self.detection_list = detection_list
        self.start_frame = detection_list[0].frame
        self.end_frame = detection_list[-1].frame
        self.start_pos = detection_list[0].pos2d
        self.end_pos = detection_list[-1].pos2d
        self.has_merged = False

    def __add__(self, other):
        self.id = np.minimum(self.id, other.id)
        if self.start_frame < other.start_frame:
            self.detection_list = self.detection_list + other.detection_list
        else:
            self.detection_list = other.detection_list + self.detection_list
        self.start_pos = self.detection_list[0].pos2d
        self.end_pos = self.detection_list[-1].pos2d

        self.start_frame = np.minimum(self.start_frame, other.start_frame)
        self.end_frame = np.maximum(self.end_frame, other.end_frame)
        other.has_merged = True
        return self


def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes
