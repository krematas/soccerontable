import numpy as np
import torch.nn as nn
import utils.geometric as cg_utils
import utils.io as io
import utils.mesh as mesh_utils
import utils.files as file_utils
import yaml
from os.path import join


def read_annotation_from_filename(path_to_data, fname):

    basename, ext = file_utils.extract_basename(fname)
    annoname = join(path_to_data, 'anno', '{0}.yml'.format(basename))

    with open(annoname) as data_file:
        data_yml = yaml.load(data_file)

    return data_yml


def convert_network_prediction_to_depthmap(netpred, cropbox, output_nc=51):

    x1, y1, x2, y2 = cropbox[:4]

    crop_w = (x2 - x1)
    crop_h = (y2 - y1)
    upsampler = nn.UpsamplingBilinear2d(size=(int(crop_h), int(crop_w)))
    final_prediction = upsampler(netpred)
    final_prediction = np.argmax(final_prediction.cpu().data.numpy(), axis=1)[0, :, :]

    estimated_mask = np.zeros_like(final_prediction)
    I, J = (final_prediction > 0).nonzero()
    estimated_mask[I, J] = 1

    bins = np.linspace(-0.5, 0.5, output_nc - 1)
    estimated_depthmap = bins[final_prediction - 1]

    return estimated_depthmap, estimated_mask


def convert_depthmap_to_pointcloud(estimated_depthmap, estimated_mask, cam, cropbox, mean_pos):

    h, w = cam.height, cam.width
    x1, y1, x2, y2 = cropbox[:4]

    view_mask = np.zeros((h, w))
    view_depth = np.zeros((h, w))

    view_depth[y1:y2, x1:x2] = estimated_depthmap
    view_mask[y1:y2, x1:x2] = estimated_mask
    I, J = (view_mask == 1).nonzero()

    # Make billboard: a vertical plane passing through the middle of the player
    p0 = mean_pos
    n0 = cam.get_direction()
    origin = cam.get_position().T
    n0[1] = 0.0
    n0 /= np.linalg.norm(n0)

    # Get the rays from the mask towards that plane
    player_points2d = np.vstack((J, I)).T
    p3 = cam.unproject(player_points2d, 0.5)
    direction = p3.T - np.tile(origin, (p3.shape[1], 1))
    direction /= np.tile(np.linalg.norm(direction, axis=1)[:, np.newaxis], (1, 3))
    billboard = cg_utils.ray_plane_intersection(origin, direction, p0, n0)

    billboard2d, billboard_depth = cam.project(billboard, dtype=np.float32)

    metric_depth = view_depth[I, J] + billboard_depth
    points2d = np.array([J, I]).T

    player3d = cam.unproject(points2d, metric_depth).T

    view_depth[I, J] += billboard_depth
    return player3d, view_depth


def get_colored_mesh_from_pointcloud(player3d, img, cam):

    player2d, depth = cam.project(player3d, dtype=np.int32)
    colors = img[player2d[:, 1], player2d[:, 0]]

    faces = mesh_utils.triangulate_depthmap_points(player2d, depth, depth_thresh=0.1)
    ply_data = io.numpy_to_ply(player3d, colors*255)
    return ply_data, faces