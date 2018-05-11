import numpy as np
import soccer3d
from os.path import join
import utils.camera as cam_utils
import utils.io as io
import utils.misc as misc_utils
import utils.geometric as geom_utils
from tqdm import tqdm
import cv2
import argparse
import torch
import torch.nn as nn


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona', help='path')
opt, _ = parser.parse_known_args()


db = soccer3d.YoutubeVideo(opt.path_to_data)
db.digest_metadata()

db.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)


h, w = db.shape[0], db.shape[1]
margin = 0
bins = np.linspace(-0.5, 0.5, 50)

for sel_frame in tqdm(range(db.n_frames)):

    img = db.get_frame(sel_frame)
    basename = db.frame_basenames[sel_frame]

    poses = db.poses[basename]
    #
    cam_data = db.calib[basename]
    cam = cam_utils.Camera(basename, cam_data['A'], cam_data['R'], cam_data['T'], db.shape[0], db.shape[1])

    for i in range(len(poses)):
        player_name = '{0}_{1:05d}'.format(db.frame_basenames[sel_frame], i)

        anno = np.loadtxt(join(db.path_to_dataset, 'players', 'anno', player_name+'.txt'))
        pred_npy = np.load(join(db.path_to_dataset, 'players', 'predictions', player_name+'.npy'))

        x1, y1, x2, y2 = anno[:4].astype(int)
        cx, cy, cz = anno[4:7]
        player_depth = anno[7]

        upsampler = nn.UpsamplingBilinear2d(size=(int(y2 - y1), int(x2 - x1)))
        _prediction = upsampler(torch.from_numpy(pred_npy))
        prediction = np.argmax(_prediction.cpu().data.numpy(), axis=1)[0, :, :]

        mask = np.zeros_like(prediction)
        I, J = (prediction > 0).nonzero()
        mask[I, J] = 1

        depthmap = bins[prediction - 1]

        # Estimate the bounding box plane: the plane that passes through the mid of the player and faces towards the camera
        p0 = np.array([[cx, cy, cz]])
        n0 = cam.get_direction()
        origin = cam.get_position().T
        n0[1] = 0.0
        n0 /= np.linalg.norm(n0)

        player_points2d = np.vstack((J, I)).T
        p3 = cam.unproject(player_points2d, 0.5)
        direction = p3.T - np.tile(origin, (p3.shape[1], 1))
        direction /= np.tile(np.linalg.norm(direction, axis=1)[:, np.newaxis], (1, 3))
        billboard = geom_utils.ray_plane_intersection(origin, direction, p0, n0)
        billboard_2d, billboard_depth = cam.project(billboard, dtype=np.float32)

        depthmap[I, J] += billboard_depth
        depthmap *= mask

        z_buffer = np.zeros((db.shape[0], db.shape[1]))
        z_buffer[y1:y2, x1:x2] = depthmap
        I, J = (z_buffer > 0).nonzero()

        player3d = cam.depthmap_to_pointcloud(z_buffer)
        color3d = img[I, J, :]

        ply_data = io.numpy_to_ply(player3d, color3d*255)
        io.write_ply(join(db.path_to_dataset, 'players', 'meshes', player_name+'.ply'), ply_data)

