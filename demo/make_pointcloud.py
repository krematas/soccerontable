import numpy as np
import soccer3d
from os.path import join
import utils.camera as cam_utils
import utils.io as io
import utils.misc as misc_utils
from tqdm import tqdm
import cv2
import argparse


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona', help='path')
opt, _ = parser.parse_known_args()


db = soccer3d.YoutubeVideo(opt.path_to_data)
db.digest_metadata()

db.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)


h, w = db.shape[0], db.shape[1]
margin = 25
bins = np.linspace(-0.5, 0.5, 51 - 1)

for sel_frame in tqdm(range(db.n_frames)):

    img = db.get_frame(sel_frame)
    basename = db.frame_basenames[sel_frame]
    poses = db.poses[basename]

    cam_data = db.calib[basename]
    cam = cam_utils.Camera(basename, cam_data['A'], cam_data['R'], cam_data['T'], db.shape[0], db.shape[1])

    for i in range(len(poses)):
        cur_pose = poses[i]
        valid = cur_pose[:, 2] > 0
        x1, y1, x2, y2 = int(np.min(cur_pose[valid, 0])), int(np.min(cur_pose[valid, 1])), \
                         int(np.max(cur_pose[valid, 0])), int(np.max(cur_pose[valid, 1]))

        x1 -= margin
        y1 -= margin
        x2 += margin
        y2 += margin
        x1, x2, y1, y2 = max(x1, 0), min(w, x2), max(y1, 0), min(h, y2)

        player_name = '{0}_{1:05d}'.format(db.frame_basenames[sel_frame], i)
        pred_npy = np.load(join(db.path_to_dataset, 'players', 'predictions', player_name+'.npy'))
        prediction = np.argmax(pred_npy, axis=1)[0, :, :]

        mask = np.zeros_like(prediction)
        I, J = (prediction > 0).nonzero()
        mask[I, J] = 1

        # Lift box in 3D
        kp_3d = misc_utils.lift_keypoints_in_3d(cam, poses[i])
        _, player_depth = cam.project(kp_3d)

        depthmap = bins[prediction - 1]
        depthmap = cv2.resize(depthmap, (x2-x1, y2-y1))
        mask = cv2.resize(mask, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)

        depthmap += np.mean(player_depth)
        depthmap *= mask

        z_buffer = np.zeros((db.shape[0], db.shape[1]))
        z_buffer[y1:y2, x1:x2] = depthmap
        I, J = (z_buffer > 0).nonzero()

        player3d = cam.depthmap_to_pointcloud(z_buffer)
        color3d = img[I, J, :]

        ply_data = io.numpy_to_ply(player3d, color3d*255)
        io.write_ply(join(db.path_to_dataset, 'players', 'meshes', player_name+'.ply'), ply_data)

