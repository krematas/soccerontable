import numpy as np
import os
import argparse
import cv2
from tqdm import tqdm

import soccer3d
import utils.io as io
import utils.files as file_utils


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona', help='path')
opt, _ = parser.parse_known_args()


db = soccer3d.YoutubeVideo(opt.path_to_data)
db.digest_metadata()

db.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)

file_utils.mkdir(os.path.join(db.path_to_dataset, 'players', 'masks'))
file_utils.mkdir(os.path.join(db.path_to_dataset, 'players', 'labels'))

# Merge masks for players
for sel_frame in tqdm(range(db.n_frames)):

    basename = db.frame_basenames[sel_frame]
    poses = db.poses[basename]

    for i in range(len(poses)):
        fname = os.path.join(db.path_to_dataset, 'players', 'cnn_masks', '{0}_{1:05d}.png'.format(db.frame_basenames[sel_frame], i))
        cnn_mask = io.imread(fname)
        fname = os.path.join(db.path_to_dataset, 'players', 'pose_masks', '{0}_{1:05d}.png'.format(db.frame_basenames[sel_frame], i))
        pose_mask = io.imread(fname)

        if len(cnn_mask.shape) == 1:
            cnn_mask = np.ones_like(pose_mask)
        mask = cnn_mask[:, :, 0]*pose_mask[:, :, 0]*255
        cv2.imwrite(os.path.join(db.path_to_dataset, 'players', 'masks', '{0}_{1:05d}.png'.format(db.frame_basenames[sel_frame], i)), mask)

        label = {'mask': mask, 'depth': mask, 'billboard': mask}
        np.save(os.path.join(db.path_to_dataset, 'players', 'labels', '{0}_{1:05d}'.format(db.frame_basenames[sel_frame], i)), label)