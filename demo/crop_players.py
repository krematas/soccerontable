import numpy as np
import os
import argparse
import cv2
from tqdm import tqdm

import soccer3d
import utils.files as file_utils
import soccer3d.instancesegm.utils as seg_utils


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona', help='path')
parser.add_argument('--margin', type=int, default=25, help='Margin around the pose')
opt, _ = parser.parse_known_args()


db = soccer3d.YoutubeVideo(opt.path_to_data)
db.digest_metadata()

db.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)


# Prepare the tree dir structure
file_utils.mkdir(os.path.join(db.path_to_dataset, 'players'))
file_utils.mkdir(os.path.join(db.path_to_dataset, 'players', 'images'))
file_utils.mkdir(os.path.join(db.path_to_dataset, 'players', 'poseimgs'))
file_utils.mkdir(os.path.join(db.path_to_dataset, 'players', 'pose_masks'))
file_utils.mkdir(os.path.join(db.path_to_dataset, 'players', 'cnn_masks'))
file_utils.mkdir(os.path.join(db.path_to_dataset, 'players', 'labels'))
file_utils.mkdir(os.path.join(db.path_to_dataset, 'players', 'predictions'))
file_utils.mkdir(os.path.join(db.path_to_dataset, 'players', 'metadata'))
file_utils.mkdir(os.path.join(db.path_to_dataset, 'players', 'meshes'))

margin = opt.margin

# Decompose frame into players
for sel_frame in tqdm(range(db.n_frames)):

    img = db.get_frame(sel_frame)
    basename = db.frame_basenames[sel_frame]
    poses = db.poses[basename]
    mask = db.get_instances_from_detectron(sel_frame, is_bool=True)

    skeleton_buffer = seg_utils.get_instance_skeleton_buffer(db.shape[0], db.shape[1], poses)

    h, w = img.shape[:2]
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

        crop = img[y1:y2, x1:x2, :]
        cv2.imwrite(os.path.join(db.path_to_dataset, 'players', 'images', '{0}_{1:05d}.jpg'.format(db.frame_basenames[sel_frame], i)), crop[:, :, (2, 1, 0)]*255)

        pose_img = seg_utils.get_poseimg_for_opt(i, skeleton_buffer[y1:y2, x1:x2], mask[y1:y2, x1:x2], n_bg=30)
        cv2.imwrite(os.path.join(db.path_to_dataset, 'players', 'poseimgs', '{0}_{1:05d}.png'.format(db.frame_basenames[sel_frame], i)), (pose_img+1).astype(np.uint8))
        cv2.imwrite(os.path.join(db.path_to_dataset, 'players', 'cnn_masks','{0}_{1:05d}.png'.format(db.frame_basenames[sel_frame], i)), mask[y1:y2, x1:x2]*255)
