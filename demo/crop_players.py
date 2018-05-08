import numpy as np
import os
import argparse
import cv2
from tqdm import tqdm

import soccer3d
import utils.camera as cam_utils
import utils.files as file_utils
import utils.misc as misc_utils
import soccer3d.instancesegm.utils as seg_utils


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona', help='path')
parser.add_argument('--margin', type=int, default=0, help='Margin around the pose')
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
file_utils.mkdir(os.path.join(db.path_to_dataset, 'players', 'anno'))

margin = opt.margin

# Decompose frame into players
for sel_frame in tqdm(range(db.n_frames)):

    img = db.get_frame(sel_frame)
    basename = db.frame_basenames[sel_frame]
    poses = db.poses[basename]
    mask = db.get_instances_from_detectron(sel_frame, is_bool=True)

    cam_mat = db.calib[basename]
    cam = cam_utils.Camera(basename, cam_mat['A'], cam_mat['R'], cam_mat['T'], db.shape[0], db.shape[1])

    skeleton_buffer = seg_utils.get_instance_skeleton_buffer(db.shape[0], db.shape[1], poses)

    h, w = img.shape[:2]
    for i in range(len(poses)):
        valid = poses[i][:, 2] > 0

        kp3 = misc_utils.lift_keypoints_in_3d(cam, poses[i][valid, :], pad=0)

        center3d = np.mean(kp3, axis=0)
        # Most of keypoitns are in the upper body so the center of the mass is closer to neck
        center3d[1] -= 0.25

        _, center_depth = cam.project(np.array([center3d]))

        bbox = misc_utils.get_box_from_3d_shpere(cam, center3d)
        x1, y1, x2, y2 = bbox[:4]

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
        with open(os.path.join(db.path_to_dataset, 'players', 'anno', '{0}_{1:05d}.txt'.format(db.frame_basenames[sel_frame], i)), 'w') as f:
            # x1, y1, x2, y2, center_x, center_y, center_z, center_d
            f.write('{0} {1} {2} {3} {4} {5} {6} {7}'.format(x1, y1, x2, y2, center3d[0], center3d[1], center3d[2], center_depth[0]))
