import numpy as np
import soccer3d
from soccer3d.tracking import Detection, find_tracks, smooth_trajectory, convert_to_MOT
from os.path import join
import utils.camera as cam_utils
import utils.misc as misc_utils
import json
import argparse
from tqdm import tqdm
import glog
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona', help='path')
parser.add_argument('--dist_thresh', type=int, default=50, help='Distance threshold for merging tracklets (in pixels)')
opt, _ = parser.parse_known_args()


db = soccer3d.YoutubeVideo(opt.path_to_data)
db.digest_metadata()

db.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)

# ----------------------------------------------------------------------------------------------------------------------
# Gather all souces, boxes
glog.info('Tracking players')

dets = []
dets_per_frame = []

for i in range(db.n_frames):
    basename = db.frame_basenames[i]
    poses = db.poses[basename]

    __detection_list = []
    for j in range(len(poses)):
        cur_det = Detection(poses[j], basename, i)
        cur_det.mesh_name = '{0}_{1:05d}'.format(basename, j)
        __detection_list.append(cur_det)
        dets.append(cur_det)
    dets_per_frame.append(__detection_list)


new_tracklets = find_tracks(dets_per_frame, db.frame_basenames, dist_thresh=opt.dist_thresh)


# ----------------------------------------------------------------------------------------------------------------------
# Save tracks

mot_matrix = convert_to_MOT(new_tracklets, db.n_frames)
db.dump_video('tracks', scale=2, mot_tracks=mot_matrix)


# ----------------------------------------------------------------------------------------------------------------------
# 3DTrajectory smoothing
glog.info('Smoothing 3D trajectories')

data_out = {i: [] for i in db.frame_basenames}

fig = plt.figure()
ax = fig.add_subplot(111)


for i in tqdm(range(len(new_tracklets))):

    neck_pos = []
    for j in range(len(new_tracklets[i])):
        frame_index = new_tracklets[i][j].frame_index
        basename = db.frame_basenames[frame_index]

        cam_data = db.calib[basename]
        cam = cam_utils.Camera(basename, cam_data['A'], cam_data['R'], cam_data['T'], db.shape[0], db.shape[1])

        kp_3d = misc_utils.lift_keypoints_in_3d(cam, new_tracklets[i][j].keypoints)
        neck_pos.append(kp_3d[1, :])
    neck_pos = np.array(neck_pos)

    # Smooth trajectory
    smoothed_positions = smooth_trajectory(new_tracklets[i], neck_pos)
    for j in range(len(new_tracklets[i])):
        data_out[new_tracklets[i][j].frame].append({'mesh': new_tracklets[i][j].mesh_name, 'x': smoothed_positions[0, j],
                                                    'y': smoothed_positions[1, j], 'z': smoothed_positions[2, j]})

    ax.plot(smoothed_positions[0, :], smoothed_positions[2, :], 'o')

plt.show()

with open(join(db.path_to_dataset, 'players', 'metadata', 'position.json'), 'w') as outfile:
    json.dump(data_out, outfile)
