import argparse
import soccer3d
import numpy as np
import cv2
from matplotlib import pyplot as plt
import utils.io as io
from soccer3d.tracking import Detection, find_tracks_simple
from scipy.optimize import minimize


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data',
                    default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Netherlands-Japan-0', help='path')
parser.add_argument('--openpose_dir', default='/home/krematas/code/openpose', help='path')
opt, _ = parser.parse_known_args()


db = soccer3d.YoutubeVideo(opt.path_to_data)
db.gather_detectron()

db.digest_metadata()

db.get_boxes_from_detectron()
db.get_ball_from_detectron(thresh=0.80)

frame_number = 0
img = db.get_frame(frame_number)
# io.show_box(img, db.ball[db.frame_basenames[frame_number]])

dets = []
dets_per_frame = []
all_balls = np.zeros((0, 3))


for i in range(db.n_frames):
    basename = db.frame_basenames[i]
    _ball = db.ball[basename]

    __detection_list = []
    if len(_ball) > 0:
        for j in range(_ball.shape[0]):
            x1, y1, x2, y2, s, = _ball[j, :]
            ball = np.array([[x1, y1, s], [x2, y2, s]])
            all_balls = np.vstack((all_balls, ball[1, :]))
            cur_det = Detection(ball, basename, i)
            cur_det.mesh_name = '{0}_{1:05d}'.format(basename, j)
            __detection_list.append(cur_det)
            dets.append(cur_det)
    dets_per_frame.append(__detection_list)


dist_thresh = 100
time_thresh = 10
new_tracklets = find_tracks_simple(dets_per_frame, db.frame_basenames, dist_thresh=dist_thresh, time_thresh=time_thresh,
                                   len_thresh=2)

tracks = []
data = []
for i in range(len(new_tracklets)):
    ball_pos = np.zeros((db.n_frames, 5))
    ball_pos[:, 4] = -1

    _data = []
    for j in range(len(new_tracklets[i])):
        det = new_tracklets[i][j]
        x1, y1, x2, y2, s = det.keypoints[0, 0], det.keypoints[0, 1], det.keypoints[1, 0], det.keypoints[1, 1], det.keypoints[0, 2]
        ball_pos[det.frame_index, :] = np.array([x1, y1, x2, y2, det.frame_index])

        _data.append([x1, y1, x2, y2, det.frame_index])
    data.append(np.array(_data))
    tracks.append(ball_pos)


tracks2 = []
# Check if track is not moving
for i in range(len(tracks)):
    track_info = tracks[i]
    valid = (track_info[:, 4] > 0).nonzero()[0]
    # io.show_box(img, track_info[valid, :])
    if np.std(track_info[valid, 0]) > 5:
        tracks2.append(track_info)
tracks = tracks2[:]

# Order tracks based on length
index = sorted([i for i in range(len(tracks))], key=lambda x: len(tracks[x]))
tracks = [tracks[i] for i in index]

# Accumulate tracks
ball_pos = np.zeros((db.n_frames, 5)) - 1

for i in range(len(tracks)):
    track_info = tracks[i]
    track_valid = (track_info[:, 4] >= 0.0).nonzero()[0]
    not_set_index = (ball_pos[track_valid, 4] == -1).nonzero()[0]
    ball_pos[track_valid[not_set_index], :] = track_info[track_valid[not_set_index], :]


valid = (ball_pos[:, 2] >= 0).nonzero()[0]


# Interpolate for the non valid

def interpolate_trajectory(ball_pos):
    ball_pos2 = ball_pos.copy()

    anchors = []
    start = - 1
    end = -1

    for i in range(db.n_frames):
        if ball_pos2[i, -1] == -1 and start == -1:
            start = i-1
        if ball_pos2[i, -1] != -1 and start != -1:
            end = i
            anchors.append([start, end])
            start = - 1
            end = -1

    for i in range(len(anchors)):
        start, end = anchors[i]
        xx1 = np.linspace(ball_pos2[start, 0], ball_pos2[end, 0], end - start)
        yy1 = np.linspace(ball_pos2[start, 1], ball_pos2[end, 1], end - start)
        xx2 = np.linspace(ball_pos2[start, 2], ball_pos2[end, 2], end - start)
        yy2 = np.linspace(ball_pos2[start, 3], ball_pos2[end, 3], end - start)
        ss = np.linspace(ball_pos2[start, 4], ball_pos2[end, 4], end - start)

        ball_pos2[start+1:end, :] = np.vstack((xx1[:-1], yy1[:-1], xx2[:-1], yy2[:-1], ss[:-1])).T

    return ball_pos2


ball_pos2 = interpolate_trajectory(ball_pos)

# Check if there is frames without points
rm = (ball_pos2[:, 4] == -1).nonzero()[0]
if len(rm) > 0:
    ball_pos2[rm, :] = ball_pos2[rm[0]-1, :]

plt.imshow(img)
plt.plot(ball_pos[:, 0], ball_pos[:, 1], 'o')
plt.plot(ball_pos2[:, 0], ball_pos2[:, 1], '.-')
plt.show()

for i, k in enumerate(db.frame_basenames):
    db.ball[k] = np.array([ball_pos2[i, :]])

path_to_save = db.path_to_dataset + '/metadata/ball.npy'
# np.save(path_to_save, db.ball)

# db.dump_video('detections')
