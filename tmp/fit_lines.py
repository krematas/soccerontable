import argparse
import soccer3d
import numpy as np
import cv2
from matplotlib import pyplot as plt
import utils.io as io
from soccer3d.tracking import Detection, find_tracks_simple
from scipy.optimize import minimize


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Japan-Something-1', help='path')
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


dist_thresh = 150
time_thresh = 30
new_tracklets = find_tracks_simple(dets_per_frame, db.frame_basenames, dist_thresh=dist_thresh, time_thresh=time_thresh,
                            len_thresh=1)

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

plt.imshow(img)
for i in range(len(tracks)):
    plt.plot(tracks[i][:, 0], tracks[i][:, 1], '.')
plt.show()

tracks2 = []
# Check if track is not moving
for i in range(len(tracks)):
    track_info = tracks[i]
    valid = (track_info[:, 4] > 0).nonzero()[0]
    if np.std(track_info[valid, 0]) > 5:
        tracks2.append(track_info)
tracks = tracks2[:]

# Accumulate tracks
ball_pos = np.zeros((db.n_frames, 5)) - 1

for i in range(len(tracks)):
    track_info = tracks[i]
    track_valid = (track_info[:, 4] >= 0.0).nonzero()[0]
    ball_pos[track_valid, :] = track_info[track_valid, :]


valid = (ball_pos[:, 2] >= 0).nonzero()[0]

plt.imshow(img)
for i in range(len(tracks)):
    plt.plot(ball_pos[:, 0], ball_pos[:, 1], 'o')

# Interpolate for the non valid
anchors = []
start = - 1
end = -1

for i in range(db.n_frames):
    if ball_pos[i, -1] == -1 and start == -1:
        start = i-1
    if ball_pos[i, -1] != -1 and start != -1:
        end = i
        anchors.append([start, end])
        start = - 1
        end = -1

for i in range(len(anchors)):
    start, end = anchors[i]
    print(ball_pos[start, 0], ball_pos[end, 0])
    xx1 = np.linspace(ball_pos[start, 0], ball_pos[end, 0], end - start)
    yy1 = np.linspace(ball_pos[start, 1], ball_pos[end, 1], end - start)
    xx2 = np.linspace(ball_pos[start, 2], ball_pos[end, 2], end - start)
    yy2 = np.linspace(ball_pos[start, 3], ball_pos[end, 3], end - start)
    ss = np.linspace(ball_pos[start, 4], ball_pos[end, 4], end - start)

    ball_pos[start+1:end, :] = np.vstack((xx1[:-1], yy1[:-1], xx2[:-1], yy2[:-1], ss[:-1])).T


for i in range(len(tracks)):
    plt.plot(ball_pos[:, 0], ball_pos[:, 1], '.')
plt.show()



# # Perform template matching
# io.show_box(img, ball_pos[0, :])
#
# ball = ball_pos[0, :]
# x1, y1, x2, y2 = ball[:4].astype(int)
# ball_img = img[y1:y2, x1:x2, :]
#
# ball_pos2 = np.zeros((db.n_frames, 5))
# ball_pos2[0, :] = ball
#
# counter = 1
# # Start tracking the ball
# for fname in db.frame_basenames[1:]:
#     if len(db.ball[fname]) > 0:
#         # Find the closest ball based on distance
#         candit_balls = db.ball[fname]
#         dist = np.linalg.norm(candit_balls - ball, axis=1)
#         closest = np.argmin(dist)
#         sel_ball = candit_balls[closest, :]
#         sel_ball[4] = counter
#         ball_pos2[counter, :] = sel_ball
#
#     counter += 1
#
#
#
#
#
#
# def _fun(params, detections, valid_index, time_index, traj_length, a=0.01, b=0.1, c=0.1):
#     X_ = np.reshape(params, (2, traj_length))
#     E_det = np.sum(np.linalg.norm(X_[:, valid_index] - detections[:, valid_index], axis=0))
#     temporal_term = X_[:, time_index[0, :]] - 2 * X_[:, time_index[1, :]] + X_[:, time_index[2, :]]
#     E_dyn = np.sum(np.linalg.norm(temporal_term, axis=0))
#     E_vel = np.sum(np.linalg.norm(X_[:, time_index[0, :]] - X_[:, time_index[1, :]], axis=0))
#     return a * E_det + b*E_dyn + c*E_vel
#
#
# time_index = np.array([range(0, db.n_frames - 2), range(0 + 1, db.n_frames - 1), range(0 + 2, db.n_frames)])
#
# detection_points = ball_pos[:, :2].T
# params = detection_points.copy()
#
# res = minimize(_fun, params, args=(detection_points, valid, time_index, db.n_frames), method='L-BFGS-B',
#                options={'gtol': 1e-6, 'disp': False, 'maxiter': 500})
# smoothed_positions = np.reshape(res.x, (2, db.n_frames)).T
#
#
# plt.imshow(img)
# for i in range(len(tracks)):
#     plt.plot(smoothed_positions[:, 0], smoothed_positions[:, 1], '.')
# plt.show()
#








# io.show_box(img, db.ball[db.frame_basenames[0]])
#
# # Select the ball with the highest score in the first frame
# ball = db.ball[db.frame_basenames[0]]
# scores = ball[:, 4]
# ball = ball[np.argmax(scores), :]
# x1, y1, x2, y2 = ball[:4].astype(int)
# ball_img = img[y1:y2, x1:x2, :]
#
# ball_pos = np.zeros((db.n_frames, 5))
# ball_pos[0, :] = ball
#
# counter = 1
# # Start tracking the ball
# for fname in db.frame_basenames[1:]:
#     if len(db.ball[fname]) > 0:
#         # Find the closest ball based on distance
#         candit_balls = db.ball[fname]
#         dist = np.linalg.norm(candit_balls - ball, axis=1)
#         closest = np.argmin(dist)
#         sel_ball = candit_balls[closest, :]
#         sel_ball[4] = counter
#         ball_pos[counter, :] = sel_ball
#
#     counter += 1
#
# io.show_box(img, ball_pos)
# # Get all points
# pos2d = np.zeros((db.n_frames, 2))
# valid = []
#
# for fname in db.frame_basenames:
#     if len(db.ball[fname]) > 0:
#         db.ball[fname] = np.array([db.ball[fname]])
#         bbox = db.ball[fname]
#         x1, y1, x2, y2 = bbox[0, :4].astype(int)


