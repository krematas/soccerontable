import numpy as np
from scipy.optimize import minimize


class Detection:
    def __init__(self, keypoints, frame, frame_index):
        self.keypoints = keypoints
        self.keypoints3d = None
        valid = keypoints[:, 2] > 0
        x1, y1 = np.min(keypoints[valid, 0]), np.min(keypoints[valid, 1])
        x2, y2 = np.max(keypoints[valid, 0]), np.max(keypoints[valid, 1])
        self.bbox_points = np.array([[x1, y1], [x2, y2]])
        self.root = keypoints[1, :]
        self.height = keypoints[1, 1] - keypoints[0, 1]

        self.mesh_name = None

        self.frame = frame
        self.frame_index = frame_index
        self.track_id = -1
        self.visited = False


def find_tracks(dets_per_frame, frame_basenames, dist_thresh=25, time_thresh=20, len_thresh=20):

    n_frames = len(frame_basenames)
    # ----------------------------------------------------------------------------------------------------------------------
    # Start connecting sources that are visible in the first frame
    tracklets = []

    track_id = 0
    for f in range(n_frames):

        for i in range(len(dets_per_frame[f])):
            s = dets_per_frame[f][i]
            if s.visited:
                continue

            dets_per_frame[f][i].visited = True
            cur_track = [dets_per_frame[f][i]]

            # Get detections from next frames
            j = f + 1
            q = 0
            while j < n_frames:
                potential_detections = [k for k in range(len(dets_per_frame[j])) if not dets_per_frame[j][k].visited]
                potential_matches = np.array([det.root for det in dets_per_frame[j] if not det.visited])
                if len(potential_detections) == 0:
                    j += 1
                    q += 1
                    continue

                dists = np.linalg.norm(s.root - potential_matches, axis=1)

                dists_sorted = sorted(dists)
                min_dist, min_id_ = np.min(dists), np.argmin(dists)

                if (min_dist < dist_thresh) and q < time_thresh:
                    min_id = potential_detections[min_id_]

                    dets_per_frame[j][min_id].visited = True
                    cur_track.append(dets_per_frame[j][min_id])
                    s = dets_per_frame[j][min_id]
                    q = 0
                else:
                    q += 1

                j += 1
            for j in range(len(cur_track)):
                cur_track[j].track_id = track_id
            track_id += 1

            tracklets.append(cur_track)

    n_tracks = len(tracklets)

    # Fill missing frames from trackes
    new_tracklets = []

    for i in range(n_tracks):
        _cur_tracklet = [tracklets[i][0]]
        for j in range(1, len(tracklets[i])):
            if tracklets[i][j].frame_index != tracklets[i][j - 1].frame_index + 1:
                kps_start = tracklets[i][j - 1].keypoints
                kps_end = tracklets[i][j].keypoints
                ps_start = tracklets[i][j - 1].bbox_points
                ps_end = tracklets[i][j].bbox_points
                f_start = tracklets[i][j - 1].frame_index
                f_end = tracklets[i][j].frame_index

                for q in range(tracklets[i][j - 1].frame_index + 1, tracklets[i][j].frame_index + 1):

                    new_points = np.zeros((2, 2))
                    for k1 in range(2):
                        for k2 in range(2):
                            new_points[k1, k2] = np.interp(q, [f_start, f_end], [ps_start[k1, k2], ps_end[k1, k2]])

                    _det = Detection(kps_start, frame_basenames[q], q)
                    _det.track_id = tracklets[i][j].track_id
                    _det.mesh_name = tracklets[i][j].mesh_name
                    _det.bbox_points = new_points
                    _cur_tracklet.append(_det)

            else:
                _cur_tracklet.append(tracklets[i][j])
        if len(_cur_tracklet) >= len_thresh:
            new_tracklets.append(_cur_tracklet)

    return new_tracklets


def _fun(params, detections, time_index, traj_length, a=0.01, b=0.1, c=0.1):
    X_ = np.reshape(params, (2, traj_length))
    E_det = np.sum(np.linalg.norm(X_ - detections, axis=0))
    temporal_term = X_[:, time_index[0, :]] - 2 * X_[:, time_index[1, :]] + X_[:, time_index[2, :]]
    E_dyn = np.sum(np.linalg.norm(temporal_term, axis=0))
    E_vel = np.sum(np.linalg.norm(X_[:, time_index[0, :]] - X_[:, time_index[1, :]], axis=0))
    return a * E_det + b*E_dyn + c*E_vel


def smooth_trajectory(tracklet, neck_pos):

    traj_length = len(tracklet)
    time_index = np.array([range(0, traj_length - 2), range(0 + 1, traj_length - 1), range(0 + 2, traj_length)])

    detection_points = neck_pos.T[(0, 2), :]
    params = detection_points.copy()

    res = minimize(_fun, params, args=(detection_points, time_index, traj_length), method='L-BFGS-B',
                   options={'gtol': 1e-6, 'disp': False, 'maxiter': 500})
    _smoothed_positions = np.reshape(res.x, (2, traj_length))
    smoothed_positions = neck_pos.T.copy()
    smoothed_positions[(0, 2), :] = _smoothed_positions
    return smoothed_positions


def convert_to_MOT(new_tracklets, n_frames):
    tracked_data = {i: [] for i in range(n_frames)}
    for i in range(len(new_tracklets)):
        cur_track = new_tracklets[i]
        if len(cur_track) < 10:
            continue
        for j in range(len(cur_track)):
            cur_det = cur_track[j]
            tracked_data[cur_det.frame_index].append(cur_det)

    mot_matrix = []
    for i in range(n_frames):
        for j in range(len(tracked_data[i])):
            x1, y1 = tracked_data[i][j].bbox_points[0, :]
            x2, y2 = tracked_data[i][j].bbox_points[1, :]
            track_id = tracked_data[i][j].track_id
            w, h = x2 - x1, y2 - y1
            mot_matrix.append([i + 1, track_id, x1, y1, w, h, 1.0, -1, -1, -1])

    mot_matrix = np.array(mot_matrix)
    return mot_matrix