import numpy as np
import scipy
import cv2


def get_pixel_neighbors(height, width):
    """
    Estimate the 4 neighbors of every pixel in an image
    :param height: image height
    :param width: image width
    :return: pixel index - neighbor index lists
    """

    pix_id = []
    neighbor_id = []
    for i in range(height):
        for j in range(width):

            n = []
            if i == 0:
                n = n + [(i + 1) * width + j]
            elif i == height - 1:
                n = n + [(i - 1) * width + j]
            else:
                n = n + [(i + 1) * width + j, (i - 1) * width + j]

            if j == 0:
                n = n + [i * width + j + 1]
            elif j == width - 1:
                n = n + [i * width + j - 1]
            else:
                n = n + [i * width + j + 1, i * width + j - 1]

            for k in n:
                pix_id.append(i*width+j)
                neighbor_id.append(k)

    return pix_id, neighbor_id


limps = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 11], [11, 12], [12, 13], [1, 8],
         [8, 9], [9, 10], [14, 15], [16, 17], [0, 14], [0, 15], [14, 16], [15, 17]])


def get_instance_skeleton_buffer(h, w, poses):
    output = np.zeros((h, w, 3), dtype=np.float32) - 1
    for i in range(len(poses)):
        keypoints = poses[i]

        lbl = i
        for k in range(limps.shape[0]):
            kp1, kp2 = limps[k, :].astype(int)
            bone_start = keypoints[kp1, :]
            bone_end = keypoints[kp2, :]
            bone_start[0] = np.maximum(np.minimum(bone_start[0], w - 1), 0.)
            bone_start[1] = np.maximum(np.minimum(bone_start[1], h - 1), 0.)

            bone_end[0] = np.maximum(np.minimum(bone_end[0], w - 1), 0.)
            bone_end[1] = np.maximum(np.minimum(bone_end[1], h - 1), 0.)

            if bone_start[2] > 0.0:
                output[int(bone_start[1]), int(bone_start[0])] = 1
                cv2.circle(output, (int(bone_start[0]), int(bone_start[1])), 2, (lbl, 0, 0), -1)

            if bone_end[2] > 0.0:
                output[int(bone_end[1]), int(bone_end[0])] = 1
                cv2.circle(output, (int(bone_end[0]), int(bone_end[1])), 2, (lbl, 0, 0), -1)

            if bone_start[2] > 0.0 and bone_end[2] > 0.0:
                cv2.line(output, (int(bone_start[0]), int(bone_start[1])), (int(bone_end[0]), int(bone_end[1])), (lbl, 0, 0), 1)

    return output[:, :, 0]


def get_poseimg_for_opt(sel_pose, poseimg, init_mask, n_bg=50):

    h, w = init_mask.shape[:2]
    bg_label = 1
    output = np.zeros((h, w, 3), dtype=np.float32) - 1
    II, JJ = (poseimg > 0).nonzero()
    Isel, J_sel = (poseimg == sel_pose).nonzero()

    output[II, JJ] = 0
    output[Isel, J_sel] = 2

    init_mask[Isel, J_sel] = 1
    # Sample also from points in the field
    init_mask = cv2.dilate(init_mask, np.ones((25, 25), np.uint8), iterations=1)

    I_bg, J_bg = (init_mask == 0).nonzero()
    rand_index = np.random.permutation(len(I_bg))[:n_bg]
    bg_points = np.array([J_bg[rand_index], I_bg[rand_index]]).T

    for k in range(bg_points.shape[0]):
        cv2.circle(output, (int(bg_points[k, 0]), int(bg_points[k, 1])), 2, (bg_label, 0, 0), -1)

    return output[:, :, 0]


def draw_poses_for_optimization(sel_pose, keypoints_list, init_mask, n_bg=50):

    h, w = init_mask.shape[:2]
    bg_label = 0
    output = np.zeros((h, w, 3), dtype=np.float32)-1

    for i in range(len(keypoints_list)):
        keypoints = keypoints_list[i]
        if i == sel_pose:
            lbl = 2
        else:
            lbl = 1
        for k in range(limps.shape[0]):
            kp1, kp2 = limps[k, :].astype(int)
            bone_start = keypoints[kp1, :]
            bone_end = keypoints[kp2, :]
            bone_start[0] = np.maximum(np.minimum(bone_start[0], w - 1), 0.)
            bone_start[1] = np.maximum(np.minimum(bone_start[1], h - 1), 0.)

            bone_end[0] = np.maximum(np.minimum(bone_end[0], w - 1), 0.)
            bone_end[1] = np.maximum(np.minimum(bone_end[1], h - 1), 0.)

            if bone_start[2] > 0.0:
                output[int(bone_start[1]), int(bone_start[0])] = 1
                cv2.circle(output, (int(bone_start[0]), int(bone_start[1])), 2, (lbl, 0, 0), -1)

            if bone_end[2] > 0.0:
                output[int(bone_end[1]), int(bone_end[0])] = 1
                cv2.circle(output, (int(bone_end[0]), int(bone_end[1])), 2, (lbl, 0, 0), -1)

            if bone_start[2] > 0.0 and bone_end[2] > 0.0:
                cv2.line(output, (int(bone_start[0]), int(bone_start[1])), (int(bone_end[0]), int(bone_end[1])), (lbl, 0, 0), 1)

    # Draw circles for the bg players keypoints
    # for k in range(bg_keypoints.shape[0]):
    #     cv2.circle(output, (int(bg_keypoints[k, 0]), int(bg_keypoints[k, 1])), 2, (bg_keypoint_lable, 0, 0), -1)

    # Sample also from points in the field
    init_mask = cv2.dilate(init_mask, np.ones((5, 5), np.uint8), iterations=1)

    I_bg, J_bg = (init_mask == 0).nonzero()
    rand_index = np.random.permutation(len(I_bg))[:n_bg]
    bg_points = np.array([J_bg[rand_index], I_bg[rand_index]]).T

    for k in range(bg_points.shape[0]):
        cv2.circle(output, (int(bg_points[k, 0]), int(bg_points[k, 1])), 2, (bg_label, 0, 0), -1)

    return output[:, :, 0]


def set_U(strokes, h, w, dim):
    N = h*w
    y = np.zeros((N, dim))

    U = scipy.sparse.lil_matrix((N, N))
    for p in range(strokes.shape[0]):
        i = strokes[p, 1]
        j = strokes[p, 0]
        index = int(i * w + j)
        for ii in range(dim):
            y[index, ii] = strokes[p, ii+2]
        U[index, index] = 1

    return U, y


def set_DW(image, edges=None, sigma1=1000., sigma2=0.01):
    image = image.astype(float)
    h, w = image.shape[0:2]

    N = h * w

    pixd, neighborid = get_pixel_neighbors(h, w)
    i, j = np.unravel_index(pixd, (h, w))
    ii, jj = np.unravel_index(neighborid, (h, w))

    pix_diff = np.squeeze((image[i, j, :] - image[ii, jj, :]) ** 2)
    if len(pix_diff.shape) == 1:
        pix_diff = pix_diff[:, np.newaxis]
    weight0 = np.exp(-(np.sum(pix_diff, axis=1)) / sigma1)
    weight1 = np.exp(-((edges[i, j]) ** 2) / sigma2)

    # neighbor_info = np.vstack((pixd, neighborid, weight0)).T

    M = len(pixd)

    D = scipy.sparse.lil_matrix((M, N))
    W = scipy.sparse.lil_matrix((M, M))

    p = np.arange(0, M, 1)
    D[p, pixd] = 1
    D[p, neighborid] = -1
    W[p, p] = weight1

    return D, W
