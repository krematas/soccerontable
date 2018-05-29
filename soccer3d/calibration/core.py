import numpy as np
import utils.io as io
import utils.camera as cam_utils
import utils.draw as draw_utils
import utils.image as image_utils
import utils.transform as transf_utils
from scipy.optimize import minimize
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def _fun_distance_transform(params_, dist_map_, points3d):
    theta_x_, theta_y_, theta_z_, fx_, tx_, ty_, tz_ = params_
    h_, w_ = dist_map_.shape[0:2]
    n_ = points3d.shape[0]

    cx_, cy_ = float(dist_map_.shape[1])/2.0, float(dist_map_.shape[0])/2.0

    R_ = transf_utils.Rz(theta_z_).dot(transf_utils.Ry(theta_y_)).dot(transf_utils.Rx(theta_x_))
    A_ = np.eye(3, 3)
    A_[0, 0], A_[1, 1], A_[0, 2], A_[1, 2] = fx_, fx_, cx_, cy_

    T_ = np.array([[tx_], [ty_], [tz_]])

    p2_ = A_.dot(R_.dot(points3d.T) + np.tile(T_, (1, n_)))
    p2_ /= p2_[2, :]
    p2_ = p2_.T[:, 0:2]
    p2_ = np.round(p2_).astype(int)
    _, valid_id_ = cam_utils.inside_frame(p2_, h_, w_)

    residual = np.zeros((n_,)) + 0.0
    residual[valid_id_] = dist_map_[p2_[valid_id_, 1], p2_[valid_id_, 0]]
    return np.sum(residual)


def _calibrate_camera_dist_transf(A, R, T, dist_transf, points3d):

    theta_x, theta_y, theta_z = transf_utils.get_angle_from_rotation(R)
    fx, fy, cx, cy = A[0, 0], A[1, 1], A[0, 2], A[1, 2]

    params = np.hstack((theta_x, theta_y, theta_z, fx, T[0, 0], T[1, 0], T[2, 0]))

    res_ = minimize(_fun_distance_transform, params, args=(dist_transf, points3d),
                    method='Powell', options={'disp': False, 'maxiter': 5000})
    result = res_.x

    theta_x_, theta_y_, theta_z_, fx_, tx_, ty_, tz_ = result

    cx_, cy_ = float(dist_transf.shape[1]) / 2.0, float(dist_transf.shape[0]) / 2.0

    R__ = transf_utils.Rz(theta_z_).dot(transf_utils.Ry(theta_y_)).dot(transf_utils.Rx(theta_x_))
    T__ = np.array([[tx_], [ty_], [tz_]])
    A__ = np.eye(3, 3)
    A__[0, 0], A__[1, 1], A__[0, 2], A__[1, 2] = fx_, fx_, cx_, cy_

    return A__, R__, T__


def _set_correspondences(img, field_img_path='./demo/data/field.png'):

    field_img = io.imread(field_img_path)

    h2, w2 = field_img.shape[0:2]
    W, H = 104.73, 67.74

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(field_img)

    ax[0].axis('off')
    ax[1].axis('off')

    points2d = []
    points3d = []

    def onclick(event):
        # print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #       (event.button, event.x, event.y, event.xdata, event.ydata))
        x, y = event.xdata, event.ydata
        if event.inaxes.axes.get_position().x0 < 0.5:
            ax[0].plot(x, y, 'r.', ms=10)
            points2d.append([x, y])
        else:
            ax[1].plot(x, y, 'b+', ms=10)
            points3d.append([x, 0, y])
        plt.show()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    points2d = np.array(points2d)
    points3d = np.array(points3d)

    points3d[:, 0] = ((points3d[:, 0] - w2 / 2.) / w2) * W
    points3d[:, 2] = ((points3d[:, 2] - h2 / 2.) / h2) * H

    return points2d, points3d


def calibrate_by_click(img, mask, edge_sfactor=0.5, field_img_path='./demo/data/field.png'):

    h, w = img.shape[0:2]

    points2d, points3d = _set_correspondences(img, field_img_path=field_img_path)

    # ------------------------------------------------------------------------------------------------------------------
    # OpenCV initial calibration
    fx, fy = cam_utils.grid_search_focal_length(points3d, points2d, h, w, same_f=True)
    A = cam_utils.intrinsic_matrix_from_focal_length(fx, fy, h, w)

    points_3d_cv = points3d[:, np.newaxis, :].astype(np.float32)
    points_2d_cv = points2d[:, np.newaxis, :].astype(np.float32)

    _, rvec, tvec, _ = cv2.solvePnPRansac(points_3d_cv, points_2d_cv, A, None)
    rvec, tvec = np.squeeze(rvec), np.squeeze(tvec)
    R, _ = cv2.Rodrigues(rvec)
    T = np.array([tvec]).T
    cam_cv = cam_utils.Camera('tmp', A, R, T, h, w)

    # ------------------------------------------------------------------------------------------------------------------
    # Photometric refinement
    A__, R__, T__, field3d = calibrate_from_initialization(img, mask, A, R, T, edge_sfactor)
    cam = cam_utils.Camera('tmp', A__, R__, T__, h, w)

    # Sanity check, project the soccer field points
    p2_cv, _ = cam.project(field3d)
    p2_cv, _ = cam_utils.inside_frame(p2_cv, cam.height, cam.width)

    p2_opt, _ = cam_cv.project(field3d)
    p2_opt, valid_id = cam_utils.inside_frame(p2_opt, cam_cv.height, cam_cv.width)

    A_out, R_out, T_out = [None], [None], [None]

    class Index(object):

        def save_opt(self, event):
            A_out.append(cam.A)
            R_out.append(cam.R)
            T_out.append(cam.T)
            plt.close()

        def save_pnp(self, event):
            A_out.append(cam_cv.A)
            R_out.append(cam_cv.R)
            T_out.append(cam_cv.T)
            plt.close()

        def discard(self, event):
            plt.close()

    fig, ax = plt.subplots(1, 2)
    io.imshow(img, ax=ax[0], points=p2_opt)
    io.imshow(img, ax=ax[1], points=p2_cv)
    callback = Index()
    axdisc = plt.axes([0.6, 0.05, 0.1, 0.075])
    axcv = plt.axes([0.7, 0.05, 0.1, 0.075])
    axopt = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axopt, 'Save opt')
    bnext.on_clicked(callback.save_opt)
    bprev = Button(axdisc, 'Discard')
    bprev.on_clicked(callback.discard)
    bpcv = Button(axcv, 'Save cv')
    bpcv.on_clicked(callback.save_pnp)
    plt.show()

    return A_out[-1], R_out[-1], T_out[-1]


def calibrate_from_initialization(img, mask, A_init, R_init, T_init, edge_sfactor=0.5, visualize=False):

    h, w = img.shape[:2]

    edges = image_utils.robust_edge_detection(cv2.resize(img, None, fx=edge_sfactor, fy=edge_sfactor))

    edges = cv2.resize(edges, None, fx=1. / edge_sfactor, fy=1. / edge_sfactor)
    edges = cv2.Canny(edges.astype(np.uint8) * 255, 100, 200) / 255.0

    mask = cv2.dilate(mask, np.ones((25, 25), dtype=np.uint8))

    edges = edges * (1 - mask)
    dist_transf = cv2.distanceTransform((1 - edges).astype(np.uint8), cv2.DIST_L2, 0)

    cam_init = cam_utils.Camera('tmp', A_init, R_init, T_init, h, w)
    template, field_mask = draw_utils.draw_field(cam_init)

    II, JJ = (template > 0).nonzero()
    synth_field2d = np.array([[JJ, II]]).T[:, :, 0]

    field3d = cam_utils.plane_points_to_3d(synth_field2d, cam_init)

    A, R, T = _calibrate_camera_dist_transf(A_init, R_init, T_init, dist_transf, field3d)

    if visualize:
        cam_res = cam_utils.Camera('tmp', A, R, T, h, w)
        field2d, __ = cam_res.project(field3d)
        io.imshow(img, points=field2d)

    return A, R, T, field3d


