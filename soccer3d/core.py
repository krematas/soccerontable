import numpy as np
import os
from os import listdir
from os.path import isfile, join, exists
import utils.io as io
from soccer3d import calibration
# from soccer3d import segmentation
# from soccer3d import tracking
import utils.misc as misc_utils
import utils.camera as cam_utils
import utils.draw as draw_utils
import utils.files as file_utils
import pickle
import glog
import cv2
from tqdm import tqdm
import yaml
import matplotlib
import pycocotools.mask as mask_util
from utils.nms.nms_wrapper import nms


class Player:

    def __init__(self):
        self.id = -1
        self.bbox = None
        self.frame = None


class YoutubeVideo:

    def __init__(self, path_to_dataset):
        image_extensions = ['jpg', 'png']

        # Load images
        self.path_to_dataset = path_to_dataset
        self.frame_basenames = [f for f in listdir(join(path_to_dataset, 'images'))
                                if isfile(join(path_to_dataset, 'images', f)) and any(i in f for i in image_extensions)]
        self.frame_fullnames = [join(path_to_dataset, 'images', f) for f in self.frame_basenames]
        self.frame_basenames = [f[:-4] for f in self.frame_basenames]

        self.frame_basenames.sort()
        self.frame_fullnames.sort()

        self.n_frames = len(self.frame_basenames)
        self.ext = self.frame_fullnames[0][-3:]

        self.bbox = {f: None for f in self.frame_basenames}
        self.mask_coarse = {f: None for f in self.frame_basenames}
        self.calib = {f: None for f in self.frame_basenames}
        self.poses = {f: None for f in self.frame_basenames}
        self.detectron = {f: None for f in self.frame_basenames}
        self.ball = {f: None for f in self.frame_basenames}
        self.tracks = None

        # Make the txt file
        txt_file = join(path_to_dataset, 'youtube.txt')
        if not exists(txt_file):
            np.savetxt(txt_file, self.frame_fullnames, fmt='%s')

        if not exists(join(path_to_dataset, 'metadata')):
            os.mkdir(join(path_to_dataset, 'metadata'))

        img_ = self.get_frame(0)
        self.shape = img_.shape

    # ------------------------------------------------------------------------------------------------------------------
    def _load_metadata(self, filename, attr):
        if exists(filename):
            with open(filename, 'rb') as f:
                setattr(self, attr, pickle.load(f))
        glog.info('{0}: {1}\tfrom {2}'.format(attr, exists(filename), file_utils.extract_basename(filename)[0]))

    def digest_metadata(self):

        calib_file = join(self.path_to_dataset, 'metadata', 'calib.p')
        self._load_metadata(calib_file, 'calib')

        bbox_coarse_file = join(self.path_to_dataset, 'metadata', 'bbox_coarse.p')
        self._load_metadata(bbox_coarse_file, 'bbox')

        seg_coarse_file = join(self.path_to_dataset, 'metadata', 'seg_coarse.p')
        self._load_metadata(seg_coarse_file, 'mask_coarse')

        bbox_fine_file = join(self.path_to_dataset, 'metadata', 'bbox_fine.p')
        self._load_metadata(bbox_fine_file, 'bbox')

        pose_coarse_file = join(self.path_to_dataset, 'metadata', 'poses_coarse.p')
        self._load_metadata(pose_coarse_file, 'poses')

        # pose_fine_file = join(self.path_to_dataset, 'metadata', 'poses_fine.p')
        # self._load_metadata(pose_fine_file, 'poses')

        detectron_file = join(self.path_to_dataset, 'metadata', 'detectron.p')
        self._load_metadata(detectron_file, 'detectron')

    # ------------------------------------------------------------------------------------------------------------------

    def get_frame(self, frame_number, dtype=np.float32, sfactor=1.0, image_type='rgb'):
        return io.imread(self.frame_fullnames[frame_number], dtype=dtype, sfactor=sfactor, image_type=image_type)

    # ------------------------------------------------------------------------------------------------------------------

    def get_coarse_mask(self, frame_number):
        mask = self.mask_coarse[self.frame_basenames[frame_number]]
        if mask.shape[0] != self.shape[0] or mask.shape[1] != self.shape[1]:
            mask = cv2.resize(mask, (self.shape[1], self.shape[0]))
        return mask

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def extract_edges(self, scale=1.0):

        if not exists(join(self.path_to_dataset, 'edges')):
            os.mkdir(join(self.path_to_dataset, 'edges'))

        # Check if there are edges
        existing_edges = [f for f in listdir(join(self.path_to_dataset, 'edges'))]
        if len(existing_edges) != self.n_frames:
            glog.info('Estimating edges')
            edge_bin = '/home/krematas/CLionProjects/EdgeDetection/build/EdgeDetection'
            for basename in self.frame_basenames:
                input_ = join(self.path_to_dataset, 'images', '{0}.{1}'.format(basename, self.ext))
                output_ = join(self.path_to_dataset, 'edges', '{0}.jpg'.format(basename))
                model_ = '/home/krematas/Downloads/model.yml.gz'
                os.system('{0} -i={1} -m={2} -o={3} -s={4}'.format(edge_bin, input_, model_, output_, scale))
                break

        return 0

    # ------------------------------------------------------------------------------------------------------------------

    def calibrate_camera(self, vis_every=-1):
        if not exists(join(self.path_to_dataset, 'calib')):
            os.mkdir(join(self.path_to_dataset, 'calib'))

        calib_file = join(self.path_to_dataset, 'metadata', 'calib.p')
        if exists(calib_file):
            glog.info('Loading coarse detections from: {0}'.format(calib_file))
            with open(calib_file, 'rb') as f:
                self.calib = pickle.load(f)

        else:

            if not self.file_lists_match(listdir(join(self.path_to_dataset, 'calib'))):

                # The first frame is estimated by manual clicking
                manual_calib = join(self.path_to_dataset, 'calib', '{0}.npy'.format(self.frame_basenames[0]))
                if exists(manual_calib):
                    calib_npy = np.load(manual_calib).item()
                    A, R, T = calib_npy['A'], calib_npy['R'], calib_npy['T']
                else:
                    img = self.get_frame(0)
                    coarse_mask = self.get_mask_from_detectron(0)
                    A, R, T = calibration.calibrate_by_click(img, coarse_mask)

                if A is None:
                    glog.error('Manual calibration failed!')
                else:
                    np.save(join(self.path_to_dataset, 'calib', '{0}'.format(self.frame_basenames[0])),
                            {'A': A, 'R': R, 'T': T})
                    for i in tqdm(range(1, self.n_frames)):
                        glog.info('Calibrating frame {0} ({1}/{2})'.format(self.frame_basenames[i], i, self.n_frames))
                        img = self.get_frame(i)
                        coarse_mask = self.get_mask_from_detectron(i)

                        if i % vis_every == 0:
                            vis = True
                        else:
                            vis = False
                        A, R, T, __ = calibration.calibrate_from_initialization(img, coarse_mask, A, R, T, vis)

                        np.save(join(self.path_to_dataset, 'calib', '{0}'.format(self.frame_basenames[i])),
                                {'A': A, 'R': R, 'T': T})

            for i, basename in tqdm(enumerate(self.frame_basenames)):
                calib_npy = np.load(join(self.path_to_dataset, 'calib', '{0}.npy'.format(basename))).item()
                A, R, T = calib_npy['A'], calib_npy['R'], calib_npy['T']
                self.calib[basename] = {'A': A, 'R': R, 'T': T}

            with open(calib_file, 'wb') as f:
                pickle.dump(self.calib, f)

    # ------------------------------------------------------------------------------------------------------------------

    def dump_video(self, vidtype, scale=4, mot_tracks=None, one_color=True):
        if vidtype not in ['calib', 'poses', 'detections', 'tracks']:
            raise Exception('Uknown video format')

        if vidtype == 'tracks' and mot_tracks is None:
            raise Exception('No MOT tracks provided')

        glog.info('Dumping {0} video'.format(vidtype))

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out_file = join(self.path_to_dataset, '{0}.mp4'.format(vidtype))
        out = cv2.VideoWriter(out_file, fourcc, 20.0, (self.shape[1] // scale, self.shape[0] // scale))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cmap = matplotlib.cm.get_cmap('hsv')
        if mot_tracks is not None:
            n_tracks = max(np.unique(mot_tracks[:, 1]))

        for i, basename in enumerate(tqdm(self.frame_basenames)):
            img = self.get_frame(i, dtype=np.uint8)

            if vidtype == 'poses':
                # Pose
                poses = self.poses[basename]
                draw_utils.draw_skeleton_on_image(img, poses, cmap, one_color=one_color)

            if vidtype == 'calib':
                # Calib
                cam = cam_utils.Camera('tmp', self.calib[basename]['A'], self.calib[basename]['R'], self.calib[basename]['T'], self.shape[0], self.shape[1])
                canvas, mask = draw_utils.draw_field(cam)
                canvas = cv2.dilate(canvas.astype(np.uint8), np.ones((15, 15), dtype=np.uint8)).astype(float)
                img = img * (1 - canvas)[:, :, None] + np.dstack((canvas*255, np.zeros_like(canvas), np.zeros_like(canvas)))

            elif vidtype == 'detections':
                # Detection
                bbox = self.bbox[basename].astype(np.int32)
                if self.ball[basename] is not None:
                    ball = self.ball[basename].astype(np.int32)
                else:
                    ball = np.zeros((0, 4), dtype=np.int32)

                for j in range(bbox.shape[0]):
                    cv2.rectangle(img, (bbox[j, 0], bbox[j, 1]), (bbox[j, 2], bbox[j, 3]), (255, 0, 0), 10)
                for j in range(ball.shape[0]):
                    cv2.rectangle(img, (ball[j, 0], ball[j, 1]), (ball[j, 2], ball[j, 3]), (0, 255, 0), 10)

            elif vidtype == 'tracks':
                # Tracks
                cur_id = mot_tracks[:, 0] - 1 == i
                current_boxes = mot_tracks[cur_id, :]

                for j in range(current_boxes.shape[0]):
                    track_id, x, y, w, h = current_boxes[j, 1:6]
                    clr = cmap(track_id / float(n_tracks))
                    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                                  (clr[0] * 255, clr[1] * 255, clr[2] * 255), 10)
                    cv2.putText(img, str(int(track_id)), (int(x), int(y)), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

            img = cv2.resize(img, (self.shape[1] // scale, self.shape[0] // scale))
            out.write(np.uint8(img[:, :, (2, 1, 0)]))

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------------------------------------------------------

    def segment_people_coarse(self, sfactor=0.5, bg_prob=0.1, compat=5, sxy=80, srgb=13):

        seg_coarse_file = join(self.path_to_dataset, 'metadata', 'seg_coarse.p')
        if exists(seg_coarse_file):
            glog.info('Loading coarse detections from: {0}'.format(seg_coarse_file))
            with open(seg_coarse_file, 'rb') as f:
                self.mask_coarse = pickle.load(f)
        else:

            if not exists(join(self.path_to_dataset, 'masks')):
                os.mkdir(join(self.path_to_dataset, 'masks'))

            # Check if the masks have been calculated
            if self.file_lists_match(listdir(join(self.path_to_dataset, 'masks'))):
                # Apply the person segmentation
                glog.info('Applying the person segmentation')
                file_dir = os.path.dirname(os.path.realpath(__file__))
                os.system('bash {0}/bash_scripts/person_segmentation.sh {1} {2}'.format(file_dir, self.path_to_dataset, sfactor))

            glog.info('Refining segmentations from {0}'.format(join(self.path_to_dataset, 'masks')))
            for i, basename in enumerate(self.frame_basenames):
                probs = np.load(join(self.path_to_dataset, 'masks', '{0}.{1}.npy'.format(basename, self.ext)))
                probs_cnn = np.zeros((2, probs.shape[1], probs.shape[2]))
                probs_cnn[1, :, :] = probs[1, :, :]
                probs_cnn[0, :, :] = bg_prob
                probs_cnn /= (np.sum(probs_cnn, axis=0))

                img_ = self.get_frame(i, sfactor=sfactor, dtype=np.uint8)
                mask = segmentation.segment_image_densecrf(img_, probs_cnn, compat=compat, sxy=sxy, srgb=srgb)
                # mask = cv2.resize(mask, None, fx=1./sfactor, fy=1./sfactor)
                self.mask_coarse[basename] = mask

            with open(seg_coarse_file, 'wb') as f:
                pickle.dump(self.mask_coarse, f)

        return 0

    # ------------------------------------------------------------------------------------------------------------------

    def detect_people_coarse(self, nms_thresh=0.5, score_thresh=0.25):

        bbox_coarse_file = join(self.path_to_dataset, 'metadata', 'bbox_coarse.p')
        if exists(bbox_coarse_file):
            glog.info('Loading coarse detections from: {0}'.format(bbox_coarse_file))
            with open(bbox_coarse_file, 'rb') as f:
                self.bbox = pickle.load(f)
        else:

            if not exists(join(self.path_to_dataset, 'bbox')):
                os.mkdir(join(self.path_to_dataset, 'bbox'))

            # Check if the boxes have been calculated
            if not self.file_lists_match(listdir(join(self.path_to_dataset, 'bbox'))):
                # Apply the person detector
                glog.info('Applying the person detector')
                file_dir = os.path.dirname(os.path.realpath(__file__))
                os.system('bash {0}/bash_scripts/person_detection.sh {1}'.format(file_dir, self.path_to_dataset))

            # Gather the txt
            glog.info('Getting detections from {0}'.format(join(self.path_to_dataset, 'bbox')))
            for basename in tqdm(self.frame_basenames):
                bbox = np.loadtxt(join(self.path_to_dataset, 'bbox', '{0}.txt'.format(basename)))
                keep = nms(bbox.astype(np.float32), nms_thresh)
                bbox = bbox[keep, :]
                keep = bbox[:, 4] > score_thresh
                bbox = bbox[keep, :]
                self.bbox[basename] = bbox

            with open(bbox_coarse_file, 'wb') as f:
                pickle.dump(self.bbox, f)

        return 0

    # ------------------------------------------------------------------------------------------------------------------

    def detect_people_fine(self, redo=False, SCORE_THRESH=0.2, NMS_THRESH=0.2, min_height=0.2):

        bbox_fine_file = join(self.path_to_dataset, 'metadata', 'bbox_fine.p')
        if exists(bbox_fine_file) and not redo:
            glog.info('Loading fine detections from: {0}'.format(bbox_fine_file))
            with open(bbox_fine_file, 'rb') as f:
                self.bbox = pickle.load(f)
        else:

            for i, basename in enumerate(tqdm(self.frame_basenames)):
                bbox = self.bbox[basename]
                cam_mat = self.calib[basename]

                cam = cam_utils.Camera(basename, cam_mat['A'], cam_mat['R'], cam_mat['T'], self.shape[0], self.shape[1])

                scores = bbox[:, -1]
                keep = scores > SCORE_THRESH
                bbox = bbox[keep, :]
                keep, __ = misc_utils.putting_objects_in_perspective(cam, bbox, min_height=min_height)
                bbox = bbox[keep, :]
                keep = nms(bbox.astype(np.float32), NMS_THRESH)
                bbox = bbox[keep, :]

                self.bbox[basename] = bbox

            with open(bbox_fine_file, 'wb') as f:
                pickle.dump(self.bbox, f)

        return 0

    # ------------------------------------------------------------------------------------------------------------------

    def estimate_pose_coarse(self, redo=False):

        pose_file_coarse = join(self.path_to_dataset, 'metadata', 'poses_coarse.p')
        if exists(pose_file_coarse) and not redo:
            glog.info('Loading fine detections from: {0}'.format(pose_file_coarse))
            with open(pose_file_coarse, 'rb') as f:
                self.poses = pickle.load(f)
        else:
            h, w = self.shape[:2]
            openposebin = './build/examples/openpose/openpose.bin'
            tmp_dir = join(self.path_to_dataset, 'tmp')
            if not exists(tmp_dir):
                os.mkdir(tmp_dir)

            for i, basename in enumerate(tqdm(self.frame_basenames)):

                # Remove previous files
                previous_files = [f for f in os.listdir(tmp_dir)]
                for f in previous_files:
                    os.remove(join(tmp_dir, f))

                img = self.get_frame(i)
                bbox = self.bbox[basename]

                pad = 150

                # save the crops in a temp file
                for j in range(bbox.shape[0]):
                    x1, y1, x2, y2 = bbox[j, 0:4]
                    x1, y1 = int(np.maximum(np.minimum(x1 - pad, w - 1), 0)), int(
                        np.maximum(np.minimum(y1 - pad, h - 1), 0))
                    x2, y2 = int(np.maximum(np.minimum(x2 + pad, w - 1), 0)), int(
                        np.maximum(np.minimum(y2 + pad, h - 1), 0))
                    crop = img[y1:y2, x1:x2, :]

                    # Save crop
                    cv2.imwrite(join(self.path_to_dataset, 'tmp', '{0}.jpg'.format(j)), crop[:, :, (2, 1, 0)] * 255)

                cwd = os.getcwd()
                os.chdir(join(file_utils.get_platform_codedir(), 'openpose'))
                command = '{0} --model_pose COCO --image_dir {1} --write_keypoint {2} --no_display'.format(openposebin,
                                                                                                           tmp_dir,
                                                                                                           tmp_dir)

                os.system(command)
                os.chdir(cwd)

                poses = []
                for j in range(bbox.shape[0]):
                    x1, y1, x2, y2 = bbox[j, 0:4]
                    x1, y1 = int(np.maximum(np.minimum(x1 - pad, w - 1), 0)), int(
                        np.maximum(np.minimum(y1 - pad, h - 1), 0))

                    with open(join(join(self.path_to_dataset, 'tmp'), '{0}_pose.yml'.format(j))) as data_file:
                        for iii in range(2):
                            _ = data_file.readline()
                        data_yml = yaml.load(data_file)

                        if 'sizes' not in data_yml:
                            continue
                        sz = data_yml['sizes']
                        n_persons = sz[0]
                        keypoints = np.array(data_yml['data']).reshape(sz)

                        for k in range(n_persons):
                            keypoints_ = keypoints[k, :, :]
                            keypoints_[:, 0] += x1
                            keypoints_[:, 1] += y1
                            poses.append(keypoints_)

                self.poses[basename] = poses

            with open(pose_file_coarse, 'wb') as f:
                pickle.dump(self.poses, f)

        return 0

    # ------------------------------------------------------------------------------------------------------------------

    def refine_poses(self, keypoint_thresh=10, score_thresh=0.5, neck_thresh=0.59):

        for i, basename in enumerate(tqdm(self.frame_basenames)):
            poses = self.poses[basename]

            # remove the poses with few keypoints or they
            keep = []
            for ii in range(len(poses)):
                keypoints = poses[ii]
                valid = (keypoints[:, 2] > 0.).nonzero()[0]
                score = np.sum(keypoints[valid, 2])

                if len(valid) > keypoint_thresh and score > score_thresh and keypoints[1, 2] > neck_thresh:
                    keep.append(ii)

            poses = [poses[ii] for ii in keep]

            root_part = 1
            root_box = []
            for ii in range(len(poses)):
                root_tmp = poses[ii][root_part, :]
                valid_keypoints = (poses[ii][:, 2] > 0).nonzero()
                root_box.append(
                     [root_tmp[0] - 10, root_tmp[1] - 10, root_tmp[0] + 10, root_tmp[1] + 10,
                     np.sum(poses[ii][valid_keypoints, 2])])
            root_box = np.array(root_box)

            # Perform Neck NMS

            if len(root_box.shape) == 1:
                root_box = root_box[None, :]
                keep2 = [0]
            else:
                keep2 = nms(root_box.astype(np.float32), 0.1)

            poses = [poses[ii] for ii in keep2]
            self.poses[basename] = poses

    def estimate_pose_fine(self, redo=False, keypoint_thresh=10, score_thresh=0.5):

        pose_file_fine = join(self.path_to_dataset, 'metadata', 'poses_fine.p')
        if exists(pose_file_fine) and not redo:
            glog.info('Loading fine detections from: {0}'.format(pose_file_fine))
            with open(pose_file_fine, 'rb') as f:
                self.poses = pickle.load(f)
        else:

            for i, basename in enumerate(tqdm(self.frame_basenames)):
                poses = self.poses[basename]

                # remove the poses with few keypoints or they
                keep = []
                for ii in range(len(poses)):
                    keypoints = poses[ii]
                    valid = (keypoints[:, 2] > 0.).nonzero()[0]
                    score = np.sum(keypoints[valid, 2])

                    if len(valid) > keypoint_thresh and score > score_thresh:
                        keep.append(ii)

                poses = [poses[ii] for ii in keep]

                root_part = 1
                root_box = []
                for ii in range(len(poses)):
                    root_tmp = poses[ii][root_part, :]
                    valid_keypoints = (poses[ii][root_part, 2] > 0).nonzero()
                    root_box.append([root_tmp[0] - 10, root_tmp[1] - 10, root_tmp[0] + 10, root_tmp[1] + 10, len(valid_keypoints)])
                root_box = np.array(root_box)

                # Perform Neck NMS
                keep2 = nms(root_box.astype(np.float32), 0.1)
                poses = [poses[ii] for ii in keep2]
                self.poses[basename] = poses

            with open(pose_file_fine, 'wb') as f:
                pickle.dump(self.poses, f)

    # ------------------------------------------------------------------------------------------------------------------

    def segment_people(self):
        pass

    # ------------------------------------------------------------------------------------------------------------------

    def file_lists_match(self, list2):
        list2 = [file_utils.extract_basename(f)[0] for f in list2]
        hash_table = dict.fromkeys(list2)
        all_included = True
        for i in self.frame_basenames:
            if i not in hash_table:
                all_included = False
                break
        return all_included

    # ------------------------------------------------------------------------------------------------------------------

    def track_people(self):
        pose_matrix = []
        for i, basename in enumerate(tqdm(self.frame_basenames)):
            poses = self.poses[basename]
            pose_matrix.append(poses)

        sol = tracking.track_from_poses(pose_matrix)
        self.tracks = sol
        # Plot the result

    # ------------------------------------------------------------------------------------------------------------------

    def dump_tracking_video_mot_mac(self, mot_tracks, scale=4):

        glog.info('Dumping tracking video')

        if file_utils.get_platform() != 'mac':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_file = join(self.path_to_dataset, 'tracks_mot.mp4')
            out = cv2.VideoWriter(out_file, fourcc, 20.0, (self.shape[1] // scale, self.shape[0] // scale))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cmap = matplotlib.cm.get_cmap('hsv')

        n_tracks = max(np.unique(mot_tracks[:, 1]))

        for i, basename in enumerate(tqdm(self.frame_basenames)):
            out_file = join(self.path_to_dataset, 'tmp', '{0:05d}.jpg'.format(i))

            img = self.get_frame(i, dtype=np.uint8)

            cur_id = mot_tracks[:, 0] - 1 == i
            current_boxes = mot_tracks[cur_id, :]

            for j in range(current_boxes.shape[0]):
                track_id, x, y, w, h = current_boxes[j, 1:6]
                clr = cmap(track_id / float(n_tracks))
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                              (clr[0] * 255, clr[1] * 255, clr[2] * 255), 10)
                cv2.putText(img, str(int(track_id)), (int(x), int(y)), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

            img = cv2.resize(img, (self.shape[1] // scale, self.shape[0] // scale))

            if file_utils.get_platform() != 'mac':
                out.write(np.uint8(img[:, :, (2, 1, 0)]))
            else:
                cv2.imwrite(out_file, np.uint8(img[:, :, (2, 1, 0)]))

        if file_utils.get_platform() != 'mac':
            out.release()
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------

    def _merge_subimages_detectron(self, _boxes, _segms, _keyps):

        H, W = self.shape[0], self.shape[1]
        subimages = []
        for x in range(3):
            for y in range(3):
                x1, y1 = x * H // 4, y * W // 4
                x2, y2 = (x + 2) * H // 4, (y + 2) * W // 4
                subimages.append([x1, y1, x2, y2])

        all_boxes = np.zeros((0, 5))
        all_segs = []
        all_keyps = None
        all_classes = []
        for index in range(len(subimages)):
            x1, y1, x2, y2 = subimages[index]

            boxes, segms, keyps, classes = misc_utils.convert_from_cls_format(_boxes[index], None, None)
            if boxes is None:
                continue

            boxes[:, 0] += y1
            boxes[:, 2] += y1
            boxes[:, 1] += x1
            boxes[:, 3] += x1

            for i in range(boxes.shape[0]):
                if segms is not None:
                    __segm = mask_util.decode(segms[i])
                    _tmp = np.zeros((H, W), dtype=np.uint8, order='F')
                    _tmp[x1:x2, y1:y2] = __segm
                    __tmp = mask_util.encode(_tmp)

                    all_segs.append(__tmp)
                else:
                    all_segs = None
                all_classes.append(classes[i])

            all_boxes = np.vstack((all_boxes, boxes))

        return all_boxes, all_segs, all_keyps, all_classes

    def gather_detectron(self):

        if not exists(join(self.path_to_dataset, 'detectron')):
            os.mkdir(join(self.path_to_dataset, 'detectron'))

        detectron_file = join(self.path_to_dataset, 'metadata', 'detectron.p')
        if exists(detectron_file):
            glog.info('Loading coarse detections from: {0}'.format(detectron_file))
            with open(detectron_file, 'rb') as f:
                self.detectron = pickle.load(f)

        else:

            for i, basename in tqdm(enumerate(self.frame_basenames)):
                # glog.info(basename)
                with open(join(self.path_to_dataset, 'detectron', '{0}.yml'.format(basename)), 'rb') as stream:
                    data = yaml.load(stream)
                boxes, classes, segms = data['boxes'], data['classes'], data['segms']

                self.detectron[basename] = {'boxes': boxes, 'segms': segms, 'keyps': None, 'classes': classes}

            with open(detectron_file, 'wb') as f:
                pickle.dump(self.detectron, f)

    def get_number_of_players(self):

        players_in_frame = np.zeros((self.n_frames,))
        for i, basename in enumerate(self.frame_basenames):
            players_in_frame[i] = len(self.bbox[basename])

        return players_in_frame

    def get_boxes_in_2d(self):
        boxes2d = []
        for i, basename in enumerate(self.frame_basenames):
            bbox = self.bbox[basename]
            boxes2d.append(bbox[:, :4].reshape(bbox.shape[0], 2, 2))
        return boxes2d

    def get_keypoints_in_2d(self):
        keypoints = []
        for i, basename in enumerate(self.frame_basenames):
            kp = self.poses[basename]
            keypoints.append(kp)
        return keypoints

    def get_boxes_in_3d(self):
        boxes3d = []
        for i, basename in enumerate(self.frame_basenames):
            bbox = self.bbox[basename]
            cam_mat = self.calib[basename]
            cam = cam_utils.Camera(basename, cam_mat['A'], cam_mat['R'], cam_mat['T'], self.shape[0], self.shape[1])
            bbox3d = misc_utils.lift_box_in_3d(cam, bbox)
            boxes3d.append(bbox3d)
        return boxes3d

    def get_mask_from_detectron(self, frame_number):
        return io.imread(join(self.path_to_dataset, 'detectron', self.frame_basenames[frame_number]+'.png'))[:, :, 0]

    def get_ball_from_detectron(self, thresh=0.0, ):
        for i, basename in enumerate(tqdm(self.frame_basenames)):
            data = self.detectron[basename]
            boxes, segms, keyps, classes = data['boxes'], data['segms'], data['keyps'], data['classes']
            valid = (boxes[:, 4] > thresh)*([j == 33 for j in classes])
            self.ball[basename] = boxes[valid, :]

    def get_color_from_detections(self, frame_number):
        basename = self.frame_basenames[frame_number]
        img = self.get_frame(frame_number)
        boxes = self.bbox[basename]
        n_boxes = boxes.shape[0]
        box_color = np.zeros((n_boxes, 3))
        segms = self.detectron[basename]['segms']

        for i in range(n_boxes):
            masks = mask_util.decode(segms[i])
            II, JJ = (masks > 0).nonzero()
            crop = img[II, JJ, :].reshape((-1, 3))
            box_color[i, :] = np.mean(crop, axis=0)
        return box_color

    def refine_detectron(self, basename, score_thresh=0.9, nms_thresh=0.5, min_height=0.0, min_area=200):

        data = self.detectron[basename]
        boxes, segms, keyps, classes = data['boxes'], data['segms'], data['keyps'], data['classes']

        valid = (boxes[:, 4] > score_thresh) * ([j == 1 for j in classes])
        valid = (valid==True).nonzero()[0]
        boxes = boxes[valid, :]
        segms = [segms[i] for i in valid]
        classes = [classes[i] for i in valid]

        cam_mat = self.calib[basename]
        cam = cam_utils.Camera(basename, cam_mat['A'], cam_mat['R'], cam_mat['T'], self.shape[0], self.shape[1])

        keep, __ = misc_utils.putting_objects_in_perspective(cam, boxes, min_height=min_height)
        boxes = boxes[keep, :]
        segms = [segms[i] for i in keep]
        classes = [classes[i] for i in keep]

        valid_nms = nms(boxes.astype(np.float32), nms_thresh)
        boxes = boxes[valid_nms, :]
        segms = [segms[i] for i in valid_nms]
        classes = [classes[i] for i in valid_nms]

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        valid_area = (areas > min_area).nonzero()[0]
        boxes = boxes[valid_area, :]
        segms = [segms[i] for i in valid_area]
        classes = [classes[i] for i in valid_area]

        return boxes, segms, keyps, classes

    def get_boxes_from_detectron(self, score_thresh=0.9, nms_thresh=0.5, min_height=0.0, min_area=200):

        for i, basename in enumerate(tqdm(self.frame_basenames)):
            boxes, segms, keyps, classes = self.refine_detectron(basename, score_thresh=score_thresh,
                                                                 nms_thresh=nms_thresh, min_height=min_height,
                                                                 min_area=min_area)
            self.bbox[basename] = boxes

    def get_instances_from_detectron(self, frame_number, sorted_inds=None, score_thresh=0.9, nms_thresh=0.5,
                                     min_height=0.0, min_area=200, is_bool=False, refine=True):

        if refine:
            boxes, segms, keyps, classes = self.refine_detectron(self.frame_basenames[frame_number],
                                                                 score_thresh=score_thresh, nms_thresh=nms_thresh,
                                                                 min_height=min_height, min_area=min_area)
        else:
            data = self.detectron[self.frame_basenames[frame_number]]
            boxes, segms, keyps, classes = data['boxes'], data['segms'], data['keyps'], data['classes']

        if sorted_inds is None:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            sorted_inds = np.argsort(-areas)

        instance_map = np.zeros((self.shape[0], self.shape[1]))

        for ii, i in enumerate(sorted_inds):
                masks = mask_util.decode(segms[i])
                instance_map += (masks*(ii + 1))

        if is_bool:
            instance_map = instance_map.astype(bool).astype(np.uint8)

        return instance_map

    def order_boxes(self, frame_number):
        basename = self.frame_basenames[frame_number]
        boxes = self.bbox[basename]

        cam_mat = self.calib[basename]
        cam = cam_utils.Camera(basename, cam_mat['A'], cam_mat['R'], cam_mat['T'], self.shape[0], self.shape[1])
        bbox3d = misc_utils.lift_box_in_3d(cam, boxes)

        points3d = np.zeros((0, 3))
        for i in range(bbox3d.shape[0]):
            points3d = np.vstack((points3d, bbox3d[i, 0, :]))

        _, depth = cam.project(points3d)

        return sorted([i for i in range(len(depth))], key=lambda x: depth[x])

    def dump_instance_video(self, scale=4):

        glog.info('Dumping tracking video')

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_file = join(self.path_to_dataset, 'instances.avi')
        out = cv2.VideoWriter(out_file, fourcc, 20.0, (self.shape[1] // scale, self.shape[0] // scale))

        for i, basename in enumerate(tqdm(self.frame_basenames)):
            ordering = self.order_boxes(i)
            instaces_gray = self.get_instances_from_detectron(i, sorted_inds=ordering, score_thresh=0.8, nms_thresh=0.25)
            instaces_gray = (instaces_gray*255/np.max(instaces_gray)).astype(np.uint8)
            instaces = cv2.applyColorMap(instaces_gray, cv2.COLORMAP_COOL)
            instaces = cv2.resize(instaces, (self.shape[1] // scale, self.shape[0] // scale))
            out.write(np.uint8(instaces[:, :, (2, 1, 0)]))

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()



















































