#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import yaml
import numpy as np

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

import pycocotools.mask as mask_util

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.' + args.output_ext)
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()

        # ======================================================================
        h, w = im.shape[:2]

        subimages = []
        for x in range(3):
            for y in range(3):
                x1, y1 = x*h//4, y*w//4
                x2, y2 = (x+2)*h//4, (y+2)*w//4
                subimages.append([x1, y1, x2, y2])

        with c2_utils.NamedCudaScope(0):
            cls_boxes = []
            cls_segms = []
            cls_keyps = []
            for index in range(len(subimages)):
                x1, y1, x2, y2 = subimages[index]
                _cls_boxes, _cls_segms, _cls_keyps = infer_engine.im_detect_all(
                    model, im[x1:x2, y1:y2, :], None, timers=timers
                )
                cls_boxes.append(_cls_boxes)
                cls_segms.append(_cls_segms)
                cls_keyps.append(_cls_keyps)
        # ======================================================================

        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )


        # ======================================================================
        t = time.time()

        out_name_yml = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name)[:-4] + '.yml')
        )

        _mask = np.zeros((h, w), dtype=np.uint8)
        all_boxes = np.zeros((0, 5))
        all_classes = []
        all_segs = []
        for index in range(len(subimages)):
            x1, y1, x2, y2 = subimages[index]

            boxes, segms, keyps, classes = vis_utils.convert_from_cls_format(cls_boxes[index], cls_segms[index], cls_keyps[index])
            if boxes is None:
                continue

            for i in range(boxes.shape[0]):
                _tmp = np.zeros((h, w), dtype=np.uint8, order='F')
                __segm = mask_util.decode(segms[i])
                _tmp[x1:x2, y1:y2] = __segm
                __tmp = mask_util.encode(_tmp)
                all_segs.append(__tmp)

                _mask[x1:x2, y1:y2] += __segm
                all_classes.append(classes[i])

            boxes[:, 0] += y1
            boxes[:, 2] += y1
            boxes[:, 1] += x1
            boxes[:, 3] += x1

            all_boxes = np.vstack((all_boxes, boxes))

        _mask = _mask.astype(bool).astype(int)
        out_name_mask = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name)[:-4] + '.png')
        )
        cv2.imwrite(out_name_mask, _mask*255)



        with open(out_name_yml, 'w') as outfile:
            yaml.dump({'boxes': all_boxes,
                       'segms': all_segs,
                       'classes': all_classes}, outfile, default_flow_style=False)


        logger.info('Saving time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        # ======================================================================


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
