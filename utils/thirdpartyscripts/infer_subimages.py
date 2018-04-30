#!/usr/bin/env python2

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

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
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
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for ii, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )

        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        h, w = im.shape[:2]

        subimages = []
        for x in range(3):
            for y in range(3):
                x1, y1 = x*h//4, y*w//4
                x2, y2 = (x+2)*h//4, (y+2)*w//4
                subimages.append([x1, y1, x2, y2])

        timers = defaultdict(Timer)
        t = time.time()
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

        logger.info('Inference time: {:.3f}s'.format(time.time() - t))

        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

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

        # break
        # vis_utils.vis_one_image(
        #     im[:, :, ::-1],  # BGR -> RGB for visualization
        #     im_name,
        #     args.output_dir,
        #     cls_boxes,
        #     cls_segms,
        #     cls_keyps,
        #     dataset=dummy_coco_dataset,
        #     box_alpha=0.3,
        #     show_class=True,
        #     thresh=0.7,
        #     kp_thresh=2
        # )


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
