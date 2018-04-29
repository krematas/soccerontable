# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# from utils.nms.nms.gpu_nms import gpu_nms
from utils.nms.nms.cpu_nms import cpu_nms


def nms(dets, thresh, gpu=-1):
    """Dispatch to either CPU or GPU NMS implementations."""

    # if dets.shape[0] == 0:
    #     return []
    # if gpu > -1:
    #     return gpu_nms(dets, thresh, device_id=gpu)
    # else:
    return cpu_nms(dets, thresh)
