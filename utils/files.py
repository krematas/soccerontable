import os
import socket
import numpy as np
import glog

__all__ = ['extract_basename', 'extract_path', 'get_dataset_info', 'gen_folder_structure']


def extract_basename(filename):
    path, tail = os.path.split(filename)
    basename, ext = os.path.splitext(tail)
    ext = ext.replace('.', '').replace('\'', '')
    return basename, ext


def extract_path(filename):
    path, _ = os.path.split(filename)
    return path


def mkdir(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)
    else:
        glog.warning('Dir {0} already exist.'.format(path_to_dir))


def get_dataset_info(path_to_dataset, info_file='info.txt'):
    fname = os.path.join(path_to_dataset, info_file)
    if os.path.exists(fname):
        info = np.loadtxt(fname, delimiter=':', dtype=str)
        out = {}
        for i in range(info.shape[0]):
            out[info[i, 0]] = info[i, 1]
        out['fps'] = int(out['fps'])
        out['height'] = int(out['height'])
        out['width'] = int(out['width'])
        out['ext'] = out['extension'][1:]

        if 'flipped' not in out:
            out['flipped'] = 0
        else:
            out['flipped'] = int(out['flipped'])

        return out
    else:
        glog.error('There is no info file in folder {0}'.format(path_to_dataset))
        return -1


def gen_folder_structure(path_to_dataset):
    os.mkdir(os.path.join(path_to_dataset, 'bbox'))
    os.mkdir(os.path.join(path_to_dataset, 'masks'))
    os.mkdir(os.path.join(path_to_dataset, 'calib'))
    os.mkdir(os.path.join(path_to_dataset, 'calib', 'corr'))

    os.mkdir(os.path.join(path_to_dataset, 'edges'))
    os.mkdir(os.path.join(path_to_dataset, 'cnn'))
    os.mkdir(os.path.join(path_to_dataset, 'cnn', 'youtube'))
    os.mkdir(os.path.join(path_to_dataset, 'cnn', 'youtube', 'labels'))
    os.mkdir(os.path.join(path_to_dataset, 'cnn', 'youtube', 'images'))
    os.mkdir(os.path.join(path_to_dataset, 'cnn', 'youtube', 'anno'))
    os.mkdir(os.path.join(path_to_dataset, 'cnn', 'youtube', 'pointcloud'))
    os.mkdir(os.path.join(path_to_dataset, 'cnn', 'youtube', 'pointcloud', 'single_player'))
    os.mkdir(os.path.join(path_to_dataset, 'cnn', 'youtube', 'pointcloud', 'smooth'))

    os.mkdir(os.path.join(path_to_dataset, 'tracks'))

    os.mkdir(os.path.join(path_to_dataset, 'tmp'))
    os.mkdir(os.path.join(path_to_dataset, 'scene'))
    os.mkdir(os.path.join(path_to_dataset, 'scene', 'meshes'))
