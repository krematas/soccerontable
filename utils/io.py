import numpy as np
import struct
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import glog
from os.path import join, exists
import sys
from . import files
import yaml

# ======================================================================================================================
# Image input and visualization
# ======================================================================================================================


def imread(filename, dtype=np.float32, sfactor=1.0, image_type='rgb', flip=False):
    if exists(filename):
        image = cv2.imread(filename)
        if image_type == 'gray':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image_type == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            glog.error('Unknown format')

        if dtype == np.float32 or dtype == np.float64:
            image = image.astype(dtype)
            image /= 255.

        if sfactor != 1.0:
            image = cv2.resize(image, None, fx=sfactor, fy=sfactor)

        if flip:
            image = image[:, ::-1, :]
    else:
        glog.error('File {0} not found'.format(filename))
        image = np.array([-1])

    return image


def imshow(image, ax=None, points=None, marker='r.'):

    show = False
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        show = True

    ax.imshow(image)
    if points is not None:
        if isinstance(points, list):
            for p in range(len(points)):
                ax.plot(points[p][:, 0], points[p][:, 1], marker[p])
        else:
            ax.plot(points[:, 0], points[:, 1], marker)

    ax.axis('off')

    if show:
        plt.show()


def show_box(img, bbox, points=None, ax=None, edgecolor='red'):

    if len(bbox.shape) == 1:
        bbox = bbox[None, :]

    show = False
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        show = True

    ax.imshow(img)
    for i in range(bbox.shape[0]):
        x1, y1, x2, y2 = bbox[i, 0:4]
        ax.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,  # remove background
                edgecolor=edgecolor
            )
        )
        plt.plot([x1, x2], [y2, y2], 'ro')
        plt.text(x1, y2, '{0}: {1:.2f}'.format(i, bbox[i, 4]), fontsize=10)

    if points is not None:
        plt.plot(points[:, 0], points[:, 1], 'c.')
    plt.axis('off')
    if show:
        plt.show()


def show_pose(img, poses, points=None, ax=None, cmap='tab10', keypoint_type='COCO'):

    limps_coco = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 11], [11, 12], [12, 13], [1, 8],
                           [8, 9], [9, 10], [14, 15], [16, 17], [0, 14], [0, 15], [14, 16], [15, 17]])

    limps_mpi = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10],
                          [14, 11], [11, 12], [12, 13]])

    edges = ['-', '-.', '--', '-*']

    if keypoint_type == 'COCO':
        limps = limps_coco
    else:
        limps = limps_mpi

    show = False
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        show = True

    ax.imshow(img)

    cmap_fun = matplotlib.cm.get_cmap(cmap)
    for i in range(len(poses)):
        if poses[i] is None:
            continue
        clr = cmap_fun(i/float(len(poses)))
        edg = np.random.randint(0, 4)
        for k in range(limps.shape[0]):
            kp1, kp2 = limps[k, :].astype(int)
            x1, y1, s1 = poses[i][kp1, :]
            x2, y2, s2 = poses[i][kp2, :]

            if s1 == 0 or s2 == 0:
                continue
            plt.plot([x1, x2], [y1, y2], edges[edg], color=clr)
            # plt.text(x2, y2, '{0:0.1f}'.format(poses[i][kp2, 2]), fontsize=8)
        x1, y1, s1 = poses[i][1, :]
        plt.text(x1+10, y1-10, '{0}: {1:0.2f}'.format(i, np.sum(poses[i][1, 2])), fontsize=10)

    if points is not None:
        plt.plot(points[:, 0], points[:, 1], 'c.')
    plt.axis('off')
    if show:
        plt.show()


def imagesc(matrix, points=None, ax=None, cmap='jet', grid=True, show_axis=True, vmin=None, vmax=None):

    if len(matrix.shape) > 2:
        glog.error('Input has 3 dimensions, maybe use imshow?')
    else:
        show = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            show = True

        if points is not None:
            ax.plot(points[:, 0], points[:, 1], 'c.')

        if vmin is None:
            vmin = np.min(matrix)

        if vmax is None:
            vmax = np.max(matrix)

        ax.imshow(matrix, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
        if grid:
            ax.grid('on')
        if not show_axis:
            ax.axis('off')
        if show:
            plt.show()


def export_figure(img, cmap='gray', output='tmp.png',  vmin=0, vmax=1):
    height, width = img.shape[:2]
    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(output, dpi=height)
    plt.close()


def exrread(filename):
    hdr = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return hdr

# ======================================================================================================================
# Mesh input and output
# ======================================================================================================================


def read_ply(full_name):

    ply_format = None
    n_vertices = None
    n_faces = 0
    properties = list()
    prop_type = list()
    properties_face = list()
    properties_face_type = list()
    header = list()
    bytes_header = 0
    bytes_ply = None

    element_flag = 'vertex'

    if sys.version_info.major == 2:
        open_fun = open(full_name)
    else:
        open_fun = open(full_name, encoding="ISO-8859-1")

    with open_fun as f:

        content = f.readlines()

        # Find header
        i = 0

        while content[i] != 'end_header\n':
            header.append(content[i])
            i += 1

        # extract info
        for i in range(0, len(header)):
            buff = header[i].strip().split()
            if buff[0] == 'format':
                ply_format = buff[1]
            if buff[0] == 'element':
                if buff[1] == 'vertex':
                    n_vertices = int(buff[2])
                elif buff[1] == 'face':
                    element_flag = 'face'
                    n_faces = int(buff[2])
                else:
                    element_flag = 'dunno'
            if buff[0] == 'property':
                if element_flag == 'vertex':
                    prop_type.append(buff[1])
                    properties.append(buff[2])
                elif element_flag == 'face':
                    properties_face = buff[2:]

        assert (ply_format is not None and n_vertices is not None)

        vertex_data = [dict.fromkeys(properties) for k in range(n_vertices)]
        bytes_ply = np.zeros((len(properties),), dtype=int)

        face_data = list()

        for i in range(len(header) + 1):
            bytes_header += len(content[i])

        prop_type_short = prop_type[:]
        for i in range(len(prop_type)):
            if prop_type[i] == 'float':
                prop_type_short[i] = 'f'
                bytes_ply[i] = 4
            elif prop_type[i] == 'uchar':
                prop_type_short[i] = 'B'
                bytes_ply[i] = 1
            elif prop_type[i] == 'int':
                prop_type_short[i] = 'i'
                bytes_ply[i] = 4
            else:
                raise ValueError('Unknown property type')

        if ply_format == 'ascii':

            for i in range(0, n_vertices):
                buff = content[i + len(header) + 1].strip().split()
                buff = [float(x) for x in buff]

                for j in range(len(properties)):
                    prop = properties[j]
                    vertex_data[i][prop] = buff[j]

            for i in range(n_faces):
                buff = content[n_vertices+i + len(header) + 1].strip().split()
                buff = [float(x) for x in buff]
                face_data.append(np.array(buff[1:]))

        elif ply_format == 'binary_little_endian':

            ff = open(full_name, "rb")
            header_ = ff.read(bytes_header)

            for i in range(n_vertices):
                for j in range(len(properties)):
                    vertex_data[i][properties[j]] = struct.unpack(prop_type_short[j], ff.read(bytes_ply[j]))[0]

            for i in range(n_faces):
                n_face_vertex = struct.unpack('B', ff.read(1))[0]
                t_ = np.zeros((n_face_vertex,))
                for j in range(n_face_vertex):
                    t_[j] = struct.unpack('i', ff.read(4))[0]
                face_data.append(t_)

            ff.close()

        else:
            raise ValueError('Unknown PLY format')

        return vertex_data, face_data, properties, prop_type


def write_ply(full_name, vertex_data, face_data=None, meshcolor=0, face_uv=None, face_colors=None,
              texture_name='Parameterization.jpg'):

    write_normals = False
    if 'nx' in list(vertex_data[0].keys()):
        write_normals = True

    write_facedata = False
    if face_data is not None:
        write_facedata = True

    fid = open(full_name, 'w')
    fid.write('ply\n')
    fid.write('format ascii 1.0\n')
    if meshcolor == 1:
        fid.write('comment TextureFile %s\n' % texture_name)
    fid.write('element vertex %d\n' % len(vertex_data))
    fid.write('property float x\n')
    fid.write('property float y\n')
    fid.write('property float z\n')
    if meshcolor == 0:
        fid.write('property uchar red\n')
        fid.write('property uchar green\n')
        fid.write('property uchar blue\n')
    if write_normals:
        fid.write('property float nx\n')
        fid.write('property float ny\n')
        fid.write('property float nz\n')

    if write_facedata:
        fid.write('element face %d\n' % len(face_data))
        fid.write('property list uchar int vertex_indices\n')
        if meshcolor == 1:
            fid.write('property list uint8 float texcoord\n')
        elif meshcolor == 2:
            fid.write('property uchar red\n')
            fid.write('property uchar green\n')
            fid.write('property uchar blue\n')
    fid.write('end_header\n')

    for i in range(len(vertex_data)):
        fid.write('%.5f %.5f %.5f' % (vertex_data[i]['x'], vertex_data[i]['y'], vertex_data[i]['z']))
        if meshcolor == 0:
            fid.write(' %d %d %d\n' % (vertex_data[i]['red'], vertex_data[i]['green'], vertex_data[i]['blue']))
        else:
            fid.write('\n')

    if write_facedata:
        for i in range(len(face_data)):
            fid.write('3 %d %d %d\n' % (face_data[i][0], face_data[i][1], face_data[i][2]))
            if meshcolor == 1:
                fid.write('6 %.5f %.5f %.5f %.5f %.5f %.5f\n' % (face_uv[i][0, 0], face_uv[i][1, 0], face_uv[i][0, 1], face_uv[i][1, 1], face_uv[i][0, 2], face_uv[i][1, 2]))
            elif meshcolor == 2:
                fid.write('%d %d %d\n' % (face_colors[i, 0] * 255, face_colors[i, 1] * 255, face_colors[i, 2] * 255))

    fid.close()


def read_obj(filename):
    with open(filename) as f:
        content = f.readlines()

        vertex = []
        normal = []
        faces = []
        for i in range(len(content)):
            line = content[i]
            if line[0] == '#':
                continue

            if line[0:2] == 'vn':
                normal_info = line.replace(' \n', '').replace('vn ', '')
            elif line[0:2] == 'v ':
                vertex_info = line.replace(' \n', '').replace('v ', '')
                vertex.append(np.array(vertex_info.split(' ')).astype(float))

            elif line[0:2] == 'f ':
                face_info = line.replace(' \n', '').replace('f ', '')
                faces.append(np.array(face_info.split(' ')).astype(int)-1)

    vertex_data = []
    for i in range(len(vertex)):
        vertex_data.append({'x': vertex[i][0], 'y': vertex[i][1], 'z': vertex[i][2]})

        face_normals = []
    for i in range(len(faces)):
        v0, v1, v2 = vertex[faces[i][0]-1], vertex[faces[i][1]-1], vertex[faces[i][2]-1]
        dir0 = v1 - v0
        dir1 = v2 - v0
        face_normal_ = np.cross(dir0 / np.linalg.norm(dir0), dir1 / np.linalg.norm(dir1))
        face_normals.append(face_normal_)

    return vertex_data, faces, face_normals


def write_obj(obj_name, vertex_data, face_data, uv_data, texture_name='Parameterization'):
    basename, ext = files.extract_basename(obj_name)
    f = open(obj_name, 'w')
    f.write('####\n')
    f.write('#\n')
    f.write('# OBJ File Generated by Kostas\n')
    f.write('#\n')
    f.write('####\n')
    f.write('#\n')
    f.write('# Vertices: %d\n' % len(vertex_data))
    f.write('# Faces: %d\n' % len(face_data))
    f.write('#\n')
    f.write('####\n')
    if texture_name is not None:
        f.write('mtllib ./%s.obj.mtl\n\n' % basename)

    for v in range(len(vertex_data)):
        f.write('v %.6f %.6f %.6f\n' % (vertex_data[v]['x'], vertex_data[v]['y'], vertex_data[v]['z']))
        # if 'red' in vertex_data[v]:
        #     f.write('%.3f %.3f %.3f\n' % (vertex_data[v]['red'], vertex_data[v]['green'], vertex_data[v]['blue']))
        # else:
        #     f.write('\n')

    f.write('# %d vertices, 0 vertices normals\n\n\n' % len(vertex_data))
    if texture_name is not None:
        f.write('usemtl material_0\n')

        for v in range(len(vertex_data)):
            f.write('vt %.6f %.6f\n' % (uv_data[v, 0], uv_data[v, 1]))

    vt_counter = 1
    for fc in range(len(face_data)):
        if texture_name is not None:
            f.write('f %d/%d %d/%d %d/%d\n' % (face_data[fc][0] + 1, face_data[fc][0] + 1,
                                               face_data[fc][1] + 1, face_data[fc][1] + 1,
                                               face_data[fc][2] + 1, face_data[fc][2] + 1))
        else:
            f.write('f %d %d %d\n' % (face_data[fc][0] + 1,
                                      face_data[fc][1] + 1,
                                      face_data[fc][2] + 1))
        vt_counter += 3

    f.write('# 4984 faces, 3290 coords texture\n\n')
    f.write('# End of File\n')
    f.close()
    if texture_name is not None:
        mtl_name = obj_name + '.mtl'

        f = open(mtl_name, 'w')
        f.write('#\n')
        f.write('# Wavefront material file\n')
        f.write('# Converted by Kostas\n')
        f.write('#\n\n')

        f.write('newmtl material_0\n')
        f.write('Ka 0.200000 0.200000 0.200000\n')
        f.write('Kd 1.000000 1.000000 1.000000\n')
        f.write('Ks 1.000000 1.000000 1.000000\n')
        f.write('Tr 1.000000\n')
        f.write('illum 2\n')
        f.write('Ns 0.000000\n')
        f.write('map_Kd %s\n' % texture_name)
        f.close()


# ======================================================================================================================
# Misc input and output
# ======================================================================================================================


def read_pose(filename):
    with open(filename) as data_file:
        for iii in range(2):
            _ = data_file.readline()
        data_yml = yaml.load(data_file)
        if 'sizes' not in data_yml:
            return -1
        sz = data_yml['sizes']
        poses = np.array(data_yml['data']).reshape(sz)
        return poses


def read_flo(filename):

    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            glog.error('Magic number incorrect. Invalid .flo file')
            flow = -1
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2 * w * h)
            flow = np.resize(data, (h, w, 2))

    return flow


def ply_to_numpy(vertex_data):

    vertex = np.zeros((len(vertex_data), 3))
    color = np.zeros((len(vertex_data), 3))
    if 'nx' in vertex_data[0]:
        normals = np.zeros((len(vertex_data), 3))
    else:
        normals = None

    for i in range(0, len(vertex_data)):
        vertex[i, :] = np.array([vertex_data[i]['x'], vertex_data[i]['y'], vertex_data[i]['z']])
        if 'red' in vertex_data[0]:
            color[i, :] = np.array([vertex_data[i]['red'], vertex_data[i]['green'], vertex_data[i]['blue']])
        if 'nx' in vertex_data[0]:
            normals[i, :] = np.array([vertex_data[i]['nx'], vertex_data[i]['ny'], vertex_data[i]['nz']])

    return vertex, color, normals


def numpy_to_ply(vertex, color=None, normals=None):
    n = vertex.shape[0]
    ply_data = []

    if color is None:
        color = 255*np.ones((n, 3), dtype=np.int8)
    add_normals = True
    if normals is None:
        add_normals = False

    for i in range(n):
        data = {'x': vertex[i, 0], 'y': vertex[i, 1], 'z': vertex[i, 2],
                'red': color[i, 0], 'green': color[i, 1], 'blue': color[i, 2]}

        if add_normals:
            data['nx'], data['ny'], data['nz'] = normals[i, :]

        ply_data.append(data)

    return ply_data


def read_colmap_points(full_name):

    with open(full_name, encoding="ISO-8859-1") as f:

        content = f.readlines()

        vertex_sfm = []
        color_sfm = []
        vertex_viewlist = []

        for i in range(len(content)):

            if content[i][0] == '#':
                continue

            line = content[i].replace('\n', '').split(' ')
            n_views = int((len(line) - 8)/2)
            vertex_sfm.append(np.array([line[1], line[2], line[3]], dtype=float))
            color_sfm.append(np.array([line[4], line[5], line[6]], dtype=int))

            view_list = []
            for j in range(n_views):
                view_list.append(int(line[2*j + 8])-1)

            vertex_viewlist.append(view_list)

    return np.array(vertex_sfm), np.array(color_sfm), vertex_viewlist
