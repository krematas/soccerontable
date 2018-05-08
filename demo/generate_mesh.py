import argparse
import soccer3d
import json
from os.path import join
from tqdm import tqdm
import utils.camera as cam_utils
import utils.io as io
import utils.mesh as mesh_utils
import utils.misc as misc_utils
import utils.files as file_utils
import openmesh as om
import numpy as np
import cv2


parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona', help='path')
parser.add_argument('--decimate_to', type=int, default=500, help='Margin around the pose')
opt, _ = parser.parse_known_args()


with open(join(opt.path_to_data, 'players', 'metadata', 'position.json')) as f:
    data = json.load(f)


db = soccer3d.YoutubeVideo(opt.path_to_data)
db.digest_metadata()

db.refine_poses(keypoint_thresh=7, score_thresh=0.4, neck_thresh=0.4)

file_utils.mkdir(join(db.path_to_dataset, 'scene3d'))

for sel_frame in tqdm(range(db.n_frames)):
    img = db.get_frame(sel_frame)
    basename = db.frame_basenames[sel_frame]

    cam_data = db.calib[basename]
    cam = cam_utils.Camera(basename, cam_data['A'], cam_data['R'], cam_data['T'], db.shape[0], db.shape[1])

    player_list = data[basename]

    frame_mesh_points = np.zeros((0, 3))
    frame_mesh_faces = np.zeros((0, 3))

    uvs_atlas = []
    textures_atlas = []

    cnt = 0
    for p in range(len(player_list)):
        mesh = om.PolyMesh()

        ply_name = join(opt.path_to_data, 'players', 'meshes', player_list[p]['mesh']+'.ply')
        vertex_data, face_data, _, _ = io.read_ply(ply_name)

        pointcloud, pc_colors, _ = io.ply_to_numpy(vertex_data)
        pc_centered = pointcloud - np.mean(pointcloud, axis=0)

        points2d, depth = cam.project(pointcloud)
        faces = mesh_utils.triangulate_depthmap_points(points2d, depth, depth_thresh=0.1)

        vertex_handle = []
        for i in range(pointcloud.shape[0]):
            vh = mesh.add_vertex(pointcloud[i, :])
            vertex_handle.append(vh)

        face_handle = []
        for i in range(len(faces)):
            fh0 = mesh.add_face(vertex_handle[faces[i][0]], vertex_handle[faces[i][1]], vertex_handle[faces[i][2]])
            face_handle.append(fh0)

        d = om.PolyMeshDecimater(mesh)
        mh = om.PolyMeshModQuadricHandle()

        # add modules
        d.add(mh)
        d.module(mh).set_max_err(0.001)

        # decimate
        d.initialize()
        d.decimate_to(opt.decimate_to)

        mesh.garbage_collection()

        _vertices = mesh.points()
        _faces = mesh.face_vertex_indices()

        uv, _ = cam.project(_vertices, dtype=np.float32)
        _vertices -= np.mean(_vertices, axis=0)
        _vertices[:, 0] += player_list[p]['x']
        # _vertices[:, 1] = _vertices[:, 1] + pc_center[1] - np.min(_vertices[:, 1])
        _vertices[:, 2] += player_list[p]['z']

        min_u, max_u = int(np.min(np.round(uv[:, 0]))), int(np.max(np.round(uv[:, 0]))) + 1
        min_v, max_v = int(np.min(np.round(uv[:, 1]))), int(np.max(np.round(uv[:, 1]))) + 1
        crop = img[min_v:max_v, min_u:max_u, :]

        tmp = uv.copy()
        tmp[:, 0] -= np.min(np.round(uv[:, 0]))
        tmp[:, 1] -= np.min(np.round(uv[:, 1]))

        uvs_atlas.append(tmp)
        textures_atlas.append(crop)

        frame_mesh_points = np.vstack((frame_mesh_points, _vertices))
        frame_mesh_faces = np.vstack((frame_mesh_faces, _faces + cnt))
        cnt += _vertices.shape[0]

    atlas, final_uvs = misc_utils.pack_textures(textures_atlas, uvs_atlas, n_rows=1)

    texture_name = join(db.path_to_dataset, 'scene3d', '{0}.jpg'.format(basename))
    cv2.imwrite(join(texture_name), atlas[:, :, (2, 1, 0)] * 255)

    ply_out = io.numpy_to_ply(frame_mesh_points)
    io.write_obj(join(db.path_to_dataset, 'scene3d', '{0}.obj'.format(basename)), ply_out, list(frame_mesh_faces), final_uvs, texture_name)
