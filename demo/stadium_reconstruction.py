from os.path import join
import argparse
import numpy as np
import soccer3d
import utils.files as file_utils
import utils.io as io
import utils.camera as cam_utils
import cv2
import utils.stadium as stadium_utils
import utils.misc as misc_utils
import utils.draw as draw_utils

parser = argparse.ArgumentParser(description='Calibrate a soccer video')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Singleview/Soccer/Japan-Something-1', help='path')
opt, _ = parser.parse_known_args()

db = soccer3d.YoutubeVideo(opt.path_to_data)
db.digest_metadata()

img = db.get_frame(0)
mask = db.get_mask_from_detectron(0)
calib_data = db.calib[db.frame_basenames[0]]

cam = cam_utils.Camera('_', calib_data['A'], calib_data['R'], calib_data['T'], db.shape[0], db.shape[1])

_, field_mask = draw_utils.draw_field(cam)
kernel = np.ones((25, 25), dtype=np.uint8)

field_mask = cv2.dilate(field_mask.astype(np.uint8), kernel, iterations=1)

mask *= field_mask
mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

# io.imshow(img*mask[:, :, None])
img = cv2.inpaint((img*255).astype(np.uint8), mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
img = img/255.
# io.imshow(img)

name = 'front'

imgs = []
uvs = []
vertices = np.zeros((0, 3))
faces = []
counter = 0

stadium_params = [stadium_utils.get_bottom_params(), stadium_utils.get_back_params(), stadium_utils.get_right_params() , stadium_utils.get_left_params()]
# stadium_params = [stadium_utils.get_bottom_params()]

for vertex, plane_origin, plane_normal in stadium_params:
    plane = vertex.copy()

    points3d, uv, _faces = stadium_utils.project_plane_to_image(plane, cam, plane_origin, plane_normal)
    if points3d is None:
        continue
    p2, _ = cam.project(points3d)

    dst, transformed_p2 = stadium_utils.rectify_image(img, cam, vertex, p2)

    x1, y1, x2, y2 = np.min(transformed_p2[:, 0]), np.min(transformed_p2[:, 1]), np.max(transformed_p2[:, 0]), np.max(transformed_p2[:, 1])
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    dst = dst[y1:y2, x1:x2, :]
    transformed_p2[:, 0] = transformed_p2[:, 0] - x1
    transformed_p2[:, 1] = transformed_p2[:, 1] - y1

    vertices = np.vstack((vertices, points3d))

    for i in range(len(_faces)):
        for j in range(3):
            _faces[i][j] += counter
        faces.append(_faces[i])

    imgs.append(dst)

    uvs.append(transformed_p2)

    counter += points3d.shape[0]
    # plt.imshow(dst)
    # plt.triplot(transformed_p2[:, 0], transformed_p2[:,1], faces)
    # plt.show()

file_utils.mkdir(join(db.path_to_dataset, 'field'))
atlas, final_uvs = misc_utils.pack_textures(imgs, uvs)
cv2.imwrite(join(db.path_to_dataset, 'field', '{0}.jpg'.format(name)), atlas[:, :, ::-1])

vertex_data_out = io.numpy_to_ply(vertices)
io.write_obj(join(db.path_to_dataset, 'field', '{0}.obj'.format(name)), vertex_data_out, faces, final_uvs, '{0}.jpg'.format(name))



# plt.imshow(img)
# # plt.plot(xy[0], xy[1])
# plt.plot(_p2[:, 0], _p2[:, 1], 'o')
# plt.triplot(_p2[:,0], _p2[:,1], faces)
# plt.show()

# io.imshow(img, points=p2)