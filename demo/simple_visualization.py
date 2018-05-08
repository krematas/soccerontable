import numpy as np
from glumpy import app, gl, gloo, data
from glumpy.transforms import Trackball, Position
import utils.io as io
import utils.files as file_utils
from os.path import join
import time
import argparse

W, H = 104.73, 67.74


def objload(filename):
    V = []  # vertex
    T = []  # texcoords
    N = []  # normals
    F = []  # face indices
    F_V = []  # face indices
    F_T = []  # face indices
    F_N = []  # face indices

    for lineno, line in enumerate(open(filename)):
        if line[0] == '#':
            continue
        values = line.strip().split(' ')
        code = values[0]
        values = values[1:]
        # vertex (v)
        if code == 'v':
            V.append([float(x) for x in values])
        # tex-coord (vt)
        elif code == 'vt':
            T.append([float(x) for x in values])
        # normal (n)
        elif code == 'vn':
            N.append([float(x) for x in values])
        # face (f)
        elif code == 'f':
            if len(values) != 3:
                raise ValueError('not a triangle at line' % lineno)
            for v in values:
                for j, index in enumerate(v.split('/')):
                    if len(index):
                        if j == 0:
                            F_V.append(int(index) - 1)
                        elif j == 1:
                            F_T.append(int(index) - 1)
                        elif j == 2:
                            F_N.append(int(index) - 1)

    # Building the vertices
    V = np.array(V)
    F_V = np.array(F_V)
    vtype = [('position', np.float32, 3)]

    if len(T):
        T = np.array(T)
        F_T = np.array(F_T)
        vtype.append(('texcoord', np.float32, 2))
    if len(N):
        N = np.array(N)
        F_N = np.array(F_N)
        vtype.append(('normal', np.float32, 3))

    vertices = np.empty(len(F_V), vtype)
    vertices["position"] = V[F_V]
    if len(T):
        vertices["texcoord"] = T[F_T]
    if len(N):
        vertices["normal"] = N[F_N]
    vertices = vertices.view(gloo.VertexBuffer)

    itype = np.uint32
    indices = np.arange(len(vertices), dtype=np.uint32)
    indices = indices.view(gloo.IndexBuffer)
    return vertices, indices


def plane():
    vtype = [('position', np.float32, 3),
             ('texcoord', np.float32, 2),
             ('normal', np.float32, 3)]
    itype = np.uint32

    y_disp = -0.8
    p = np.array([[-W / 2, y_disp, -H / 2], [-W / 2., y_disp, H / 2], [W / 2, y_disp, H / 2], [W / 2, y_disp, -H / 2], ], dtype=float)

    n = np.array([[0, 1, 0]])
    t = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

    faces_p = [0, 1, 2, 3]
    faces_n = [0, 0, 0, 0]
    faces_t = [0, 1, 2, 3]

    _vertices = np.zeros(4, vtype)
    _vertices['position'] = p[faces_p]
    _vertices['normal'] = n[faces_n]
    _vertices['texcoord'] = t[faces_t]

    filled = np.resize(np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6 * (2 * 3))
    filled += np.repeat(4 * np.arange(6, dtype=itype), 6)
    _vertices = _vertices.view(gloo.VertexBuffer)
    filled = filled.view(gloo.IndexBuffer)

    return _vertices, filled


vertex_tex = """
attribute vec3 position;
attribute vec2 texcoord;      // Vertex texture coordinates
varying vec2   v_texcoord;      // Interpolated fragment texture coordinates (out)
void main()
{
    v_texcoord  = texcoord;
    gl_Position = <transform>;
}
"""

fragment_tex = """
uniform sampler2D u_texture;  // Texture 
varying vec2      v_texcoord; // Interpolated fragment texture coordinates (in)
void main()
{
    // Get texture color
    vec4 t_color = texture2D(u_texture, v_texcoord);
    // Final color
    gl_FragColor = t_color;
}
"""

parser = argparse.ArgumentParser(description='Track camera given an initial estimate')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/barcelona', help='path')
parser.add_argument('--start', type=int, default=0, help='Starting frame')
parser.add_argument('--end', type=int, default=-1, help='Ending frame')
parser.add_argument('--fps', type=int, default=15, help='Ending frame')

opt, _ = parser.parse_known_args()


filenames = np.genfromtxt(join(opt.path_to_data, 'youtube.txt'), dtype=str)
if opt.end == -1:
    opt.end = len(filenames)
filenames = filenames[opt.start:opt.end]

fps = opt.fps
n_frames = len(filenames)

trackball = Trackball(Position("position"), aspect=1, theta=45, phi=45, distance=100, zoom=135)

vertices_field, indices_field = plane()

field = gloo.Program(vertex_tex, fragment_tex)
field.bind(vertices_field)
field['position'] = vertices_field
field['u_texture'] = data.get(join(opt.path_to_data, 'texture.png'))
field['transform'] = trackball

all_programs = []

for fid, fname in enumerate(filenames):

    (basename, ext) = file_utils.extract_basename(fname)
    print('Loading model {0}/{1}: {2}'.format(fid, len(filenames), basename))

    path_to_pc = join(opt.path_to_data, 'scene3d')
    img = io.imread(join(path_to_pc, '{0}.jpg'.format(basename)), dtype=np.float32)

    vertices, indices = objload(join(path_to_pc, '{0}.obj'.format(basename)))
    vertices['texcoord'][:, 1] = 1.0-vertices['texcoord'][:, 1]

    tex_program = gloo.Program(vertex_tex, fragment_tex)
    tex_program.bind(vertices)
    tex_program['u_texture'] = img
    tex_program['transform'] = trackball

    all_programs.append(tex_program)

trackball.theta, trackball.phi, trackball.zoom = -10, 0, 15


window = app.Window(width=512, height=512, color=(0.30, 0.30, 0.35, 1.00))

time_counter = 0
play_or_pause = 0


@window.event
def on_draw(dt):

    global time_counter, n_frames, play_or_pause
    window.clear()
    field.draw(gl.GL_TRIANGLES, indices_field)
    all_programs[time_counter].draw(gl.GL_TRIANGLES)

    if time_counter == n_frames-1:
        time_counter = -1

    if play_or_pause:
        time_counter += 1

    time.sleep(1./fps)


@window.event
def on_init():
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)


@window.event
def on_character(text):
    global play_or_pause, time_counter, fps
    if text == ' ':
        if play_or_pause == 1:
            play_or_pause = 0
        else:
            play_or_pause = 1
    if text == '+':
        fps += 1

    if text == '-':
        fps -= 1

    if text == 'a':
        time_counter -= 1

    if text == 'd':
        time_counter += 1

    print(time_counter)


window.attach(field['transform'])
app.run()
