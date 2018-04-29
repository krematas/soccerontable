import numpy as np
import cv2


def robust_edge_detection(img):
    # Find edges
    kernel_size = 5
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    # io.imagesc(blur_gray)
    edges = cv2.Canny((blur_gray * 255).astype(np.uint8), 10, 200, apertureSize=5)
    # io.imagesc(edges)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(edges)[0]  # Position 0 of the returned tuple are the detected lines

    long_lines = []
    for j in range(lines.shape[0]):
        x1, y1, x2, y2 = lines[j, 0, :]
        if np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) > 50:
            long_lines.append(lines[j, :, :])

    lines = np.array(long_lines)
    edges = 1 * np.ones_like(img)
    drawn_img = lsd.drawSegments(edges, lines)
    edges = (drawn_img[:, :, 2] > 1).astype(np.float32)

    kernel = np.ones((7, 7), np.uint8)

    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    return edges


def get_pixel_ids(height, width):
    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    xx, yy = np.meshgrid(x, y)
    index = yy.ravel()*width+xx.ravel()
    return index


def get_pixel_neighbors(height, width):

    pix_id = []
    neighbor_id = []
    for i in range(height):
        for j in range(width):

            n = []
            if i == 0:
                n = n + [(i + 1) * width + j]
            elif i == height - 1:
                n = n + [(i - 1) * width + j]
            else:
                n = n + [(i + 1) * width + j, (i - 1) * width + j]

            if j == 0:
                n = n + [i * width + j + 1]
            elif j == width - 1:
                n = n + [i * width + j - 1]
            else:
                n = n + [i * width + j + 1, i * width + j - 1]

            for k in n:
                pix_id.append(i*width+j)
                neighbor_id.append(k)

    return pix_id, neighbor_id


def mask_aware_point_sampling(mask, n_points=50, pad=50):
    """ Select n_points from the image that do not overlap with the mask.
    :param path_to_data: 
    :param frame: 
    :param n_points: 
    :param pad: 
    :return: 
    """

    # img = io.imread(join(path_to_data, 'images', '%05d.jpg' % frame))
    # mask = io.imread(join(path_to_data, 'masks', '%05d_dcrf.png' % frame))[:, :, 0]

    h, w = mask.shape[0:2]

    kernel = np.ones((pad, pad), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    i = 0
    points = []
    while i < n_points:
        x = np.random.randint(pad, w-pad)
        y = np.random.randint(pad, h-pad)

        if mask[y, x] == 0:
            points.append([x, y])
            i += 1

    points = np.array(points)
    return points


# def main():
#     import scipy.io as sio
#
#     dataset = 'Barcelona-Atlentico-1'
#     path_to_data = file_utils.get_platform_path(join('Singleview/Soccer/', dataset))
#     filenames = np.loadtxt(join(path_to_data, 'youtube.txt'), dtype=str)[::10]
#
#     for fname in filenames:
#         basename, ext = file_utils.extract_basename(fname)
#         img = io.imread(fname)
#         sfactor = 0.5
#         img = cv2.resize(img, None, fx=sfactor, fy=sfactor)
#         edges = sio.loadmat(join(path_to_data, 'edges', '%s.mat' % basename))['E']
#         ret, edges = cv2.threshold(edges, 0.05, 1, cv2.THRESH_BINARY)
#
#         edges = robust_edge_detection(img)
#         io.imshow(edges)
#
# if __name__ == "__main__":
#     main()




