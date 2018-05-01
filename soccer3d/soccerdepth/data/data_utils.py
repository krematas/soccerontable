import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils


upsampler = nn.UpsamplingBilinear2d(scale_factor=4)


def convert_test_prediction(batch_img, batch_mask, batch_label, batch_prediction):

    batch_img = batch_img.cpu().data[:, 0:3, :, :]
    batch_img[:, 0, :, :] = batch_img[:, 0, :, :] * 0.1159472 + 0.3402085
    batch_img[:, 1, :, :] = batch_img[:, 1, :, :] * 0.10461029 + 0.42575407
    batch_img[:, 2, :, :] = batch_img[:, 2, :, :] * 0.13433486 + 0.23771574
    batch_img = batch_img.numpy()

    batch_mask = batch_mask.data.cpu().numpy()

    batch_prediction = upsampler(batch_prediction)
    batch_prediction = np.argmax(batch_prediction.cpu().data.numpy(), axis=1)
    batch_prediction = batch_prediction * batch_mask

    batch_label = batch_label.data.cpu().numpy()

    img = np.transpose(batch_img[0, :, :, :], (1, 2, 0))
    prediction = np.transpose(batch_prediction[0, :, :, :], (1, 2, 0))
    label = np.transpose(batch_label[0, :, :, :], (1, 2, 0))
    return img, prediction[:, :, 0], label[:, :, 0], batch_mask


def image_logger_converter_visdom(batch_img, batch_mask, batch_label, batch_prediction, nlabels=51):
    batch_img = batch_img.cpu().data[:, 0:3, :, :]
    batch_img[:, 0, :, :] = batch_img[:, 0, :, :] * 0.1159472 + 0.3402085
    batch_img[:, 1, :, :] = batch_img[:, 1, :, :] * 0.10461029 + 0.42575407
    batch_img[:, 2, :, :] = batch_img[:, 2, :, :] * 0.13433486 + 0.23771574
    batch_img = batch_img.numpy()

    batch_mask = batch_mask.data.cpu().numpy()

    batch_prediction = upsampler(batch_prediction)
    batch_prediction = np.argmax(batch_prediction.cpu().data.numpy(), axis=1)
    batch_prediction = batch_prediction[:, None, :, :].astype(np.float32)/float(nlabels)
    batch_prediction = batch_prediction*batch_mask
    # batch_prediction = torch.from_numpy(batch_prediction*batch_mask)

    # batch_label = upsampler(batch_label)

    bs, dim, h, w = batch_label.size()
    if dim > 1:
        batch_label = upsampler(batch_label)
        batch_label = np.argmax(batch_label.cpu().data.numpy(), axis=1)
        batch_label = batch_label[:, None, :, :].astype(np.float32) / float(nlabels)
        batch_label = batch_label * batch_mask
    else:
        batch_label = batch_label.data.cpu().numpy() / nlabels
        batch_label = batch_label.astype(np.float32)

    return batch_img, batch_mask, batch_label, batch_prediction


def image_logger_converter(batch_img, batch_mask, batch_label, batch_prediction, nlabels=51):
    batch_img = batch_img.cpu().data[:, 0:3, :, :]
    batch_img[:, 0, :, :] = batch_img[:, 0, :, :] * 0.1159472 + 0.3402085
    batch_img[:, 1, :, :] = batch_img[:, 1, :, :] * 0.10461029 + 0.42575407
    batch_img[:, 2, :, :] = batch_img[:, 2, :, :] * 0.13433486 + 0.23771574
    x = vutils.make_grid(batch_img)

    batch_mask = batch_mask.data.cpu().numpy()

    batch_prediction = upsampler(batch_prediction)
    batch_prediction = np.argmax(batch_prediction.cpu().data.numpy(), axis=1)
    batch_prediction = batch_prediction[:, None, :, :].astype(np.float32)/float(nlabels)
    batch_prediction = torch.from_numpy(batch_prediction*batch_mask)
    y = vutils.make_grid(batch_prediction)

    # batch_label = upsampler(batch_label)

    bs, dim, h, w = batch_label.size()
    if dim > 1:
        batch_label = upsampler(batch_label)
        batch_label = np.argmax(batch_label.cpu().data.numpy(), axis=1)
        batch_label = batch_label[:, None, :, :].astype(np.float32) / float(nlabels)
        batch_label = torch.from_numpy(batch_label * batch_mask)
    else:
        batch_label = batch_label.data.cpu().numpy() / nlabels
        batch_label = torch.from_numpy(batch_label.astype(np.float32))
    z = vutils.make_grid(batch_label)

    return x, y, z


def show_batch(batch_img, batch_label, batch_mask):

    batch_img[:, 0, :, :] = batch_img[:, 0, :, :] * 0.1159472 + 0.3402085
    batch_img[:, 1, :, :] = batch_img[:, 1, :, :] * 0.10461029 + 0.42575407
    batch_img[:, 2, :, :] = batch_img[:, 2, :, :] * 0.13433486 + 0.23771574

    batch_images = batch_img.cpu().data.numpy()
    batch_labels = batch_label.cpu().data.numpy()
    batch_masks = batch_mask.cpu().data.numpy()
    imgs = np.transpose(batch_images, (0, 2, 3, 1))
    fig, ax_arr = plt.subplots(3, imgs.shape[0])
    for i in range(imgs.shape[0]):
        ax_arr[0, i].imshow(imgs[i, :, :, :])
        ax_arr[1, i].imshow(batch_labels[i, 0, :, :])
        ax_arr[2, i].imshow(batch_masks[i, 0, :, :])
    plt.show()





