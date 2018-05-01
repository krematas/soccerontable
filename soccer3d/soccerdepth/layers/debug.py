from torch.autograd import gradcheck, Variable
from soccerdepth.layers.losses import SoftmaxCrossEntropyLoss
import torch
import numpy as np
import utils.io as io
import torch.nn as nn

softmax = nn.Softmax()
logsoftmax = nn.LogSoftmax()

label = np.load('/home/krematas/Mountpoints/grail/data/play_for_data/cnn/train/labels/00000_r.npy')
depth = label[:, :, 0]
mask = label[:, :, 1]
nbins = 51
bins = np.linspace(-0.5, 0.5, nbins-2)
depth_q = (np.digitize(depth, bins)+1)*mask

h, w = mask.shape

I, J = (mask > 0).nonzero()
C = depth_q[I, J].ravel().astype(int)

index = (I*w*nbins + J*nbins + C).astype(int)
label_cube = np.zeros((h, w, nbins))+1.e-10
label_cube[I, J, C] = 1

label_cube_smoothed = np.zeros_like(label_cube)+1.e-10

import scipy.ndimage
import matplotlib.pyplot as plt

sigma = 1.5
for i, j in zip(I, J):
    # print(i, j)
    input_signal = label_cube[i, j, 1:]
    output_signal = scipy.ndimage.filters.gaussian_filter1d(input_signal, sigma)/(1./(np.sqrt(2*np.pi*sigma**2)))
    label_cube_smoothed[i, j, 1:] = output_signal

    # plt.plot(input_signal, 'b-')
    # plt.plot(output_signal, 'r-.')
    # break


# io.imagesc(depth_q)
# io.imagesc(np.argmax(label_cube, axis=2))
# io.imagesc(np.argmax(label_cube_smoothed, axis=2))


I, J = (mask == 0).nonzero()
label_cube_smoothed[I, J, 0] = 1
label_cube[I, J, 0] = 1

a = np.transpose(label_cube_smoothed, (2, 0, 1))[None, :, :, :]
b = np.transpose(label_cube_smoothed, (2, 0, 1))[None, :, :, :]

# gradchek takes a tuple of tensor as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.
c = a.copy()
c[0, 10:40, 40, 54] = a[0, 15:45, 40, 54]
plt.plot(a[0, :, 40, 54])
plt.plot(c[0, :, 40, 54])
plt.show()

input, label = Variable(torch.from_numpy(c).double(), requires_grad=True), Variable(torch.from_numpy(b).double())
# criterion = SoftmaxCrossEntropyLoss()
criterion = nn.MSELoss()

loss = criterion(input, label)
print(loss.backward())
print(loss.data[0])

# print(myloss.backward())
    # test = gradcheck(Linear(), input, eps=1e-6, atol=1e-4)
    # print(test)

# depth = np.random.rand(6, 1, 64, 64)
#
# pix = np.random.rand(6, 3, 64, 64)
# A_inv = np.random.rand(6, 1, 3, 3)
#
# pix_inv = np.random.rand(6, 3, 64*64)
# R_inv = np.random.rand(6, 3, 3)
# T = np.random.rand(6, 3, 1)
#
# bs, dim, sx, sy = output.size()
#
# A_inv_var = Variable(torch.from_numpy(A_inv).double(), requires_grad=False)
# pix_inv_var = Variable(torch.from_numpy(pix_inv).double(), requires_grad=False)
# R_inv_var = Variable(torch.from_numpy(R_inv).double(), requires_grad=False)
# T_var = Variable(torch.from_numpy(T).double(), requires_grad=False)
# depth_var = Variable(torch.from_numpy(depth).double(), requires_grad=True)
# pix_var = Variable(torch.from_numpy(pix).double(), requires_grad=True)
#
# output = torch.bmm(R_inv_var, depth_var.resize(bs, 1, sx*sy).repeat(1, 3, 1)*pix_inv_var - T_var.repeat(1, 1, sx*sy)).resize(bs, 3, sx, sy)
#
# input = (Variable(torch.from_numpy(a).double(), requires_grad=True), Variable(torch.from_numpy(b).double(), requires_grad=False),)