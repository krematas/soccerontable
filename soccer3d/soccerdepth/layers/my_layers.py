import torch
import torch.nn as nn
from torch.autograd import Function, Variable, gradcheck
import numpy as np


class Unproject3DMSE(nn.Module):
    def __init__(self):
        super(Unproject3DMSE, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input, target, mask, dist, A, R, T, pix, iteration):

        bs, _, _, _ = input.size()
        pix_ = pix.view([bs, 3, 128*128])
        d = input.view([bs, 1, 128*128]) + dist.repeat(1, 1, 128*128)
        dd = torch.cat((d, d, d), 1)

        result = torch.bmm(R, torch.mul(dd, torch.bmm(A, pix_)) - T.repeat(1, 1, 128*128))
        result = result.view([bs, 3, 128, 128])

        # if iteration == 1:
        #     tmp = (pix * mask.repeat(1, 3, 1, 1)).cpu().data.numpy()
        #     tmp2 = (target * mask.repeat(1, 3, 1, 1)).cpu().data.numpy()
        #     tmp3 = (input * mask).cpu().data.numpy()
        #     tmp4 = pix_.cpu().data.numpy()
        #     fix, ax = plt.subplots(3, 3)
        #     ax[0, 0].imshow(tmp2[0, 0, :, :])
        #     ax[0, 1].imshow(tmp2[0, 1, :, :])
        #     ax[0, 2].imshow(tmp2[0, 2, :, :])
        #     ax[1, 0].imshow(tmp[0, 0, :, :])
        #     ax[1, 1].imshow(tmp[0, 1, :, :])
        #     ax[1, 2].imshow(tmp[0, 2, :, :])
        #     ax[2, 0].imshow(tmp3[0, 0, :, :])
        #     ax[2, 1].plot(tmp4[0, 0, :], tmp4[0, 1, :])
        #     ax[2, 1].axis([0, 1920, 0, 1080])
        #     plt.show()

        self.loss = self.criterion(result, target * mask.repeat(1, 3, 1, 1))
        return self.loss


