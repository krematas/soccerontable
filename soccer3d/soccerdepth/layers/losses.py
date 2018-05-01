import torch.nn as nn
from torch.autograd import Function


class MaskedMSE(nn.Module):
    def __init__(self):
        super(MaskedMSE, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, input, target, mask):
        self.loss = self.criterion(input, target*mask)
        return self.loss


class MaskedSmoothL1(nn.Module):
    def __init__(self):
        super(MaskedSmoothL1, self).__init__()
        self.criterion = nn.SmoothL1Loss(size_average=True)

    def forward(self, input, target, mask):
        self.loss = self.criterion(input, target*mask)
        return self.loss


class MaskedBCE(nn.Module):
    def __init__(self):
        super(MaskedBCE, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, input, target, mask):
        self.loss = self.criterion(input, target*mask)
        return self.loss


# class DepthTo3D(Function):
#
#     def forward(self, input, pix_inv, R_inv, T):
#         self.save_for_backward(input, pix_inv, R_inv, T)
#         return torch.bmm(R_inv, input.resize(bs, 1, sx * sy).repeat(1, 3, 1) * pix_inv - T_var.repeat(1, 1,sx * sy)).resize(bs, 3, sx, sy)
#
#     def backward(self):
#
#         pix_inv, R_inv, T = self.saved_tensors
#
#         return grad_input,

softmax = nn.Softmax2d()


class SoftmaxCrossEntropyLoss(Function):

    def __init__(self):
        super(SoftmaxCrossEntropyLoss, self).__init__()

    # bias is an optional argument
    def forward(self, input, label):

        # loss -= target[i] * (log(softmax_output_data[i]) - log(target[i]));
        self.save_for_backward(input, label)
        bs, dim, h, w = input.size()
        n = bs*dim*h*w
        softmax_output = softmax(input).data

        output = label.mul(softmax_output.log()-label.log())
        output = -output.sum()/n
        return input.new((output,))

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        input, label = self.saved_tensors
        softmax_output = softmax(input).data
        bs, dim, h, w = input.size()
        grad_input = (softmax_output - label)/bs

        # print('grad output: {0}'.format(grad_input))
        return grad_input, None
