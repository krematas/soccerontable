import torch
import torch.nn as nn


class ColorizeNet(nn.Module):
    def __init__(self, input_nc, output_nc, nf=64):
        super(ColorizeNet, self).__init__()
        # Input: 128 x 128 x 3
        self.conv1a = nn.Conv2d(input_nc, nf, 3, 1, 1)
        # 128 x 128 x 64
        self.conv1b = nn.Conv2d(nf, nf, 3, 2, 1)
        # 64 x 64 x 64
        self.conv2a = nn.Conv2d(nf, nf*2, 3, 1, 1)
        # 64 x 64 x 128
        self.conv2b = nn.Conv2d(nf*2, nf*2, 3, 2, 1)
        # 32 x 32 x 128
        self.conv3a = nn.Conv2d(nf*2, nf*4, 3, 1, 1)
        # 32 x 32 x 256
        self.conv3b = nn.Conv2d(nf*4, nf*4, 3, 2, 1)
        # 16 x 16 x 256
        self.conv4a = nn.Conv2d(nf*4, nf*8, 3, 1, 1)
        # 16 x 16 x 512
        self.conv4b = nn.Conv2d(nf*8, nf*8, 3, 1, 1)
        # 16 x 16 x 512
        self.conv5a = nn.Conv2d(nf*8, nf*8, 3, 1, 2, 2)
        # 16 x 16 x 512
        self.conv5b = nn.Conv2d(nf*8, nf*8, 3, 1, 2, 2)
        # 16 x 16 x 512
        self.conv6a = nn.Conv2d(nf*8, nf*8, 3, 1, 2, 2)
        # 16 x 16 x 512
        self.conv6b = nn.Conv2d(nf*8, nf*8, 3, 1, 2, 2)
        # 16 x 16 x 256
        self.conv7a = nn.Conv2d(nf*8, nf*8, 3, 1, 1)
        # 16 x 16 x 512
        self.conv7b = nn.Conv2d(nf*8, nf*4, 3, 1, 1)
        # 16 x 16 x 256
        self.conv8a = nn.ConvTranspose2d(nf*4*2, nf*4, 4, 2, 1)
        # 32 x 32 x 256
        self.conv8b = nn.Conv2d(nf*4, nf*2, 3, 1, 1)
        # 32 x 32 x 128
        self.conv9a = nn.ConvTranspose2d(nf*2*2, nf*2, 4, 2, 1)
        # 64 x 64 x 128
        self.conv9b = nn.Conv2d(nf*2, nf, 3, 1, 1)
        # 64 x 64 x 64
        self.conv10a = nn.ConvTranspose2d(nf*2, nf, 4, 2, 1)
        # 128 x 128 x 64
        self.conv10b = nn.Conv2d(nf, output_nc, 3, 1, 1)
        # 128 x 128 x 64
        # self.conv10c = nn.Conv2d(64, output_nc, 3, 1, 1)

        self.batch_norm1 = nn.BatchNorm2d(nf)
        self.batch_norm2 = nn.BatchNorm2d(nf * 2)
        self.batch_norm3 = nn.BatchNorm2d(nf * 4)
        self.batch_norm4 = nn.BatchNorm2d(nf * 8)
        self.batch_norm5 = nn.BatchNorm2d(nf * 8)
        self.batch_norm6 = nn.BatchNorm2d(nf * 8)
        self.batch_norm7 = nn.BatchNorm2d(nf * 4)
        self.batch_norm8 = nn.BatchNorm2d(nf * 2)
        self.batch_norm9 = nn.BatchNorm2d(nf * 1)
        # self.batch_norm10 = nn.BatchNorm2d(64 * 1)

        self.log_softmax = nn.LogSoftmax()
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 128 x 128
        e1 = self.batch_norm1(self.relu(self.conv1b(self.relu(self.conv1a(input)))))
        # 64 x 64 x 64
        e2 = self.batch_norm2(self.relu(self.conv2b(self.relu(self.conv2a(e1)))))
        # 32 x 32 x 128
        e3 = self.batch_norm3(self.relu(self.conv3b(self.relu(self.conv3a(e2)))))
        # 16 x 16 x 256
        e4 = self.batch_norm4(self.relu(self.conv4b(self.relu(self.conv4a(e3)))))
        # 16 x 16 x 512
        e5 = self.batch_norm5(self.relu(self.conv5b(self.relu(self.conv5a(e4)))))
        # 16 x 16 x 512
        e6 = self.batch_norm6(self.relu(self.conv6b(self.relu(self.conv6a(e5)))))
        # 16 x 16 x 512
        e7 = self.batch_norm7(self.relu(self.conv7b(self.relu(self.conv7a(e6)))))
        # 16 x 16 x 256

        e8_ = torch.cat((e7, e3), 1)
        e8 = self.batch_norm8(self.relu(self.conv8b(self.relu(self.conv8a(e8_)))))
        e9_ = torch.cat((e8, e2), 1)
        e9 = self.batch_norm9(self.relu(self.conv9b(self.relu(self.conv9a(e9_)))))
        e10_ = torch.cat((e9, e1), 1)
        e10 = self.conv10b(self.relu(self.conv10a(e10_)))

        output = self.log_softmax(e10)
        return output