import torch
import torch.nn as nn


class G(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64):
        super(G, self).__init__()
        # 128 x 128
        self.conv1 = nn.Conv2d(input_nc, ngf, 4, 2, 1)
        # 64 x 64 x1
        self.conv2 = nn.Conv2d(ngf, ngf*2, 4, 2, 1)
        # 32 x 32 x 2
        self.conv3 = nn.Conv2d(ngf*2, ngf*4, 4, 2, 1)
        # 16 x 16 x 4
        self.conv4 = nn.Conv2d(ngf*4, ngf*8, 4, 2, 1)
        # 8 x 8 x 8
        self.conv5 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1)
        # 4 x 4 x 8 --------------------------------------------------------
        self.conv6 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1)
        # 2 x 2 x 8---------------------------------------------------     |
        self.conv7 = nn.Conv2d(ngf*8, ngf*8, 4, 2, 1)               # |    |
        # 1 x 1                                                       |    |
        self.dconv1 = nn.ConvTranspose2d(ngf*8, ngf*8, 4, 2, 1)     # |    |
        # 2 x 2 x 8---------------------------------------------------     |
        self.dconv2 = nn.ConvTranspose2d(ngf*8 * 2, ngf*8, 4, 2, 1)
        # 4 x 4 x 8 --------------------------------------------------------
        self.dconv3 = nn.ConvTranspose2d(ngf*8 * 2, ngf*8, 4, 2, 1)
        # 8 x 8 x 8
        self.dconv4 = nn.ConvTranspose2d(ngf*8 * 2, ngf*4, 4, 2, 1)
        # 16 x 16
        self.dconv5 = nn.ConvTranspose2d(ngf*4 * 2, ngf*2, 4, 2, 1)
        # 32 x 32
        self.dconv6 = nn.ConvTranspose2d(ngf*2 * 2, ngf, 4, 2, 1)
        # 64 x 64
        self.dconv7 = nn.ConvTranspose2d(ngf, output_nc, 4, 2, 1)
        # 128 x 128

        self.batch_norm = nn.BatchNorm2d(ngf)
        self.batch_norm2 = nn.BatchNorm2d(ngf*2)
        self.batch_norm4 = nn.BatchNorm2d(ngf*4)
        self.batch_norm8 = nn.BatchNorm2d(ngf*8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 128 x 128
        e1 = self.conv1(input)
        # state size is (ngf) x 64 x 64
        e2 = self.batch_norm2(self.conv2(self.leaky_relu(e1)))
        # state size is (ngf x 2) x 32 x 32
        e3 = self.batch_norm4(self.conv3(self.leaky_relu(e2)))
        # state size is (ngf x 4) x 16 x 16
        e4 = self.batch_norm8(self.conv4(self.leaky_relu(e3)))
        # state size is (ngf x 8) x 8 x 8
        e5 = self.batch_norm8(self.conv5(self.leaky_relu(e4)))
        # state size is (ngf x 8) x 4 x 4
        e6 = self.batch_norm8(self.conv6(self.leaky_relu(e5)))
        # state size is (ngf x 8) x 2 x 2
        # No batch norm on output of Encoder
        e7 = self.conv7(self.leaky_relu(e6))

        # Decoder
        # Deconvolution layers:
        # state size is (ngf x 8) x 1 x 1
        d1_ = self.dropout(self.batch_norm8(self.dconv1(self.relu(e7))))
        # state size is (ngf x 8) x 2 x 2
        d1 = torch.cat((d1_, e6), 1)
        d2_ = self.dropout(self.batch_norm8(self.dconv2(self.relu(d1))))
        # state size is (ngf x 8) x 4 x 4
        d2 = torch.cat((d2_, e5), 1)
        d3_ = self.dropout(self.batch_norm8(self.dconv3(self.relu(d2))))
        # state size is (ngf x 8) x 8 x 8
        d3 = torch.cat((d3_, e4), 1)
        d4_ = self.batch_norm4(self.dconv4(self.relu(d3)))
        # state size is (ngf x 8) x 16 x 16
        d4 = torch.cat((d4_, e3), 1)
        d5_ = self.batch_norm2(self.dconv5(self.relu(d4)))
        # state size is (ngf x 4) x 32 x 32
        d5 = torch.cat((d5_, e2), 1)
        d6 = self.batch_norm(self.dconv6(self.relu(d5)))
        # state size is (ngf x 2) x 64 x 64
        # d6 = torch.cat((d6_, e1), 1)
        d7 = self.dconv7(self.relu(d6))
        # state size is (ngf) x 128 x 128
        # output = self.tanh(d7)
        output = self.log_softmax(d7)
        # output = self.sigmoid(d7)
        # output = d7

        return output