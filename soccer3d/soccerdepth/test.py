import torch
import torch.nn as nn
from soccer3d.soccerdepth.models.hourglass import hg8
from torch.autograd import Variable
import utils.files as file_utils
from soccer3d.soccerdepth.data.dataset_loader import get_set
from torch.utils.data import DataLoader
import argparse
import warnings
from os.path import join
from soccer3d.soccerdepth.data.data_utils import convert_test_prediction
import numpy as np
from soccer3d.soccerdepth.data.transforms import *
from torchvision import transforms
from visdom import Visdom
from tqdm import tqdm


warnings.filterwarnings("ignore")


# Testing settings
parser = argparse.ArgumentParser(description='Depth estimation using Stacked Hourglass')
parser.add_argument('--path_to_data', default='/home/krematas/Mountpoints/grail/data/Multiview/Portland/b6/players', help='path')
parser.add_argument('--dataset', default='a6', help='path')


parser.add_argument('--modelpath', type=str, default='/home/krematas/Mountpoints/grail/tmp/cnn/model.pth', help='model file to use')
parser.add_argument('--epoch', type=int, default=292, help='testing batch size')
parser.add_argument('--input_nc', type=int, default=4, help='input image channels')
parser.add_argument('--output_nc', type=int, default=51, help='output image channels')
parser.add_argument('--img_size', type=int, default=256, help='output image channels')
parser.add_argument('--label_size', type=int, default=64, help='output image channels')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--additional_input_type', default='estmask', choices=['estmask', 'trimap'], help='The type of addtional type to load [estmask, trimap]')


opt, _ = parser.parse_known_args()
opt.cuda = True

viz = Visdom()
win0 = viz.images(np.ones((1, 3, 256, 256)))
win1 = viz.images(np.ones((1, 3, 256, 256)))
win2 = viz.images(np.ones((1, 3, 256, 256)))

print(opt)

checkpoint = torch.load(opt.modelpath)
netG_state_dict = checkpoint['state_dict']
netG = hg8(input_nc=opt.input_nc, output_nc=opt.output_nc)
netG.load_state_dict(netG_state_dict)
if opt.cuda:
    netG.cuda()

path_to_data = opt.path_to_data
file_utils.mkdir(join(path_to_data, 'predictions'))

composed = transforms.Compose([Rescale(opt.img_size, opt.label_size), ToTensor(), NormalizeImage()])
test_set = get_set(path_to_data, nbins=opt.output_nc, transform=composed, additional_input_type=opt.additional_input_type)
testing_data_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=1, shuffle=False)

logsoftmax = nn.LogSoftmax()

for iteration, batch in enumerate(tqdm(testing_data_loader)):

    input, target, mask = Variable(batch['image']).float(), Variable(batch['target']).long(), Variable(batch['mask']).float()

    if opt.input_nc > 3:
        input = torch.cat((input, mask), 1)

    if opt.cuda:
        input = input.cuda()
        target = target.cuda()

    output = netG(input)
    final_prediction = logsoftmax(output[-1])

    img, prediction, label, mask = convert_test_prediction(input, mask, target, final_prediction)

    viz.image(
        img.transpose(2, 0, 1)*mask[0, :, :, :],
        win=win0,
        opts=dict(title='Testing {0}: Input image'.format(opt.dataset))
    )
    viz.heatmap(
        prediction[::-1, :],
        win=win1,
        opts=dict(title='Testing {0}: Depth estimation'.format(opt.dataset))
    )
    viz.heatmap(
        label[::-1, :],
        win=win2,
        opts=dict(title='Testing {0}: Label'.format(opt.dataset))
    )

    # Save predictions
    fname = testing_data_loader.dataset.image_filenames[iteration]
    basename, ext = file_utils.extract_basename(fname)
    np.save(join(path_to_data, 'predictions', basename), final_prediction.cpu().data.numpy())
