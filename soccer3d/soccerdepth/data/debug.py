from os.path import join
from soccerdepth.data.dataset_loader import get_set
from torch.utils.data import DataLoader
from soccerdepth.data.data_utils import show_batch
from torch.autograd import Variable
import argparse
import warnings
import utils.files as file_utils
import numpy as np

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Depth AutoEncoder')
parser.add_argument('--dataset', default='play_for_data', help='facades')
parser.add_argument('--batchSize', type=int, default=6, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=6, help='testing batch size')
parser.add_argument('--input_nc', type=int, default=4, help='input image channels')
parser.add_argument('--output_nc', type=int, default=51, help='output image channels')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
opt, _ = parser.parse_known_args()

path_to_data = join(file_utils.get_platform_datadir(opt.dataset), 'cnn2')

print('===> Loading datasets')
train_set = get_set(path_to_data, splitdir='train', nbins=opt.output_nc)
test_set = get_set(path_to_data,  splitdir='test', nbins=opt.output_nc)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=6, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('start!')
for iteration, batch in enumerate(training_data_loader, 1):
    input, target, mask = Variable(batch['image']).float(), Variable(batch['target']).long(), Variable(batch['mask']).float()
    show_batch(input, target, mask)
    break
    data = input.data.numpy()/255.

    means = []
    stdevs = []
    for i in range(3):
        pixels = data[:, i, :, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    print("means: {}".format(means))
    print("stdevs: {}".format(stdevs))
    print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
    break
