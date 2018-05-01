from soccerdepth.data.dataset_loader import get_set
import numpy as np
import utils.files as file_utils
from os.path import join
import argparse
from soccerdepth.models.hourglass import  hg8
from soccerdepth.models.utils import weights_init
from soccerdepth.data.data_utils import image_logger_converter_visdom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from soccerdepth.data.transforms import *
from torchvision import transforms
import warnings
from visdom import Visdom

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Depth AutoEncoder')
parser.add_argument('--dataset', default='', help='Dataset to train on')
parser.add_argument('--batchSize', type=int, default=6, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=2, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=4, help='input image channels')
parser.add_argument('--output_nc', type=int, default=51, help='output image channels')
parser.add_argument('--nf', type=int, default=64, help='number of filters')
parser.add_argument('--lr', type=float, default=0.00003, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--run', type=int, default=44, help='Run number for tensorboard')
parser.add_argument('--output_path', default='/home/krematas/Mountpoints/grail/CNN', help='Where the files WILL be stored')
parser.add_argument('--modeldir', default='/home/krematas/Mountpoints/grail/CNN', help='Where the files ARE being stored')
parser.add_argument('--additional_input', default='mask', help='filepostfix')
parser.add_argument('--postfix', default='hg_estmask', help='filepostfix')
parser.add_argument('--resume', type=int, default=0, help='Resume training')
parser.add_argument('--port', type=int, default=9876, help='Run number for tensorboard')
parser.add_argument('--additional_input_type', default='estmask', help='The type of addtional type to load [estmap, trimap]')


opt, _ = parser.parse_known_args()
opt.cuda = True
print(opt)

# Initiate 5 windows
viz = Visdom(env='FIFA CNN training', port=opt.port)
viz.close()
win0 = viz.images(np.ones((4, 3, 128, 128)))
win1 = viz.images(np.ones((4, 3, 128, 128)))
win2 = viz.images(np.ones((4, 3, 128, 128)))
win3 = viz.images(np.ones((4, 3, 128, 128)))
epoch_lot = viz.line(X=torch.zeros((1,)).cpu(), Y=torch.zeros((1,)).cpu(),
                     opts=dict(
                            xlabel='Epoch',
                            ylabel='Loss',
                            title='Epoch Training Loss',
                            legend=['Loss'])
                    )
lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current Training Loss',
                legend=['Loss']
            )
        )


# writer = SummaryWriter("runs/run%d" % opt.run)


if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)


dataset = 'play_for_data'
path_to_data = join(file_utils.get_platform_datadir(dataset), 'cnn2')

print('===> Loading datasets')
composed = transforms.Compose([RandomRotation(), RandomCrop(), Rescale(256, 64), ColorOffset(), ToTensor(), NormalizeImage()])
train_set = get_set(join(path_to_data, 'train'), nbins=opt.output_nc, transform=composed, additional_input_type=opt.additional_input_type)

composed = transforms.Compose([Rescale(256, 64), ToTensor(), NormalizeImage()])
test_set = get_set(join(path_to_data, 'test'), nbins=opt.output_nc, transform=composed, additional_input_type=opt.additional_input_type)

training_data_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = hg8(input_nc=opt.input_nc, output_nc=opt.output_nc)
model.apply(weights_init)

print('===> The loss function is cross entropy loss')
class_weights = np.ones((opt.output_nc, ))
class_weights[0] = 0.1
weights = torch.from_numpy(class_weights)
weights = torch.FloatTensor(weights.size()).copy_(weights).cuda()

criterion = nn.NLLLoss2d(weight=weights)
logsoftmax = nn.LogSoftmax()


# Setup the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.003)


# Resume if available
if opt.resume > 0:
    checkpoint = torch.load(join(opt.modeldir, 'model_epoch_%d_%s_%s.pth' % (opt.resume, opt.additional_input, opt.postfix)))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


if opt.cuda:
    model = model.cuda()
    criterion = criterion.cuda()


def train(epoch):

    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, mask = Variable(batch['image']).float(), Variable(batch['target']).long(), Variable(batch['mask']).float()
        if opt.input_nc > 3:
            input = torch.cat((input, mask), 1)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        loss = criterion(logsoftmax(output[0]), target.squeeze())
        for j in range(1, len(output)):
            loss += criterion(logsoftmax(output[j]), target.squeeze())

        epoch_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

        if iteration % 50 == 0:
            prediction = logsoftmax(output[-1])

            x, y, z, w = image_logger_converter_visdom(input, mask, target, prediction)

            viz.images(
                x,
                win=win0,
            )
            viz.images(
                y,
                win=win1,
            )
            viz.images(
                w,
                win=win2,
            )
            viz.images(
                z,
                win=win3,
            )

            # writer.add_image('Train_image', x, epoch)
            # writer.add_image('Train_prediction', y, epoch)
            # writer.add_image('Train_label', z, epoch)

        # print(torch.ones((1,)).cpu().size())
        # print(torch.Tensor([loss.data[0]]).unsqueeze(0).cpu().size())
        viz.line(
            X=torch.ones((1, 1)).cpu() * ((epoch-1)*len(training_data_loader) + iteration),
            Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
            win=lot,
            update='append'
        )

        # hacky fencepost solution for 0th epoch plot
        if epoch == 1 and iteration == len(training_data_loader)-1:
            viz.line(
                X=torch.zeros((1, 1)).cpu(),
                Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                win=epoch_lot,
                update=True
            )

    viz.line(
        X=torch.ones((1, 1)).cpu()*epoch,
        Y=torch.Tensor([epoch_loss/len(training_data_loader)]).unsqueeze(0).cpu(),
        win=epoch_lot,
        update='append'
    )
    print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss / len(training_data_loader)


def test():

    epoch_loss = 0
    for iteration, batch in enumerate(testing_data_loader):
        input, target, mask = Variable(batch['image']).float(), Variable(batch['target']).long(), Variable(batch['mask']).float()

        if opt.input_nc > 3:
            input = torch.cat((input, mask), 1)

        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        output = model(input)

        loss = 0
        for o in output:
            loss += criterion(logsoftmax(o), target.squeeze())

        epoch_loss += loss.data[0]

        if iteration == 1:
            prediction = logsoftmax(output[-1])

            x, y, z, w = image_logger_converter_visdom(input, mask, target, prediction)

            viz.images(
                x,
                win=win0,
            )
            viz.images(
                y,
                win=win1,
            )
            viz.images(
                w,
                win=win2,
            )
            viz.images(
                z,
                win=win3,
            )

    print("===> Validation Complete: Avg. Loss: {:.6f}".format(epoch_loss / len(testing_data_loader)))
    return epoch_loss / len(testing_data_loader)


def checkpoint(epoch):
    model_out_path = "{0}/model_epoch_{1}_{2}_{3}.pth".format(opt.output_path, epoch, opt.additional_input, opt.postfix)
    dict_to_save = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(dict_to_save, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(opt.resume+1, opt.nEpochs + 1):
    v1 = train(epoch)
    v2 = test()
    writer.add_scalar('train_loss', v1, epoch)
    writer.add_scalar('val_loss', v2, epoch)
    checkpoint(epoch)

writer.close()
