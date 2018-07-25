import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import face, AnnotationTransform, Detection, detection_collate
from utils.augmentations import PyramidAugmentation
from layers.modules import MultiBoxLoss

from pyramid import build_sfd,SFD,SSHContext,ContextTexture
import numpy as np
import time
from layers import *


os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--annoPath', default="/home/guoqiushan/share/face_box/wider.list", help='Location of wider face')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = face

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
# train_sets = 'train'
ssd_dim = 640  # only support 300 now
means = (104, 117, 123)  # only support voc now
num_classes = 1 + 1
batch_size = args.batch_size
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
max_iter = 120000
weight_decay = 0.0001
stepvalues = (80000, 100000, 120000)
gamma = 0.1
momentum = 0.9

if args.visdom:
    import visdom
    viz = visdom.Visdom()

ssd_net = build_sfd('train', 640, num_classes)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True


if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if 'bias' in m.state_dict().keys():
            m.bias.data.zero_()

    if isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        if 'bias' in m.state_dict().keys():
            m.bias.data.zero_()

    if isinstance(m, nn.BatchNorm2d):
        m.weight.data[...] = 1
        m.bias.data.zero_()


for layer in net.modules():
    layer.apply(weights_init)

if not args.resume:
    print('Initializing weights...')
    
if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    pass
    
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 3, 0.35, False, False, args.cuda)
criterion1 = MultiBoxLoss(num_classes, 0.35, True, 0, True, 3, 0.35, False, True, args.cuda)

def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    dataset = Detection(args.annoPath, PyramidAugmentation(ssd_dim, means), AnnotationTransform())

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    for iteration in range(args.start_iter, max_iter):
        t0 = time.time()
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        t1 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(tuple(out[0:3]), targets)
        loss_l_head, loss_c_head = criterion(tuple(out[3:6]), targets)
        
        loss = loss_l + loss_c + 0.5 * loss_l_head + 0.5 * loss_c_head
        optimizer.step()
        t2 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if iteration % 10 == 0:
            print('front and back Timer: {} sec.' .format((t2 - t1)))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]))
            print('Loss conf: {} Loss loc: {}'.format(loss_c.data[0],loss_l.data[0]))
            print('Loss head conf: {} Loss head loc: {}'.format(loss_c_head.data[0],loss_l_head.data[0]))
            print('lr: {}'.format(optimizer.param_groups[0]['lr']))
            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())
        if args.visdom:
            viz.line(
                X=torch.ones((1, 3)).cpu() * iteration,
                Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                    loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                win=lot,
                update='append'
            )
            # hacky fencepost solution for 0th epoch plot
            if iteration == 0:
                viz.line(
                    X=torch.zeros((1, 3)).cpu(),
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu(),
                    win=epoch_lot,
                    update=True
                )
        if iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), args.save_folder + 'Res50_pyramid_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), args.save_folder + 'Res50_pyramid' + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma


if __name__ == '__main__':
    train()
