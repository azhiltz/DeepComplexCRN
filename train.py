#coding: utf-8
#author: ltz
#date: 2021.1.9

from dc_crn import DCCRN
import wav_loader as loader 
import argparse
import os
import logging as logger
import torch
from torch.utils.data import DataLoader
import time 
import random

def _get_sub_paths( all_paths, index_list ):
    sub_paths = []
    for i in index_list:
        sub_paths.append(all_paths[i] )
    return sub_paths

def train(dns_home, mask_type="E", log_path="logs/", batch_size = 128, test_ratio=0.1, frame_dur=37.5, epochs=100, args = None ):
    #set data loader
    noisy_paths, clean_paths = loader.get_all_file_name( dns_home )
    if len(noisy_paths) < 1:
        logger.error("load data file wrong")
        return
    total_num = len(noisy_paths)
    tl = [x for x in range(0, total_num, 1)]

    #shuffle
    random.shuffle(tl)
    train_num = int((1.0-test_ratio)*total_num)
    test_num = total_num - train_num

    #split the dataset 
    train_idx = tl[:train_num-1]
    test_idx = tl[train_num:]
    train_noisy_paths = _get_sub_paths(noisy_paths, train_idx)
    train_clean_paths = _get_sub_paths(clean_paths, train_idx)
    test_noisy_paths = _get_sub_paths(noisy_paths, test_idx)
    test_clean_paths = _get_sub_paths(clean_paths, test_idx )
    
    train_dataset = loader.WavDataset( train_noisy_paths, train_clean_paths )
    test_dataset = loader.WavDataset( test_noisy_paths, test_clean_paths )
    train_loader = DataLoader( train_dataset, batch_size=batch_size, shuffle=True )
    test_loader = DataLoader( test_dataset, batch_size=batch_size, shuffle=True )
    
    #construct the model
    model = DCCRN(rnn_units=128,masking_mode=mask_type,use_clstm=True, kernel_size=5, kernel_num=[32, 64, 128, 256])
    model.to(torch.device("cuda:0"))

    optimizer = torch.optim.SGD(model.parameters(), args.lr )

    #train 
    for epoch in range(0, epochs):
        adjust_learning_rate( optimizer, epoch, args )

        train_epoch( train_loader, model, model.loss, optimizer, args )
        
        save_checkpoint( 
                    {
                    'epoch':epoch+1,
                    'state_dict':model.state_dict(),
                    'optimizer':optimizer.state.state_dict() 
                    }, True )


def train_epoch( train_loader, model, criterion, optimizer, args ):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses]
    )

    end = time.time()
    for i, ( x, y ) in enumerate(train_loader):
        data_time.update(time.time()-end)

        x = x.cuda(0, non_blocking=True)
        y = y.cuda(0, non_blocking=True)

        output = model(x)[1]
        loss = model.loss(outputs, y, loss_mode='SI-SNR')
        losses.update( loss.item(), x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dns_home", type=str, default="/home/ltz/Disk_B/train_disk/DNS-Challenge/datasets/training_set_sept12/" )
    parser.add_argument("--mask_type", type=str, default="E")
    parser.add_argument("--log_path", type=str, default="logs/")
    parser.add_argument("--test_ratio", type=float, default=0.1 )

    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    all_paras = parser.parse_args()

    if not os.path.exists(all_paras.log_path):
        os.mkdir(all_paras.log_path)
    
    train( all_paras.dns_home, all_paras.mask_type, all_paras.log_path, batch_size=all_paras.batch_size, args=all_paras )


