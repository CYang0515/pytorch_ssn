import numpy as np
import argparse
import sys
from torch.utils import data
import torchvision.transforms as tf
from dataset import Dataset
from model import create_ssn_net, Loss
import torch
import os
import time
import logging
from tensorboardX import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'

class loss_logger():
    def __init__(self):
        self.loss = 0
        self.loss1 = 0
        self.loss2 = 0
        self.count = 0
    def add(self, l, l1, l2):
        self.loss += l
        self.loss1 +=l1
        self.loss2 +=l2
        self.count +=1
    def ave(self):
        self.loss /= self.count
        self.loss1 /= self.count
        self.loss2 /= self.count
    def clear(self):
        self.__init__()


def train_net(args, writer, dtype='train'):
    is_shuffle = dtype == 'train'
    dataloader = data.DataLoader(Dataset(num_spixel=100, patch_size=[200, 200], root=args.root_dir, dtype=dtype),
                                 batch_size=16, shuffle=is_shuffle, num_workers=4)

    # build model
    model = create_ssn_net(num_spixels=100, num_iter=args.num_steps, num_spixels_h=10, num_spixels_w=10, dtype=dtype)
    # loss function
    criten = Loss()

    device = torch.device('cpu')
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model.cuda()
        device = torch.device('cuda')
    optim = torch.optim.Adam(model.parameters(), lr=args.l_rate)

    if dtype == 'train' or dtype == 'test':
        if dtype == 'train':
            model.train()
            logger = loss_logger()
            for epoch in range(100000):
                logger.clear()
                for iter, [inputs, num_h, num_w, init_index, cir_index, p2sp_index_, invisible] in enumerate(dataloader):
                    with torch.autograd.set_detect_anomaly(True):
                        t0 = time.time()
                        img = inputs['img'].to(device)
                        label = inputs['label'].to(device)
                        problabel = inputs['problabel'].to(device)
                        num_h = num_h.to(device)
                        num_w = num_w.to(device)
                        init_index = [x.to(device) for x in init_index]
                        cir_index = [x.to(device) for x in cir_index]
                        p2sp_index_ = p2sp_index_.to(device)
                        invisible = invisible.to(device)

                        t1 = time.time()
                        recon_feat2, recon_label = model(img, p2sp_index_, invisible, init_index, cir_index, problabel, num_h, num_w, device)
                        loss, loss_1, loss_2 = criten(recon_feat2, img, recon_label, label)
                        t2 = time.time()

                        # optimizer
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        t3 = time.time()
                        print(f'epoch:{epoch}, iter:{iter}, total_loss:{loss}, pos_loss:{loss_1}, rec_loss:{loss_2}')
                        print(f'forward time:{t2-t1:.3f}, backward time:{t3-t2:.3f}, total time:{t3-t0:.3f}')
                        logger.add(loss.data, loss_1.data, loss_2.data)

                logger.ave()
                writer.add_scalar('train/total_loss', logger.loss, epoch)
                writer.add_scalar('train/pos_loss', logger.loss1, epoch)
                writer.add_scalar('train/rec_loss', logger.loss2, epoch)

                if epoch % 100 == 0 and epoch != 0:
                    torch.save(model.state_dict(), f'./checkpoints/checkpoints/{epoch}_{loss:.3f}_model.pt')
        else:
            pass

    else:
        pass










def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--l_rate', type=float, default=0.0001)
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--root_dir', type=str, default='/home/yc/ssn_superpixels/data')

    var_args = parser.parse_args()
    writer = SummaryWriter('log')
    train_net(var_args, writer)

if __name__ == '__main__':
    main()