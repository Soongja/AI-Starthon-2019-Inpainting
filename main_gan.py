import os
import time
import random
import cv2
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torchvision.utils as vutils

from src.loss import l1_loss, inpainting_loss, AdversarialLoss
from src.dataloader import get_dataloader
from src.utils import compose, load_image, normalize

from models._count_params import count_parameters
from models.PConvUNet import PConvUNet, PConvUNetNew
from models.edgeconnect import InpaintGeneratorLight, InpaintUnet, Discriminator
from models.DilatedUnetResnet import DilatedUnetResnet

use_nsml = False

if use_nsml:
    import nsml
    dir_data_root = nsml.DATASET_PATH
else:
    from tensorboardX import SummaryWriter
    dir_data_root = 'data'

try:
    from tqdm import tqdm, trange
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x, desc='', **kwargs):
        if len(desc) > 0:
            print(desc, end=' ')
        return x

    def trange(x, desc='', **kwargs):
        if len(desc) > 0:
            print(desc, end=' ')
        return range(x)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--lr_decay_epoch', type=int, default=40)
    parser.add_argument('--bbox_epochs', default=[10,20])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--mask_channels', type=int, default=1)

    parser.add_argument('--eval_every', type=int, default=675)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--nickname', type=str, default='edgeconnect_dis_weak')

    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--load', type=bool, default=True)
    parser.add_argument('--load_epoch', type=int, default=0)

    # parser.add_argument()
    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)
    args = parser.parse_args()
    print(args)
    return args


def seed_everything():
    seed = 2019
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_everything()
    args = get_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    netG = InpaintGeneratorLight()
    netD = Discriminator()
    print('################################################################')
    print('Total number of parameters * 4:', (count_parameters(netG) + count_parameters(netD)) * 4)
    print('################################################################')
    netG = netG.to(device)
    netD = netD.to(device)


    optimG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.0, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=args.lr * 0.1, betas=(0.0, 0.999))
    save, load = bind_nsml(netG, optimG)
    if args.pause == 1:
        nsml.paused(scope=locals())

    adversarial_loss = AdversarialLoss()
    l1_loss = nn.L1Loss()

    # load
    current_epoch = 0
    if not use_nsml:
        writer = SummaryWriter(os.path.join('logs', args.nickname))
        if args.load:
            netG_name = os.path.join('checkpoints', args.nickname, 'netG_%03d.pth' % args.load_epoch)
            netD_name = os.path.join('checkpoints', args.nickname, 'netD_%03d.pth' % args.load_epoch)
            netG_dict = torch.load(netG_name)
            netD_dict = torch.load(netD_name)
            netG.load_state_dict(netG_dict['state_dict'])
            netD.load_state_dict(netD_dict['state_dict'])
            current_epoch = args.load_epoch + 1
            print('loaded')

    if args.mode == 'train':
        path_train = os.path.join(dir_data_root, 'train')
        path_train_data = os.path.join(dir_data_root, 'train', 'train_data')

        # fold
        fnames = os.listdir(path_train_data)
        if args.debug:
            fnames = fnames[:1000]
        random.shuffle(fnames)
        val_ratio = 0.1
        train_fnames = fnames[:-int(len(fnames) * val_ratio)]
        val_fnames = fnames[-int(len(fnames) * val_ratio):]


        postfix = dict()
        total_step = 0
        start = time.time()
        # for epoch in trange(args.num_epochs, disable=use_nsml):
        for epoch in range(current_epoch, args.num_epochs):
            if epoch < args.bbox_epochs[0]:
                bbox_constraint = 0.25
            elif epoch < args.bbox_epochs[1]:
                bbox_constraint = 0.75
            else:
                bbox_constraint = 1.0

            tr_loader = get_dataloader(path_train_data, train_fnames, 'train', bbox_constraint, args.mask_channels, args.batch_size, args.num_workers)
            val_loader = get_dataloader(path_train_data, val_fnames, 'val', bbox_constraint, args.mask_channels, args.batch_size, args.num_workers)
            print('train:', len(tr_loader) * args.batch_size, 'val:', len(val_loader) * args.batch_size)

            # if epoch >= args.lr_decay_epoch:
            #     optim.param_groups[0]['lr'] *= 0.1


            pbar = tqdm(enumerate(tr_loader), total=len(tr_loader), disable=True)
            for step, (_, x_input, mask, x_GT) in pbar:
                total_step += 1

                x_input = x_input.to(device)
                mask = mask.to(device)
                x_GT = x_GT.to(device)

                x_mask = torch.cat([x_input, mask], dim=1)
                x_hat = netG(x_mask)
                x_composed = compose(x_input, x_hat, mask)

                ###########################################
                # update D network
                ###########################################
                netD.zero_grad()

                netD_real = netD(x_GT)
                net_D_real_loss = adversarial_loss(netD_real, True)

                netD_fake = netD(x_hat)
                netD_fake_loss = adversarial_loss(netD_fake, False)

                netD_loss = net_D_real_loss + netD_fake_loss
                netD_loss.backward(retain_graph=True)
                optimD.step()

                ###########################################
                # update G network
                ###########################################
                netD.zero_grad()

                netG_fake = netD(x_hat)  #.view(-1) 해야할 수도
                netG_fake_loss = adversarial_loss(netG_fake, True) * 0.1

                # netG_L1_loss = inpainting_loss(x_hat, x_GT, mask)
                netG_L1_loss = l1_loss(x_hat, x_GT) / torch.mean(mask)

                netG_loss = netG_fake_loss + netG_L1_loss
                netG_loss.backward()
                optimG.step()

                postfix['netD_loss'] = netD_loss.item()
                postfix['netG_loss'] = netG_loss.item()
                postfix['epoch'] = epoch
                postfix['step_'] = step
                postfix['total_step'] = total_step
                postfix['steps_per_epoch'] = len(tr_loader)

                if step != 0 and step % (args.eval_every - 1) == 0:
                    metric_eval = local_eval(netG, val_loader, path_train_data)
                    postfix['metric_eval'] = metric_eval
                    print('metric eval:', metric_eval)

                    if not use_nsml:
                        sample_dir = os.path.join('samples', args.nickname)
                        os.makedirs(sample_dir, exist_ok=True)
                        vutils.save_image(x_GT, os.path.join(sample_dir, 'x_GT_%03d.png' % epoch), normalize=True)
                        vutils.save_image(x_input, os.path.join(sample_dir, 'x_input_%03d.png' % epoch), normalize=True)
                        vutils.save_image(x_hat, os.path.join(sample_dir, 'x_hat_%03d.png' % epoch), normalize=True)
                        vutils.save_image(mask, os.path.join(sample_dir, 'mask_%03d.png' % epoch), normalize=True)
                        vutils.save_image(x_composed, os.path.join(sample_dir, 'x_composed_%03d_%.1f.png' % (epoch, metric_eval)), normalize=True)
                        writer.add_scalar('train/netD_loss', netD_loss.item(), epoch)
                        writer.add_scalar('train/netG_loss', netG_loss.item(), epoch)

                if step % args.print_every == 0:
                    print("[%d/%d][%d/%d] time: %.2f,"
                          "netG_gan_loss: %.2f, netG_L1_loss: %.2f, netD_loss: %.2f" % (
                        epoch, args.num_epochs, step, len(tr_loader), time.time() - start,
                        netG_fake_loss.item(), netG_L1_loss.item(), netD_loss.item()))

                if use_nsml:
                    nsml.report(**postfix, scope=locals(), step=total_step)

            if use_nsml:
                nsml.save(epoch)
            else:
                checkpoint_dir = os.path.join('checkpoints', args.nickname)
                os.makedirs(checkpoint_dir, exist_ok=True)

                netG_dict = {'state_dict': netG.state_dict()}
                netD_dict = {'state_dict': netD.state_dict()}
                torch.save(netG_dict, os.path.join(checkpoint_dir, 'netG_%03d.pth' % epoch))
                torch.save(netD_dict, os.path.join(checkpoint_dir, 'netD_%03d.pth' % epoch))
                print('saved')


def _infer(model, root_path, test_loader=None):
    args = get_args()

    is_test = False
    if test_loader is None:
        is_test = True
        test_loader = get_dataloader(root=os.path.join(root_path, 'test_data'), fnames=None, split='test', bbox_constraint=None,
                                     mask_channels=args.mask_channels, batch_size=args.batch_size, num_workers=args.num_workers)

    x_hats = []
    fnames = []
    desc = 'infer...'
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, desc=desc, total=len(test_loader), disable=not use_nsml):
            if not is_test:
                fname, x_input, mask, _ = data
            else:
                fname, x_input, mask = data
            x_input = x_input.cuda()
            mask = mask.cuda()
            x_mask = torch.cat([x_input, mask], dim=1)
            x_hat = model(x_mask)
            x_hat = compose(x_input, x_hat, mask)
            x_hats.append(x_hat.cpu())
            fnames = fnames + list(fname)

    x_hats = torch.cat(x_hats, dim=0)

    return fnames, x_hats


def read_prediction_gt(dname, fnames):
    images = []
    for fname in fnames:
        fname = os.path.join(dname, fname)
        image = load_image(fname)
        image = normalize(image)
        images.append(image)
    return torch.stack(images, dim=0)


def local_eval(model, test_loader, path_GT):
    fnames, x_hats = _infer(model, None, test_loader=test_loader)
    x_GTs = read_prediction_gt(path_GT, fnames)
    loss = float(l1_loss(x_hats, x_GTs))
    # print('local_eval', loss)
    return loss


def bind_nsml(model, optimizer):
    def save(dir_name, *args, **kwargs):
        if not isinstance(dir_name, str):
            dir_name = str(dir_name)
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        fname = os.path.join(dir_name, 'model.pth')
        torch.save(state, fname)
        print('saved')

    def load(dir_name, *args, **kwargs):
        if not isinstance(dir_name, str):
            dir_name = str(dir_name)
        fname = os.path.join(dir_name, 'model.pth')
        state = torch.load(fname)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    if use_nsml:
        nsml.bind(save=save, load=load, infer=infer)
    return save, load


if __name__ == '__main__':
    main()
