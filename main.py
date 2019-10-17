import os
import time
import random
import cv2
import numpy as np
from argparse import ArgumentParser

import torch
import torchvision.utils as vutils

from src.loss import l1_loss, inpainting_loss
from src.dataloader import get_dataloader
# from src.data_local_loader import data_loader, data_loader_with_split
from src.utils import compose, load_image, normalize, save_image

from models._count_params import count_parameters
from models.PConvUNet import PConvUNet, PConvUNetNew
from models.edgeconnect import InpaintGeneratorLight, InpaintUnet
from models.DilatedUnetResnet import DilatedUnetResnet, FuckNet
from models.baselinenet import Inpaint


use_nsml = True

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
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--bn_freeze_epoch', type=int, default=50)
    parser.add_argument('--lr_decay_epoch', type=int, default=2000)
    parser.add_argument('--bbox_epochs', default=[0,0])
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=40) # batchsize 8이면 36!
    parser.add_argument('--val_ratio', type=float, default=0.02)

    # model dependent
    parser.add_argument('--mask_channels', type=int, default=3)

    # parser.add_argument('--eval_every', type=int, default=713)  # 713
    parser.add_argument('--print_every', type=int, default=1000)  # 100
    parser.add_argument('--nickname', type=str, default='pconvnet_final')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--test_debug', type=bool, default=False)
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--load_epoch', type=int, default=0)

    # parser.add_argument()
    # reserved for nsml
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--iteration", type=str, default='0')
    parser.add_argument("--pause", type=int, default=0)
    args = parser.parse_args()
    # print(args)
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

    model = PConvUNetNew()

    print('################################################################')
    print('Total number of parameters * 4:', count_parameters(model) * 4)
    print('################################################################')
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0, 0.999))
    save, load = bind_nsml(model)
    if args.pause == 1:
        nsml.paused(scope=locals())

    # load
    current_epoch = 0
    if not use_nsml:
        writer = SummaryWriter(os.path.join('logs', args.nickname))
        if args.load:
            fname = os.path.join('checkpoints', args.nickname, 'model_%03d.pth' % args.load_epoch)
            state = torch.load(fname)
            model.load_state_dict(state['model'])
            current_epoch = args.load_epoch + 1
            print('loaded')

        path_test_data = 'data/test_data_original'
        test_loader = get_dataloader(root=os.path.join('data', 'test_data'), fnames=None, split='test',
                                     mask_channels=args.mask_channels, batch_size=args.batch_size,
                                     num_workers=args.num_workers)
        # test_loader = data_loader(root=os.path.join('data', 'test_data'), phase='test',
        #                           batch_size=args.batch_size, num_workers=args.num_workers)

    if args.mode == 'train':
        path_train = os.path.join(dir_data_root, 'train')
        path_train_data = os.path.join(dir_data_root, 'train', 'train_data')
        # tr_loader, val_loader = data_loader_with_split(path_train, train_split=(1 - args.val_ratio),
        #                                                batch_size=args.batch_size, num_workers=args.num_workers)

        # fold
        fnames = os.listdir(path_train_data)
        if args.debug:
            fnames = fnames[:1000]
        random.shuffle(fnames)
        train_fnames = fnames[:-int(len(fnames) * args.val_ratio)]
        val_fnames = fnames[-int(len(fnames) * args.val_ratio):]

        tr_loader = get_dataloader(path_train_data, train_fnames, 'train', args.mask_channels, args.batch_size, args.num_workers)
        val_loader = get_dataloader(path_train_data, val_fnames, 'val', args.mask_channels, args.batch_size, args.num_workers)
        print('train:', len(tr_loader) * args.batch_size, 'val:', len(val_loader) * args.batch_size)

        if args.test_debug:
            metric_eval = local_eval(model, test_loader, path_test_data)
            # metric_eval = local_eval(model, val_loader, path_train_data)
            return

        postfix = dict()
        total_step = 0
        start = time.time()

        best_val_loss = float('inf')

        for epoch in range(current_epoch, args.num_epochs):
            if epoch >= args.lr_decay_epoch:
                optim.param_groups[0]['lr'] = 0.0001

            if epoch >= args.bn_freeze_epoch:
                model.freeze_enc_bn = True
                optim.param_groups[0]['lr'] = 0.00005

            model.train()

            # if epoch < args.bbox_epochs[0]:
            #     bbox_constraint = 0.3
            # elif epoch < args.bbox_epochs[1]:
            #     bbox_constraint = 0.7
            # else:
            #     bbox_constraint = 1.0

            # tr_loader = get_dataloader(path_train_data, train_fnames, 'train', bbox_constraint, args.mask_channels, args.batch_size, args.num_workers)
            # val_loader = get_dataloader(path_train_data, val_fnames, 'val', bbox_constraint, args.mask_channels, args.batch_size, args.num_workers)
            # print('train:', len(tr_loader) * args.batch_size, 'val:', len(val_loader) * args.batch_size)

            for step, (fname, x_input, mask, x_GT) in enumerate(tr_loader):
                total_step += 1

                x_GT = x_GT.to(device)
                x_input = x_input.to(device)
                mask = mask.to(device)

                model.zero_grad()

                x_hat, _ = model(x_input, mask)  # PConvnet
                # x_mask = torch.cat([x_input, mask], dim=1)  #else
                # x_hat = model(x_mask)  #else

                # x_composed = compose(x_input, x_hat, mask)

                # loss = l1_loss(x_composed, x_GT)
                # loss = l1_loss(x_hat, x_GT)
                loss = inpainting_loss(x_hat, x_GT, mask)
                loss.backward()
                optim.step()

                if use_nsml:
                    postfix['loss'] = loss.item()
                    postfix['epoch'] = epoch
                    postfix['step_'] = step
                    postfix['total_step'] = total_step
                    postfix['steps_per_epoch'] = len(tr_loader)
                    nsml.report(**postfix, scope=locals(), step=total_step)

                if step % args.print_every == 0:
                    print("[%d/%d][%d/%d] time: %.2f, train_loss: %.6f, lr: %f" % (
                        epoch, args.num_epochs, step, len(tr_loader), time.time() - start,
                        loss.item(), optim.param_groups[0]['lr']))

            metric_eval = local_eval(model, val_loader, path_train_data)

            if use_nsml:
                postfix['metric_eval'] = metric_eval
                nsml.report(**postfix, scope=locals(), step=total_step)
            else:
                writer.add_scalar('train/metric_eval', metric_eval, epoch)
                writer.add_scalar('train/loss', loss.item(), epoch)

                # sample_dir = os.path.join('samples', args.nickname)
                # os.makedirs(sample_dir, exist_ok=True)
                # vutils.save_image(x_GT, os.path.join(sample_dir, 'x_GT_%03d.png' % epoch), normalize=True)
                # vutils.save_image(x_input, os.path.join(sample_dir, 'x_input_%03d.png' % epoch), normalize=True)
                # vutils.save_image(x_hat, os.path.join(sample_dir, 'x_hat_%03d.png' % epoch), normalize=True)
                # vutils.save_image(mask, os.path.join(sample_dir, 'mask_%03d.png' % epoch), normalize=True)
                # vutils.save_image(x_composed, os.path.join(sample_dir, 'x_composed_%03d.png' % epoch), normalize=True)
                # save_image(x_GT, os.path.join(sample_dir, 'x_GT_%03d.png' % epoch))
                # save_image(x_input, os.path.join(sample_dir, 'x_input_%03d.png' % epoch))
                # save_image(x_hat, os.path.join(sample_dir, 'x_hat_%03d.png' % epoch))
                # save_image(mask, os.path.join(sample_dir, 'x_mask_%03d.png' % epoch))
                # save_image(x_composed, os.path.join(sample_dir, 'x_composed_%03d_%.2f.png' % (epoch, metric_eval)))

            if use_nsml:
                if metric_eval < best_val_loss:
                    nsml.save(epoch)
                    best_val_loss = metric_eval
            else:
                checkpoint_dir = os.path.join('checkpoints', args.nickname)
                os.makedirs(checkpoint_dir, exist_ok=True)

                state = {'model': model.state_dict()}
                torch.save(state, os.path.join(checkpoint_dir, 'model_%03d.pth' % epoch))
                print('saved')


def _infer(model, root_path, test_loader=None):
    args = get_args()

    is_test = True if args.test_debug else False # False로 바꿔야대~~~~~~~~~~~~~~~~~~~~~~~~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if test_loader is None:
        is_test = True
        test_loader = get_dataloader(root=os.path.join(root_path, 'test_data'), fnames=None, split='test',
                                     mask_channels=args.mask_channels, batch_size=args.batch_size, num_workers=args.num_workers)
        # test_loader = data_loader(root=os.path.join(root_path, 'test_data'), phase='test',
        #                           batch_size=args.batch_size, num_workers=args.num_workers)

    x_hats = []
    fnames = []
    desc = 'infer...'
    model.eval()
    with torch.no_grad():
        for step, data in tqdm(enumerate(test_loader), desc=desc, total=len(test_loader), disable=use_nsml):
            if not is_test:
                fname, x_input, mask, _ = data
            else:
                fname, x_input, mask = data
            x_input = x_input.cuda()
            mask = mask.cuda()

            x_hat, _ = model(x_input, mask)  # PConvnet
            # x_mask = torch.cat([x_input, mask], dim=1)  # else
            # x_hat = model(x_mask)  # else

            x_hat = compose(x_input, x_hat, mask)

            # save_image(x_hat, os.path.join('test_output', 'x_hat_%03d.png' % step))
            # save_image(x_hat, os.path.join('val_output', 'x_hat_%03d.png' % step))

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
    print('local_eval', loss)
    return loss


def bind_nsml(model):
    def save(dir_name, *args, **kwargs):
        if not isinstance(dir_name, str):
            dir_name = str(dir_name)
        os.makedirs(dir_name, exist_ok=True)
        state = {
            'model': model.state_dict(),
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
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    if use_nsml:
        nsml.bind(save=save, load=load, infer=infer)
    return save, load


if __name__ == '__main__':
    main()
