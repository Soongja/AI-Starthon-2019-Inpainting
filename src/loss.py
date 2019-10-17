import numpy as np
import torch
import torch.nn as nn


def l1_loss(x_composed, x_GT):
    loss = torch.abs(x_composed - x_GT).sum(dim=[1, 2, 3]).mean()
    return loss


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = l1_loss(image[:, :, :, :-1], image[:, :, :, 1:]) + l1_loss(image[:, :, :-1, :], image[:, :, 1:, :])
    # loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
    #        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


def inpainting_loss(x_hat, x_GT, mask):
    # mask는 mask 쳐진 부분이 검은색, 나머지가 흰색
    # l1 = nn.L1Loss()
    l1 = l1_loss

    l1_hole = l1((1 - mask) * x_hat, (1 - mask) * x_GT)
    l1_valid = l1(mask * x_hat, mask * x_GT)

    x_composed = mask * x_GT + (1- mask) * x_hat

    tv = total_variation_loss(x_composed)

    # print('L1_hole:', 6 * l1_hole.item(), 'L1_valid:', l1_valid.item(), 'total_variation:', 0.1 * tv.item())
    return 6 * l1_hole + 1 * l1_valid + 0.1 * tv


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs).cuda()
            loss = self.criterion(outputs, labels)
            return loss
