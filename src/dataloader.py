import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.utils import cutout


class MyDataset(Dataset):
    def __init__(self, root, fnames, mask_channels, augment=False, transform=None):
        self.root = root
        self.fnames = fnames
        self.mask_channels = mask_channels
        self.augment = augment
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        GT = Image.open(os.path.join(self.root, fname)).convert('RGB')

        # if self.augment:
        #     augmentation
            # aug = transforms.Compose([
            #                           transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0),
                                      # transforms.RandomResizedCrop(size=128, scale=(0.9, 1.0), ratio=(0.9, 1.1))
                                      # ])
            # GT = aug(GT)

        if self.transform is not None:
            GT = self.transform(GT)

        masked, mask = cutout(GT)

        if self.mask_channels == 1:
            mask = mask[:1, :, :] # 0과 1 값

        # print('masked:', np.unique(masked.numpy()))  # -1~1, 마스크영역 0
        # print('mask:', np.unique(mask.numpy()))  # 0,1
        # print('GT:', np.unique(GT.numpy()))  # -1~1

        return fname, masked, mask, GT


class TestDataset(Dataset):
    def __init__(self, root_dir, mask_channels, transform=None):
        self.root_dir = root_dir
        self.mask_channels = mask_channels
        self.transform = transform
        self.dname_masked = os.path.join(root_dir, 'masked')
        self.dname_mask = os.path.join(root_dir, 'mask')
        self.samples = sorted(os.listdir(self.dname_masked))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (masked, mask)
        """
        fname = self.samples[index]
        fname_masked = os.path.join(self.dname_masked, fname)
        fname_mask = os.path.join(self.dname_mask, fname)

        masked = Image.open(fname_masked).convert('RGB')
        mask = Image.open(fname_mask).convert('RGB')
        if self.transform is not None:
            masked = self.transform(masked)
            mask = self.transform(mask)

        # gooooooooooooooooooooooooooooooooooood
        mask = mask.masked_fill(mask < 0.5, 0.0)
        mask = mask.masked_fill(mask > 0.5, 1.0)
        masked = masked * mask

        if self.mask_channels == 1:
            mask = mask[:1, :, :]

        # print('masked:', np.unique(masked.numpy()))  # -1~1, 마스크영역 0
        # print('mask:', np.unique(mask.numpy()))  # 0,1
        # masked_save = np.uint8((np.transpose(masked.numpy(), (1, 2, 0)) + 1)/2 * 255)
        # cv2.imwrite('masked.png', masked_save)

        return fname, masked, mask


def get_dataloader(root, fnames, split, mask_channels, batch_size, num_workers):
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if split == 'train':
        is_train = True
        dataset = MyDataset(root, fnames, mask_channels, augment=False, transform=transform)
    elif split == 'val':
        is_train = False
        dataset = MyDataset(root, fnames, mask_channels, transform=transform)
    elif split == 'test':
        is_train = False
        dataset = TestDataset(root, mask_channels, transform=transform)

    dataloader = DataLoader(dataset,
                            shuffle=is_train,
                            batch_size=batch_size,
                            num_workers=num_workers)

    return dataloader
