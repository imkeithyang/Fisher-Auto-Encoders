# this was taken from the repository here: https://github.com/jpowie01/DCGAN_CelebA/blob/master/dataset.py 
import torch
import torchvision
import torchvision.transforms as transforms

import math

import numpy as np

import random

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
def _init_fn(worker_id):
    np.random.seed(1)
torch.backends.cudnn.deterministic=True

# TODO Make these data getters to mask by quadrant, using mask=-1 (Need to think about high-dimensions, maybe 0)

class MaskImages:
    """This torchvision image transformation prepares the MNIST digits to be
    used in the tutorial. Depending on the number of quadrants to be used as
    inputs (1, 2, or 3), the transformation masks the remaining (3, 2, 1)
    quadrant(s) setting their pixels with -1. Additionally, the transformation
    adds the target output in the sample dict as the complementary of the input
    """
    def __init__(self, num_quadrant_inputs, mask_ind=-1):
        if num_quadrant_inputs <= 0 or num_quadrant_inputs >= 4:
            raise ValueError('Number of quadrants as inputs must be 1, 2 or 3')
        self.num = num_quadrant_inputs
        self.mask_ind = mask_ind

    def __call__(self, sample):
        tensor = sample.squeeze(0)
        out = tensor.detach().clone()
        h, w = tensor.shape

        # removes the bottom left quadrant from the target output
        out[h // 2:, :w // 2] = self.mask_ind
        # if num of quadrants to be used as input is 2,
        # also removes the top left quadrant from the target output
        if self.num == 2:
            out[:, :w // 2] = self.mask_ind
        # if num of quadrants to be used as input is 3,
        # also removes the top right quadrant from the target output
        if self.num == 3:
            out[:h // 2, :] = self.mask_ind

        # now, sets the input as complementary
        inp = tensor.clone()
        inp[out != -1] = self.mask_ind
        sample_dict = {}
        sample_dict['input'] = inp
        sample_dict['output'] = out
        return sample_dict

def get_celeba(batch_size, dataset_directory, dataloader_workers=4, train_size = 0.8):
    # 1. Download this file into dataset_directory and unzip it:
    #  https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
    # 2. Put the `img_align_celeba` directory into the `celeba` directory!
    # 3. Dataset directory structure should look like this (required by ImageFolder from torchvision):
    #  +- `dataset_directory`
    #     +- celeba
    #        +- img_align_celeba
    #           +- 000001.jpg
    #           +- 000002.jpg
    #           +- 000003.jpg
    #           +- ...
    train_transformation = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    celeb_dataset = torchvision.datasets.ImageFolder(dataset_directory + 'celeba', train_transformation)

    lengths_ = [math.floor(len(celeb_dataset) * train_size), math.ceil(len(celeb_dataset) * ( 1 - train_size ) / 2 ),\
            math.ceil(len(celeb_dataset) * (1 - train_size ) / 2 )]

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(celeb_dataset, lengths_)


    # Use sampler for randomization
    training_sampler = torch.utils.data.SubsetRandomSampler(range(len(train_dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(range(len(valid_dataset)))
    test_sampler = torch.utils.data.SubsetRandomSampler(range(len(test_dataset)))

    # Prepare Data Loaders for training and validation
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=training_sampler,
                                               pin_memory=True, num_workers=dataloader_workers, drop_last=True, worker_init_fn=_init_fn)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler,
                                               pin_memory=True, num_workers=dataloader_workers, drop_last=True, worker_init_fn=_init_fn)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler,
                                               pin_memory=True, num_workers=dataloader_workers, drop_last=True, worker_init_fn=_init_fn)
    return train_loader, valid_loader, test_loader

def get_mnist(batch_size, num_quadrant_inputs, dataset_directory, mask_ind=-1):

    transform = transforms.Compose([transforms.ToTensor(), 
                                    MaskImages(num_quadrant_inputs=num_quadrant_inputs, mask_ind=mask_ind)])

    train_set = torchvision.datasets.MNIST(root=dataset_directory, train=True, download=False, transform=transform)
    lengths_ = [math.floor(len(train_set) * 0.8), math.ceil(len(train_set) * (1 - 0.8))]

    train_dataset, valid_dataset = torch.utils.data.random_split(train_set, lengths_)
    test_dataset = torchvision.datasets.MNIST(root=dataset_directory, train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader

    
    