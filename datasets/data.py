from datasets import get_datasets
from config_utils import load_config
import numpy as np
import torch
import torchvision

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.001):
        self.std = std
        self.mean = mean
                                    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
                                                    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)




class RepeatSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, samp, repeat):
        self.samp = samp
        self.repeat = repeat
    def __iter__(self):
        for i in self.samp:
            for j in range(self.repeat):
                yield i
    def __len__(self):
        return self.repeat*len(self.samp)


def get_data(dataset, data_loc, trainval, batch_size, augtype, repeat, user, args, pin_memory=True):
    file = 'Dirichlet_5_User_0.5_Dataset_Cifar10_non_iid_setting.npy'
    user_data = np.load(file, allow_pickle=True).item()
    train_data, valid_data, xshape, class_num = get_datasets(dataset, data_loc, cutout=0)
    if augtype == 'gaussnoise':
        train_data.transform.transforms = train_data.transform.transforms[2:]
        train_data.transform.transforms.append(AddGaussianNoise(std=args.sigma))
    elif augtype == 'cutout':
        train_data.transform.transforms = train_data.transform.transforms[2:]
        train_data.transform.transforms.append(torchvision.transforms.RandomErasing(p=0.9, scale=(0.02, 0.04)))
    elif augtype == 'none':
        train_data.transform.transforms = train_data.transform.transforms[2:]
    
    if dataset == 'cifar10':
        acc_type = 'ori-test'
        val_acc_type = 'x-valid'
    
    else:
        acc_type = 'x-test'
        val_acc_type = 'x-valid'
    
    if trainval and 'cifar10' in dataset:
        cifar_split = load_config('config_utils/cifar-split.txt', None, None)
        train_split, valid_split = cifar_split.train, cifar_split.valid
        if repeat > 0:
            # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
            #                                            num_workers=0, pin_memory=pin_memory, sampler= RepeatSampler(torch.utils.data.sampler.SubsetRandomSampler(train_split), repeat))
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, #shuffle=True,
                                                       num_workers=0, pin_memory=pin_memory,
                                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                           user_data[user]['train']), )
            test_loader = torch.utils.data.DataLoader(valid_data, batch_size=len(user_data[user]['test']), #shuffle=True,
                                                       num_workers=0, pin_memory=pin_memory,
                                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                           user_data[user]['test']), )
            print('non-iid setting success')
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                       num_workers=0, pin_memory=pin_memory, sampler= torch.utils.data.sampler.SubsetRandomSampler(train_split))
        
    
    else:
        if repeat > 0:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, #shuffle=True,
                                                       num_workers=0, pin_memory=pin_memory, sampler= RepeatSampler(torch.utils.data.sampler.SubsetRandomSampler(range(len(train_data))), repeat))
        else:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                                       num_workers=0, pin_memory=pin_memory)
    return train_loader, test_loader
