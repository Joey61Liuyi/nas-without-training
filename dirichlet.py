# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 21:29
# @Author  : LIU YI

import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import random


def data_organize(idxs_labels, labels):
    data_dict = {}

    labels = np.unique(labels, axis=0)
    for one in labels:
        data_dict[one] = []

    for i in range(len(idxs_labels[1, :])):
        data_dict[idxs_labels[1, i]].append(idxs_labels[0, i])
    return data_dict


def data_partition(training_data, testing_data, alpha, user_num):
    idxs_train = np.arange(len(training_data))
    idxs_valid = np.arange(len(testing_data))

    if hasattr(training_data, 'targets'):
        labels_train = training_data.targets
        labels_valid = testing_data.targets
    elif hasattr(training_data, 'img_label'):
        labels_train = training_data.img_label
        labels_valid = testing_data.img_label

    idxs_labels_train = np.vstack((idxs_train, labels_train))
    idxs_labels_train = idxs_labels_train[:, idxs_labels_train[1,:].argsort()]
    idxs_labels_valid = np.vstack((idxs_valid, labels_valid))
    idxs_labels_valid = idxs_labels_valid[:, idxs_labels_valid[1,:].argsort()]

    labels = np.unique(labels_train, axis=0)

    data_train_dict = data_organize(idxs_labels_train, labels)
    data_valid_dict = data_organize(idxs_labels_valid, labels)

    data_partition_profile_train = {}
    data_partition_profile_valid = {}


    for i in range(user_num):
        data_partition_profile_train[i] = []
        data_partition_profile_valid[i] = []

    # ## Setting the public data
    # public_data = set([])
    # for label in data_train_dict:
    #     tep = set(np.random.choice(data_train_dict[label], int(len(data_train_dict[label])/20), replace = False))
    #     public_data = set.union(public_data, tep)
    #     data_train_dict[label] = list(set(data_train_dict[label])-tep)
    #
    # public_data = list(public_data)
    # np.random.shuffle(public_data)
    public_data = None

    ## Distribute rest data
    for label in data_train_dict:
        proportions = np.random.dirichlet(np.repeat(alpha, user_num))
        proportions_train = len(data_train_dict[label])*proportions
        proportions_valid = len(data_valid_dict[label]) * proportions

        for user in data_partition_profile_train:

            data_partition_profile_train[user]   \
                = set.union(set(np.random.choice(data_train_dict[label], int(proportions_train[user]) , replace = False)), data_partition_profile_train[user])
            data_train_dict[label] = list(set(data_train_dict[label])-data_partition_profile_train[user])


            data_partition_profile_valid[user] = set.union(set(
                np.random.choice(data_valid_dict[label], int(proportions_valid[user]),
                                 replace=False)), data_partition_profile_valid[user])
            data_valid_dict[label] = list(set(data_valid_dict[label]) - data_partition_profile_valid[user])


        while len(data_train_dict[label]) != 0:
            rest_data = data_train_dict[label][0]
            user = np.random.randint(0, user_num)
            data_partition_profile_train[user].add(rest_data)
            data_train_dict[label].remove(rest_data)

        while len(data_valid_dict[label]) != 0:
            rest_data = data_valid_dict[label][0]
            user = np.random.randint(0, user_num)
            data_partition_profile_valid[user].add(rest_data)
            data_valid_dict[label].remove(rest_data)

    for user in data_partition_profile_train:
        data_partition_profile_train[user] = list(data_partition_profile_train[user])
        data_partition_profile_valid[user] = list(data_partition_profile_valid[user])
        np.random.shuffle(data_partition_profile_train[user])
        np.random.shuffle(data_partition_profile_valid[user])
    return data_partition_profile_train, data_partition_profile_valid, public_data

if __name__ == '__main__':

    dataset = 'Cifar10'
    alpha = 0.5
    user_num = 5
    random.seed(61)
    np.random.seed(61)
    valid_use = False

    if dataset == "Cifar10":
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]

        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 32, 32)
        train_set = torchvision.datasets.CIFAR10(
            "../dataset/Cifar10/rawdata", train=True, transform=train_transform, download=True
        )
        test_set = torchvision.datasets.CIFAR10(
            "../dataset/Cifar10/rawdata", train=False, transform=test_transform, download=True
        )

        # trainloader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set.data), shuffle=False)
        # testloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set.data), shuffle=False)
        # for _, train_data in enumerate(trainloader, 0):
        #     train_set.data, train_set.targets = train_data
        # for _, train_data in enumerate(testloader, 0):
        #     test_set.data, test_set.targets = train_data
    elif dataset == "Cifar100":
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 32, 32)
        train_set = torchvision.datasets.CIFAR100(
            "../dataset/{}/rawdata".format(dataset), train=True, transform=train_transform, download=True
        )
        test_set = torchvision.datasets.CIFAR100(
            "../dataset/{}/rawdata".format(dataset), train=False, transform=test_transform, download=True
        )
        assert len(train_set) == 50000 and len(test_set) == 10000
        # trainloader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set.data), shuffle=False)
        # testloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set.data), shuffle=False)
        # for _, train_data in enumerate(trainloader, 0):
        #     train_set.data, train_set.targets = train_data
        # for _, train_data in enumerate(testloader, 0):
        #     test_set.data, test_set.targets = train_data

    elif dataset == 'mnist' or 'fmnist':

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_set = torchvision.datasets.MNIST("../dataset/{}/rawdata".format(dataset), train=True, download=True,
                                               transform=apply_transform)

        test_set = torchvision.datasets.MNIST("../dataset/{}/rawdata".format(dataset), train=False, download=True,
                                              transform=apply_transform)

    tep_train, tep_valid, tep_public = data_partition(train_set, test_set, alpha, user_num)

    user_data = {}
    for one in tep_train:
        # a = np.random.choice(tep_train[one], int(len(tep_train[one]) / 2), replace=False)
        user_data[one] = {'train': tep_train[one], 'test': tep_valid[one]}

    user_data["public"] = tep_public
    np.save('Dirichlet_{}_User_{}_Dataset_{}_non_iid_setting.npy'.format(user_num, alpha, dataset), user_data)

    print('ok')