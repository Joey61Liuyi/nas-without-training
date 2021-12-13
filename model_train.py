# -*- coding: utf-8 -*-
# @Time    : 2021/12/13 11:02
# @Author  : LIU YI

import argparse
import nasspace
import datasets
import random
import numpy as np
import torch
import os
from scores import get_score_func
from scipy import stats
from pycls.models.nas.nas import Cell
from utils import add_dropout, init_network
import wandb






def main_func(user, searched):
    wandb_project = "Score_trail"
    parser = argparse.ArgumentParser(description='NAS Without Training')
    parser.add_argument('--data_loc', default='../cifardata/', type=str, help='dataset folder')
    parser.add_argument('--api_loc', default='../NAS-Bench-201-v1_0-e61699.pth', type=str, help='path to API')
    parser.add_argument('--save_loc', default='results', type=str, help='folder to save results')
    parser.add_argument('--save_string', default='naswot', type=str, help='prefix of results file')
    parser.add_argument('--score', default='hook_logdet', type=str, help='the score to evaluate')
    parser.add_argument('--nasspace', default='nasbench201', type=str, help='the nas search space to use')
    parser.add_argument('--batch_size', default= 128, type=int)
    parser.add_argument('--repeat', default=1, type=int, help='how often to repeat a single image with a batch')
    parser.add_argument('--augtype', default='none', type=str, help='which perturbations to use')
    parser.add_argument('--sigma', default=0.05, type=float, help='noise level if. augtype is "gaussnoise"')
    parser.add_argument('--GPU', default='0', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--init', default='', type=str)
    parser.add_argument('--trainval', default= True, action='store_true')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--maxofn', default=1, type=int, help='score is the max of this many evaluations of the network')
    parser.add_argument('--n_samples', default=100, type=int)
    parser.add_argument('--n_runs', default=500, type=int)
    parser.add_argument('--stem_out_channels', default=16, type=int, help='output channels of stem convolution (nasbench101)')
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules (nasbench101)')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack (nasbench101)')
    parser.add_argument('--num_labels', default=1, type=int, help='#classes (nasbench101)')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def get_batch_jacobian(net, x, target, device, args=None):
        net.zero_grad()
        x.requires_grad_(True)
        y, out = net(x)
        y.backward(torch.ones_like(y))
        jacob = x.grad.detach()
        return jacob, target.detach(), y.detach(), out.detach()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    savedataset = args.dataset
    dataset = 'fake' if 'fake' in args.dataset else args.dataset
    args.dataset = args.dataset.replace('fake', '')
    if args.dataset == 'cifar10':
        args.dataset = args.dataset + '-valid'
    searchspace = nasspace.get_search_space(args)
    if 'valid' in args.dataset:
        args.dataset = args.dataset.replace('-valid', '')
    train_loader, test_loader = datasets.get_data(args.dataset, args.data_loc, args.trainval, args.batch_size, args.augtype, args.repeat, user, args)
    os.makedirs(args.save_loc, exist_ok=True)
    # filename = f'{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{savedataset}{"_" + args.init + "_" if args.init != "" else args.init}_{"_dropout" if args.dropout else ""}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}'
    # filename = f'{args.save_loc}/{args.save_string}_{args.score}_{args.nasspace}_{savedataset}{"_" + args.init + "_" if args.init != "" else args.init}_{"_dropout" if args.dropout else ""}_{args.augtype}_{args.sigma}_{args.repeat}_{args.trainval}_{args.batch_size}_{args.maxofn}_{args.seed}_{user}_new'
    # accfilename = f'{args.save_loc}/{args.save_string}_accs_{args.nasspace}_{savedataset}_{args.trainval}_{user}_new'
    if args.dataset == 'cifar10':
        acc_type = 'ori-test'
        val_acc_type = 'x-valid'
    else:
        acc_type = 'x-test'
        val_acc_type = 'x-valid'
    scores = np.zeros(len(searchspace))
    for i, (uid, network) in enumerate(searchspace):
        if i == searched:

            run_name = "User{}_model{}".format(user, i)
            wandb.init(project=wandb_project, name = run_name)
            if args.dropout:
                add_dropout(network, args.sigma)
            if args.init != '':
                init_network(network, args.init)
            # if 'hook_' in args.score:
            #     network.K = np.zeros((args.batch_size, args.batch_size))
            #     for name, module in network.named_modules():
            #         if 'ReLU' in str(type(module)):
            #             # hooks[name] = module.register_forward_hook(counting_hook)
            #             module.register_forward_hook(counting_forward_hook)
            #             module.register_backward_hook(counting_backward_hook)
            #     # print('done')
            network = network.to(device)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            # optimizer = torch.optim.SGD(network.parameters(), lr=0.001)
            optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            for epoch in range(100):

                test_acc = 0
                test_num = 0
                test_loss = 0
                network.eval()
                for batch_id, (images, labels) in enumerate(test_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    pred = network(images)
                    pred = pred[0]
                    loss = criterion(pred, labels)
                    test_acc += (torch.sum(torch.argmax(pred, dim=1) == labels)).item()
                    test_num += labels.shape[0]
                    test_loss += loss.item() * labels.shape[0]

                print("current epoch: {}, test accuracy is {:.3f}, test loss is {:.3f}".format(epoch, test_acc/test_num, test_loss/test_num))

                train_acc = 0
                train_num = 0
                train_loss = 0
                network.train()
                for batch_id, (images, labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    images = images.to(device)
                    labels = labels.to(device)
                    pred = network(images)
                    pred = pred[0]
                    loss = criterion(pred, labels)
                    loss.backward()
                    optimizer.step()
                    train_acc += (torch.sum(torch.argmax(pred, dim=1) == labels)).item()
                    train_num += labels.shape[0]
                    train_loss += loss.item() * labels.shape[0]

                print(
                    "current epoch: {}, the train accuracy is {:.3f}, train loss is {:.3f}".format(
                        epoch, train_acc / train_num, train_loss / train_num))
                info_dict = {
                    "epoch": epoch,
                    "train_acc": train_acc/train_num,
                    "train_loss": train_loss/train_num,
                    "test_acc": test_acc/test_num,
                    "test_loss": test_loss/test_num
                }
                wandb.log(info_dict)
            wandb.finish()
            # torch.cuda.empty_cache()






        # Reproducibility
        # try:
        #     if args.dropout:
        #         add_dropout(network, args.sigma)
        #     if args.init != '':
        #         init_network(network, args.init)
        #     if 'hook_' in args.score:
        #         network.K = np.zeros((args.batch_size, args.batch_size))
        #         def counting_forward_hook(module, inp, out):
        #             try:
        #                 if not module.visited_backwards:
        #                     return
        #                 if isinstance(inp, tuple):
        #                     inp = inp[0]
        #                 inp = inp.view(inp.size(0), -1)
        #                 x = (inp > 0).float()
        #                 K = x @ x.t()
        #                 K2 = (1.-x) @ (1.-x.t())
        #                 network.K = network.K + K.cpu().numpy() + K2.cpu().numpy()
        #             except:
        #                 pass
        #         def counting_backward_hook(module, inp, out):
        #             module.visited_backwards = True
        #         for name, module in network.named_modules():
        #             if 'ReLU' in str(type(module)):
        #                 #hooks[name] = module.register_forward_hook(counting_hook)
        #                 module.register_forward_hook(counting_forward_hook)
        #                 module.register_backward_hook(counting_backward_hook)
        #         # print('done')
        #     network = network.to(device)
        #     random.seed(args.seed)
        #     np.random.seed(args.seed)
        #     torch.manual_seed(args.seed)
        #     s = []
        #     for j in range(args.maxofn):
        #         data_iterator = iter(train_loader)
        #         x, target = next(data_iterator)
        #         x2 = torch.clone(x)
        #         x2 = x2.to(device)
        #         x, target = x.to(device), target.to(device)
        #         jacobs, labels, y, out = get_batch_jacobian(network, x, target, device, args)
        #         if 'hook_' in args.score:
        #             network(x2.to(device))
        #             s.append(get_score_func(args.score)(network.K, target))
        #         else:
        #             s.append(get_score_func(args.score)(jacobs, labels))
        #     scores[i] = np.mean(s)
        #     accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        #     accs_ = accs[~np.isnan(scores)]
        #     scores_ = scores[~np.isnan(scores)]
        #     numnan = np.isnan(scores).sum()
        #     tau, p = stats.kendalltau(accs_[:max(i-numnan, 1)], scores_[:max(i-numnan, 1)])
        #     print(f'{tau}, {i}/{len(searchspace)}')
        # except Exception as e:
        #     print(e)
        #     accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        #     scores[i] = np.nan

if __name__ == '__main__':
    user_list = [1, 2, 3, 4]
    searched = [1,3]
    for user in user_list:
        main_func(user, searched[0])
        main_func(user, searched[1])
