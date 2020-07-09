import argparse
import os, sys
import os.path as osp
import torchvision
from torchvision import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def image_test(resize_size=256, crop_size=224):
  return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def data_load(args): 
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets['target'] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders['target'] = DataLoader(dsets['target'], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets['test'] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders['test'] = DataLoader(dsets['test'], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, net, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = net(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(all_output.size(1))
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item() 
    
    return accuracy, mean_ent

def train_target(args):
    dset_loaders = data_load(args)

    param_group = []    
    model_resnet = network.Res50().cuda()
    for k, v in model_resnet.named_parameters():
        if k.__contains__('fc'):
            v.requires_grad = False
        else:
            param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)

    for epoch in tqdm(range(args.max_epoch), leave=False):

        model_resnet.eval()
        mem_label = obtain_label(dset_loaders['test'], model_resnet, args)
        mem_label = torch.from_numpy(mem_label).cuda()
        model_resnet.train()

        iter_test = iter(dset_loaders['target'])
        for _, (inputs_test, _, tar_idx) in tqdm(enumerate(iter_test), leave=False):
            if inputs_test.size(0) == 1:
                continue
            inputs_test = inputs_test.cuda()

            pred = mem_label[tar_idx]
            features_test, outputs_test = model_resnet(inputs_test)

            classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0)(outputs_test, pred)
            classifier_loss *= args.cls_par

            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(loss.Entropy(softmax_out))
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    entropy_loss -= gentropy_loss
                classifier_loss += entropy_loss * args.ent_par

            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

        model_resnet.eval()
        acc, ment = cal_acc(dset_loaders['test'], model_resnet)
        log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.dset, epoch+1, args.max_epoch, acc*100)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str+'\n')
    
    # torch.save(model_resnet.state_dict(), osp.join(args.output_dir, 'target.pt'))
    return model_resnet

def print_args(args):
    s = '==========================================\n'
    for arg, content in args.__dict__.items():
        s += '{}:{}\n'.format(arg, content)
    return s

def obtain_label(loader, net, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas, outputs = net(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return pred_label.astype('int')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help='device id to run')
    parser.add_argument('--s', type=int, default=0, help='source')
    parser.add_argument('--t', type=int, default=1, help='target')
    parser.add_argument('--max_epoch', type=int, default=30, help='max iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--worker', type=int, default=4, help='number of workers')
    parser.add_argument('--distance', type=str, default='cosine', choices=['euclidean', 'cosine'])
    parser.add_argument('--dset', type=str, default='imagenet_caltech', choices=['imagenet_caltech'])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--net', type=str, default='resnet50', help='resnet50')
    parser.add_argument('--seed', type=int, default=2019, help='random seed')
    parser.add_argument('--epsilon', type=float, default=1e-5)  
    parser.add_argument('--gent', type=bool, default=False)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=30)

    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--output', type=str, default='res50')
    parser.add_argument('--da', type=str, default='pda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--savename', type=str, default='')
    args = parser.parse_args()

    args.prep = {'params':{'resize_size':256, 'crop_size':224, 'alexnet':False}}           

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.class_num = 1000
    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + 'imagenet' + '_list.txt'
    if args.da == 'pda':
        args.t_dset_path = folder + args.dset + '/' + 'caltech_84' + '_list.txt'
    args.test_dset_path = args.t_dset_path

    current_folder = './ckps/'
    args.output_dir = osp.join(current_folder, args.da, args.output, args.dset)
    args.name = args.dset

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    train_target(args)
    args.out_file.close()