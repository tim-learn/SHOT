import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList
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
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if args.da == 'oda':
        label_map_s = {}
        for i in range(len(args.share_classes)):
            label_map_s[args.share_classes[i]] = i
        for i in range(len(args.src_classes)):
            if args.tar_classes[i] not in args.share_classes:
                label_map_s[args.src_classes[i]] = len(args.share_classes)
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()

        #########################
        label_map_t = {}
        for i in range(len(args.share_classes)):
            label_map_t[args.share_classes[i]] = i
        for i in range(len(args.tar_classes)):
            if args.tar_classes[i] not in args.share_classes:
                label_map_t[args.tar_classes[i]] = len(args.share_classes)

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                line = reci[0] + ' ' + str(label_map_t[int(reci[1])]) + '\n'   
                new_tar.append(line)
        txt_tar = new_tar.copy()

        new_test = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                line = reci[0] + ' ' + str(label_map_t[int(reci[1])]) + '\n'   
                new_test.append(line)
        txt_test = new_test.copy()

    if args.trte == 'val':
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        tr_txt = txt_src
        te_txt = txt_src   

    dsets['source_tr'] = ImageList(tr_txt, transform=image_train())
    dset_loaders['source_tr'] = DataLoader(dsets['source_tr'], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets['source_te'] = ImageList(te_txt, transform=image_test())
    dset_loaders['source_te'] = DataLoader(dsets['source_te'], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets['target'] = ImageList(txt_tar, transform=image_test())
    dset_loaders['target'] = DataLoader(dsets['target'], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets['test'] = ImageList(txt_test, transform=image_test())
    dset_loaders['test'] = DataLoader(dsets['test'], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    return accuracy, all_label, nn.Softmax(dim=1)(all_output)

def print_args(args):
    s = '==========================================\n'
    for arg, content in args.__dict__.items():
        s += '{}:{}\n'.format(arg, content)
    return s

def test_target_srconly(args, zz):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F_' + str(zz) + '.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B_' + str(zz) + '.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C_' + str(zz) + '.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, y, py = cal_acc(dset_loaders['test'], netF, netB, netC)
    log_str = '\nZz: {}, Task: {}, Accuracy = {:.2f}%'.format(zz, args.name, acc*100)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    return y, py

def test_target(args, zz):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_ori + '/target_F_' + args.savename + '.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_ori + '/target_B_' + args.savename + '.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_ori + '/target_C_' + args.savename + '.pt'  
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, y, py = cal_acc(dset_loaders['test'], netF, netB, netC)
    log_str = '\nZz: {}, Task: {}, Accuracy = {:.2f}%'.format(zz, args.name, acc*100)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    return y, py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help='device id to run')
    parser.add_argument('--tencrop', type=bool, default=False)
    parser.add_argument('--s', type=int, default=0, help='source')
    parser.add_argument('--t', type=int, default=1, help='target')
    parser.add_argument('--max_epoch', type=int, default=20, help='max iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--worker', type=int, default=4, help='number of workers')
    parser.add_argument('--dset', type=str, default='office-caltech', choices=['office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--net', type=str, default='resnet101', help='resnet50, resnet101')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=-1)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=1.0)
    parser.add_argument('--lr_decay2', type=float, default=10.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default='wn', choices=['linear', 'wn'])
    parser.add_argument('--classifier', type=str, default='bn', choices=['ori', 'bn'])
    parser.add_argument('--distance', type=str, default='cosine', choices=['euclidean', 'cosine'])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--zz', type=str, default='val', choices=['5', '10', '15', '20', '25', '30', 'val'])  
    parser.add_argument('--savename', type=str, default='san')
    args = parser.parse_args()
       
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    score_srconly = 0
    score = 0

    current_folder = './ckps/'
    args.output_dir = osp.join(current_folder, args.da, args.output, args.dset, str(0)+names[args.t][0].upper())
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(osp.join(args.output_dir, 'log_' + str(args.zz) + '_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    for i in range(len(names)):
        if i == args.t:
            continue
        args.s = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = args.t_dset_path

        args.output_dir_src = osp.join(current_folder, args.da, args.output, args.dset, names[args.s][0].upper())
        args.output_dir_ori = osp.join(current_folder, args.da, args.output, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        label, output_srconly = test_target_srconly(args, args.zz)
        score_srconly += output_srconly

        _, output = test_target(args, args.zz)
        score += output

    _, predict = torch.max(score_srconly, 1)
    acc = torch.sum(torch.squeeze(predict).float() == label).item() / float(label.size()[0])
    log_str = '\nZz: {}, Task: {}, Accuracy = {:.2f}%'.format(args.zz, '->' + names[args.t][0].upper(), acc*100)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    _, predict = torch.max(score, 1)
    acc = torch.sum(torch.squeeze(predict).float() == label).item() / float(label.size()[0])
    log_str = '\nZz: {}, Task: {}, Accuracy = {:.2f}%'.format(args.zz, '->' + names[args.t][0].upper(), acc*100)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)