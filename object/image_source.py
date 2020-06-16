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
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
import loss
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

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

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

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
    dset_loaders['test'] = DataLoader(dsets['test'], batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)

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

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
        ent = ent.float().cpu()
        initc = np.array([[0], [1]])
        kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
        threshold = (kmeans.cluster_centers_).mean()

        predict[ent>threshold] = args.class_num

        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        matrix = matrix[np.unique(all_label).astype(int),:]

        acc = matrix.diagonal()/matrix.sum(axis=1)
        unknown_acc = acc[-1:].item()
        return np.mean(acc), np.mean(acc[:-1])
    else:
        return accuracy, mean_ent

def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*10}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*10}]   
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)

    acc_init = 0
    for epoch in tqdm(range(args.max_epoch), leave=False):
        netF.train()
        netB.train()
        netC.train()
        iter_source = iter(dset_loaders['source_tr'])
        for _, (inputs_source, labels_source) in tqdm(enumerate(iter_source), leave=False):
            if inputs_source.size(0) == 1:
                continue
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            outputs_source = netC(netB(netF(inputs_source)))
            classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)            
            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0 and args.trte == 'full':
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, args.dset=='visda17')
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, epoch+1, args.max_epoch, acc_s_te*100)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            best_netF = netF.state_dict()
            best_netB = netB.state_dict()
            best_netC = netC.state_dict()
            torch.save(best_netF, osp.join(args.output_dir_src, 'source_F_' + str(epoch + 1) + '.pt'))
            torch.save(best_netB, osp.join(args.output_dir_src, 'source_B_' + str(epoch + 1) + '.pt'))
            torch.save(best_netC, osp.join(args.output_dir_src, 'source_C_' + str(epoch + 1) + '.pt'))

        if args.trte == 'val':
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], netF, netB, netC)
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%'.format(args.name_src, epoch+1, args.max_epoch, acc_s_tr*100, acc_s_te*100)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te > acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()
                
    torch.save(best_netF, osp.join(args.output_dir_src, 'source_F_val.pt'))
    torch.save(best_netB, osp.join(args.output_dir_src, 'source_B_val.pt'))
    torch.save(best_netC, osp.join(args.output_dir_src, 'source_C_val.pt'))
    return netF, netB, netC

def test_target(args, zz):
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

    acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, args.da=='oda')
    log_str = '\nZz: {}, Task: {}, Accuracy = {:.2f}%'.format(zz, args.name, acc*100)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def print_args(args):
    s = '==========================================\n'
    for arg, content in args.__dict__.items():
        s += '{}:{}\n'.format(arg, content)
    return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help='device id to run')
    parser.add_argument('--s', type=int, default=0, help='source')
    parser.add_argument('--t', type=int, default=1, help='target')
    parser.add_argument('--max_epoch', type=int, default=20, help='max iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--worker', type=int, default=4, help='number of workers')
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--net', type=str, default='resnet50', help='resnet50, resnet101')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default='wn', choices=['linear', 'wn'])
    parser.add_argument('--classifier', type=str, default='bn', choices=['ori', 'bn'])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda', 'opda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
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

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = args.t_dset_path

    if args.dset == 'office':
        if args.da == 'pda':
            args.class_num = 31
            args.src_classes = [i for i in range(31)]
            args.tar_classes = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22]          
        if args.da == 'oda':
            args.class_num = 10
            args.src_classes = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22] 
            args.tar_classes = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22] + [19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30]
        if args.da == 'opda':
            args.class_num = 20
            args.src_classes = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22] + [2, 3, 4, 6, 7, 8, 9, 13, 14, 18]
            args.tar_classes = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22] + [19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30]

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]
        if args.da == 'opda':
            args.class_num = 15
            args.src_classes = [i for i in range(15)]
            args.tar_classes = [i for i in range(10)] + [i for i in range(15, 65)]

    current_folder = './ckps/'
    args.output_dir_src = osp.join(current_folder, args.da, args.output, args.dset, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    if args.trte == 'val':
        args.out_file = open(osp.join(args.output_dir_src, 'log_val.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_source(args)
    else:
        args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_source(args)

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        folder = '/Checkpoint/liangjian/tran/data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = args.t_dset_path

        if args.dset == 'office':
            if args.da == 'pda':
                args.class_num = 31
                args.src_classes = [i for i in range(31)]
                args.tar_classes = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22]          
            if args.da == 'oda':
                args.class_num = 10
                args.src_classes = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22] 
                args.tar_classes = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22] + [19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30]
            if args.da == 'opda':
                args.class_num = 20
                args.src_classes = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22] + [2, 3, 4, 6, 7, 8, 9, 13, 14, 18]
                args.tar_classes = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22] + [19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30]

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]
            if args.da == 'oda':
                args.class_num = 25
                args.src_classes = [i for i in range(25)]
                args.tar_classes = [i for i in range(65)]
            if args.da == 'opda':
                args.class_num = 15
                args.src_classes = [i for i in range(15)]
                args.tar_classes = [i for i in range(10)] + [i for i in range(15, 65)]
        
        if args.trte == 'val':
            test_target(args, 'val')
        else:            
            n = args.max_epoch // 5
            for e in range(1, 1 + n):
                test_target(args, 5*e)