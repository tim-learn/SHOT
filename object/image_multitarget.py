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
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()

    txt_tar = []
    for i in range(len(args.t_dset_path)):
        tmp = open(args.t_dset_path[i]).readlines()
        txt_tar.extend(tmp)
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
    dsets['target'] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders['target'] = DataLoader(dsets['target'], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets['test'] = ImageList(txt_test, transform=image_test())
    dset_loaders['test'] = DataLoader(dsets['test'], batch_size=train_bs*2, shuffle=False, num_workers=args.worker, drop_last=False)

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

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        pdb.set_trace()
        acc = matrix.diagonal()/matrix.sum(axis=1)
        return np.mean(acc), acc
    else:
        return accuracy, mean_ent

def print_args(args):
    s = '==========================================\n'
    for arg, content in args.__dict__.items():
        s += '{}:{}\n'.format(arg, content)
    return s

def train_target(args, zz=''):
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
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)

    for epoch in tqdm(range(args.max_epoch), leave=False):
        netF.eval()
        netB.eval()
        mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
        mem_label = torch.from_numpy(mem_label).cuda()
        netF.train()
        netB.train()
        iter_test = iter(dset_loaders['target'])

        for _, (inputs_test, _, tar_idx) in tqdm(enumerate(iter_test), leave=False):
            if inputs_test.size(0) == 1:
                continue
            inputs_test = inputs_test.cuda()
            pred = mem_label[tar_idx]
            features_test = netB(netF(inputs_test))
            outputs_test = netC(features_test)

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

        netF.eval()
        netB.eval()
        flag = (args.dset=='visda17') or (args.da == 'oda')
        acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, flag)
        if args.tencrop and epoch == args.max_epoch - 1 and not args.net[0:4] == 'alex':
            acc2 = cal_acc_tencrop(dset_loaders['test'], netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}% / {:.2f}%'.format(args.name, epoch+1, args.max_epoch, acc*100, acc2*100)
        else:
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, epoch+1, args.max_epoch, acc*100)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str+'\n')
    
    # torch.save(netF.state_dict(), osp.join(args.output_dir, 'target_F.pt'))
    # torch.save(netB.state_dict(), osp.join(args.output_dir, 'target_B.pt'))
    # torch.save(netC.state_dict(), osp.join(args.output_dir, 'target_C.pt'))
    return netF, netB, netC

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
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
    parser.add_argument('--max_epoch', type=int, default=20000, help='max iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--worker', type=int, default=4, help='number of workers')
    parser.add_argument('--dset', type=str, default='office-home', choices=['office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--net', type=str, default='resnet50', help='resnet50, resnet101')
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
    parser.add_argument('--trte', type=str, default='full', choices=['full', 'val'])
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

    t_dset = []
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        t_dset.append(args.t_dset_path)

    args.t_dset_path = t_dset
    args.test_dset_path = args.t_dset_path
    current_folder = './ckps/'
    args.output_dir_src = osp.join(current_folder, args.da, args.output, args.dset, names[args.s][0].upper())
    args.output_dir = osp.join(current_folder, args.da, args.output, args.dset, names[args.s][0].upper() + str(0))
    args.name = names[args.s][0].upper() + str(0)

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file = open(osp.join(args.output_dir, 'log_' + str(args.zz) + '_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args, args.zz)