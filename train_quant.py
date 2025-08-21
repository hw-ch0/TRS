import argparse
import logging
import os
import random

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from custom_models import *
from custom_modules import *
from custom_optims import *
from utils import *

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="PyTorch Implementation of EWGS (CIFAR)")
# data and model
parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10','cifar100'), 
                                 help='dataset to use CIFAR10|CIFAR100')
parser.add_argument('--arch', type=str, default='resnet20_quant', help='model architecture')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--seed', type=int, default=None, help='seed for initialization')

# training settings
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size for training')
parser.add_argument('--epochs', type=int, default=400, help='number of epochs for training')
parser.add_argument('--optimizer', type=str, default='SGD', choices=('SGD','Adam'), help='type of an optimizer')
parser.add_argument('--scheduler', type=str, default='cosine', choices=('step','cosine'), help='')
parser.add_argument('--decay_schedule', type=str, default="100-200-300", help='')
parser.add_argument('--gamma', type=float, default=0.1, help='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

# custom settings
parser.add_argument('--bit_weight', type=int, default=1, help='')
parser.add_argument('--bit_act', type=int, default=1, help='')
parser.add_argument('--symmetric_fxp', type=str2bool, default=False, help='')
parser.add_argument('--scale_lr', type=float, default=0.01, help='')
parser.add_argument('--initial_tr', type=float, default=0.005, help='initial transition ratio')
parser.add_argument('--momentum_tr', type=float, default=0.99, help='')
parser.add_argument('--baseline', type=str2bool, default=False, help='')

# logging and misc
parser.add_argument('--gpu_id', type=str, default='0', help='target GPU to use')
parser.add_argument('--log_dir', type=str, default='../results/ResNet20_CIFAR10/W1A1/')
parser.add_argument('--load_pretrain', type=str2bool, default=True, help='load pretrained full-precision model')
parser.add_argument('--pretrain_path', type=str, default='../results/ResNet20_CIFAR100/fp/checkpoint/last_checkpoint.pth', 
                                       help='path for pretrained full-preicion model')
args = parser.parse_args()
arg_dict = vars(args)

### make log directory
if not os.path.exists(args.log_dir):
    os.makedirs(os.path.join(args.log_dir, 'checkpoint'))

logging.basicConfig(filename=os.path.join(args.log_dir, "log.txt"),
                    level=logging.INFO,
                    format='')
log_string = 'configs\n'
for k, v in arg_dict.items():
    log_string += "{}: {}\t".format(k,v)
    print("{}: {}".format(k,v), end='\t')
logging.info(log_string+'\n')
print('')

### GPU setting
# os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### set the seed number
if args.seed is not None:
    print("The seed number is set to", args.seed)
    logging.info("The seed number is set to {}".format(args.seed))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic=True

def _init_fn(worker_id):
    seed = args.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

### train/test datasets
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
    args.num_classes = 10
    train_dataset = dsets.CIFAR10(root='../data/CIFAR10/',
                                train=True, 
                                transform=transform_train,
                                download=True)
    test_dataset = dsets.CIFAR10(root='../data/CIFAR10/',
                            train=False, 
                            transform=transform_test)
elif args.dataset == 'cifar100':
    args.num_classes = 100
    train_dataset = dsets.CIFAR100(root='../data/CIFAR100/',
                                train=True, 
                                transform=transform_train,
                                download=True)
    test_dataset = dsets.CIFAR100(root='../data/CIFAR100/',
                            train=False, 
                            transform=transform_test)
else:
    raise NotImplementedError

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers,
                                           worker_init_fn=None if args.seed is None else _init_fn)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False,
                                          num_workers=args.num_workers)

### initialize model
model_class = globals().get(args.arch)
model = model_class(args)
model.to(device)

num_total_params = sum(p.numel() for p in model.parameters())
print("The number of parameters : ", num_total_params)
logging.info("The number of parameters : {}".format(num_total_params))

if args.load_pretrain:
    trained_model = torch.load(args.pretrain_path)
    current_dict = model.state_dict()
    print("Pretrained full precision weights are initialized")
    logging.info("\nFollowing modules are initialized from pretrained model")
    log_string = ''
    for key in trained_model['model'].keys():
        if key in current_dict.keys():
            log_string += '{}\t'.format(key)
            current_dict[key].copy_(trained_model['model'][key])
    logging.info(log_string+'\n')
    model.load_state_dict(current_dict)
    ###
    for m in model.modules():
        if isinstance(m, QConv):
            m.init.fill_(1)
    ###

### initialize optimizer, scheduler, loss function
if args.bit_weight == 32 and args.bit_act == 32:
    optimizer = quant_SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, baseline=True)
else:
    quant_weights, fp_weights, scale_params, other_params = get_trainable_params(model)
    if args.optimizer == 'SGD':
        if args.baseline:
            optimizer = quant_SGD([{'params':quant_weights},
                                    {'params':fp_weights},
                                    {'params':scale_params, 'lr':args.scale_lr, 'momentum':0, 'weight_decay':0},
                                    {'params':other_params, 'momentum':0, 'weight_decay':0}],
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, baseline=True)
        else:
            optimizer = quant_SGD([{'params':quant_weights, 'lr':args.initial_tr, # lr here represents transition ratio
                                    'momentum_tr': args.momentum_tr, 'alpha':args.lr},
                                    {'params':fp_weights},
                                    {'params':scale_params, 'lr':args.scale_lr, 'momentum':0, 'weight_decay':0},
                                    {'params':other_params, 'momentum':0, 'weight_decay':0}],
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, baseline=False)
    elif args.optimizer == 'Adam':
        if args.baseline:
            optimizer = quant_Adam([{'params':quant_weights},
                                    {'params':fp_weights},
                                    {'params':scale_params, 'lr':args.scale_lr, 'weight_decay':0},
                                    {'params':other_params, 'weight_decay':0}],
                                    lr=args.lr, weight_decay=args.weight_decay, baseline=True)
        else:
            optimizer = quant_Adam([{'params':quant_weights, 'lr':args.initial_tr, # lr here represents transition ratio
                                        'momentum_tr': args.momentum_tr, 'alpha':args.lr},
                                    {'params':fp_weights},
                                    {'params':scale_params, 'lr':args.scale_lr, 'weight_decay':0},
                                    {'params':other_params, 'weight_decay':0}],
                                    lr=args.lr, weight_decay=args.weight_decay, baseline=False)
    
if args.scheduler == 'step':
    if args.decay_schedule is not None:
        milestones = list(map(lambda x: int(x), args.decay_schedule.split('-')))
    else:
        milestones = [args.epochs+1]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)
elif args.scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(args.log_dir)


quant_modules = []
for m in model.modules():
    if isinstance(m, QConv):
        quant_modules.append(m)
### train
total_iter = 0
best_acc = 0
for ep in range(args.epochs):
    model.train()
    if args.bit_weight == 32 and args.bit_act == 32:
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], ep)
    else:
        if args.baseline:
            writer.add_scalar('train/quant_lr', optimizer.param_groups[0]['lr'], ep)
        else:
            writer.add_scalar('train/target_tr', optimizer.param_groups[0]['lr'], ep)
        writer.add_scalar('train/fp_lr', optimizer.param_groups[1]['lr'], ep)
        writer.add_scalar('train/scale_lr', optimizer.param_groups[2]['lr'], ep)
        writer.add_scalar('train/other_lr', optimizer.param_groups[3]['lr'], ep)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
            
        pred = model(images)
        loss = criterion(pred, labels)
        
        loss.backward()
        
        optimizer.step()
        
        #######################
        if total_iter % 100 == 0 and total_iter != 0:
            writer.add_scalar('train/loss', loss.item(), total_iter)
            if args.bit_weight == 32:
                for j, m in enumerate(quant_modules):
                    current_w = m.weight.detach().clone().view(-1)
                    prev_w = m.prev_weight.detach().clone().view(-1)
                    if isinstance(optimizer, quant_SGD) or isinstance(optimizer, quant_Adam):
                        d_p = m.weight.d_p
                        MAG = d_p.abs().mean()
                        writer.add_scalar("z_{}th_module/MAG".format(j+1), MAG.item(), total_iter)


                    fp_mean_abs_diff = (current_w-prev_w).abs().mean()
                    fp_mean_abs_diff_per_mean_abs_weight = fp_mean_abs_diff / current_w.abs().mean()
                    fp_mean_abs_diff_per_max = fp_mean_abs_diff / current_w.abs().max()

                    writer.add_scalar("z_{}th_module/fp_mean_abs_diff".format(j+1), fp_mean_abs_diff.item(), total_iter)
                    writer.add_scalar("z_{}th_module/fp_mean_abs_diff_per_mean_abs_weight".format(j+1), fp_mean_abs_diff_per_mean_abs_weight.item(), total_iter)
                    writer.add_scalar("z_{}th_module/fp_mean_abs_diff_per_max".format(j+1), fp_mean_abs_diff_per_max.item(), total_iter)
            else:
                for j, p in enumerate(optimizer.param_groups[0]['params']):
                    if not args.baseline:
                        EMA_transition = optimizer.state[p]['EMA_transition'].item()
                        quant_lr = optimizer.state[p]['lr_scale'].item() * args.lr
                        writer.add_scalar("z_{}th_module/EMA_transition".format(j+1), EMA_transition, total_iter)
                        writer.add_scalar("z_{}th_module/quant_lr".format(j+1), quant_lr, total_iter)
                    writer.add_scalar("z_{}th_module/num_transitions".format(j+1), p.transition.sum().item(), total_iter)
                    writer.add_scalar("z_{}th_module/transition_ratio".format(j+1), p.transition.sum().item() / p.numel(), total_iter)
                for j, m in enumerate(quant_modules):
                    if isinstance(optimizer, quant_SGD) or isinstance(optimizer, quant_Adam):
                        d_p = m.weight.d_p
                        MAG = d_p.abs().mean()
                        writer.add_scalar("z_{}th_module/MAG".format(j+1), MAG.item(), total_iter)
                    current_w = m.weight.detach().clone().view(-1)
                    prev_w = m.prev_weight.detach().clone().view(-1)
                    current_Qw = m.weight_quantization(m.weight).detach().clone().view(-1)
                    prev_Qw = m.prev_Qweight.detach().clone().view(-1)

                    fp_mean_abs_diff = (current_w-prev_w).abs().mean()
                    fp_mean_abs_diff_per_mean_abs_weight = fp_mean_abs_diff / current_w.abs().mean()
                    fp_mean_abs_diff_per_max = fp_mean_abs_diff / current_w.abs().max()
                    
                    Q_mean_abs_diff = (current_Qw-prev_Qw).abs().mean()
                    Q_mean_abs_diff_per_mean_abs_weight = Q_mean_abs_diff / current_Qw.abs().mean()
                    Q_mean_abs_diff_per_max = Q_mean_abs_diff / current_Qw.abs().max()

                    writer.add_scalar("z_{}th_module/fp_mean_abs_diff".format(j+1), fp_mean_abs_diff.item(), total_iter)
                    writer.add_scalar("z_{}th_module/fp_mean_abs_diff_per_mean_abs_weight".format(j+1), fp_mean_abs_diff_per_mean_abs_weight.item(), total_iter)
                    writer.add_scalar("z_{}th_module/fp_mean_abs_diff_per_max".format(j+1), fp_mean_abs_diff_per_max.item(), total_iter)

                    writer.add_scalar("z_{}th_module/Q_mean_abs_diff".format(j+1), Q_mean_abs_diff.item(), total_iter)
                    writer.add_scalar("z_{}th_module/Q_mean_abs_diff_per_mean_abs_weight".format(j+1), Q_mean_abs_diff_per_mean_abs_weight.item(), total_iter)
                    writer.add_scalar("z_{}th_module/Q_mean_abs_diff_per_max".format(j+1), Q_mean_abs_diff_per_max.item(), total_iter)

                    writer.add_scalar("z_{}th_module/MD2TP".format(j+1), m.MD2TP.item(), total_iter)

        if total_iter % 1000 == 0:
            for j, m in enumerate(quant_modules):
                fig = plt.figure()
                plt.hist(m.weight.view(-1).detach().cpu().numpy(), bins=100)
                # plt.xlim([-1,1])
                writer.add_figure("z_{}th_module/w_fig".format(j+1), fig, total_iter//1000)
        total_iter += 1
    
    scheduler.step()

    with torch.no_grad():
        model.eval()
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified/total*100
        writer.add_scalar('train/acc', test_acc, ep)

        model.eval()
        correct_classified = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            total += pred.size(0)
            correct_classified += (predicted == labels).sum().item()
        test_acc = correct_classified/total*100
        print("Current epoch: {:03d}".format(ep), "\t Test accuracy:", test_acc, "%")
        logging.info("Current epoch: {:03d}\t Test accuracy: {}%".format(ep, test_acc))
        writer.add_scalar('test/acc', test_acc, ep)

        torch.save({
            'epoch':ep,
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'criterion':criterion.state_dict()
        }, os.path.join(args.log_dir,'checkpoint/last_checkpoint.pth'))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch':ep,
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'criterion':criterion.state_dict()
            }, os.path.join(args.log_dir,'checkpoint/best_checkpoint.pth'))  
    layer_num = 0
    for m in model.modules():
        if isinstance(m, QConv):
            layer_num += 1
            writer.add_scalar("z_{}th_module/sW".format(layer_num), m.sW.item(), ep)
            writer.add_scalar("z_{}th_module/sA".format(layer_num), m.sA.item(), ep)
            logging.info("{}th_module/sW: {}".format(layer_num,m.sW))
            logging.info("{}th_module/sA: {}".format(layer_num,m.sA))
            logging.info('\n')

### Test accuracy @ last checkpoint
trained_model = torch.load(os.path.join(args.log_dir,'checkpoint/last_checkpoint.pth'))
model.load_state_dict(trained_model['model'])
print("The last checkpoint is loaded")
logging.info("The last checkpoint is loaded")
model.eval()
with torch.no_grad():
    correct_classified = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        _, predicted = torch.max(pred.data, 1)
        total += pred.size(0)
        correct_classified += (predicted == labels).sum().item()
    test_acc = correct_classified/total*100
    print("Test accuracy: {}%".format(test_acc))
    logging.info("Test accuracy: {}%".format(test_acc))