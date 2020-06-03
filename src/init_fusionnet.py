"""
"""
from __future__ import print_function

import os
from math import log10
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision

from utils.helper import Helper
from utils.provider import FusionDataset
from configs import args


############ Settings ##############
TRAIN_FRACTION = 0.8
cudnn.benchmark = True
torch.cuda.manual_seed(args.seed)
device = torch.device("cuda:0")

args.fnet_name = "fnet_{}_{}_{}_{}".format(args.fname, args.dataset, args.iname, args.nactors)
args.inet_name = "inet_{}_{}".format(args.dataset, args.iname)

checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
args.inet_save_dir = os.path.join(checkpoint_dir, args.inet_name)
args.fnet_save_dir = os.path.join(checkpoint_dir, args.fnet_name)
Helper.try_make_dir(args.save_dir)
Helper.try_make_dir(checkpoint_dir)
Helper.try_make_dir(args.fnet_save_dir)
Helper.try_make_dir(args.inet_save_dir)

if args.dataset == 'cifar10':
    args.input_nc = 3
    args.output_nc = 3
	in_shape = (3, 32, 32)
else:
    args.input_nc = 1
    args.output_nc = 1
    in_shape = (1, 32, 32)

train_path = os.path.join(args.data_dir, "fusion/inet_{}_{}_{}_{}_data_train.npy".format(args.dataset, args.iname, args.nactors, args.reduction))
test_path = os.path.join(args.data_dir, "fusion/inet_{}_{}_{}_{}_data_test.npy".format(args.dataset, args.iname, args.nactors, args.reduction))
val_path = os.path.join(args.data_dir, "fusion/inet_{}_{}_{}_{}_data_val.npy".format(args.dataset, args.iname, args.nactors, args.reduction))
train_dataset = FusionDataset(train_path)
test_dataset = FusionDataset(test_path)
val_dataset = FusionDataset(val_path)

# train_size = int(nsamples * TRAIN_FRACTION)
# validation_size = nsamples - train_size
print("{}: Train {} Val {} Test {}".format(args.dataset, len(train_dataset), len(val_dataset), len(test_dataset)))

# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
train_loader = DataLoader(dataset=train_dataset,
    num_workers=args.threads,
    batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
    num_workers=args.threads,
    batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(dataset=val_dataset,
    num_workers=args.threads,
    batch_size=args.batch_size, shuffle=False)
