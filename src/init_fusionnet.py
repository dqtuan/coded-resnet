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
from configs_fusionnet import args

from fusionnet.networks import define_G

############ Settings ##############
TRAIN_FRACTION = 0.75
cudnn.benchmark = True
torch.cuda.manual_seed(args.seed)
device = torch.device("cuda:0" if args.cuda else "cpu")

# define paths
args.model_name = args.dataset + '_' + args.model_name
data_path = os.path.join(args.data_dir, 'data.npy')
checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
net_path = os.path.join(checkpoint_dir, "{}.pth".format(args.model_name))
sample_dir = os.path.join(args.save_dir, 'samples')
stat_path = os.path.join(args.save_dir, "{}.npy".format(args.model_name))

Helper.try_make_dir(args.save_dir)
Helper.try_make_dir(checkpoint_dir)
Helper.try_make_dir(sample_dir)

# data
if not(os.path.isfile(data_path)):
    Helper.try_make_dir(args.data_dir)
    Helper.combine_npys(args.npy_dir, data_path)

############ Data ##############
dict_data = np.load(data_path, allow_pickle=True)
dict_data = dict_data.item()
data = dict_data['data'].astype('float32')
targets = dict_data['targets']
nsamples = data.shape[0]
ntrains = int(nsamples * TRAIN_FRACTION)
stats = {
    'target': {},
    'input':{}
}
stats['target']['mean'] = np.mean(data[:, -1, ...], axis=(0, 2,3))
stats['target']['std'] = np.std(data[:, -1, ...], axis=(0, 2,3))
stats['target']['min'] = np.min(data[:, -1, ...], axis=(0, 2,3))
stats['target']['max'] = np.max(data[:, -1, ...], axis=(0, 2,3))

stats['input']['mean'] = np.mean(data[:, 0, ...], axis=(0, 2,3))
stats['input']['std'] = np.std(data[:, 0, ...], axis=(0, 2,3))
stats['input']['min'] = np.min(data[:, 0, ...], axis=(0, 2,3))
stats['input']['max'] = np.max(data[:, 0, ...], axis=(0, 2,3))
# STORE stats
print(stats)
np.save(stat_path, stats)
############ Standarlize ##############
snames = ['input', 'input', 'target']
for k in range(3):
    s = stats[snames[k]]
    for i in range(3):
        data[:, k, i, ...] = (data[:, k, i, ...] - s['mean'][i]) / s['std'][i]
print('Dataset: {}, {}'.format(nsamples, ntrains))
# apply transform??? check the range
train_set = torch.Tensor(data[:ntrains, ...])
test_set = torch.Tensor(data[ntrains:, ...])
train_loader = DataLoader(dataset=train_set,
    num_workers=args.threads,
    batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set,
    num_workers=args.threads,
    batch_size=args.batch_size, shuffle=False)

######## Losss ####
criterionMSE = nn.MSELoss(reduction='sum')
