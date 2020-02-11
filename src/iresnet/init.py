"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import numpy as np
import visdom, os, sys, time, pdb, random, json

from utils.utils_cifar import train, test, std, mean, get_hms, interpolate
from models.conv_iResNet import conv_iResNet as iResNet
from models.conv_iResNet import multiscale_conv_iResNet as multiscale_iResNet
from configs import args

checkpoint_path = os.path.join(args.save_dir, 'checkpoints')
param_path = os.path.join(args.save_dir, 'params.txt')
trainlog_path = os.path.join(args.save_dir, "train_log.txt")
testlog_path = os.path.join(args.save_dir, "test_log.txt")
final_path = os.path.join(args.save_dir, 'final.txt')
sample_path = os.path.join(args.save_dir, 'samples')

dens_est_chain = [
    lambda x: (255. * x) + torch.zeros_like(x).uniform_(0., 1.),
    lambda x: x / 256.,
    lambda x: x - 0.5
]

train_chain = [transforms.Pad(4, padding_mode="symmetric"),
               transforms.RandomCrop(32),
               transforms.RandomHorizontalFlip(),
               transforms.ToTensor()]

test_chain = [transforms.ToTensor()]

clf_chain = [transforms.Normalize(mean[args.dataset], std[args.dataset])]
transform_train = transforms.Compose(train_chain + clf_chain)
transform_test = transforms.Compose(test_chain + clf_chain)

def try_make_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)

def save_images(sample_path, train_epoch, bs, samples, sample_name):
    torchvision.utils.save_image(samples.cpu(), os.path.join(sample_path, "epoch_{}_{}.jpg".format(train_epoch, sample_name)), int(bs**.5), normalize=True)


def anaylse_trace_estimation(model, testset, use_cuda, extension):
    # setup range for analysis
    numSamples = np.arange(10)*10 + 1
    numIter = np.arange(10)
    # setup number of datapoints
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    # TODO change

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        inputs, targets = Variable(inputs, requires_grad=True), Variable(targets)
        # compute trace
        out_bij, p_z_g_y, trace, gt_trace = model(inputs[:, :, :8, :8],
                                                       exact_trace=True)
        trace = [t.cpu().numpy() for t in trace]
        np.save('gtTrace'+extension, gt_trace)
        np.save('estTrace'+extension, trace)
        return

def test_spec_norm_v0(model, in_shapes, extension):
    i = 0
    j = 0
    params = [v for v in model.module.state_dict().keys() \
              if "bottleneck" and "weight" in v \
              and not "weight_u" in v \
              and not "weight_orig" in v \
              and not "bn1" in v and not "linear" in v\
              and not "weight_sigma" in v]
    print(len(params))
    print(len(in_shapes))
    svs = []
    for param in params:
      if i == 0:
        input_shape = in_shapes[j]
      else:
        input_shape = in_shapes[j]
        # input_shape[1] = int(input_shape[1] // 4)

      convKernel = model.module.state_dict()[param].cpu().numpy()
      # input_shape = input_shape[2:]
      # fft_coeff = np.fft.fft2(convKernel.reshape(input_shape), input_shape, axes=[2, 3])
      from IPython import embed; embed()
      fft_coeff = np.fft.fft2(convKernel.reshape(input_shape))
      t_fft_coeff = np.transpose(fft_coeff)
      U, D, V = np.linalg.svd(t_fft_coeff, compute_uv=True, full_matrices=False)
      Dflat = np.sort(D.flatten())[::-1]
      print("Layer "+str(j)+" Singular Value "+str(Dflat[0]))
      svs.append(Dflat[0])
      if i == 2:
        i = 0
        j+= 1
      else:
        i+=1
    np.save('singular_values'+extension, svs)
    return

def test_spec_norm(model, in_shapes, extension):
  i = 0
  j = 0
  params = [v for v in model.module.state_dict().keys()
            if "bottleneck" in v and "weight_orig" in v
            and not "weight_u" in v
            and not "bn1" in v
            and not "linear" in v]
  print(len(params))
  print(len(in_shapes))
  svs = []
  for param in params:
    input_shape = tuple(in_shapes[j])
    # get unscaled parameters from state dict
    convKernel_unscaled = model.module.state_dict()[param].cpu().numpy()
    # get scaling by spectral norm
    sigma = model.module.state_dict()[param[:-5] + '_sigma'].cpu().numpy()
    convKernel = convKernel_unscaled / sigma
    # compute singular values
    input_shape = input_shape[1:]
    fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
    t_fft_coeff = np.transpose(fft_coeff)
    D = np.linalg.svd(t_fft_coeff, compute_uv=False, full_matrices=False)
    Dflat = np.sort(D.flatten())[::-1]
    print("Layer "+str(j)+" Singular Value "+str(Dflat[0]))
    svs.append(Dflat[0])
    if i == 2:
      i = 0
      j+= 1
    else:
      i+=1
  return svs

def get_init_batch(dataloader, batch_size):
    """
    gets a batch to use for init
    """
    batches = []
    seen = 0
    for x, y in dataloader:
        batches.append(x)
        seen += x.size(0)
        if seen >= batch_size:
            break
    batch = torch.cat(batches)
    return batch
