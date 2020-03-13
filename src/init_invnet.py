"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
	Loading data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import visdom, os, sys, time, pdb, random, json

from utils.helper import Helper, AverageMeter
from utils.plotter import Plotter
from utils.tester import Tester
from utils.provider import Provider
from configs_invnet import args

from invnet.iresnet import conv_iResNet as iResNet

########## Setting ####################
device = torch.device("cuda:0")
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True

########## Paths ####################
args.model_name = args.dataset + '_' + args.model_name
checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
# param_path = os.path.join(args.save_dir, 'params.txt')
# trainlog_path = os.path.join(args.save_dir, "train_log.txt")
# testlog_path = os.path.join(args.save_dir, "test_log.txt")
# final_path = os.path.join(args.save_dir, 'final.txt')
sample_dir = os.path.join(args.save_dir, 'samples')
resume_path = os.path.join(checkpoint_dir, '{}_e{}.pth'.format(args.model_name, args.resume))

# setup logging with visdom
viz = visdom.Visdom(port=args.vis_port, server="http://" + args.vis_server)
assert viz.check_connection(), "Could not make visdom"

# with open(param_path, 'w') as f:
# 	f.write(json.dumps(args.__dict__))
# train_log = open(trainlog_path, 'w')
Helper.try_make_dir(args.save_dir)
Helper.try_make_dir(checkpoint_dir)
Helper.try_make_dir(sample_dir)

trainset, testset = Provider.load_data(args.dataset, args.data_root)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)
in_shape = (3, 32, 32)
args.nClasses = 10

########## Loss ####################
bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()
criterionCE = nn.CrossEntropyLoss()
criterionMSE = nn.MSELoss(reduction='mean')

def analyse(args, model, in_shapes, trainloader, testloader):
	if args.evaluate:
		test_log = open(os.path.join(args.save_dir, "test_log.txt"), 'w')
		Tester.test(best_objective, args, model, start_epoch, testloader, viz, use_cuda, test_log)
		return True

	if args.eval_invertibility:
		Tester.eval_invertibility(model, testloader, sample_dir, args.model_name, num_epochs=args.resume)
		return True

	if args.generate_fused_image:
		data_dir = '../data/fusion'
		Helper.try_make_dir(data_dir)
		Tester.generate_inversed_images(model, trainloader, data_dir)
		return True

	if args.eval_fusion_net:
		from fusionnet.networks import define_G
		fnet = define_G(
				input_nc=3,
				output_nc=3,
				ngf=8,
				norm='batch',
				use_dropout=False,
				init_type='normal',
				init_gain=0.02,
				gpu_id=device)
		net_path = '../results/fusion/checkpoints/cifar10_mixup_fusion_z.pth'
		stats = np.load('../results/fusion/cifar10_mixup_fusion_z.npy', allow_pickle=True)
		stats = stats.item()
		fnet.load_state_dict(torch.load(net_path))
		Tester.evaluate_fusion_net(model, fnet, testloader, stats)
		return True

	if args.eval_sensitivity:
		data_path = '../../data/pix2pix/data.npy'
		data = np.load(data_path).astype('float32')
		train_set = torch.Tensor(data)
		test_loader = torch.utils.data.DataLoader(dataset=train_set,
		    batch_size=args.batch, shuffle=True)
		Tester.eval_sensitivity(model, test_loader, args.eps)
		return True

	if args.test_inversed_images:
		# from torch.utils.data.sampler import SubsetRandomSampler
		labels = trainset.targets
		indices = np.argwhere(np.asarray(labels) == 1)
		indices = indices.reshape(len(indices))
		trainset = torch.utils.data.Subset(trainset, indices)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, num_workers=2)
		Tester.test_inversed_images(model, trainloader)
		return True

	if args.plot_tnse:
		num_classes = 7
		labels = trainset.targets
		indices = np.argwhere(np.asarray(labels) < num_classes)
		indices = indices.reshape(len(indices))
		trainset = torch.utils.data.Subset(trainset, indices)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, num_workers=2)
		Tester.plot_latent(model, trainloader, num_classes)
		return True

	if args.analysisTraceEst:
		Tester.anaylse_trace_estimation(model, testset, use_cuda, args.extension)
		return True

	if args.norm:
		svs = Tester.test_spec_norm(model, in_shapes, args.extension)
		svs = svs[0]
		print(np.min(svs), np.median(svs), np.max(svs))
		return True

	if args.interpolate:
		Tester.interpolate(model, testloader, testset, start_epoch, use_cuda, best_objective, args.dataset)
		return True

	return False
