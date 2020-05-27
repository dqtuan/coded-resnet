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
from torch.optim import lr_scheduler

import numpy as np
import visdom, os, sys, time, pdb, random, json

from utils.helper import Helper, AverageMeter
from utils.plotter import Plotter
from utils.tester import Tester
from utils.provider import Provider
from configs import args

from invnet.iresnet import conv_iResNet as iResNet

########## Setting ####################
device = torch.device("cuda:0")
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True

########## Paths ####################
root = 'inet'
checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
inet_sample_dir = os.path.join(args.save_dir, 'samples/inet')

# setup logging with visdom
viz = visdom.Visdom(port=args.vis_port, server="http://" + args.vis_server)
assert viz.check_connection(), "Could not make visdom"

Helper.try_make_dir(args.save_dir)
Helper.try_make_dir(checkpoint_dir)
Helper.try_make_dir(inet_sample_dir)

########### DATA #################
trainset, testset, in_shape = Provider.load_data(args.dataset, args.data_dir)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

########## Loss ####################
bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()
criterionCE = nn.CrossEntropyLoss().cuda()
criterionMSE = nn.MSELoss(reduction='sum').cuda()

if args.dataset == 'cifar10':
    args.input_nc = 3
    args.output_nc = 3
else:
    args.input_nc = 1
    args.output_nc = 1

# Evaluation
def analyse(args, model, in_shapes, trainloader, testloader):
	if args.evaluate:
		Tester.eval_inv(model, testloader, inet_sample_dir, args.inet_name, nactors=args.nactors, num_epochs=args.resume)
		return True

	if args.plot_fused:
		sample_path = os.path.join(args.save_dir, 'samples')
		Tester.plot_fused_images(args.inet_name, sample_path, model, testloader, args.nactors)
		return True

	if args.evalInv:
		Tester.eval_inv(model, testloader, inet_sample_dir, args.inet_name, nactors=args.nactors, num_epochs=args.resume)
		return True

	if args.sampleImg:
		data_dir = '../data/fusion'
		train_path = os.path.join(data_dir, args.inet_name + '_{}_{}_data_train.npy'.format(args.nactors, args.reduction))
		test_path = os.path.join(data_dir, args.inet_name + '_{}_{}_data_test.npy'.format(args.nactors, args.reduction))
		val_path = os.path.join(data_dir, args.inet_name + '_{}_{}_data_val.npy'.format(args.nactors, args.reduction))
		Helper.try_make_dir(data_dir)
		print("Generating train set")
		Tester.sample_fused_data(model, trainloader, train_path, nactors=args.nactors, reduction=args.reduction, niters=3)
		print("Generating test set")
		Tester.sample_fused_data(model, testloader, test_path, nactors=args.nactors, reduction=args.reduction)
		print("Generating validation set")
		Tester.sample_fused_data(model, trainloader, val_path, nactors=args.nactors, reduction=args.reduction)

		return True

	if args.evalFnet:
		from fusionnet.networks import define_G
		if args.concat_input:
			ninputs = 1
			input_nc = args.input_nc * args.nactors
		else:
			ninputs = args.nactors
			input_nc = args.input_nc

		fnet = define_G(
				input_nc=input_nc,
				output_nc=in_shape[0],
				nactors=ninputs,
				ngf=8,
				norm='batch',
				use_dropout=False,
				init_type='normal',
				init_gain=0.02,
				gpu_id=device)
		# fname = 'cifar10_mixup_fusion_z'
		stat_path = os.path.join(args.data_dir, "fnet/{}_{}_stats.npy".format(args.inet_name, args.nactors))
		fnet_path = os.path.join(checkpoint_dir, 'fnet/fnet_{}_{}_e{}.pth'.format(args.inet_name, args.nactors,  args.resume_g))
		stats = np.load(stat_path, allow_pickle=True)
		stats = stats.item()
		fnet.load_state_dict(torch.load(fnet_path))
		Tester.evaluate_fusion_net(model, fnet, testloader, stats, nactors=args.nactors, nchannels=in_shape[0], concat_input=args.concat_input)

		return True

	if args.evalFnet2:
		from invnet.fresnet import conv_iResNet as fResNet
		args.init_ds = 1
		fnet = fResNet(nBlocks=args.nBlocks,
						nStrides=args.nStrides,
						nChannels=args.nChannels,
						nClasses=10,
						init_ds=args.init_ds,
						inj_pad=args.inj_pad,
						in_shape=in_shape,
						nactors=args.nactors,
						input_nc=args.input_nc,
						coeff=args.coeff,
						numTraceSamples=args.numTraceSamples,
						numSeriesTerms=args.numSeriesTerms,
						n_power_iter = args.powerIterSpectralNorm,
						density_estimation=args.densityEstimation,
						actnorm=(not args.noActnorm),
						learn_prior=(not args.fixedPrior),
						nonlin=args.nonlin).to(device)

		fnet = torch.nn.DataParallel(fnet, range(torch.cuda.device_count()))

		# fname = 'cifar10_mixup_fusion_z'
		stat_path = os.path.join(args.data_dir, "fnet/{}_mixup_hidden_{}_stats.npy".format(args.dataset, args.nactors))
		fnet_path = os.path.join(checkpoint_dir, 'fnet/fnet_{}_{}_e{}.pth'.format(args.inet_name, args.nactors,  args.resume_g))
		stats = np.load(stat_path, allow_pickle=True)
		stats = stats.item()
		if os.path.isfile(fnet_path):
			print("-- Loading checkpoint '{}'".format(fnet_path))
			# fnet = torch.load(fnet_path)
			fnet.load_state_dict(torch.load(fnet_path))
		else:
			print("No checkpoint found at: ", fnet_path)
		Tester.evaluate_fusion_net(model, fnet, testloader, stats, nactors=args.nactors, nchannels=in_shape[0], concat_input=args.concat_input)

		return True

	if args.evalFgan:
		from fusionnet.pix2pix import Pix2PixModel
		# args.input_nc = args.input_nc * args.nactors
		args.norm='batch'
		args.dataset_mode='aligned'
		args.gpu_ids = [0]
		#train
		args.pool_size=0
		args.gan_mode='lsgan'
		args.netD = 'pixel'
		args.name = 'large'
		args.fnet_name = "{}_{}_{}_{}".format(args.dataset, args.name, args.nactors, args.gan_mode)
		args.isTrain = False

		pix2pix = Pix2PixModel(args)
		pix2pix.setup(args)              # regular setup: load and print networks; create schedulers
		pix2pix.load_networks(epoch=int(args.resume_g))
		fnet = pix2pix.netG

		sample_path = os.path.join(args.save_dir, 'samples')
		if args.flag_test:
			data_loader = testloader
			print('Evaluate on Test set')
		else:
			data_loader = trainloader
			print('Evaluate on Train set')
		Tester.plot_reconstruction(args.fnet_name, sample_path, model, fnet, data_loader, args.nactors)

		Tester.evaluate_fgan(model, fnet, testloader, stats=None, nactors=args.nactors, nchannels=in_shape[0], concat_input=True)

		return True

	if args.evalDistill:
		from fusionnet.networks import define_G
		ninputs = args.nactors
		input_nc = args.input_nc

		fnet = define_G(
				input_nc=input_nc,
				output_nc=in_shape[0],
				nactors=ninputs,
				ngf=8,
				norm='batch',
				use_dropout=False,
				init_type='normal',
				init_gain=0.02,
				gpu_id=device)
		fname = 'fnet_integrated_{}_{}_perturb_e{}.pth'.format(args.inet_name, args.nactors, args.resume_g)
		# fname = 'fnet_integrated_cifar10_mixup_hidden_4.pth'
		fnet_path = os.path.join(checkpoint_dir, fname)
		fnet.load_state_dict(torch.load(fnet_path))
		# fnet = torch.load(fnet_path)
		Tester.evaluate_fnet(model, fnet, testloader, nactors=args.nactors, nchannels=in_shape[0], concat_input=args.concat_input)

		return True

	if args.evalSen:
		data_path = '../data/pix2pix/data.npy'
		data = np.load(data_path).astype('float32')
		train_set = torch.Tensor(data)
		test_loader = torch.utils.data.DataLoader(dataset=train_set,
		    batch_size=args.batch_size, shuffle=True)
		Tester.eval_sensitivity(model, test_loader, args.eps)
		return True

	if args.testInv:
		# from torch.utils.data.sampler import SubsetRandomSampler
		labels = trainset.targets
		indices = np.argwhere(np.asarray(labels) == 1)
		indices = indices.reshape(len(indices))
		trainset = torch.utils.data.Subset(trainset, indices)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)
		Tester.test_inversed_images(model, trainloader)
		return True

	if args.plotTnse:
		num_classes = 7
		labels = trainset.targets
		indices = np.argwhere(np.asarray(labels) < num_classes)
		indices = indices.reshape(len(indices))
		trainset = torch.utils.data.Subset(trainset, indices)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)
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
