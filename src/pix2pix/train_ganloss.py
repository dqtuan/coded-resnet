from __future__ import print_function
import argparse
import os
from math import log10
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--data_path',  default='../data/', required=False, help='facades')
parser.add_argument('--result_path',  default='../results/', required=False, help='facades')
parser.add_argument('--dataset',  default='cifar10', required=False, help='facades')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=8, help='8- 32/ 64-256generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=8, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()
print(opt)

checkpoint_path = os.path.join(opt.result_path, 'fusion/checkpoint/cifar10')
net_g_model_out_path = os.path.join(checkpoint_path, "netG_model_gan.pth")
net_d_model_out_path = os.path.join(checkpoint_path, "netD_model_gan.pth")
if not os.path.exists(checkpoint_path):
	os.mkdir(checkpoint_path)
sample_path = os.path.join(opt.result_path, 'fusion/sample-ganloss')
if not os.path.exists(sample_path):
	os.mkdir(sample_path)
if opt.cuda and not torch.cuda.is_available():
	raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True
torch.manual_seed(opt.seed)
if opt.cuda:
	torch.cuda.manual_seed(opt.seed)
device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Loading datasets')

data_path = os.path.join(opt.data_path, 'cifar10/data.npy')
data = np.load(data_path).astype('float32')
total_samples = data.shape[0]
ntrains = int(total_samples * 0.75)
train_set = torch.Tensor(data[:ntrains, ...])
test_set = torch.Tensor(data[ntrains:, ...])
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=False)

def norm(img):
	min = float(img.min())
	max = float(img.max())
	img = img.clamp_(min=min, max=max)
	img = img.add_(-min).div_(max - min + 1e-5)
	return (img - 0.5) * 2

print('===> Building models')
net_g = define_G(opt.input_nc*2, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_d = define_D(opt.input_nc, opt.ndf, 'basic', gpu_id=device)
if os.path.isfile(net_g_model_out_path):
	print('Load existing models')
	net_g = torch.load(net_g_model_out_path).to(device)
	net_d = torch.load(net_d_model_out_path).to(device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
# setup optimizer
optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

	for iteration, batch in enumerate(training_data_loader, 1):
		# forward
		batch = batch.to(device)
		# [-2.7; 2.3]
		input_a = batch[:, 0, ...]
		input_b = batch[:, 1, ...]
		target = batch[:, 2, ...]
		input = torch.cat((input_a, input_b), dim=1)

		fake = net_g(input)
		mse = criterionMSE(fake, target)
		######################
		# (1) Update D network
		######################

		optimizer_d.zero_grad()
		# train with fake
		pred_fake = net_d.forward(fake.detach())
		loss_d_fake = criterionGAN(pred_fake, False)
		# train with real
		pred_real = net_d.forward(target)
		loss_d_real = criterionGAN(pred_real, True)
		# Combined D loss
		loss_d = (loss_d_fake + loss_d_real) * 0.5
		loss_d.backward()
		optimizer_d.step()

		######################
		# (2) Update G network
		######################
		optimizer_g.zero_grad()
		# First, G(A) should fake the discriminator
		pred_fake = net_d.forward(fake)
		loss_g_gan = criterionGAN(pred_fake, True)
		# Second, G(A) = B
		loss_g_l1 = criterionL1(fake, target) * opt.lamb
		loss_g = loss_g_gan + loss_g_l1
		loss_g.backward()
		optimizer_g.step()

		print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} MSE {:.4f}".format(
			epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item(), mse.item()))

	update_learning_rate(net_g_scheduler, optimizer_g)
	update_learning_rate(net_d_scheduler, optimizer_d)


	#checkpoint
	if epoch % 5 == 0:
		torch.save(net_g, net_g_model_out_path)
		print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

	# train
	lst = ['a', 'b', 'target', 'fake']
	samples = [input_a, input_b, target, fake]
	for i in range(len(lst)):
		save_path = os.path.join(sample_path, "epoch_{}_{}.jpg".format(epoch, lst[i]))
		torchvision.utils.save_image(samples[i].cpu(), save_path, int(opt.batch_size**.5), normalize=True)

lossf = []
for iteration, batch in enumerate(testing_data_loader, 1):
	# forward
	batch = batch.to(device)
	# [-2.7; 2.3]
	input_a = batch[:, 0, ...]
	input_b = batch[:, 1, ...]
	target = batch[:, 2, ...]
	input = torch.cat((input_a, input_b), dim=1)
	# train with fake
	# [-1, 1]
	pred = net_g(input)
	loss = criterionMSE(pred, target)
	lossf.append(loss.detach().cpu().data)
print('Test - MSE: ', np.mean(lossf))
