"""
	Doing fusion 2 images to match target images
	Supervised tasks
"""
from init_fusionnet import *
from fusionnet.unet import UnetModel
from torch.utils.tensorboard import SummaryWriter

args.fnet_name = "unet_{}_{}_{}_{}_{}".format(args.dataset, args.name, args.nactors, args.gan_mode, args.reduction)

writer = SummaryWriter('../results/runs/' + args.fnet_name)

#defaults
args.norm='batch'
args.dataset_mode='aligned'
args.gpu_ids = [0]
args.pool_size=0

fnet = UnetModel(args)
fnet.setup(args)              # regular setup: load and print networks; create schedulers

init_epoch = int(args.resume_g)
if init_epoch > 0:
	fnet.load_networks(epoch=init_epoch)

total_steps = len(iter(train_loader))
for epoch in range(1 + init_epoch, args.epochs + 1 + init_epoch):
	for batch_idx, (images, targets) in enumerate(train_loader):
		inputs = images.to(device)
		targets = targets.to(device)
		N, A, C, H, W = inputs.shape
		# inputs = inputs[:, torch.randperm(A), ...]
		inputs = inputs.view(N, C*A, H, W)

		# forward
		fnet.set_input(inputs, targets)         # unpack data from dataset and apply preprocessing
		fnet.optimize_parameters()   # calculate loss functions, get gradients, update network weights

		if batch_idx % args.log_steps == 0:
			losses = fnet.get_current_losses()
			print("===> Epoch[{}]({}/{}): G-L1 {:.4f}".format(
			epoch, batch_idx, total_steps,  losses['G_L1']))

	writer.add_scalar('L1 loss', losses['G_L1'], epoch)
	#checkpoint
	if (epoch - 1) % args.save_steps == 0 or epoch == args.epochs + init_epoch:
		print('saving the latest fnet (epoch %d, total_iters %d)' % (epoch, total_steps))
		fnet.save_networks(epoch)

	fnet.update_learning_rate()                     # update learning rates at the end of every epoch.
print('Done')
