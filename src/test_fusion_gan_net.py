"""
	Doing fusion 2 images to match target images
	Supervised tasks
"""
from init_fusionnet import *
from fusionnet.pix2pix import Pix2PixModel
from invnet.iresnet import conv_iResNet as iResNet
from utils.provider import Provider

in_shape = (1, 32, 32)
trainset, testset, in_shape = Provider.load_data(args.dataset, args.data_dir)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

inet = iResNet(nBlocks=args.nBlocks,
				nStrides=args.nStrides,
				nChannels=args.nChannels,
				nClasses=args.nClasses,
				init_ds=args.init_ds,
				inj_pad=args.inj_pad,
				in_shape=in_shape,
				coeff=args.coeff,
				numTraceSamples=args.numTraceSamples,
				numSeriesTerms=args.numSeriesTerms,
				n_power_iter = args.powerIterSpectralNorm,
				density_estimation=args.densityEstimation,
				actnorm=(not args.noActnorm),
				learn_prior=(not args.fixedPrior),
				nonlin=args.nonlin).to(device)

init_batch = Helper.get_init_batch(trainloader, args.init_batch)
print("initializing actnorm parameters...")
with torch.no_grad():
	inet(init_batch.to(device), ignore_logdet=True)
print("initialized")
inet = torch.nn.DataParallel(inet, range(torch.cuda.device_count()))

save_filename = '%s_net.pth' % (args.resume)
args.inet_name = "inet_{}_{}".format(args.dataset, data_name)
args.inet_save_dir = os.path.join(checkpoint_dir, args.inet_name)
inet_path = os.path.join(args.inet_save_dir, save_filename)
if os.path.isfile(inet_path):
	print("-- Loading checkpoint '{}'".format(inet_path))
	inet.load_state_dict(torch.load(inet_path))
else:
	print("--- No checkpoint found at '{}'".format(inet_path))
	sys.exit('Done')

#defaults
# args.input_nc = args.input_nc * args.nactors
args.norm='batch'
args.dataset_mode='aligned'
args.gpu_ids = [0]
#train
args.pool_size=0
args.lambda_L1 = args.lamb

args.fnet_name = "{}_{}_{}_{}_{}".format(args.dataset, args.name, args.nactors, args.gan_mode, args.reduction)
fnet = Pix2PixModel(args)
fnet.setup(args)              # regular setup: load and print networks; create schedulers
init_epoch = int(args.resume_g)
fnet.load_networks(epoch=init_epoch)
# Training

if args.flag_test:
	print('Evaluate on Test Set')
	data_loader = test_loader
elif args.flag_val:
	print('Evaluate on Validation Set')
	data_loader = val_loader
else:
	print('Evaluate on Train Set')
	data_loader = train_loader

total_steps = len(iter(data_loader))
lossf = [0, 0]
corrects = 0
def atanh(x):
	return 0.5*torch.log((1+x)/(1-x))

def cart2polar(p):
	# p = [N, D]
	# angles: reverse [x_d, ... , x_1]
	r = torch.norm(p, dim=1)
	s = torch.sqrt(torch.cumsum(p**2, dim=1))
	angles = torch.acos(p/s)
	## [N, D-1]
	angles = angles[:, 1:]
	return r, angles

def polar2cart(r, angles):
	# r= [N], angles=[N, D-1]
	p = torch.zeros(angles.shape[0], angles.shape[1] + 1).cuda()
	prev = torch.ones(angles.shape[0]).cuda()
	for i in range(1, p.shape[1]):
		k = p.shape[1] - i
		p[:, k] = prev * torch.cos(angles[:, k-1])
		prev = prev * torch.sin(angles[:, k-1])
	p[:, 0] = prev
	p = p * r.unsqueeze(1).expand(p.shape)
	return p

for batch_idx, (images, targets) in enumerate(data_loader):
	inputs = images.to(device)
	targets = targets.to(device)
	N, A, C, H, W = inputs.shape
	# inputs = inputs[:, torch.randperm(A), ...]
	inputs = inputs.view(N, C*A, H, W)

	fnet.set_input(inputs, targets)         # unpack data from dataset and apply preprocessing
	"""Calculate GAN and L1 loss for the generator"""
	# First, G(A) should fake the discriminator
	fnet.fake_B = fnet.netG(fnet.real_A)
	fake_AB = torch.cat((fnet.real_A, fnet.fake_B), 1)
	pred_fake = fnet.netD(fake_AB)
	fnet.loss_G_GAN = fnet.criterionGAN(pred_fake, True)
	# Second, G(A) = B
	fnet.loss_G_L1 = fnet.criterionL1(fnet.fake_B, fnet.real_B) * fnet.opt.lambda_L1
	# combine loss and calculate gradients
	fnet.loss_G = fnet.loss_G_GAN + fnet.loss_G_L1

	""" Classification evaluation """
	inputs = inputs.view(N, A, C, H, W)
	inputs = inputs.view(N * A, C, H, W)
	_, z_c, _ = inet(inputs)
	_, C_z, H_z, W_z = z_c.shape
	D = C_z*H_z*W_z

	# [NA, D-1]
	z_c = z_c.view(N, A, C_z, H_z, W_z)
	z_0 = z_c[:, 0, ...]
	z_c = z_c[:, 1:, ...]
	img_fused = atanh(fnet.fake_B)
	_, z_fused, _ = inet(img_fused)

	# r_c, angle_c = cart2polar(z_c.contiguous().view(N * (A-1), D))
	# r_c = r_c.view(N, A-1)
	# angle_c = angle_c.view(N, A-1, D-1)
	#
	# r_fused, angle_fused = cart2polar(z_fused.view(N, D))
	#
	# angle_missed = angle_fused * A - angle_c.sum(dim=1)
	# r_missed = torch.exp(torch.log(r_fused)*A - torch.log(r_c).sum(dim=1))
	#
	# z_hat = polar2cart(r_missed, angle_missed)
	# z_hat = z_hat.view(N, C_z, H_z, W_z)

	if args.reduction == 'mean':
		z_hat = A * z_fused - z_c.sum(dim=1)
	elif args.reduction == 'sum':
		z_hat = z_fused - z_c

	out_hat = inet.module.classifier(z_hat)
	out = inet.module.classifier(z_0)

	_, y = torch.max(out.data, 1)
	_, y_hat = torch.max(out_hat.data, 1)

	correct = y_hat.eq(y.data).sum().cpu().item()/N
	# from IPython import embed; embed()
	if batch_idx % args.log_steps == 0:
		print("===> ({}/{}): Loss_GAN: {:.4f} L1 {:.4f} Match {:.4f}".format(
		batch_idx, total_steps, fnet.loss_G_GAN.item(), fnet.loss_G_L1.item()/args.lambda_L1, correct))

	lossf[0] += fnet.loss_G_L1.item()/args.lambda_L1
	lossf[1] += fnet.loss_G_GAN.item()
	corrects += correct

print('GAN: {:.4f} L1 {:.4f} Match: {:.4f}'.format(lossf[1]/total_steps, lossf[0]/total_steps, corrects/total_steps))
print('Done')
