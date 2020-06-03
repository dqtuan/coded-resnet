"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""
from init_fusionnet import *
from fusionnet.unet import UnetModel
from invnet.iresnet import conv_iResNet as iResNet
from utils.provider import Provider

in_shape = (3, 32, 32)
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

inet_path = os.path.join(checkpoint_dir, 'inet/inet_{}_{}_e{}.pth'.format(args.dataset, data_name, args.resume))
if os.path.isfile(inet_path):
	print("-- Loading checkpoint '{}'".format(inet_path))
	inet.load_state_dict(torch.load(inet_path))
else:
	print("--- No checkpoint found at '{}'".format(inet_path))

#defaults
args.fnet_name = "unet_{}_{}_{}_{}_{}".format(args.dataset, args.name, args.nactors, args.gan_mode, args.reduction)
#defaults
args.norm='batch'
args.dataset_mode='aligned'
args.gpu_ids = [0]
args.pool_size=0
fnet = UnetModel(args)
fnet.setup(args)

init_epoch = int(args.resume_g)
fnet.load_networks(epoch=init_epoch)

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
lossf = 0
corrects = 0
def atanh(x):
	return 0.5*torch.log((1+x)/(1-x))

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
	# Second, G(A) = B
	fnet.loss_G_L1 = fnet.criterionL1(fnet.fake_B, fnet.real_B)
	# combine loss and calculate gradients
	fnet.loss_G = fnet.loss_G_L1

	""" Classification evaluation """
	inputs = inputs.view(N, A, C, H, W)
	inputs = inputs.view(N * A, C, H, W)
	_, z_c, _ = inet(inputs)
	z_c = z_c.view(N, A, z_c.shape[1], z_c.shape[2], z_c.shape[3])
	z_0 = z_c[:, 0, ...]
	z_c = z_c[:, 1:, ...].sum(dim=1)
	img_fused = atanh(fnet.fake_B)
	_, z_fused, _ = inet(img_fused)
	if args.reduction == 'mean':
		z_hat = A * z_fused - z_c
	elif args.reduction == 'sum':
		z_hat = z_fused - z_c
	out_hat = inet.module.classifier(z_hat)
	out = inet.module.classifier(z_0)

	_, y = torch.max(out.data, 1)
	_, y_hat = torch.max(out_hat.data, 1)

	correct = y_hat.eq(y.data).sum().cpu().item()/N
	if batch_idx % args.log_steps == 0:
		print("===> ({}/{}): L1 {:.4f} Match {:.4f}".format(
		batch_idx, total_steps,  fnet.loss_G_L1.item(), correct))

	lossf += fnet.loss_G_L1.item()
	corrects += correct

print('L1 {:.4f} Match: {:.4f}'.format(lossf/total_steps, corrects/total_steps))
print('Done')
