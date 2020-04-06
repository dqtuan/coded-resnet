"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
"""

from init_invnet import *
from fusionnet.networks import define_G
from torch.optim import lr_scheduler


def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
	"""
	Compute the knowledge-distillation (KD) loss given outputs, labels.
	"Hyperparameters": temperature and alpha
	NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
	and student expects the input tensor to be log probabilities! See Issue #2
	"""
	T = temperature # small T for small network student
	beta = (1. - alpha) * T * T # alpha for student: small alpha is better
	teacher_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
							 F.softmax(teacher_outputs/T, dim=1))
	student_loss = F.cross_entropy(outputs, labels)
	KD_loss =  beta * teacher_loss + alpha * student_loss

	return KD_loss


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
print("-- Loading checkpoint '{}'".format(inet_path))
inet.load_state_dict(torch.load(inet_path))

if args.concat_input:
	ninputs = 1
	args.input_nc = args.input_nc * args.nactors
else:
	ninputs = args.nactors
# define fnet
fnet = define_G(
		input_nc=args.input_nc,
		output_nc=args.output_nc,
		nactors=ninputs,
		ngf=args.ngf,
		norm='batch',
		use_dropout=False,
		init_type='normal',
		init_gain=0.02,
		gpu_id=device)

fname = 'fnet_integrated_{}_{}.pth'.format(args.model_name, args.nactors)
fnet_path = os.path.join(checkpoint_dir, fname)
if os.path.isfile(fnet_path):
	print("-- Loading checkpoint '{}'".format(fnet_path))
	fnet.load_state_dict(torch.load(fnet_path))

in_shapes = inet.module.get_in_shapes()
optim_fnet = optim.Adam(fnet.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
fnet_scheduler = Helper.get_scheduler(optim_fnet, args)

########################## Training ##################
print('|  Train Epochs: ' + str(args.epochs))
print('|  Initial Learning Rate: ' + str(args.lr))
elapsed_time = 0
total_steps = len(iter(trainloader))
criterion = torch.nn.CrossEntropyLoss()
criterionMSE = torch.nn.MSELoss(reduction='mean')

for epoch in range(1, 1+args.epochs):
	inet.eval()
	start_time = time.time()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		cur_iter = (epoch - 1) * len(trainloader) + batch_idx
		# if first epoch use warmup
		if epoch - 1 <= args.warmup_epochs:
			this_lr = args.lr * float(cur_iter) / (args.warmup_epochs * len(trainloader))
			Helper.update_lr(optim_fnet, this_lr)

		targets = Variable(targets).cuda()
		inputs = Variable(inputs).cuda()
		batch_size = inputs.shape[0]
		f_x = inputs.clone().unsqueeze(1)
		for i in range(1, args.nactors):
			x = inputs[torch.randperm(batch_size), :]
			f_x = torch.cat([f_x, x.clone().unsqueeze(1)], dim=1)
			_, z, _ = inet(x)
			if i == 1:
				z_c = z
			else:
				z_c += z

		img_fused = fnet(f_x)
		_, z_fused, _ = inet(img_fused)
		z_hat = args.nactors * z_fused - z_c
		out_hat = inet.module.classifier(z_hat)
		out, _, target_reweighted = inet(inputs, targets)

		# inet loss
		loss_classify = bce_loss(softmax(out_hat), target_reweighted)
		loss_distill = loss_fn_kd(out_hat, targets, out, alpha=0.1, temperature=6)
		# loss_z = criterionMSE(input_2, input_2_hat)
		loss = loss_distill + loss_classify

		# measure accuracy and record loss
		prec1, prec5 = Helper.accuracy(out_hat, targets, topk=(1, 5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1.item(), inputs.size(0))
		top5.update(prec5.item(), inputs.size(0))

		optim_fnet.zero_grad()
		loss.backward()
		optim_fnet.step()

		if batch_idx % args.log_steps == 0:
			sys.stdout.write('\r')
			sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f Acc@5: %.3f Distill: %.3f'
						 % (epoch, args.epochs, batch_idx+1,
							(len(trainset)//args.batch_size)+1, loss.data.item(),
							top1.avg, top5.avg, loss_distill.item()))
			sys.stdout.flush()

	if epoch % args.save_steps == 0:
		torch.save(fnet, fnet_path)
########################## Store ##################
torch.save(fnet, fnet_path)
