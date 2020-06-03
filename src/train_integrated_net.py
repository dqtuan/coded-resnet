"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
"""

from init_invnet import *
from invnet.ops.mixup_ops import to_one_hot
from fusionnet.networks import define_G, get_scheduler
from torch.utils.tensorboard import SummaryWriter
writer_train = SummaryWriter('../results/runs/' + args.fnet_name + '_train')
writer_val = SummaryWriter('../results/runs/' + args.fnet_name + '_val')
########################## Helpers ##################

def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
	"""
	Compute the knowledge-distillation (KD) loss given outputs, labels.
	"Hyperparameters": temperature and alpha
	NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
	and student expects the input tensor to be log probabilities! See Issue #2
	"""
	T = temperature # small T for small network student
	beta = (1. - alpha) * T * T # alpha for student: small alpha is better
	teacher_loss = criterionKLD(F.log_softmax(outputs/T, dim=1),
							 F.softmax(teacher_outputs/T, dim=1))
	# student_loss = F.cross_entropy(outputs, labels)
	student_loss = bce_loss(softmax(outputs), labels)
	KD_loss =  beta * teacher_loss + alpha * student_loss

	return KD_loss

def atanh(x, eps=1e-4):
	return 0.5*torch.log((1+x + eps)/(1-x + eps))

########################## Load ##################

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
inet = torch.nn.DataParallel(inet, range(torch.cuda.device_count()))
if args.resume > 0:
	save_filename = '%s_net.pth' % (args.resume)
	inet_path = os.path.join(args.inet_save_dir, save_filename)
	if os.path.isfile(inet_path):
		print("-- Loading checkpoint '{}'".format(inet_path))
		inet.load_state_dict(torch.load(inet_path))
	else:
		print("--- No checkpoint found at '{}'".format(inet_path))
		sys.exit('Done')


####################### Fusion ###########################
#defaults
args.norm='batch'
args.dataset_mode='aligned'
args.gpu_ids = [0]
args.pool_size=0
A = args.nactors
N = args.batch_size
M = N // A
C, H, W = in_shape

fnet = define_G(args.input_nc, args.output_nc, args.ngf, args.netG, args.norm, not args.no_dropout, args.init_type, args.init_gain, args.gpu_ids, nactors=args.nactors)
init_epoch = int(args.resume_g)
if init_epoch > 0:
	save_filename = '%s_net.pth' % (args.resume_g)
	fnet_path = os.path.join(args.fnet_save_dir, save_filename)
	fnet.load_state_dict(torch.load(fnet_path))
	print("-- Loading checkpoint '{}'".format(fnet_path))
	# fnet.load_networks(epoch=init_epoch)

optim_fnet = optim.Adam(fnet.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
fnet_scheduler = get_scheduler(optim_fnet, args)

########################## Training ###################################
print('|  Train Epochs: ' + str(args.epochs))
print('|  Initial Learning Rate: ' + str(args.lr))
elapsed_time = 0
total_steps = len(iter(trainloader))
for epoch in range(init_epoch + 1, init_epoch + 1 + args.epochs):
	inet.eval()
	start_time = time.time()
	losses = AverageMeter()
	losses_D = AverageMeter()
	losses_L = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	run_times = np.zeros(8)

	lr = Helper.learning_rate(args.lr, epoch - init_epoch, factor=40)
	Helper.update_lr(optim_fnet, lr)
	lr = optim_fnet.param_groups[0]['lr']
	print('learning rate = %.7f' % lr)
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		# # counters
		inputs = Variable(inputs).cuda()
		targets = Variable(targets).cuda()
		# fusion inputs
		x_g = inputs.view(M, A, C, H, W).view(M, A*C, H, W)
		x_fused_g = atanh(fnet(x_g))
		# mixup target at z layer

		_, z, _ = inet(inputs)
		N, Cz, Hz, Wz = z.shape
		z = z.view(M, A, Cz, Hz, Wz).mean(dim=1)
		x_fused = inet.module.inverse(z)
		out = inet.module.classifier(z)
		out_hat, _, _ = inet(x_fused_g)

		target_reweighted = to_one_hot(targets, args.nClasses).view(M, A, args.nClasses).mean(dim=1)
		loss_distill = loss_fn_kd(out_hat, target_reweighted, out, alpha=0.1, temperature=6)
		loss_xl1 = criterionL1(x_fused_g, x_fused)
		loss_tl1 = criterionL1(torch.tanh(x_fused_g), torch.tanh(x_fused))

		loss = loss_distill * args.lambda_distill + args.lambda_L1 * loss_xl1

		# measure accuracy and record loss
		_, labels = torch.max(target_reweighted.data, 1)
		prec1, prec5 = Helper.accuracy(out_hat, labels, topk=(1, 5))
		losses.update(loss.item(), out_hat.size(0))
		top1.update(prec1.item(), out_hat.size(0))
		top5.update(prec5.item(), out_hat.size(0))
		losses_L.update(loss_xl1.item(), out_hat.size(0))
		losses_D.update(loss_distill.item(), out_hat.size(0))

		optim_fnet.zero_grad()
		loss.backward()
		optim_fnet.step()

		if batch_idx % args.log_steps == 0:
			print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f Acc@5: %.3f Distill: %.3f L1: %.3f Tanh-L1: %.3f'
						 % (epoch, args.epochs, batch_idx+1,
							(len(trainset)//args.batch_size)+1, losses.avg,
							top1.avg, top5.avg, losses_D.avg, losses_L.avg, loss_tl1.item()))

	val_acc1, val_acc5 = Tester.evaluate_fgan_loss(inet, fnet, valloader, args.nactors)
	print("Val data: Acc@1: %.4f Acc@5: %.4f" % (val_acc1, val_acc5))
	writer_train.add_scalar('loss', loss, epoch)
	writer_train.add_scalar('l1', loss_xl1, epoch)
	writer_train.add_scalar('acc1', top1.avg, epoch)
	writer_train.add_scalar('acc5', top5.avg, epoch)
	writer_val.add_scalar('acc1', val_acc1, epoch)
	writer_val.add_scalar('acc5', val_acc1, epoch)

	epoch_time = time.time() - start_time
	elapsed_time += epoch_time
	print('| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))
	if (epoch - 1) % args.save_steps == 0 or epoch == init_epoch + args.epochs:
		Helper.save_networks(fnet, args.fnet_save_dir, epoch)
	print('')

########################## Store ##################
Tester.evaluate_fgan(inet, fnet, testloader, args.nactors)
print('Done')
