"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 08/14/2019
"""

from init_invnet import *
from torch.optim import lr_scheduler

model = iResNet(nBlocks=args.nBlocks,
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
	model(init_batch.to(device), ignore_logdet=True)
print("initialized")
model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))

if os.path.isfile(resume_path):
	print("-- Loading checkpoint '{}'".format(resume_path))
	"""
	checkpoint = torch.load(resume_path)
	# model = checkpoint['model']
	best_model = checkpoint['model']
	# load dict only
	model.load_state_dict(best_model.state_dict())
	best_objective = checkpoint['objective']
	print('----- Objective: {:.4f}'.format(best_objective))
	"""
	model.load_state_dict(torch.load(resume_path))
else:
	print("--- No checkpoint found at '{}'".format(resume_path))
	args.resume = 0

in_shapes = model.module.get_in_shapes()
if args.optimizer == "adam":
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
	optimizer = optim.SGD(model.parameters(), lr=args.lr,
						  momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)

scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
##### Analysis ######
if analyse(args, model, in_shapes, trainloader, testloader):
	sys.exit('Done')

##### Training ######
print('|  Train: mixup: {} mixup_hidden {} alpha {} epochs {}'.format(args.mixup, args.mixup_hidden, args.mixup_alpha, args.epochs))
print('|  Initial Learning Rate: ' + str(args.lr))
elapsed_time = 0
test_objective = -np.inf
epoch = int(args.resume)
for epoch in range(epoch + 1, epoch + 1 + args.epochs):
	start_time = time.time()
	model.train()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	# update lr for this epoch (for classification only)
	# lr = Helper.learning_rate(args.lr, epoch)
	# Helper.update_lr(optimizer, lr)
	# print('|Learning Rate: ' + str(lr))
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		cur_iter = (epoch - 1) * len(trainloader) + batch_idx
		# if first epoch use warmup
		if epoch - 1 <= args.warmup_epochs:
			this_lr = args.lr * float(cur_iter) / (args.warmup_epochs * len(trainloader))
			Helper.update_lr(optimizer, this_lr)

		inputs = Variable(inputs, requires_grad=True).cuda()
		targets = Variable(targets).cuda()

		# logits, _, reweighted_target = model(inputs, targets=targets, mixup_hidden=args.mixup_hidden, mixup=args.mixup, mixup_alpha=args.mixup_alpha)

		reweighted_target = targets

		loss = bce_loss(softmax(logits), reweighted_target)

		# measure accuracy and record loss
		prec1, prec5 = Helper.accuracy(logits, targets, topk=(1, 5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1.item(), inputs.size(0))
		top5.update(prec5.item(), inputs.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_steps == 0:
			sys.stdout.write('\r')
			sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f Acc@5: %.3f'
						 % (epoch, args.epochs, batch_idx+1,
							(len(trainset)//args.batch)+1, loss.data.item(),
							top1.avg, top5.avg))
			sys.stdout.flush()

	scheduler.step()
	epoch_time = time.time() - start_time
	elapsed_time += epoch_time
	print('| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))

	if (epoch - 1) % args.save_steps == 0:
		test_objective = Tester.eval_invertibility(model, testloader, sample_dir, args.model_name, num_epochs=epoch)
		Helper.save_checkpoint(model, test_objective, checkpoint_dir, args.model_name, epoch)

########################## Store ##################
test_objective = Tester.eval_invertibility(model, testloader, sample_dir, args.model_name, num_epochs=args.epochs)
Helper.save_checkpoint(model, test_objective, checkpoint_dir, args.model_name, args.epochs)
print('Done')
