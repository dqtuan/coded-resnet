"""
Code for "Invertible Residual Networks"
http://proceedings.mlr.press/v97/behrmann19a.html
ICML, 2019
"""
from init import *
sys.path.append('../')
from pix2pix.models.networks import define_G, update_learning_rate, get_scheduler
from utils.utils_cifar import update_lr, learning_rate


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


if args.deterministic:
	print("MODEL NOT FULLY DETERMINISTIC")
	torch.manual_seed(1234)
	torch.cuda.manual_seed(1234)
	np.random.seed(1234)
	random.seed(1234)
	torch.backends.cudnn.deterministic=True

# 'cifar10':
train_chain = [transforms.Pad(4, padding_mode="symmetric"),
			   transforms.RandomCrop(32),
			   transforms.RandomHorizontalFlip(),
			   transforms.ToTensor()]

test_chain = [transforms.ToTensor()]

clf_chain = [transforms.Normalize(mean[args.dataset], std[args.dataset])]
transform_train = transforms.Compose(train_chain + clf_chain)
transform_test = transforms.Compose(test_chain + clf_chain)

trainset = torchvision.datasets.CIFAR10(
	root='../../data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(
	root='../../data', train=False, download=True, transform=transform_test)
args.nClasses = 10
in_shape = (3, 32, 32)


if args.deterministic:
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
											  shuffle=True, num_workers=2, worker_init_fn=np.random.seed(1234))
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
											 shuffle=False, num_workers=2, worker_init_fn=np.random.seed(1234))
else:
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=2)
	testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch, shuffle=False, num_workers=2)

model_name = 'inv_unit_full'
model_path = os.path.join(checkpoint_path, '{}_e{}.pkt'.format(model_name, args.resume))
print("-- Loading checkpoint '{}'".format(model_path))
checkpoint = torch.load(model_path)
model = checkpoint['model']

netG_checkpoint_path = '../../results/pix2pix/checkpoints/'
net_path = os.path.join(netG_checkpoint_path, 'netG_finv_e{}.pth'.format(args.resume_g))
if os.path.isfile(net_path) and args.flag_retrain:
	print('Load existing models')
	net_G = torch.load(net_path).to(device)
else:
	net_G = define_G(
			input_nc=3,
			output_nc=3,
			ngf=8,
			norm='batch',
			use_dropout=False,
			init_type='normal',
			init_gain=0.02,
			gpu_id=device)

cudnn.benchmark = True
in_shapes = model.module.get_in_shapes()

if args.optimizer == "adam":
	optimizer = optim.Adam(net_G.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == "adamax":
	optimizer = optim.Adamax(net_G.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
	optimizer = optim.SGD(net_G.parameters(), lr=args.lr,
						  momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)
# log
# setup logging with visdom
viz = visdom.Visdom(port=args.vis_port, server="http://" + args.vis_server)
assert viz.check_connection(), "Could not make visdom"

with open(param_path, 'w') as f:
	f.write(json.dumps(args.__dict__))
train_log = open(trainlog_path, 'w')
try_make_dir(args.save_dir)
try_make_dir(sample_path)

########################## Training ##################
print('|  Train Epochs: ' + str(args.epochs))
print('|  Initial Learning Rate: ' + str(args.lr))
elapsed_time = 0
test_objective = -np.inf
epoch = 0
total_steps = len(iter(trainloader))
net_G_scheduler = get_scheduler(optimizer, args)
criterion = torch.nn.CrossEntropyLoss()
criterionMSE = torch.nn.MSELoss(reduction='mean')
for epoch in range(1, 1+args.epochs):
	model.eval()
	lr = learning_rate(args.lr, epoch)
	update_lr(optimizer, lr)
	start_time = time.time()
	# modeling
	for batch_idx, (inputs, labels) in enumerate(trainloader):
		# This affect GPU
		# print(Helper.get_norm(net_G))
		if inputs.shape[0] % 2 == 1:
			inputs = inputs[1:, ...]
			labels = labels[1:, ...]
		bs = inputs.shape[0] // 2
		with torch.no_grad():
			inputs = Variable(inputs).cuda()
			labels = Variable(labels[bs:, ...]).cuda()
		# data
		out, out_bij = model(inputs)
		z_1 = out_bij[:bs, ...]
		z_2 = out_bij[bs:, ...].detach() # for loss
		teacher_outputs = out[bs:, ...]

		# fusion
		input_1 = inputs[:bs, ...]
		input_2 = inputs[bs:, ...]
		# training
		optimizer.zero_grad()
		x_hat = net_G(input_1, input_2)
		_, z_hat = model(x_hat)
		z_2_hat = 2*z_hat - z_1
		outputs = model.module.classifier(z_2_hat)
		input_2_hat = model.module.inverse(z_2_hat)
		#loss here - cross entropy
		# loss = criterion(out_preds, targets)
		loss_distill = loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.1, temperature=6)
		loss_z = criterionMSE(input_2, input_2_hat)
		loss = loss_distill + loss_z
		loss.backward()
		optimizer.step()

		if batch_idx % 50 == 0:
			print("===> Epoch[{}/{}]({}/{}): Loss: distill {:.4f} mse-z {:.4f} ".format(
			epoch, args.epochs, batch_idx, total_steps, loss_distill.item(), loss_z.item()))

	# update_learning_rate(net_G_scheduler, optimizer)
	epoch_time = time.time() - start_time
	elapsed_time += epoch_time
	print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))
	# test inverse function
	if epoch % 10 == 0:
		net_path = os.path.join(netG_checkpoint_path, 'netG_finv_e{}.pth'.format(epoch))
		torch.save(net_G, net_path)
		# with torch.no_grad():
		# 	evaluate_fusion_net(model, net_G, testloader)


########################## Store ##################
print('Save checkpoint')
net_path = os.path.join(netG_checkpoint_path, 'netG_finv_e{}.pth'.format(args.epochs))
torch.save(net_G, net_path)
evaluate_fusion_net(model, net_G, testloader)
