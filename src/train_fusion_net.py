"""
	Doing fusion 2 images to match target images
	Supervised tasks
"""
from init_fusionnet import *

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

if os.path.isfile(fnet_path) and args.flag_retrain:
	print(fnet_path)
	fnet.load_state_dict(torch.load(fnet_path))

optim_fnet = optim.Adam(fnet.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
# optim_fnet = optim.SGD(fnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)
fnet_scheduler = Helper.get_scheduler(optim_fnet, args)

# Training
total_steps = len(iter(train_loader))
num_epochs = args.niter + args.niter_decay
for epoch in range(1, num_epochs + 1):
	for batch_idx, batch_data in enumerate(train_loader):
		batch_data = batch_data.to(device)
		# [-2.7; 2.3]
		inputs = batch_data[:, :-1, ...]
		if 1 == ninputs:
			# concat input
			inputs = inputs.permute(0, 2, 1, 3, 4)
		# [-30; 30]
		targets = batch_data[:, -1, ...]
		optim_fnet.zero_grad()
		# [-2.3, 2.3]
		preds = fnet(inputs)
		loss = criterionMSE(preds, targets) / args.batch_size
		loss.backward()
		optim_fnet.step()

		if batch_idx % args.log_steps == 0:
			print("===> Epoch[{}]({}/{}): MSE-pixel: {:.4f}".format(
			epoch, batch_idx, total_steps, loss.item() / npixels))

	Helper.update_learning_rate(fnet_scheduler, optim_fnet)
	#checkpoint
	if (epoch - 1) % args.save_steps == 0:
		print("Saving")
		torch.save(fnet.state_dict(), fnet_path)
		Helper.save_images(targets, sample_dir, args.model_name, 'target', epoch)
		Helper.save_images(preds, sample_dir, args.model_name, 'pred', epoch)
		# evaluate classification
		# Tester.evaluate_fnet(inet, fnet, dataloader, stats)

# final model
torch.save(fnet.state_dict(), fnet_path)
print('Done')
