"""
	Doing fusion 2 images to match target images
	Supervised tasks
"""
from init_fusionnet import *

# define fnet
fnet = define_G(
		input_nc=args.input_nc,
		output_nc=args.output_nc,
		ngf=args.ngf,
		norm='batch',
		use_dropout=False,
		init_type='normal',
		init_gain=0.02,
		gpu_id=device)

if os.path.isfile(net_path):
	print(net_path)
	fnet.load_state_dict(torch.load(net_path))

optim_G = optim.Adam(fnet.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
# optim_G = optim.SGD(fnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)
fnet_scheduler = Helper.get_scheduler(optim_G, args)

# Training
total_steps = len(iter(train_loader))
num_epochs = args.niter + args.niter_decay
for epoch in range(1, num_epochs + 1):
	for batch_idx, batch_data in enumerate(train_loader):
		batch_data = batch_data.to(device)
		# [-2.7; 2.3]
		input_1 = batch_data[:, 0, ...]
		input_2 = batch_data[:, 1, ...]
		# inputs = torch.cat((input_1, input_2), dim=1)
		# [-30; 30]
		targets = batch_data[:, 2, ...]
		optim_G.zero_grad()
		# [-2.3, 2.3]
		preds = fnet(input_1, input_2)
		loss = criterionMSE(preds, targets) / args.batch_size
		loss.backward()
		optim_G.step()

		if batch_idx % args.log_steps == 0:
			print("===> Epoch[{}]({}/{}): MSE-pixel: {:.4f}".format(
			epoch, batch_idx, total_steps, loss.item() / (32*32*3)))

	Helper.update_learning_rate(fnet_scheduler, optim_G)
	#checkpoint
	if (epoch - 1) % args.save_steps == 0:
		print("Saving")
		torch.save(fnet.state_dict(), net_path)
		Helper.save_images(targets, sample_dir, args.model_name, 'target', epoch)
		Helper.save_images(preds, sample_dir, args.model_name, 'pred', epoch)
		# evaluate classification
		# Tester.evaluate_fnet(inet, fnet, dataloader, stats)

# final model
torch.save(fnet.state_dict(), net_path)
print('Done')
