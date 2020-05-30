import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from utils.helper import Helper, AverageMeter
import os, time, sys
import numpy as np
from utils.helper import Helper

criterionMSE = nn.MSELoss(reduction='mean')
criterionL1 = torch.nn.L1Loss(reduction='mean')

def atanh(x):
	return 0.5*torch.log((1+x)/(1-x))

def scale(imgs, stats=None):
	if stats == None:
		return torch.tanh(imgs)
	else:
		s = stats['input']
		for k in range(nchannels):
			imgs[:, k, ...] = (imgs[:, k, ...] - s['mean'][k])/s['std'][k]
		return imgs

def rescale(imgs, stats=None):
	if stats == None:
		return atanh(imgs)
	else:
		s = stats['target']
		for k in range(nchannels):
			imgs[:, k, ...] = imgs[:, k, ...] * s['std'][k] + s['mean'][k]
		return imgs

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

def cart2sphere(p):
	# x = [N, D]
	r = torch.norm(p, dim=1)**2
	y = (r - 1)/(r + 1)
	r = r.unsqueeze(1).expand(p.shape)
	x = 2*p/(r + 1)
	s = torch.cat([x, y.unsqueeze(1)], dim=1)
	return s

def sphere2cart(s):
	x = s[:, :-1]
	y = s[:, -1:].expand(x.shape)
	p = x/(1 - y)
	return p

def convert_polar_to_cart(z):
	p0 = [cart2polar(z[i, :]) for i in range(M*A)]
	p1 = torch.stack(p0, dim=0)
	r = torch.norm(z, dim=1)
	p = p1.view(M, A, C*H*W-1).mean(dim=1)
	r = r.view(M, A).mean(dim=1)
	z_fused = torch.stack([polar2cart(r[i], p[i, :]) for i in range(M)], dim=0)

	# hypersphere
	# [NA, d+1]
	p = cart2sphere(z)
	p = p.view(M, A, C_z*H_z*W_z + 1)
	# [N, d+1]
	p_fused = p.mean(dim = 1)
	z_fused = sphere2cart(p_fused)

def cart2angle(p):
	# input: p=[N, D], output: c = [N, D]
	r = torch.norm(p, dim=1)
	s = r.unsqueeze(1).expand(p.shape)
	angles = torch.acos(p/s)
	return r, angles

def angle2cart(r, angles):
	# input: r=[N], angles=[N, D], output: p = [N, D]
	r = r.unsqueeze(1).expand(angles.shape)
	p = torch.cos(angles) * r
	return p


def batch_process(inet, fnet, inputs, targets, nactors):
	# counters
	inputs = Variable(inputs).cuda()
	targets = Variable(targets).cuda()
	N, C, H, W = inputs.shape
	A = nactors
	M = N // A
	N = M * A
	inputs = inputs[:N, ...].view(M, A, C, H, W)
	targets = targets[:N, ...].view(M, A)

	# indices = torch.randperm(A)
	# inputs = inputs[:, indices, ...]
	# targets = targets[:, indices]
	# fusion inputs
	x_g = inputs.view(M, A*C, H, W)
	x_fused_g = rescale(fnet(x_g))

	# Z
	_, z, _ = inet(inputs.view(M*A, C, H, W))
	z = z.view((M, A, z.shape[1], z.shape[2], z.shape[3]))
	# missed server
	target_missed = targets[:, 0]
	z_missed = z[:, 0, ...]
	z_rest = z[:, 1:, ...].sum(dim=1)
	z_fused = z.mean(dim=1)

	_, z_fused_g, _ = inet(x_fused_g)
	z_missed_g = A * z_fused_g - z_rest

	return z_missed, z_missed_g, target_missed, z_fused

def batch_score(inet, z_missed, z_missed_g, target_missed):
	# classification
	out_missed_g = inet.module.classifier(z_missed_g)
	out_missed = inet.module.classifier(z_missed)

	# evaluation
	_, y = torch.max(out_missed.data, 1)
	_, y_g = torch.max(out_missed_g.data, 1)

	acc = y.eq(target_missed.data).sum().cpu().item()
	acc_g = y_g.eq(target_missed.data).sum().cpu().item()
	match_g = y_g.eq(y.data).sum().cpu().item()

	return np.asarray([acc, acc_g, match_g])


class Tester:
	@staticmethod
	def evaluate(model, testloader):
		model.eval()
		# loss
		top1 = AverageMeter()
		top5 = AverageMeter()
		for _, (inputs, targets) in enumerate(testloader):
			# print(batch_idx)
			targets = Variable(targets).cuda()
			inputs = Variable(inputs).cuda()
			logits, _, _ = model(inputs)
			prec1, prec5 = Helper.accuracy(logits, targets, topk=(1, 5))

			top1.update(prec1.item(), inputs.size(0))
			top5.update(prec5.item(), inputs.size(0))

		sys.stdout.write('Acc@1: %.3f Acc@5: %.3f' % (top1.avg, top5.avg))

	@staticmethod
	def evaluate_fgan(inet, fnet, dataloader, nactors):
		print('Evaluate Fusion Network')
		inet.eval()
		fnet.eval()
		ntests = 0
		corrects = np.zeros(3)
		for batch_idx, (inputs, targets) in enumerate(dataloader):
			z_missed, z_missed_g, target_missed, _ = batch_process(inet, fnet, inputs, targets, nactors)
			s = batch_score(inet, z_missed, z_missed_g, target_missed)
			corrects += s
			ntests += z_missed.shape[0]

			del z_missed, z_missed_g, target_missed, _

		corrects = 100 * corrects / ntests
		print('\t == Correctly classified: X {:.4f} X_hat {:.4f} Match {:.4f}'.format(corrects[0], corrects[1], corrects[2]))

	@staticmethod
	def evaluate_fnet(inet, fnet, dataloader, stats, nactors, nchannels=3, targets=None, concat_input=False):
		print('Evaluate fusion network')
		inet.eval()
		fnet.eval()

		total = 0
		nbatches = 0
		corrects = [0, 0, 0]
		runtimes = [0, 0, 0]
		for batch_idx, (inputs, targets) in enumerate(dataloader):
			# data
			targets = Variable(targets).cuda()
			inputs = Variable(inputs).cuda()
			# counters
			batch_size, C, H, W = inputs.shape
			nbatches += 1
			total += batch_size

			# clone inputs (N-1) times
			x = scale(inputs.clone(), stats).unsqueeze(0).expand(nactors - 1, batch_size, C, H, W).contiguous().view((nactors-1)*batch_size, C, H, W)
			# shuffle samples
			x = x[torch.randperm(x.shape[0]), ...]

			# latents
			_, z_c, _ = inet(x)
			z_c = z_c.view(nactors - 1, batch_size, z_c.shape[1], z_c.shape[2], z_c.shape[3]).sum(dim=0)

			# fusion inputs
			x_g = torch.cat([scale(inputs.clone(), stats).unsqueeze(1), x.view(nactors - 1, batch_size, C, H, W).permute(1, 0, 2, 3, 4)], dim=1)

			start_time = time.time()
			if concat_input:
				x_g = x_g.permute(0, 2, 1, 3, 4)

			# N, A, C, H, W = x_g.shape
			# x_g = x_g.view(N, A * C, H, W)
			img_fused = fnet(x_g)
			# img_fused = img_fused.view(inputs.shape)
			# img_fused = fnet(x_g)
			img_fused = rescale(img_fused.clone(), stats)
			runtimes[0] += time.time() - start_time

			# time f
			start_time = time.time()
			_, z_fused, _ = inet(img_fused)
			runtimes[1] += time.time() - start_time

			start_time = time.time()
			z_hat = nactors * z_fused - z_c
			out_hat = inet.module.classifier(z_hat)
			runtimes[2] += time.time() - start_time

			out, _, _ = inet(inputs)
			_, y = torch.max(out.data, 1)
			_, y_hat = torch.max(out_hat.data, 1)

			corrects[0] += y.eq(targets.data).sum().cpu().item()
			corrects[1] += y_hat.eq(targets.data).sum().cpu().item()
			corrects[2] += y_hat.eq(y.data).sum().cpu().item()

			del _, z_c, z_hat, z_fused, inputs, targets, img_fused, out, out_hat, y, y_hat

		# Evaluate time and classification performance
		corrects = 100 * np.asarray(corrects) / total
		print('\t == Correctly classified: X {:.4f} X_hat {:.4f} Match {:.4f}'.format(corrects[0], corrects[1], corrects[2]))
		runtimes = np.asarray(runtimes)/ nbatches
		print('Average time G: {:.4f}, E: {:.4f}, C: {:.4f} over {} batches of size {}'.format(runtimes[0], runtimes[1], runtimes[2], nbatches, batch_size))

	@staticmethod
	def evaluate_fnet_multiround(inet, fnet, dataloader, nactors, nchannels=3, targets=None, concat_input=False):
		print('Evaluate fusion network')
		inet.eval()
		fnet.eval()
		total = 0
		nbatches = 0
		corrects = [0, 0, 0]
		runtimes = [0, 0, 0]
		sum_mae = 0
		sum_mse = 0
		criterionMSE = nn.MSELoss(reduction='mean')
		for batch_idx, (inputs, targets) in enumerate(dataloader):
			nbatches += 1
			targets = Variable(targets).cuda()
			inputs = Variable(inputs).cuda()
			batch_size, C, H, W = inputs.shape
			total += batch_size

			x = inputs.unsqueeze(0).expand(nactors, batch_size, C, H, W).contiguous().view(nactors*batch_size, C, H, W)
			x = x[torch.randperm(x.shape[0]), ...]

			_, z_c, _ = inet(x)
			z_c = z_c.view(nactors, batch_size, z_c.shape[1], z_c.shape[2], z_c.shape[3])
			sum_z = z_c.sum(dim=0)

			x = x.view(nactors, batch_size, C, H, W)
			x_g = x[torch.randperm(nactors), ...].permute(1, 0, 2, 3, 4)
			img_fused = fnet(x_g)
			_, z_fused, _ = inet(img_fused)

			for k in range(nactors):
				z = sum_z - z_c[k, ...]
				z_hat = nactors * z_fused - z
				out_hat = inet.module.classifier(z_hat)

				inputs = x[k, ...]
				out, _, _ = inet(inputs)
				_, y = torch.max(out.data, 1)
				_, y_hat = torch.max(out_hat.data, 1)

				corrects[2] += y_hat.eq(y.data).sum().cpu().item()/nactors

				input_hat = inet.module.inverse(z_hat)
				mae = torch.norm((input_hat - inputs).view(batch_size, -1), dim=1) / torch.norm(inputs.view(batch_size, -1), dim=1)
				sum_mae += mae.mean().cpu().item()/nactors

				sum_mse += criterionMSE(input_hat, inputs).mean().cpu().item() /nactors

				max = inputs.abs().max().cpu().item()
				min = (input_hat - inputs).abs().max().cpu().item()

			del _, z_c, z_hat, z_fused, inputs, targets, img_fused, out, out_hat, y, y_hat

		# Evaluate time and classification performance
		corrects = 100 * np.asarray(corrects) / (total)
		print('\t == Correctly classified: X {:.4f} X_hat {:.4f} Match {:.4f}'.format(corrects[0], corrects[1], corrects[2]))

		print('\t == MAE: {:.4f}, MSE: {:.4f} in {:.4f} -> {:.4f}'.format(sum_mae/nbatches, sum_mse/nbatches, min, max))

		runtimes = np.asarray(runtimes)/ nbatches
		print('Average time G: {:.4f}, E: {:.4f}, C: {:.4f} over {} batches of size {}'.format(runtimes[0], runtimes[1], runtimes[2], nbatches, batch_size))

	@staticmethod
	def evaluate_fnet_distill(inet, fnet, dataloader, nactors, nchannels=3, targets=None, concat_input=False):
		print('Evaluate fusion network')
		inet.eval()
		fnet.eval()

		total = 0
		nbatches = 0
		corrects = [0, 0, 0]
		runtimes = [0, 0, 0]
		sum_mae = 0
		sum_mse = 0
		for batch_idx, (inputs, targets) in enumerate(dataloader):
			nbatches += 1

			targets = Variable(targets).cuda()
			inputs = Variable(inputs).cuda()
			batch_size, C, H, W = inputs.shape
			total += batch_size

			x = inputs.unsqueeze(0).expand(nactors - 1, batch_size, C, H, W).contiguous().view((nactors-1)*batch_size, C, H, W)
			x = x[torch.randperm(x.shape[0]), ...]

			_, z_c, _ = inet(x)
			z_c = z_c.view(nactors - 1, batch_size, z_c.shape[1], z_c.shape[2], z_c.shape[3]).sum(dim=0)

			x_g = torch.cat([inputs.unsqueeze(1), x.view(nactors - 1, batch_size, C, H, W).permute(1, 0, 2, 3, 4)], dim=1)

			start_time = time.time()
			if concat_input:
				x_g = x_g.permute(0, 2, 1, 3, 4)

			# randomly permute batch_size
			x_g = x_g[:, torch.randperm(nactors), ...]
			# new
			N, A, C, H, W = x_g.shape
			x_g = x_g.view(N, A * C, H, W)
			img_fused = fnet(x_g)
			img_fused = fnet(img_fused).view(inputs.shape)
			runtimes[0] += time.time() - start_time

			# time f
			start_time = time.time()
			_, z_fused, _ = inet(img_fused)
			runtimes[1] += time.time() - start_time

			start_time = time.time()
			z_hat = nactors * z_fused - z_c
			out_hat = inet.module.classifier(z_hat)
			runtimes[2] += time.time() - start_time

			out, z, _ = inet(inputs)
			_, y = torch.max(out.data, 1)
			_, y_hat = torch.max(out_hat.data, 1)

			corrects[0] += y.eq(targets.data).sum().cpu().item()
			corrects[1] += y_hat.eq(targets.data).sum().cpu().item()
			corrects[2] += y_hat.eq(y.data).sum().cpu().item()

			del _, z_c, z_fused, inputs, targets, img_fused, out, out_hat, y, y_hat

		# input_hat = inet.module.inverse(z_hat)
		mae = torch.norm((z_hat - z).view(batch_size, -1), dim=1) / torch.norm(z.view(batch_size, -1), dim=1)
		sum_mae = mae.mean().cpu().item()
		sum_mse = criterionMSE(z_hat, z).mean().cpu().item()

		mean = (z_hat - z).abs().mean().cpu().item()
		median = (z_hat - z).abs().median().cpu().item()
		min = z.abs().mean().cpu().item()
		max = z.abs().median().cpu().item()

		# Evaluate time and classification performance
		corrects = 100 * np.asarray(corrects) / total
		print('\t == Correctly classified: X {:.4f} X_hat {:.4f} Match {:.4f}'.format(corrects[0], corrects[1], corrects[2]))

		print('\t == MAE: {:.4f}, MSE: {:.4f} Mean {:.4f}, Median {:.4f}, Z-Range {:.4f} {:.4f}'.format(sum_mae, sum_mse, mean, median, min, max))

		runtimes = np.asarray(runtimes)/ nbatches
		print('Average time G: {:.4f}, E: {:.4f}, C: {:.4f} over {} batches of size {}'.format(runtimes[0], runtimes[1], runtimes[2], nbatches, batch_size))


	@staticmethod
	def plot_fused_images(model_name, sample_path, inet, dataloader, nactors, concat_input=False):
		print('Evaluate fusion network: ' + model_name)
		inet.eval()

		iters = iter(dataloader)
		(inputs, _) = iters.next()
		inputs = Variable(inputs).cuda()
		N, C, H, W = inputs.shape
		A = nactors
		M = N // A
		N = M * A
		inputs = inputs[:N, ...].view(M, A, C, H, W)

		_, z, _ = inet(inputs.view(M *A, C, H, W))
		_, C_z, H_z, W_z = z.shape
		# z-mean
		z_mean = z.view(M, A, C_z, H_z, W_z).mean(dim=1)
		x_mean = inet.module.inverse(z_mean)
		# polar
		z = z.view((M * A, C_z*H_z*W_z))
		D = C_z*H_z*W_z

		# [NA, D-1]
		r, angles = cart2polar(z)
		r = r.view(M, A)
		angles = angles.view(M, A, D-1)
		r = torch.exp(torch.log(r).mean(dim=1))
		angles = angles.mean(dim=1)
		z_fused = polar2cart(r, angles)
		# from IPython import embed; embed()
		z_fused = z_fused.view(M, C_z, H_z, W_z)
		x_fused = inet.module.inverse(z_fused)

		# visualize
		model_name = model_name + '_' + str(nactors)
		Helper.save_images(x_fused, sample_path, model_name, 'x_polar_fused', 0)
		Helper.save_images(x_mean, sample_path, model_name, 'x_mean', 0)

		norms = torch.norm(z.view(z.shape[0], -1), dim=1)
		l_norm = (torch.max(norms) - torch.min(norms))

		print("L1-norm: {:.4f}".format(l_norm))

	@staticmethod
	def plot_reconstruction(model_name, sample_path, inet, fnet, dataloader, nactors, concat_input=False):
		print('Evaluate fusion network: ' + model_name)
		inet.eval()
		fnet.eval()
		corrects = [0, 0, 0]
		criterionL1 = torch.nn.L1Loss(reduction='mean')

		iters = iter(dataloader)
		(inputs, targets) = iters.next()
		inputs = Variable(inputs).cuda()
		targets = Variable(targets).cuda()
		N, C, H, W = inputs.shape

		M = N // nactors
		N = M * nactors
		inputs = inputs[:N, ...].view(M, nactors, C, H, W)
		targets = targets[:N, ...].view(M, nactors)

		# fusion inputs
		x_g = inputs.view(M, nactors*C, H, W)
		x_fused_g = rescale(fnet(x_g))

		# Z
		_, z, _ = inet(inputs.view(M *nactors, C, H, W))
		z = z.view((M, nactors, z.shape[1], z.shape[2], z.shape[3]))
		target_missed = targets[:, 0]
		z_missed = z[:, 0, ...]
		z_rest = z[:, 1:, ...].sum(dim=1)
		z_fused = z.mean(dim=1)
		_, z_fused_g, _ = inet(x_fused_g)
		z_missed_g = nactors * z_fused_g - z_rest

		# classification
		out_missed_g = inet.module.classifier(z_missed_g)
		out_missed = inet.module.classifier(z_missed)

		# evaluation
		_, y = torch.max(out_missed.data, 1)
		_, y_g = torch.max(out_missed_g.data, 1)

		corrects[0] = y.eq(target_missed.data).sum().cpu().item()
		corrects[1] = y_g.eq(target_missed.data).sum().cpu().item()
		corrects[2] = y_g.eq(y.data).sum().cpu().item()

		# inverse
		x_fused_g = inet.module.inverse(z_fused_g)
		x_fused = inet.module.inverse(z_fused)
		x_missed = inet.module.inverse(z_missed)
		x_missed_g = inet.module.inverse(z_missed_g)

		# visualize
		Helper.save_images(x_fused_g, sample_path, model_name, 'x_fused_g', 0)
		Helper.save_images(x_fused, sample_path, model_name, 'x_fused', 0)
		Helper.save_images(x_missed_g, sample_path, model_name, 'x_missed_g', 0)
		Helper.save_images(x_missed, sample_path, model_name, 'x_missed', 0)

		print("L1: {:.4f}".format(criterionL1(torch.tanh(x_fused_g), torch.tanh(x_fused))))
		print(np.asarray(corrects)/M)


	@staticmethod
	def eval_inv(model, testloader, sample_path, model_name, num_epochs, nactors, debug=False):
		model_name = "{}_epoch_{}".format(model_name, num_epochs)
		print('Evaluate Invertibility on ', model_name)

		model.eval()
		# loss
		criterion = torch.nn.MSELoss(reduction='sum')
		corrects = [0, 0, 0] # correct-x, correct-x-hat, match
		total = 0
		in_shapes = -1
		for batch_idx, (inputs, targets) in enumerate(testloader):
			# print(batch_idx)
			targets = Variable(targets).cuda()
			inputs = Variable(inputs).cuda()
			batch_size, C, H, W = inputs.shape
			total += batch_size
			out, z_input, _ = model(inputs)

			x = inputs.clone().unsqueeze(0).expand(nactors - 1, batch_size, C, H, W).contiguous().view((nactors-1)*batch_size, C, H, W)
			x = x[torch.randperm(x.shape[0]), :]

			_, z_c, _ = model(x)
			z_c = z_c.view(nactors - 1, batch_size, z_input.shape[1], z_input.shape[2], z_input.shape[3]).sum(dim=0)

			z_fused = (z_input + z_c) / nactors
			x_fused = model.module.inverse(z_fused)

			_, z_fused_hat, _ = model(x_fused)
			z_hat = z_fused_hat * nactors - z_c

			# invert
			x_inv = model.module.inverse(z_input)
			x_inv_hat = model.module.inverse(z_hat)

			# classification
			out_hat = model.module.classifier(z_hat)
			_, y = torch.max(out.data, 1)
			_, y_hat = torch.max(out_hat.data, 1)

			corrects[0] += y.eq(targets.data).sum().cpu().item()
			corrects[1] += y_hat.eq(targets.data).sum().cpu().item()
			corrects[2] += y_hat.eq(y.data).sum().cpu().item()

			# save samples
			if batch_idx == 0:
				Helper.save_images(x_inv, sample_path, model_name, 'inv', batch_idx)
				Helper.save_images(x_inv_hat, sample_path, model_name, 'inv_hat', batch_idx)
				Helper.save_images(x_fused, sample_path, model_name, 'fused', batch_idx)

			del out, out_hat, z_c, z_fused, z_hat, z_fused_hat, inputs, x_inv, x_inv_hat, targets, y, y_hat

		# print('\t {} images of {} pixels'.format(total, in_shapes))

		corrects = 100 * np.asarray(corrects) / total
		print('\t Correctly classified: X {:.4f} X_hat {:.4f} Match {:.4f}'.format(corrects[0], corrects[1], corrects[2]))

		return corrects[0]

	@staticmethod
	def sample_fused_data(inet, dataloader, out_path, nchannels=1, nactors=2, niters=1, reduction='mean'):
		inet.eval()
		for epoch in range(niters):
			for batch_idx, (inputs, targets) in enumerate(dataloader):
				# This affect GPU
				inputs = Variable(inputs).cuda()
				targets = Variable(targets).cuda()
				N, C, H, W = inputs.shape

				M = N // nactors
				N = M * nactors
				A = nactors

				inputs = inputs[:N, ...].view(M, nactors, C, H, W)
				targets = targets[:N, ...].view(M, nactors)

				_, z, _ = inet(inputs.view(M *A, C, H, W))
				_, C_z, H_z, W_z = z.shape
				# z-mean
				# polar
				z = z.view((M * A, C_z*H_z*W_z))
				D = C_z*H_z*W_z

				# [NA, D-1]
				r, angles = cart2polar(z)
				r = r.view(M, A)
				angles = angles.view(M, A, D-1)
				r = torch.exp(torch.log(r).mean(dim=1))
				angles = angles.mean(dim=1)
				z_fused = polar2cart(r, angles)
				# from IPython import embed; embed()
				z_fused = z_fused.view(M, C_z, H_z, W_z)

				# if reduction == 'sum':
				# 	z_fused = z.sum(dim=1)
				# elif reduction == 'mean':
				# 	z_fused = z.mean(dim=1)
				x_fused = inet.module.inverse(z_fused)

				data = torch.cat([inputs, x_fused.unsqueeze(1)], dim=1)
				if 0 == epoch and 0 == batch_idx:
					images = data
					labels = targets
				else:
					images = torch.cat([images, data], dim=0)
					labels = torch.cat([labels, targets], dim=0)

				print('Save ', epoch, batch_idx)
				del data, z, z_fused, x_fused, inputs, targets
		try:
			img_dict = {
				'images': images.cpu().numpy(),
				'labels': labels.cpu().numpy()
				}
			np.save(out_path, img_dict)
			print('Done')
		except:
			from IPython import embed; embed()

	@staticmethod
	def plot_latent(model, dataloader, num_classes):
		model.eval()
		for batch_idx, (inputs, batch_labels) in enumerate(dataloader):
			# This affect GPU
			with torch.no_grad():
				_, out_bij = model(Variable(inputs).cuda())
				# z3 = (out_bij[:bs, ...] + out_bij[bs:, ...]) / 2
				#invert
				# batch_data = torch.cat([out_bij, z3], dim=1)
				batch_data = out_bij
				if batch_idx == 0:
					data = batch_data
					labels = batch_labels
				else:
					data = torch.cat([data, batch_data], dim=0)
					labels = torch.cat([labels, batch_labels], dim=0)
			del out_bij, batch_data
			if batch_idx > 40:
				break
		data = data.view(data.shape[0], -1)
		img_path = os.path.join(args.save_dir, 'tnse_{}.png'.format(5))
		Plotter.plot_tnse(data.cpu().numpy(), labels.cpu().numpy(), img_path, num_classes=5)

	@staticmethod
	def test_invert_images(model, dataloader, name, noise=True):
		model.eval()
		for batch_idx, (inputs, targets) in enumerate(dataloader):
			# This affect GPU
			if batch_idx < 2:
				continue
			inputs = inputs.cuda()
			with torch.no_grad():
				inputs = Variable(inputs)

			# if inputs.shape[0] % 2 == 1:
			# 	break
			bs = inputs.shape[0] // 2
			# modelling
			out, out_bij = model(inputs)
			# divide out_bij into 2 parts
			z1 = out_bij[:bs, ...]
			z2 = out_bij[bs:, ...]

			# x1 = inputs[:bs, ...]
			# x2 = inputs[:bs, ...] * 1.01
			# _, z1 = model(x1)
			# _, z2 = model(x2)

			# z2 = out_bij[bs:, ...]
			z3 = (z1 + z2) / 2
			x3 = model.module.inverse(z3)
			# torchvision.utils.save_image(x1_hat.cpu(), os.path.join(sample_path, "test_inv_car_identity.jpg"), int(bs**.5), normalize=True)
			# x_inv_noise = model.module.inverse(z2 * 1.01)
			# torchvision.utils.save_image(x_inv_noise.cpu(), os.path.join(sample_path, "test_inv_car_noise.jpg"), int(bs**.5), normalize=True)
			torchvision.utils.save_image(x3.cpu(), os.path.join(sample_path, "{}_fused.jpg".format(name)), int(bs**.5), normalize=True)

			# x1 = inputs[88:89, ...]
			# x2 = inputs[90:91, ...]
			# z1 = out_bij[88:89, ...]
			# z2 = out_bij[90:91, ...]
			# z3 = (z1 + z2) / 2
			# x_inv_3 = model.module.inverse(z3)
			# out = torch.cat((x1, x2, x_inv_3), dim=0)
			# name = 'mix50_full'
			# torchvision.utils.save_image(out.cpu(), os.path.join(sample_path, "test_inv_car_{}.jpg".format(name)), 1, normalize=True)

			# del out, out_bij, z1, z2, z3, x_inv_3
			break
			if batch_idx > 2:
				break

	@staticmethod
	def eval_sensitivity(model, dataloader, eps=0.01):
		print('Evaluate ensitivity')
		model.eval()
		total = 0
		corrects = [0, 0]
		for batch_idx, batch_data in enumerate(dataloader):
			batch_data = batch_data.to(device)
			# [-2.7; 2.3]
			img1 = batch_data[:, 0, ...]
			img2 = batch_data[:, 1, ...]
			img3 = batch_data[:, 2, ...]
			img3_noise = img3 + torch.rand(img3.shape).to(device) * 0.25 + eps

			_, z1 = model(img1)
			_, z3 = model(img3)
			_, z3_noise = model(img3_noise)
			z2_hat = 2*z3 - z1
			z2_hat_noise = 2*z3_noise - z1

			out, _ = model(img2)
			out_hat = model.module.classifier(z2_hat)
			out_hat_noise = model.module.classifier(z2_hat_noise)
			_, y = torch.max(out.data, 1)
			_, y_hat = torch.max(out_hat.data, 1)
			_, y_hat_noise = torch.max(out_hat_noise.data, 1)

			corrects[0] += y_hat.eq(y.data).sum().cpu().item()
			corrects[1] += y_hat_noise.eq(y.data).sum().cpu().item()
			total += 1

			del _, z1, z3, z2_hat, z3_noise, img1, img2, img3, out, out_hat, y, y_hat

		# Evaluate time and classification performance
		corrects = 100 * np.asarray(corrects) / (total * args.batch)
		print('\t == Match: X_hat {:.4f} X_hat_noise {:.4f}'.format(corrects[0], corrects[1]))
