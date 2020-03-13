import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms

import os, time
import numpy as np
from utils.helper import Helper

class Tester:

	@staticmethod
	def eval_invertibility(model, testloader, sample_path, model_name, num_epochs, debug=False):
		model_name = "{}_epoch_{}".format(model_name, num_epochs)
		print('Evaluate Invertibility on ', model_name)
		def update_pixel_range(pixels, inputs):
			# range of pixel values
			if pixels[1] < inputs.max():
				pixels[1] = inputs.max().cpu().item()
			if pixels[0] > inputs.min():
				pixels[0] = inputs.min().cpu().item()
			pixels[2] += inputs.mean().cpu().item()
			return pixels

		model.eval()
		# loss
		criterion = torch.nn.MSELoss(reduction='sum')
		corrects = [0, 0, 0] # correct-x, correct-x-hat, match
		mses = [0, 0, 0] # x-inv, x-inv-hat, z
		total = 0
		# range of pixels
		pixel_x = [1000, -1000, 0]
		pixel_x_inv = [1000, -1000, 0]
		pixel_x_inv_hat = [1000, -1000, 0]
		pixel_z = [1000, -1000, 0]
		pixel_z_hat = [1000, -1000, 0]
		in_shapes = -1
		for batch_idx, (inputs, targets) in enumerate(testloader):
			batch_size = inputs.shape[0]
			total += batch_size
			index = torch.randperm(batch_size).cuda()
			targets = Variable(targets).cuda()
			# careful here
			x1 = Variable(inputs).cuda()
			x2 = x1[index, :]

			# modelling
			out1, z1, _ = model(x1)
			out2, z2, _ = model(x2)
			z3 = (z1 + z2) / 2
			x3_hat = model.module.inverse(z3)
			_, z3_hat, _ = model(x3_hat)
			z1_hat = z3_hat * 2 - z2

			# invert
			x1_org = model.module.inverse(z1)
			x1_hat = model.module.inverse(z1_hat)

			# MSE measurement
			# mses[0] += criterion(x2, x1_org).cpu().item()
			# mses[1] += criterion(x2, x1_hat).cpu().item()
			# mses[2] += criterion(z2, z2_hat).cpu().item()
			#
			# # pixel range
			# pixel_x = update_pixel_range(pixel_x, x2)
			# pixel_x_inv = update_pixel_range(pixel_x_inv, x1_org)
			# pixel_x_inv_hat = update_pixel_range(pixel_x_inv_hat, x1_hat)
			# pixel_z = update_pixel_range(pixel_z, z2)
			# pixel_z_hat = update_pixel_range(pixel_z_hat, z2_hat)

			# classification
			out1_hat = model.module.classifier(z1_hat)
			_, y1 = torch.max(out1.data, 1)
			_, y1_hat = torch.max(out1_hat.data, 1)

			corrects[0] += y1.eq(targets.data).sum().cpu().item()
			corrects[1] += y1_hat.eq(targets.data).sum().cpu().item()
			corrects[2] += y1_hat.eq(y1.data).sum().cpu().item()

			# save samples
			if batch_idx == 0:
				Helper.save_images(x1_org, sample_path, model_name, 'inv', batch_idx)
				Helper.save_images(x1_hat, sample_path, model_name, 'inv_hat', batch_idx)
				Helper.save_images(x3_hat, sample_path, model_name, 'fused', batch_idx)

			del out1, out2, z1, z2, z3, z3_hat, x1, x2, x1_org, x1_hat, x3_hat,  inputs, targets, y1, y1_hat

		# print('\t {} images of {} pixels'.format(total, in_shapes))

		corrects = 100 * np.asarray(corrects) / total
		print('\t Correctly classified: X {:.4f} X_hat {:.4f} Match {:.4f}'.format(corrects[0], corrects[1], corrects[2]))

		# mses = np.asarray(mses) / total
		# print('\t Avg.MSE of one image: X_inv {:.4f} X_inv_hat {:.4f} Z_match {:.4f}'.format(mses[0], mses[1], mses[2]))
		# print('\t Range of pixel [Min, Max, Mean, Median]:')
		# def print_out_pixel(pname, pixels):
		# 	print('\t \t {}: {:.4f} {:.4f} {:.4f}'.format(pname, pixels[0], pixels[1], pixels[2]/total))
		# print_out_pixel('pixel_x', pixel_x)
		# print_out_pixel('pixel_x_inv', pixel_x_inv)
		# print_out_pixel('pixel_x_inv_hat', pixel_x_inv_hat)
		# print_out_pixel('pixel_z', pixel_z)
		# print_out_pixel('pixel_z_hat', pixel_z_hat)

		return corrects[0]

	@staticmethod
	def generate_inversed_images(model, dataloader, data_path):
		model.eval()
		for batch_idx, (inputs, targets) in enumerate(dataloader):
			# This affect GPU
			batch_size = inputs.shape[0]
			index = torch.randperm(batch_size).cuda()
			x1 = Variable(inputs).cuda()
			x2 = x1[index, :]

			# modelling
			out1, z1, _ = model(x1)
			out2, z2, _ = model(x2)
			z3 = (z1 + z2) / 2
			x3_hat = model.module.inverse(z3)

			# Store data
			x1 = x1.detach().cpu().numpy()
			x2 = x2.detach().cpu().numpy()
			x3_hat = x3_hat.detach().cpu().numpy()
			img_dict = {'img1': x1, 'img2': x2, 'fused': x3_hat, 'targets': targets.numpy()}
			save_path = os.path.join(data_path, "data_{}.npy".format(batch_idx))
			np.save(save_path, img_dict)
			print('Save ', batch_idx)
			del x1, x2, x3_hat, z1, z2, z3, img_dict, targets

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
	def test_inversed_images(model, dataloader, name, noise=True):
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
	def evaluate_fusion_net(inet, fnet, dataloader, stats, targets=None):
		print('Evaluate fusion network')
		inet.eval()
		fnet.eval()
		def scale(imgs):
			s = stats['input']
			for k in range(3):
				imgs[:, k, ...] = (imgs[:, k, ...] - s['mean'][k])/s['std'][k]
			return imgs

		def rescale(imgs):
			s = stats['target']
			for k in range(3):
				imgs[:, k, ...] = imgs[:, k, ...] * s['std'][k] + s['mean'][k]
			return imgs

		total = 0
		nbatches = 0
		corrects = [0, 0, 0]
		runtimes = [0, 0, 0]
		for batch_idx, (inputs, targets) in enumerate(dataloader):
			batch_size = inputs.shape[0]
			total += batch_size
			nbatches += 1
			index = torch.randperm(batch_size).cuda()
			targets = Variable(targets).cuda()
			img1 = Variable(inputs).cuda()
			img2 = img1[index, :]

			# this is important to clone, otherwise, img1 will change its value
			scaled_img1 = scale(img1.clone())
			scaled_img2 = scale(img2.clone())

			start_time = time.time()
			img3 = fnet(scaled_img1, scaled_img2)
			img3 = rescale(img3.clone())
			runtimes[0] += time.time() - start_time

			_, z2, _ = inet(img2)
			# time f
			start_time = time.time()
			_, z3, _ = inet(img3)
			runtimes[1] += time.time() - start_time

			start_time = time.time()
			z1_hat = 2*z3 - z2
			out_hat = inet.module.classifier(z1_hat)
			runtimes[2] += time.time() - start_time

			out, _, _ = inet(img1)
			_, y = torch.max(out.data, 1)
			_, y_hat = torch.max(out_hat.data, 1)

			corrects[0] += y.eq(targets.data).sum().cpu().item()
			corrects[1] += y_hat.eq(targets.data).sum().cpu().item()
			corrects[2] += y_hat.eq(y.data).sum().cpu().item()

			del _, z2, z3, z1_hat, inputs, targets, img1, img2, img3, out, out_hat, y, y_hat

		# Evaluate time and classification performance
		corrects = 100 * np.asarray(corrects) / total
		print('\t == Correctly classified: X {:.4f} X_hat {:.4f} Match {:.4f}'.format(corrects[0], corrects[1], corrects[2]))
		runtimes = np.asarray(runtimes)/ nbatches
		print('Average time G: {:.4f}, E: {:.4f}, C: {:.4f} over {} batches of size {}'.format(runtimes[0], runtimes[1], runtimes[2], nbatches, batch_size))

	@staticmethod
	def evaluate_fnet(inet, fnet, dataloader, stats):
		print('Evaluate fusion network')
		inet.eval()
		fnet.eval()
		def rescale(imgs, key):
			s = stats[key]
			for k in range(3):
				imgs[:, k, ...] = imgs[:, k, ...] * s['std'][k] + s['mean'][k]
			return imgs
		total = 0
		corrects = 0
		for batch_idx, (inputs, _) in enumerate(dataloader):
			batch_size = inputs.shape[0]
			total += batch_size
			index = torch.randperm(batch_size).cuda()
			img1 = Variable(inputs).cuda()
			img2 = img1[index, :]
			img3 = fnet(img1, img2)

			img1 = rescale(img1, 'input')
			img2 = rescale(img2, 'input')
			img3 = rescale(img3, 'target')

			out, z1, _ = inet(img1)
			_, z2, _ = inet(img2)
			_, z3, _ = inet(img3)

			z1_hat = 2*z3 - z2
			out_hat = inet.module.classifier(z1_hat)

			_, y = torch.max(out.data, 1)
			_, y_hat = torch.max(out_hat.data, 1)
			corrects += y_hat.eq(y.data).sum().cpu().item()

			del _, z1, z3, z1_hat, inputs, img1, img2, img3, out, out_hat, y, y_hat

		# Evaluate time and classification performance
		corrects = 100 * corrects / total
		print('\t == Correctly Match {:.4f}'.format(corrects))

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


	@staticmethod
	def anaylse_trace_estimation(model, testset, use_cuda, extension):
		# setup range for analysis
		numSamples = np.arange(10) * 10 + 1
		numIter = np.arange(10)
		# setup number of datapoints
		testloader = torch.utils.data.DataLoader(
			testset, batch_size=128, shuffle=False, num_workers=2)
		# TODO change

		for batch_idx, (inputs, targets) in enumerate(testloader):
			if use_cuda:
				inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
			inputs, targets = Variable(
				inputs, requires_grad=True), Variable(targets)
			# compute trace
			out_bij, p_z_g_y, trace, gt_trace = model(inputs[:, :, :8, :8],
													  exact_trace=True)
			trace = [t.cpu().numpy() for t in trace]
			np.save('gtTrace' + extension, gt_trace)
			np.save('estTrace' + extension, trace)
			return

	@staticmethod
	def test_spec_norm_v0(model, in_shapes, extension):
		i = 0
		j = 0
		params = [v for v in model.module.state_dict().keys()
				  if "bottleneck" and "weight" in v
				  and not "weight_u" in v
				  and not "weight_orig" in v
				  and not "bn1" in v and not "linear" in v
				  and not "weight_sigma" in v]
		print(len(params))
		print(len(in_shapes))
		svs = []
		for param in params:
			if i == 0:
				input_shape = in_shapes[j]
			else:
				input_shape = in_shapes[j]
				# input_shape[1] = int(input_shape[1] // 4)

			convKernel = model.module.state_dict()[param].cpu().numpy()
			# input_shape = input_shape[2:]
			# fft_coeff = np.fft.fft2(convKernel.reshape(input_shape), input_shape, axes=[2, 3])
			from IPython import embed
			embed()
			fft_coeff = np.fft.fft2(convKernel.reshape(input_shape))
			t_fft_coeff = np.transpose(fft_coeff)
			U, D, V = np.linalg.svd(
				t_fft_coeff, compute_uv=True, full_matrices=False)
			Dflat = np.sort(D.flatten())[::-1]
			print("Layer " + str(j) + " Singular Value " + str(Dflat[0]))
			svs.append(Dflat[0])
			if i == 2:
				i = 0
				j += 1
			else:
				i += 1
		np.save('singular_values' + extension, svs)
		return

	@staticmethod
	def test_spec_norm(model, in_shapes, extension):
		i = 0
		j = 0
		params = [v for v in model.module.state_dict().keys()
				  if "bottleneck" in v and "weight_orig" in v
				  and not "weight_u" in v
				  and not "bn1" in v
				  and not "linear" in v]
		print(len(params))
		print(len(in_shapes))
		svs = []
		for param in params:
			input_shape = tuple(in_shapes[j])
			# get unscaled parameters from state dict
			convKernel_unscaled = model.module.state_dict()[param].cpu().numpy()
			# get scaling by spectral norm
			sigma = model.module.state_dict()[param[:-5] + '_sigma'].cpu().numpy()
			convKernel = convKernel_unscaled / sigma
			# compute singular values
			input_shape = input_shape[1:]
			fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
			t_fft_coeff = np.transpose(fft_coeff)
			D = np.linalg.svd(t_fft_coeff, compute_uv=False, full_matrices=False)
			Dflat = np.sort(D.flatten())[::-1]
			print("Layer " + str(j) + " Singular Value " + str(Dflat[0]))
			svs.append(Dflat[0])
			if i == 2:
				i = 0
				j += 1
			else:
				i += 1
		return svs

	##### -- iResNet test ------

	def run_x(x, criterion, model):
		model.eval()
		_, z = model(x)
		x_inv = model.module.inverse(z)

		mse = criterion(x, x_inv).cpu().item()
		pixel_x = [x.min().cpu().item(), x.max().cpu().item(), x.mean().cpu().item()]
		pixel_x_inv = [x_inv.min().cpu().item(), x_inv.max().cpu().item(), x_inv.mean().cpu().item()]
		pixel_z = [z.min().cpu().item(), z.max().cpu().item(), z.mean().cpu().item()]

		def print_out_pixel(pname, pixels):
			print('\t \t {}: {:.4f} {:.4f} {:.4f}'.format(pname, pixels[0], pixels[1], pixels[2]))

		print('Avg.MSE of one image: {:.4f}'.format(mse/128))
		print_out_pixel('pixel_x', pixel_x)
		print_out_pixel('pixel_x_inv', pixel_x_inv)
		print_out_pixel('pixel_z', pixel_z)

	def test_x_range(model):
		in_shapes = [128, 3, 32, 32]
		x_range = [-2.4291, 2.7537]

		criterion = torch.nn.MSELoss(reduction='sum')
		x0 = -torch.rand(in_shapes) + 100 * x_range[0]
		x1 = torch.rand(in_shapes) + 100 * x_range[1]
		x2 = torch.rand(in_shapes) * (x_range[1] - x_range[0]) + x_range[0]
		x3 = torch.zeros(in_shapes)

		print('\nBelow:')
		run(x0.cuda(), criterion, model)
		print('\nAbove')
		run(x1.cuda(), criterion, model)
		print('\nInrange')
		run(x2.cuda(), criterion, model)
		print('\nZeros')
		run(x3.cuda(), criterion, model)

	def run_z(z, criterion, model):
		model.eval()
		x = model.module.inverse(z)
		_, z_inv = model(x)

		mse = criterion(z, z_inv).cpu().item()
		pixel_x = [x.min().cpu().item(), x.max().cpu().item(), x.mean().cpu().item()]
		pixel_z_inv = [z_inv.min().cpu().item(), z_inv.max().cpu().item(), z_inv.mean().cpu().item()]
		pixel_z = [z.min().cpu().item(), z.max().cpu().item(), z.mean().cpu().item()]

		def print_out_pixel(pname, pixels):
			print('\t \t {}: {:.4f} {:.4f} {:.4f}'.format(pname, pixels[0], pixels[1], pixels[2]))

		print('Avg.MSE of one image: {:.4f}'.format(mse/128))
		print_out_pixel('pixel_x', pixel_x)
		print_out_pixel('pixel_z_inv', pixel_z_inv)
		print_out_pixel('pixel_z', pixel_z)

	def test_z_range(model):
		in_shapes = [64, 256, 8, 8]
		x_range = [-896.5399, 1055.2096]

		criterion = torch.nn.MSELoss(reduction='sum')
		x0 = -torch.rand(in_shapes) + 100 * x_range[0]
		x1 = torch.rand(in_shapes) + 100 * x_range[1]
		x2 = torch.rand(in_shapes) * (x_range[1] - x_range[0]) + x_range[0]
		x3 = torch.zeros(in_shapes)

		print('\nBelow:')
		run_z(x0.cuda(), criterion, model)
		print('\nAbove')
		run_z(x1.cuda(), criterion, model)
		print('\nInrange')
		run_z(x2.cuda(), criterion, model)
		print('\nZeros')
		run_z(x3.cuda(), criterion, model)

	def test_z(model, testloader):
		(inputs, _) = iter(testloader).next()
		with torch.no_grad():
			inputs = Variable(inputs.cuda())
		# modelling
		bs = inputs.shape[0] // 2
		_, out_bij = model(inputs)
		# divide out_bij into 2 parts
		z1 = out_bij[:bs, ...]
		z2 = out_bij[bs:, ...]

		mses = np.zeros([3, 10])
		criterion = torch.nn.MSELoss(reduction='sum')
		for i in range(10):
			alpha = (1 + i) * 0.1
			z3 = (1 - alpha) * z1 + alpha * z2
			#invert
			x_inv_3 = model.module.inverse(z3)
			_, z3_hat = model(x_inv_3)
			z2_hat = (z3_hat - z1 * (1 - alpha))/alpha

			# invert
			x_real = inputs[bs:, ...]
			x_inv = model.module.inverse(z2)
			x_inv_hat = model.module.inverse(z2_hat)

			# MSE measurement
			mses[0][i] = criterion(x_real, x_inv).cpu().item() /bs
			mses[1][i] = criterion(x_real, x_inv_hat).cpu().item()/bs
			mses[2][i] = criterion(z2, z2_hat).cpu().item()/bs

		from matplotlib import pyplot as plt
		fig = plt.figure()
		alphas = (np.arange(10) + 1) * 0.1
		plt.title('MSE comparision over Primary Image Weight (alpha)')
		# plt.plot(alphas, mses[0, :], label=r'X_inv', c='r')
		# plt.plot(alphas, mses[1, :], label=r'X_inv_hat', c='b')
		plt.plot(alphas, mses[2, :], label=r'Z_inv', c='g')
		plt.legend()
		plt.xlabel('alpha')
		plt.ylabel("MSE")
		plt.show()

		fig.tight_layout()
		fig.savefig('spectralnorm.jpg', bbox_inches='tight')
