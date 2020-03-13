
import torch
import torchvision
import torchvision.transforms as transforms


class Provider:
    @staticmethod
    def load_data(dataset, data_root):
        ########## Data ####################
        if dataset == 'mnist':
        	dens_est_chain = [
        		lambda x: (255. * x) + torch.zeros_like(x).uniform_(0., 1.),
        		lambda x: x / 256.,
        		lambda x: x - 0.5
        	]
        	mnist_transforms = [transforms.Pad(2, 0), transforms.ToTensor(), lambda x: x.repeat((3, 1, 1))]
        	transform_train_mnist = transforms.Compose(mnist_transforms + dens_est_chain)
        	transform_test_mnist = transforms.Compose(mnist_transforms + dens_est_chain)
        	trainset = torchvision.datasets.MNIST(
        		root=data_root, train=True, download=True, transform=transform_train_mnist)
        	testset = torchvision.datasets.MNIST(
        		root=data_root, train=False, download=False, transform=transform_test_mnist)
        else:
        	# 'cifar10':
        	mean = {
        		'cifar10': (0.4914, 0.4822, 0.4465),
        		'cifar100': (0.5071, 0.4867, 0.4408),
        	}

        	std = {
        		'cifar10': (1, 1, 1),
        		# 'cifar10': (0.2023, 0.1994, 0.2010),
        		'cifar100': (0.2675, 0.2565, 0.2761),
        	}

        	# Only for cifar-10
        	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        	train_chain = [transforms.Pad(4, padding_mode="symmetric"),
        				   transforms.RandomCrop(32),
        				   transforms.RandomHorizontalFlip(),
        				   transforms.ToTensor()]
        	test_chain = [transforms.ToTensor()]
        	clf_chain = [transforms.Normalize(mean[dataset], std[dataset])]
        	transform_train = transforms.Compose(train_chain + clf_chain)
        	transform_test = transforms.Compose(test_chain + clf_chain)
        	trainset = torchvision.datasets.CIFAR10(
        		root=data_root, train=True, download=True, transform=transform_train)
        	testset = torchvision.datasets.CIFAR10(
        		root=data_root, train=False, download=True, transform=transform_test)

        return trainset, testset
