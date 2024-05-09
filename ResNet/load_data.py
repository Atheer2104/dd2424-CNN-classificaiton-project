import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Subset
import random

# this is the same as the ResNet paper
train_batch_size = 128

# setting batch size at test time to be 100 have division since we have 10k images at test time
test_batch_size = 100


def load_CIFAR10_train_validation(validation_set_size):
	train_transformations = v2.Compose(
		[
			v2.ToImage(),
			# add 4 pixel zero padding
			v2.Pad(4),
			# 50% of horizontal flip
			v2.RandomHorizontalFlip(),
			# perform random crop back to size 32x32
			v2.RandomCrop(size=32),
			# change type of number and re-scale to [0.0, 1.0]
			v2.ToDtype(torch.float32, scale=True),
			# ! have to perform per-pixel mean subtraction
			v2.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
		]
	)

	training_data = datasets.CIFAR10(
		root="../root", train=True, download=True, transform=train_transformations
	)

	validation_set_inidices = random.sample(
		range(len(training_data)), validation_set_size
	)
	
	training_set_inidices = list(set([x for x in range(len(training_data))]) - set(validation_set_inidices))
	assert(len(training_set_inidices) == len(training_data) - validation_set_size)

	validation_set = Subset(training_data, validation_set_inidices)
	training_set = Subset(training_data, training_set_inidices)

	validation_set_loader = DataLoader(validation_set, batch_size=train_batch_size, shuffle=True, num_workers=2)
	training_set_loader = DataLoader(training_set, batch_size=train_batch_size, shuffle=True, num_workers=2)
 
	return validation_set_loader, training_set_loader


def load_CIFAR10_test():
	test_transformations = v2.Compose(
		[
			v2.ToImage(),
			# change type of number and re-scale to [0.0, 1.0]
			v2.ToDtype(torch.float32, scale=True),
			# ! have to perform per-pixel mean subtraction
			v2.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
		]
	)

	test_data = datasets.CIFAR10(
		root="../root", train=False, download=True, transform=test_transformations
	)
 
	return DataLoader(
		test_data, batch_size=test_batch_size, shuffle=True, num_workers=2
	)
