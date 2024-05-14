from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import random



# ----------------------- CIFAR 10 Dataset functions ----------------------------------------------------

def load_CIFAR10_train_validation(batch_size, train_transformations, validation_set_size):
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

	validation_set_loader = DataLoader(validation_set, batch_size, shuffle=True, num_workers=2)
	training_set_loader = DataLoader(training_set, batch_size, shuffle=True, num_workers=2)
 
	return validation_set_loader, training_set_loader

def load_CIFAR10_train(batch_size, train_transformation):
	# load dataset
	training_data = datasets.CIFAR10(
		"../root",
		train=True,
		download=True,
		transform=train_transformation,
	)
 
	return DataLoader(training_data, batch_size, shuffle=True, num_workers=2)

def load_CIFAR10_test(batch_size, test_transformations):
	test_data = datasets.CIFAR10(
		root="../root", train=False, download=True, transform=test_transformations
	)
 
	return DataLoader(
		test_data, batch_size, shuffle=True, num_workers=2
	)

# ----------------------- CIFAR 100 Dataset functions ----------------------------------------------------

def load_CIFAR100_train_validation(batch_size, train_transformations, validation_set_size):
	training_data = datasets.CIFAR100(
		root="../cifar100", train=True, download=True, transform=train_transformations
	)

	validation_set_inidices = random.sample(
		range(len(training_data)), validation_set_size
	)
	
	training_set_inidices = list(set([x for x in range(len(training_data))]) - set(validation_set_inidices))
	assert(len(training_set_inidices) == len(training_data) - validation_set_size)

	validation_set = Subset(training_data, validation_set_inidices)
	training_set = Subset(training_data, training_set_inidices)

	validation_set_loader = DataLoader(validation_set, batch_size, shuffle=True, num_workers=2)
	training_set_loader = DataLoader(training_set, batch_size, shuffle=True, num_workers=2)
 
	return validation_set_loader, training_set_loader

def load_CIFAR100_train(batch_size, train_transformation):
	# load dataset
	training_data = datasets.CIFAR100(
		"../cifar100",
		train=True,
		download=True,
		transform=train_transformation,
	)
 
	return DataLoader(training_data, batch_size, shuffle=True, num_workers=2)

def load_CIFAR100_test(batch_size, test_transformations):
	test_data = datasets.CIFAR100(
		root="../cifar100", train=False, download=True, transform=test_transformations
	)
 
	return DataLoader(
		test_data, batch_size, shuffle=True, num_workers=2
	)

