from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import random

from torchvision.transforms import v2
from torch.utils.data import default_collate


def cifar10_cutmix_mixup_collate_fn(batch):
	cutmix = v2.CutMix(num_classes=10)
	mixup = v2.MixUp(num_classes=10)
	cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
	return cutmix_or_mixup(*default_collate(batch))


def cifar100_cutmix_mixup_collate_fn(batch):
	cutmix = v2.CutMix(num_classes=100)
	mixup = v2.MixUp(num_classes=100)
	cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
	return cutmix_or_mixup(*default_collate(batch))


# ----------------------- CIFAR 10 Dataset functions ----------------------------------------------------

def load_CIFAR10_train_validation(
	batch_size, train_transformations, validation_set_size, use_collate_fn=False
):
	# the collate_fn is only applied for the training dataset but this function will also return an unmodified version
	if use_collate_fn is True:
		fn = cifar10_cutmix_mixup_collate_fn
	else:
		fn = None

	training_data = datasets.CIFAR10(
		root="../root", train=True, download=True, transform=train_transformations
	)

	validation_set_inidices = random.sample(
		range(len(training_data)), validation_set_size
	)
	
	training_set_inidices = list(
		set([x for x in range(len(training_data))]) - set(validation_set_inidices)
	)
	assert len(training_set_inidices) == len(training_data) - validation_set_size

	validation_set = Subset(training_data, validation_set_inidices)
	training_set = Subset(training_data, training_set_inidices)

	validation_set_loader = DataLoader(
		validation_set, batch_size, shuffle=True, num_workers=2
	)
	training_set_loader = DataLoader(
		training_set, batch_size, shuffle=True, num_workers=2, collate_fn=fn
	)
 
	if use_collate_fn is True:
		unmodified_training_set_loader = DataLoader(
			training_set, batch_size, shuffle=True, num_workers=2
		)
		
		return validation_set_loader, training_set_loader, unmodified_training_set_loader
	else:
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
	return DataLoader(test_data, batch_size, shuffle=True, num_workers=2)

# ----------------------- CIFAR 100 Dataset functions ----------------------------------------------------


def load_CIFAR100_train_validation(
	batch_size, train_transformations, validation_set_size, use_collate_fn=False
):
    
	# the collate_fn is only applied for the training dataset but this function will also return an unmodified version
	if use_collate_fn is True:
		fn = cifar100_cutmix_mixup_collate_fn
	else:
		fn = None
    
	training_data = datasets.CIFAR100(
		root="../cifar100", train=True, download=True, transform=train_transformations
	)

	validation_set_inidices = random.sample(
		range(len(training_data)), validation_set_size
	)

	training_set_inidices = list(
		set([x for x in range(len(training_data))]) - set(validation_set_inidices)
	)
	assert len(training_set_inidices) == len(training_data) - validation_set_size

	validation_set = Subset(training_data, validation_set_inidices)
	training_set = Subset(training_data, training_set_inidices)

	validation_set_loader = DataLoader(
		validation_set, batch_size, shuffle=True, num_workers=2
	)
	training_set_loader = DataLoader(
		training_set, batch_size, shuffle=True, num_workers=2, collate_fn=fn
	)
 
	
	if use_collate_fn is True:
		unmodified_training_set_loader = DataLoader(
			training_set, batch_size, shuffle=True, num_workers=2
		)
			
		return validation_set_loader, training_set_loader, unmodified_training_set_loader
	else:
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

	return DataLoader(test_data, batch_size, shuffle=True, num_workers=2)
