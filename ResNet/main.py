import torch
import load_data


if __name__ == "__main__":
	device = (
		"cuda"
		if torch.cuda.is_available()
		else "mps" if torch.backends.mps.is_available() else "cpu"
	)
	print(f"Using {device} device")
 
	validation_set_size = 5000

	validation_set_loader, training_set_loader = load_data.load_CIFAR10_train_validation(validation_set_size)
	test_set_dataloader = load_data.load_CIFAR10_test()
 
	
 
