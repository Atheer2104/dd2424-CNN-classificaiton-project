import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

def compute_mean_std(dataset):
	dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
  
	mean = torch.zeros(3)
	std = torch.zeros(3)
	
	# the images will have the shape [batch size, # channels, width, height]
	for image, _label in dataloader:
		for channel in range(3):
			mean[channel] += image[:, channel, :, :].mean()
			std[channel] += image[:, channel, :, :].std()
	
 	# dividing by the total images that we have 
	# div_ is inplace division means that it will modify the existing tensor
	mean.div_(len(dataset))
	std.div_(len(dataset))
	return mean, std

		
if __name__ == "__main__":
	training_data_set_CIFAR10 = datasets.CIFAR10(root="../root", train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))
	mean, std = compute_mean_std(training_data_set_CIFAR10)
 
	print(f"mean: {mean}")
	print(f"std: {std}")

 