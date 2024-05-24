from VGG_dropout_BN_models.VGG3_BN_dropout import VGG3_BN_dropout
from SymmetricCrossEntropyLearning import SymmetricCrossEntropyLearning
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import time
import numpy as np
import copy

train_model_training_loss_ls = []
train_model_training_accuracy_ls = []
validation_model_training_loss_ls = []
validation_model_training_accuracy_ls = []

np.random.seed(256) # seed for consistency

def plot_training_validation_loss_and_accuracy():
	epochs = range(1, len(train_model_training_loss_ls) + 1)

	plt.figure(figsize=(12, 6))

	plt.subplot(1, 2, 1)
	plt.plot(epochs, train_model_training_loss_ls, "c", label="Training Loss")
	plt.plot(epochs, validation_model_training_loss_ls, "r", label="Validation Loss")
	plt.title("Training and Validation Loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(epochs, train_model_training_accuracy_ls, "c", label="Training Acc.")
	plt.plot(
		epochs, validation_model_training_accuracy_ls, "r", label="Validation Acc."
	)
	plt.title("Training and Validation Accuracy")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.legend()

	plt.show()

	#plt.show()
	fname = f"figs/{time.strftime('%Y%m%d-%H%M%S')}"
	plt.savefig(fname)
	plt.close()

# load train and test dataset
def load_dataset():

    # load dataset
    training_vali_data = datasets.CIFAR10(
        "root",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )
    test_data = datasets.CIFAR10(
        "root",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    return training_vali_data, test_data

def contaminate_labels(dataset, noise=0.0, num_classes=10):
	num_samples = len(dataset)
	num_to_contaminate = int(noise * num_samples)
	indices_to_contaminate = np.random.choice(num_samples, num_to_contaminate, replace=False)

	for i in indices_to_contaminate:
		original_label = dataset.targets[i]
		possible_labels = list(range(num_classes))
		
		# excludes original label from being selected as new one
		possible_labels.remove(original_label)

		new_label = np.random.choice(possible_labels)
		dataset.targets[i] = new_label
		
	return dataset


def create_dataloaders(batch_size, training_data, validation_data, test_data):
	train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
	validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
	test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

	return train_dataloader, validation_dataloader, test_dataloader


def he_initalization(m):
	for i in m.modules():
		if isinstance(i, nn.Conv2d) or isinstance(i, nn.Linear):
			nn.init.kaiming_normal_(i.weight, mode="fan_in", nonlinearity="relu")
			nn.init.zeros_(i.bias)


def evaluate(model, dataloader, loss_fn):
	model.eval()

	dataset_size = len(dataloader.dataset)
	num_batches = len(dataloader)

	total_loss, num_correct = 0, 0

	# Disable gradient computation and reduce memory consumption.
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)

			# making predictions
			pred = model(X)

			# getting total loss
			total_loss += loss_fn(pred, y).item()

			num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

	total_loss /= num_batches
	num_correct /= dataset_size
	print(
		f"Test Data: \n Accuracy: {(100*num_correct):>0.3f}%, Avg loss: {total_loss:>8f} \n"
	)


def compute_loss_on_whole_dataloader(model, dataloader, loss_fn, device):
	running_loss = 0.0

	for valX, valy in dataloader:
		valX, valy = valX.to(device), valy.to(device)

		# make predictions
		validation_pred = model(valX.to(device))
		# compute loss
		validation_loss = loss_fn(validation_pred, valy.to(device).long())
		# update running loss
		running_loss += validation_loss.item()

	# dividing running loss with number total batches made on the data
	return running_loss / len(dataloader)

def compute_accuracy_on_whole_dataloader(model, dataloader, device):
	dataset_size = len(dataloader.dataset)

	num_correct = 0
	with torch.no_grad():
		for X, y in dataloader:
			X, y = X.to(device), y.to(device)

			# making predictions
			pred = model(X)

			num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

	return num_correct / dataset_size

def train(
	epochs, training_dataloader, validation_dataloader, model, loss_fn, optimizer
):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	# set the model on training model
	for current_epoch in range(0, epochs):
		print(f"current epoch: {current_epoch}")

		if current_epoch == 40 or current_epoch == 80:
			print("learning rate updated from ", optimizer.param_groups[0]['lr'], end=" ")
			update_learning_rate(optimizer, 0.1)
			print("to ", optimizer.param_groups[0]['lr'])

		for batch_index, (X, y) in enumerate(training_dataloader):
			model.train()

			X, y = X.to(device), y.to(device)

			# compute prediction error
			# here we are making the prediction
			trainig_pred = model(X)
			# computing the loss from our prediction and true val
			training_loss = loss_fn(trainig_pred, y)

			# have to zero out the gradients, for each batch since they can be accumulated
			optimizer.zero_grad()

			# backpropagation
			training_loss.backward()

			# Adjust learning weights
			optimizer.step()

		# getting valiation loss now
		model.eval()

		training_loss = compute_loss_on_whole_dataloader(model, training_dataloader, loss_fn, device)
		validation_loss = compute_loss_on_whole_dataloader(model, validation_dataloader, loss_fn, device)

		training_acc = compute_accuracy_on_whole_dataloader(model, training_dataloader, device)
		validation_acc = compute_accuracy_on_whole_dataloader(model, validation_dataloader, device)

		train_model_training_loss_ls.append(training_loss)
		validation_model_training_loss_ls.append(validation_loss)

		train_model_training_accuracy_ls.append(training_acc)
		validation_model_training_accuracy_ls.append(validation_acc)

		# Print information out
		print(f"Epoch: {current_epoch}, Loss: {training_loss:.4f}")

	print("training finished")

def update_learning_rate(optimizer, multiplier):
	for param_group in optimizer.param_groups:
		param_group['lr'] *= multiplier

if __name__ == "__main__":
	device = (
		"cuda"
		if torch.cuda.is_available()
		else "mps" if torch.backends.mps.is_available() else "cpu"
	)
	print(f"Using {device} device")

	batch_size = 64

	train_vali_data, test_data = load_dataset()

	validation_set_size = 5000
	training_set_size = len(train_vali_data.targets) - validation_set_size

	training_data, validation_data = random_split(train_vali_data, [training_set_size, validation_set_size])
	training_data_copy = copy.deepcopy(training_data)

	# Contaminate training data labels
	noise = 0.6
	contaminate_labels(training_data_copy.dataset, noise)

	training_dataloader, validation_loader, test_dataloader = create_dataloaders(batch_size, training_data_copy, validation_data, test_data)

	VGG3 = VGG3_BN_dropout().to(device)
	VGG3.apply(he_initalization)

	# defining loss function and optimizer
	loss_fn = SymmetricCrossEntropyLearning()
	#loss_fn = nn.CrossEntropyLoss()

	optimize = torch.optim.SGD(VGG3.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

	start_total_time = time.time()

	train(120, training_dataloader, validation_loader, VGG3, loss_fn, optimize)  # training
	train_time = time.time() - start_total_time
	print(f"Training Time: {train_time}")

	start_evaluation_time = time.time()

	evaluate(VGG3, test_dataloader, loss_fn)  # evaluating

	evaluate_time = time.time() - start_evaluation_time
	print(f"Evaluation Time: {evaluate_time}")

	print(f"Total Time: {time.time() - start_total_time}")

	# print(f"training lost list: {train_model_training_loss_ls}")
	# print(f"validation lost list: {validation_model_training_loss_ls}")

	plot_training_validation_loss_and_accuracy()
