
import torch

import matplotlib.pyplot as plt
import time

train_model_training_loss_ls = []
train_model_training_accuracy_ls = []
validation_model_training_loss_ls = []
validation_model_training_accuracy_ls = []

def clear_histogram():
	global train_model_training_loss_ls, train_model_training_accuracy_ls
	global validation_model_training_loss_ls, validation_model_training_accuracy_ls

	train_model_training_loss_ls = []
	train_model_training_accuracy_ls = []
	validation_model_training_loss_ls = []
	validation_model_training_accuracy_ls = []


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

	# plt.show()
	fname = f"figs/{time.strftime('%Y%m%d-%H%M%S')}"
	plt.savefig(fname)
	plt.close()
 
def compute_loss_on_whole_dataloader(model, dataloader, loss_fn, device):
	running_loss = 0.0

	for valX, valy in dataloader:
		valX, valy = valX.to(device), valy.to(device)

		# make predictions
		validation_pred = model(valX.to(device))
		# compute loss
		validation_loss = loss_fn(validation_pred, valy.to(device))
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
 
 
def compute_train_validation_loss_accuracy(current_epoch, model, loss_fn, device, training_dataloader, validation_dataloader):
	# getting valiation loss now
	model.eval()

	training_loss = compute_loss_on_whole_dataloader(model, training_dataloader, loss_fn, device)
	validation_loss = compute_loss_on_whole_dataloader(model, validation_dataloader, loss_fn, device)

	train_model_training_loss_ls.append(training_loss)
	validation_model_training_loss_ls.append(validation_loss)

	training_acc = compute_accuracy_on_whole_dataloader(model, training_dataloader, device)
	validation_acc = compute_accuracy_on_whole_dataloader(
		model, validation_dataloader, device
	)

	train_model_training_accuracy_ls.append(training_acc)
	validation_model_training_accuracy_ls.append(validation_acc)

	# Print information out
	print(f"Epoch: {current_epoch}, Loss: {training_loss:.4f}")
 
def evaluate(model, dataloader, loss_fn, device):
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