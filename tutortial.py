from VGG1 import VGG1
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader


# load train and test dataset
def load_dataset():

    # load dataset
    training_data = datasets.CIFAR10(
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

    return training_data, test_data


def create_dataloaders(batch_size, training_data, test_data):
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def he_initalization(m):
    for i in m.modules():
        if isinstance(i, nn.Conv2d) or isinstance(i, nn.Linear):
            nn.init.kaiming_normal_(i.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(i.bias)


def evaluate(model, dataloader, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
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
    num_correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*num_correct):>0.1f}%, Avg loss: {total_loss:>8f} \n"
    )


def train(epochs, dataloader, model, loss_fn, optimizer):
    # set the model on training model
    model.train()

    number_of_batch_til_print_loss = 25

    for current_epoch in range(epochs):
        print(f"current epoch: {current_epoch}")

        running_loss = 0.0
        for batch_index, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # compute prediction error
            # here we are making the prediction
            pred = model(X)
            # computing the loss from our prediction and true val
            loss = loss_fn(pred, y)

            # have to zero out the gradients, for each batch since they can be accumulated
            optimizer.zero_grad()

            # backpropagation
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            running_loss += loss.item()

            # print loss for every 25 batches
            if batch_index % number_of_batch_til_print_loss == 0:
                print(
                    f"Batch Index: {batch_index} loss: {running_loss / number_of_batch_til_print_loss:.3f} \n"
                )
                running_loss = 0.0

    print("training finished")


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device} device")

    batch_size = 64

    trainig_data, test_data = load_dataset()
    train_dataloader, test_dataloader = create_dataloaders(
        batch_size, trainig_data, test_data
    )

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    model = VGG1().to(device)
    # print(model)

    model.apply(he_initalization)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

    # defining loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimize = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(100, train_dataloader, model, loss_fn, optimize)
    evaluate(model, test_dataloader, loss_fn)
    