import argparse
import sys
from collections import OrderedDict

import torch
from torchvision import datasets
from torch import nn
from torch import optim
import torch.nn.functional as F

from common import model_dicts, common_transform

def _get_model(arch, hidden_units):
    print("Getting Model")
    model_to_use = model_dicts[arch]
    model = model_to_use.get("model")

    # freeze params
    for param in model.parameters():
        param.requires_grad = False

    h = hidden_units if hidden_units is not None else model_to_use.get("hidden")
    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(model_to_use.get("input"), h)), 
                        ('relu', nn.ReLU()),
                        ('fc2', nn.Linear(4096, 102)),
                        # use softmax to get probability of the 102 output classes
                        ('output', nn.LogSoftmax(dim=1))
                        ]))

    model.classifier = classifier
    return model


def _get_loaders(data_dir):
    print("Getting Loaders")
    train_dir = f'{data_dir}/train'
    valid_dir =  f'{data_dir}/valid'
    test_dir = f'{data_dir}/test'

    image_datasets = datasets.ImageFolder(data_dir, transform=common_transform)
    train_datasets = datasets.ImageFolder(train_dir, transform=common_transform)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=common_transform)
    test_datasets = datasets.ImageFolder(test_dir, transform=common_transform)

    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)
    return dataloader, trainloader, validloader, testloader, train_datasets

def _test_nn(model, testloader, device):
    print("Testing NN")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            # get the highest probability index
            probability, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
        print(f'Accuracy: {100 * correct / total}')


def _prepare_and_train(data_directory, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    print("Prepare and Train")
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

    dataloader, trainloader, validloader, testloader, train_datasets = _get_loaders(data_directory)
    model = _get_model(arch, hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    model.to(device)

    print_every = 40
    steps = 0

    for e in range(epochs):
        r_loss = 0

        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            r_loss += loss.item()

            if steps % print_every == 0:
                n_correct = 0
                total = 0
                accuracy = 0

                with torch.no_grad():
                    for data in validloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = model(images)
                        # get the highest probability index
                        probability, prediction = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        n_correct += (prediction == labels).sum().item()

                    accuracy = n_correct / total 
                    
                print(f"Epoch: {e+1}/{epochs}... ",
                        f"Loss: {r_loss/print_every}",
                        f"Accuracy: {round(accuracy,4)}")

                r_loss = 0

    _test_nn(model, testloader, device)

        # Ideas from udacity community boards
    checkpoint = {
                    'model_used': arch,
                    'input_size': 25088,
                    'output_size': 102,
                    'classifier': model.classifier,
                    'state_dict': model.state_dict(),
                    'idx_to_class': {v: k for k, v in train_datasets.class_to_idx.items()}
                }

    torch.save(checkpoint, f'{save_dir}/model_checkpoint.pth')


def _get_arguments(sys_args):
    parser = argparse.ArgumentParser(
        description="Submitting measurement sets to the Submissions API."
    )

    parser.add_argument("data_directory")

    parser.add_argument(
        "--save_dir",
        default="./",
        help="Directory to save the checkpoint",
    )

    parser.add_argument(
        "--arch",
        default="vgg19",
        help="Trained Model to Use",
    )

    parser.add_argument(
        "--learning_rate",
        default=0.3,
        help="Learning Rate for the Model",
        type=int
    )

    parser.add_argument(
        "--hidden_units",
        default=4096,
        help="Hidden Units for the Model",
        type=int
    )

    parser.add_argument(
        "--epochs",
        default=3,
        help="Number of epocjs",
        type=int
    )

    parser.add_argument("--gpu", action="store_true",
                    help="Run on GPU")

    return parser.parse_args(sys_args)


def cmd_line_entry():
    args = _get_arguments(sys.argv[1:])
    print("Running with args", args)
    _prepare_and_train(args.data_directory, args.save_dir, args.arch, args.learning_rate,
                        args.hidden_units, args.epochs, args.gpu)


if __name__ == "__main__":
    cmd_line_entry()