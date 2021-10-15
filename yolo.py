import torch
import torch.nn as nn

# Verify GPU connectivity
import torchvision.datasets

if not torch.cuda.is_available():
    raise Exception("CUDA not available!!")
print("CUDA availability verified")

# Get gpu device
gpu = torch.device("cuda")

# Create dataset transforms and loaders
# TODO normalize input data based on mean and standard deviation of dataset?
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.VOCDetection("/data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.VOCDetection("/data", train=False, download=True, transform=transform)

batch_size_train = 64
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()

        # Define layers here

    def forward(self, x):
        # Transform x through layers here
        return 0

def train(classifier, optimizer, epoch):

    # Set to train mode
    classifier.train()
    classifier = classifier.to(gpu)

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(gpu)
        targets = targets.to(gpu)

        optimizer.zero_grad()
        output = classifier(images)
        loss = F.nll_loss(output, targets)
        loss.backward()
        optimizer.step()

        # TODO add progress indication here


test_losses = []
test_counter = []


def test(classifier, epoch):

    # Set to evaluation mode
    classifier.eval()
    classifier = classifier.to(gpu)

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(gpu)
            targets = targets.to(gpu)
            output = classifier(images)
            test_loss += F.nll_loss(output, targets, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_counter.append(len(train_loader.dataset)*epoch)

    # TODO add progress / results indication here


# TODO train and test
