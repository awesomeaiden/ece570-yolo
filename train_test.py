from __future__ import division

import torch

from util import *
from network import Network
import torchvision.datasets
import torch.optim as optim

CUDA = torch.cuda.is_available()

yolo = Network("yolo.cfg")
optimizer = optim.Adam(yolo.parameters(), lr=1e-4)
loss_fn = nn.MSELoss(reduction='sum')

confidence = 0.5
nms_thresh = 0.4
num_classes = 80
classes = load_classes("coco.names")

# Create dataset transforms and loaders
# Custom transform to normalize size of input images
class Normalize(object):
    def __init__(self, inp_dim=416):
        self.inp_dim = inp_dim

    def __call__(self, img):
        return prep_image(img.numpy().transpose((1, 2, 0)), self.inp_dim)

    def __repr__(self):
        return "Normalize images"


def collate_fn(batch):
    temp = list(zip(*batch))
    temp[0] = torch.cat(temp[0], 0)
    new_temp_one = None
    for img in range(len(temp[1])):
        objects = temp[1][img]["annotation"]["object"]
        for det in objects:
            box = det["bndbox"]
            new_tens = torch.tensor([float(img), float(box["xmin"]), float(box["ymax"]), float(box["xmax"]), float(box["ymin"]), float(classes.index(det["name"]))]).view(1, 6)
            if new_temp_one is None:
                new_temp_one = new_tens
            else:
                new_temp_one = torch.cat((new_temp_one, new_tens), 0)
    temp[1] = new_temp_one
    return tuple(temp)


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), Normalize(int(yolo.net_info["height"]))])

batch_size_train = int(yolo.net_info["batch"])
batch_size_test = 4

train_dataset = torchvision.datasets.VOCDetection("/data", image_set="train", download=False, transform=transform)
test_dataset = torchvision.datasets.VOCDetection("/data", image_set="val", download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True, collate_fn=collate_fn)

test_losses = []
test_counter = []
train_losses = []
train_counter = []

device = None
if (CUDA):
    device = torch.device("cuda")

torch.cuda.empty_cache()


def get_predictions(model, confidence, num_classes, nms_thresh, batch):
    # Iterate over the batches, generate the prediction, and concatenate the prediction tensors
    write = 0
    prediction = model(Variable(batch), CUDA)
    prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thresh)

    return prediction


def train(classifier, optimizer, epoch, CUDA):
    # Set to train mode
    classifier.train()
    if CUDA:
        classifier = classifier.cuda()

    for batch_idx, (images, targets) in enumerate(train_loader):
        if CUDA:
            images = images.cuda()
            #targets = targets.to(device)

        optimizer.zero_grad()
        output = get_predictions(model=yolo, confidence=confidence, num_classes=num_classes, nms_thresh=nms_thresh, batch=images)
        loss = F.nll_loss(output, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:  # We record our output every 10 batches
            train_losses.append(loss.item() / batch_size_train)  # item() is to get the value of the tensor directly
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
        if batch_idx % 100 == 0:  # We visulize our output every 100 batches
            print(
                f'Epoch {epoch}: [{batch_idx * len(images)}/{len(train_loader.dataset)}] Loss: {loss.item() / batch_size_train}')


def test(classifier, epoch, CUDA):
    # Set to evaluation mode
    classifier.eval()
    if CUDA:
        classifier = classifier.cuda()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for images, targets in test_loader:
            if CUDA:
                images = images.to(device)
                targets = targets.to(device)
            output = classifier(images)
            test_loss += F.nll_loss(output, targets, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_counter.append(len(train_loader.dataset) * epoch)

    print(f'Test result on epoch {epoch}: Avg loss is {test_loss}')


max_epoch = 3
for epoch in range(1, max_epoch + 1):
    train(classifier=yolo, optimizer=optimizer, epoch=epoch, CUDA=CUDA)
    test(classifier=yolo, epoch=epoch, CUDA=CUDA)
