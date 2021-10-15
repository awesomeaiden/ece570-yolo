from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.datasets


# NOTICE:
# This code is inspired by this tutorial on YOLO pytorch implementation here:
# https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/


# Takes a configuration file
# Returns a list of blocks.  Each block describes a block in the neural network to be built.
# Each block is represented as a dictionary in the list
def parse_cfg(cfg_file):
    with open(cfg_file, "r") as config_file:
        lines = config_file.read().split("\n")
        # Filter out empty lines, comments, and fringe whitespaces
        filtered_lines = [x.rstrip().lstrip() for x in lines if (len(x) > 0 and x[0] != "#")]

        # Now get blocks from filtered lines
        blocks = []
        block = dict()
        for line in filtered_lines:
            # Start of a new block?
            if line[0] == "[":
                # Block not empty?  Need to store before starting new block!
                if len(block) != 0:
                    blocks.append(block)
                    block = dict()
                # Set type of new block
                block["type"] = line[1:-1].rstrip()
            # Otherwise, add attribute to current block
            else:
                key, value = line.split("=")
                block[key.rstrip()] = value.lstrip()
        # Append last block
        blocks.append(block)

        return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


# Takes a list of blocks and generates nn modules for each
def create_modules(blocks):
    # Network info is in the first block
    net_info = blocks[0]

    # Instantiate ModuleList
    module_list = nn.ModuleList()

    # The depth of the kernel is the number of filters in the previous layer
    # Use this variable to keep track of that as we progress through the layers
    # Initial value is 3, as the image as RGB input channels
    prev_filters = 3

    # Need to keep all filter numbers in case there is a route layer
    # Use this list to do that
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        # Since a block may contain multiple layers, we can package them together with a sequential module
        module = nn.Sequential()

        # If this is a convolutional block
        if x["type"] == "convolutional":
            activation = x["activation"]

            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1)
            else:
                pad = 0

            # Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_" + str(index), conv)

            # Add batch norm layer, if present
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_" + str(index), bn)

            # Check the activation
            # If linear or leaky ReLU:
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_" + str(index), activn)

        # Otherwise, if its an upsampling layer
        elif x["type"] == "upsample":
            stride = int(x["stride"])

            # Use Bilinear2dUpsampling
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_" + str(index), upsample)

        # Otherwise if it is a route layer
        elif x["type"] == "route":
            split_layers = x["layers"].split(",")

            # Start of a route
            start = int(split_layers[0])

            # End of a route, if it exists
            try:
                end = int(split_layers[1])
            except:
                end = 0

            # Positive annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_" + str(index), route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filter = output_filters[index + start]

        # Otherwise if it is a shortcut layer
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_" + str(index), shortcut)


# Verify GPU connectivity
if not torch.cuda.is_available():
    raise Exception("CUDA not available!!")
print("CUDA availability verified")

# Get gpu device
gpu = torch.device("cuda")

blocks = parse_cfg("yolo.cfg")



#
# # Create dataset transforms and loaders
# # TODO normalize input data based on mean and standard deviation of dataset?
# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#
# train_dataset = torchvision.datasets.VOCDetection("/data", train=True, download=True, transform=transform)
# test_dataset = torchvision.datasets.VOCDetection("/data", train=False, download=True, transform=transform)
#
# batch_size_train = 64
# batch_size_test = 1000
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)
#
# class Yolo(nn.Module):
#     def __init__(self):
#         super(Yolo, self).__init__()
#
#         # Define layers here
#
#     def forward(self, x):
#         # Transform x through layers here
#         return 0
#
# def train(classifier, optimizer, epoch):
#
#     # Set to train mode
#     classifier.train()
#     classifier = classifier.to(gpu)
#
#     for batch_idx, (images, targets) in enumerate(train_loader):
#         images = images.to(gpu)
#         targets = targets.to(gpu)
#
#         optimizer.zero_grad()
#         output = classifier(images)
#         loss = F.nll_loss(output, targets)
#         loss.backward()
#         optimizer.step()
#
#         # TODO add progress indication here
#
#
# test_losses = []
# test_counter = []
#
#
# def test(classifier, epoch):
#
#     # Set to evaluation mode
#     classifier.eval()
#     classifier = classifier.to(gpu)
#
#     test_loss = 0
#     correct = 0
#
#     with torch.no_grad():
#         for images, targets in test_loader:
#             images = images.to(gpu)
#             targets = targets.to(gpu)
#             output = classifier(images)
#             test_loss += F.nll_loss(output, targets, reduction="sum").item()
#             pred = output.data.max(1, keepdim=True)[1]
#             correct += pred.eq(targets.data.view_as(pred)).sum()
#
#     test_loss /= len(test_loader.dataset)
#     test_losses.append(test_loss)
#     test_counter.append(len(train_loader.dataset)*epoch)
#
#     # TODO add progress / results indication here
#
#
# # TODO train and test
