from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *
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
        filtered_lines = [x.rstrip().lstrip() for x in lines if (len(x) > 0 and x[0] != "#" and x[0] != ";")]

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


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")

    # Resize to input dimension
    img = cv2.resize(img, (416, 416))

    # BGR -> RGB | H x W x C -> C x H x W
    rgb_img = img[:, :, ::-1].transpose((2, 0, 1))

    # Add a channel at 0 (for batch) | Normalize
    rgb_img = rgb_img[np.newaxis, :, :, :] / 255.0

    # Convert to float
    rgb_img = torch.from_numpy(rgb_img).float()

    # Convert to variable
    rgb_img = Variable(rgb_img)

    return rgb_img


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


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
                pad = (kernel_size - 1) // 2
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
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            module.add_module("upsample_" + str(index), upsample)

        # Otherwise if it is a route layer
        elif x["type"] == "route":
            x["layers"] = x["layers"].split(",")

            # Start of a route
            start = int(x["layers"][0])

            # End of a route, if it exists
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            # Positive annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            # Use empty layer here, as we will perform routing directly in forward function of the overall network
            route = EmptyLayer()
            module.add_module("route_" + str(index), route)

            # Update filters to hold the number of filters outputted by the route layer
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        # Otherwise if it is a shortcut layer
        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_" + str(index), shortcut)

        # Otherwise if it is a detection layer
        elif x["type"] == "yolo":
            mask = [int(x) for x in x["mask"].split(",")]

            anchors = [int(a) for a in x["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_" + str(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


class Network(nn.Module):
    def __init__(self, cfgfile):
        super(Network, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = dict()

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = [int(a) for a in module["layers"]]

                if layers[0] > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors

                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                # If no collector has been initialized
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections


    def load_weights(self, weightfile):
        with open(weightfile, "rb") as wfile:
            # First 5 values are the header information
            # 1. Major version number
            # 2. Minor version number
            # 3. Subversion number
            # 4 and 5: Images seen by the network in training
            header = np.fromfile(wfile, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]

            # Rest of the bits represent the weights
            weights = np.fromfile(wfile, dtype=np.float32)

            # Iterate through the weights and load into the modules
            weights_ind = 0
            for i in range(len(self.module_list)):
                module_type = self.blocks[i + 1]["type"]

                # If module type is convolutional, load weights (otherwise ignore)
                if module_type == "convolutional":
                    model = self.module_list[i]
                    try:
                        batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                    except:
                        batch_normalize = 0
                    conv = model[0]

                    if (batch_normalize):
                        bn = model[1]

                        # Get the number of weights of the batchnorm layer
                        num_bn_biases = bn.bias.numel()

                        # Load the weights
                        bn_biases = torch.from_numpy(weights[weights_ind:weights_ind + num_bn_biases])
                        weights_ind += num_bn_biases

                        bn_weights = torch.from_numpy(weights[weights_ind:weights_ind + num_bn_biases])
                        weights_ind += num_bn_biases

                        bn_running_mean = torch.from_numpy(weights[weights_ind:weights_ind + num_bn_biases])
                        weights_ind += num_bn_biases

                        bn_running_var = torch.from_numpy(weights[weights_ind:weights_ind + num_bn_biases])
                        weights_ind += num_bn_biases

                        # Cast loaded weights into dims of model weights
                        bn_biases = bn_biases.view_as(bn.bias.data)
                        bn_weights = bn_weights.view_as(bn.weight.data)
                        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                        bn_running_var = bn_running_var.view_as(bn.running_var)

                        # Copy data to the model
                        bn.bias.data.copy_(bn_biases)
                        bn.weight.data.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_var)

                    else:
                        # Number of biases
                        num_biases = conv.bias.numel()

                        # Load the weights
                        conv_biases = torch.from_numpy(weights[weights_ind:weights_ind + num_biases])
                        weights_ind += num_biases

                        # Reshape loaded weights according to dims of models weights
                        conv_biases = conv_biases.view_as(conv.bias.data)

                        # Copy the data
                        conv.bias.data.copy_(conv_biases)

                    # Finally, load the weights for the convolutional layers
                    num_weights = conv.weight.numel()

                    conv_weights = torch.from_numpy(weights[weights_ind:weights_ind + num_weights])
                    weights_ind += num_weights

                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)



# # Verify GPU connectivity
# if not torch.cuda.is_available():
#     raise Exception("CUDA not available!!")
# print("CUDA availability verified")
#
# # Get gpu device
# gpu = torch.device("cuda")
#
# yolo_blocks = parse_cfg("yolo.cfg")
# yolo_modules = create_modules(yolo_blocks)
#
# model = Network("yolo.cfg")
# model.load_weights("yolo.weights")
# inp = get_test_input()
# pred = model(inp, gpu)
#
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
