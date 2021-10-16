from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


# Takes a detection feature map and turns it into a 2D tensor,
# where each row of the tensor corresponds to attributes of a bounding box
def predict_transform(prediction, inp_dim, anchors, num_classes):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]

    # Transform output according to the equations discussed in part 1
    # Sigmoid the center_X, center_Y, and object confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add grid offsets to the center coordinates prediction
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1).cuda()
    y_offset = torch.FloatTensor(b).view(-1, 1).cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # Apply the anchors to the dimensions of the bounding box
    anchors = torch.FloatTensor(anchors).cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Apply sigmoid activation to the class scores
    prediction[:, :, 5:(5 + num_classes)] = torch.sigmoid((prediction[:, :, 5:(5 + num_classes)]))

    # Resize the detections map to teh size of the input image
    # The bounding box attributes are sized according to the feature map (say 13 x 13)
    # If the input image is 416 x 416, we multiply the attributes by 32, or the "stride" variable
    prediction[:, :, :4] *= stride

    return prediction
