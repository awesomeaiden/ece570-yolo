from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


# Takes a detection feature map and turns it into a 2D tensor,
# where each row of the tensor corresponds to attributes of a bounding box
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
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

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # Apply the anchors to the dimensions of the bounding box
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Apply sigmoid activation to the class scores
    prediction[:, :, 5:(5 + num_classes)] = torch.sigmoid((prediction[:, :, 5:(5 + num_classes)]))

    # Resize the detections map to the size of the input image
    # The bounding box attributes are sized according to the feature map (say 13 x 13)
    # If the input image is 416 x 416, we multiply the attributes by 32, or the "stride" variable
    prediction[:, :, :4] *= stride

    return prediction


def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


# Returns the IoU of two bounding boxes
def bbox_iou(box1, box2):
    # Get the coordinates of the bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    int_rect_x1 = torch.max(b1_x1, b2_x1)
    int_rect_y1 = torch.max(b1_y1, b2_y1)
    int_rect_x2 = torch.max(b1_x2, b2_x2)
    int_rect_y2 = torch.max(b1_y2, b2_y2)

    # Intersection area
    int_area = torch.clamp(int_rect_x2 - int_rect_x1 + 1, min=0) * torch.clamp(int_rect_y2 - int_rect_y1 + 1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = int_area / (b1_area + b2_area - int_area)

    return iou


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    # Prediction tensor contains information about batch_size x 10647 bounding boxes
    # For each of the bounding boxes having an objectness score below a threshold,
    # we should set the values of its every attribute to zero
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # Perform non-maximum suppression:
    # Transform (center x, center y, height, width) to
    # (top-left corner x, top-left corner y, bottom-right corner x, bottom-right corner y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    # Thresholding must be done one image at a time
    batch_size = prediction.size(0)
    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]

        # Get the maximum class score
        max_conf, max_conf_score = torch.max(image_pred[:, 5:(5 + num_classes)], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # Remove bounding box rows with object confidence under threshold (previously set to 0)
        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
        except:
            continue

        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:, -1])

        # Perform NMS classwise
        for cls in img_classes:
            # Get the detections with one particular class
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            cls_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_cls = image_pred_[cls_mask_ind].view(-1, 7)

            # Sort the detections such that the entry with the maximum
            # objectness confidence is at the top
            conf_sort_ind = torch.sort(image_pred_cls[:, 4], descending=True)[1]
            image_pred_cls = image_pred_cls[conf_sort_ind]
            idx = image_pred_cls.size(0)  # Number of detections

            # Now perform NMS
            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at in the loop
                try:
                    ious = bbox_iou(image_pred_cls[i].unsqueeze(0), image_pred_cls[(i + 1):])
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all the detections that have IoU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_cls[(i + 1):] *= iou_mask

                # Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_cls[:, 4]).squeeze()
                image_pred_cls = image_pred_cls[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_cls.new(image_pred_cls.size(0), 1).fill_(ind)

            # Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_cls

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0


def load_classes(namesfile):
    with open(namesfile, "r") as nf:
        names = nf.read().split("\n")[:-1]
        return names


def letterbox_images(img, inp_dim):
    # Resize image with unchanged aspect ratio using padding
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[((h - new_h) // 2):((h - new_h) // 2 + new_h), ((w - new_w) // 2):((w - new_w) // 2 + new_w), :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    # Prepare image for inputting to the neural network

    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)

    return img
