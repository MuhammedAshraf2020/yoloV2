import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes


def yolo_head(feats , anchors , num_classes):

    # convert anchors to shape 1 , 1 , 1 , len of anchors , 2
    num_anchors = len(anchors)
    anchors_tensor = K.reshape(K.variable(anchors) , [1 , 1 , 1 , num_anchors , 2])
    # conv_dims , width and hight of the grid
    _, conv_height, conv_width, _ = K.int_shape(feats)
    conv_dims = K.variable([conv_width, conv_height])
    # reshape yolo network output to None , grid_width , grid_hight , num of amnchors , num of classes + 5
    feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
    # convert conv_dims after casting it to feats datatype to 1 , 1 , 1 , 2
    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))
    # create grid from (0 , 0 ) to (width , hight)
    conv_index = np.array([_ for _ in np.ndindex(conv_width, conv_height)])
    conv_index = conv_index[:, [1, 0]]  # swap columns for YOLO ordering.
    conv_index = K.variable(conv_index.reshape(1, conv_height, conv_width, 1, 2))

    box_confidence = K.sigmoid(feats[..., 4:5])
    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_class_probs = K.softmax(feats[..., 5:])

    box_xy = (box_xy + conv_index) / conv_dims
    box_wh = box_wh * anchors_tensor / conv_dims        
 
    return box_confidence, box_xy, box_wh, box_class_probs


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    # get the p(class = x given object = true) = p(class = x) * p(object = true)
    box_scores = box_confidence * box_class_probs # 19x19x80

    # box_classes indeces of highest probability  
    box_classes = K.argmax(box_scores, axis = -1)  # 19x19x5x1  (1 class index)
    # box class scores of highest probabilites  
    box_class_scores = K.max(box_scores, axis = -1)  # 19x19x5x1 (1 class score)
    # make filter  of boxes with have scores more than threshold
    filtering_mask = box_class_scores >= threshold
    # choice from box_classes that is exist in our filter 
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
     
    return scores, boxes, classes

def iou(box1, box2):
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width  = xi2 - xi1
    inter_height = yi2 - yi1
    inter_area   = inter_width * inter_height if inter_width > 0 and inter_height > 0 else 0

    box1_area  = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area  = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area

    return iou

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.compat.v1.variables_initializer([max_boxes_tensor])) 
    

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    scores  = K.gather(scores,  nms_indices)
    boxes   = K.gather(boxes,   nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes

def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.65, iou_threshold=.5):

    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    
    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
        
    return scores, boxes, classes

def predict(sess, image_file):

    image, image_data = preprocess_image(image_file, model_image_size = (608, 608))

    out_scores, out_boxes, out_classes = sess.run([scores , boxes, classes], feed_dict={yolo_model.input: image_data,
                                                                                       K.learning_phase(): 0})

    print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    colors = generate_colors(class_names)

    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    image.save(os.path.join("out", image_file), quality=90)

    output_image = plt.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes