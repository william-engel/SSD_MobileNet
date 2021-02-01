import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np 
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore', category = np.VisibleDeprecationWarning)

tf.get_logger().setLevel('INFO')

def readLabels(label_path):
  '''
  Args:
    label_path: Path to raw kitti label (e.g. '...\000000.txt')
                The raw label contains one or more lines. Each line defines a ground truth bounding box.
                  e.g.
                  Car 0.00 0 1.64 542.05 175.55 565.27 193.79 1.46 1.66 4.05 -4.71 1.71 60.52 1.56
                  Cyclist 0.00 0 1.89 330.60 176.09 355.61 213.60 1.72 0.50 1.95 -12.63 1.88 34.09 1.54
  Returns:
    label: An array containing structured arrays. Where each structured array represents a ground truth bounding box.
  '''
  label = np.genfromtxt(label_path, delimiter=' ', dtype=None,
                        names=('type',
                               'truncation',
                               'occlusion',
                               'alpha',
                               'x1',
                               'y1',
                               'x2',
                               'y2',
                               'box_width',
                               'box_height',
                               'box_length',
                               'location_x',
                               'location_y',
                               'location_z',
                               'yaw_angle'))
  
  if label.ndim == 0: label = np.expand_dims(label, axis = 0)

  return label

def preprocess_label(labels, label2id, original_size):
  '''
  Args:
    labels: An array containing structured arrays. Where each structured array represents a ground truth bounding box.
    label2id: A dictionary mapping labels to integer id's. e.g. {'car': 1, 'cyclist': 2, ....}
    original_size: Since the bounding box coordinates are given in absolute coordinates we need the original image size to normalize the coordinates.
  Returns:
    gt_boxes: An array containing the normalized box coordinates and the class id. [[x1,y1,x2,y2,c]]
  '''
  h, w = original_size

  gt_boxes = np.array([]).reshape(0,5)
  for label in labels:
    if label['type'] != b'DontCare':
      x1 = label['x1'] / w                          
      y1 = label['y1'] / h                              
      x2 = label['x2'] / w                         
      y2 = label['y2'] / h                      
      c = label2id[label['type'].decode('utf-8')] # class

      gt_boxes = np.vstack((gt_boxes, np.array([x1, y1, x2, y2, c]).T))
      gt_boxes = tf.convert_to_tensor(gt_boxes, dtype = tf.float32)

  return gt_boxes

def get_box_specs_list(aspect_ratios = None, scale_factors = None, smin = 0.2, smax = 0.95, K = 6):
  '''
  Args:
    aspect_ratios: list containing the bounding box aspect ratios
    scale_factors: list containing the bounding box scale factors
  Returns:
    box_specs_list: List of list of tuples. Each tuple defines the scale factor and aspect ratio of a bbbox (scale factor, aspect ratio).
                    Each list entry definies all bbox for one feature layer.
  '''
  if aspect_ratios is None: aspect_ratios = [1, 2, 3, 1/2, 1/3] # default
  if scale_factors is None: scale_factors = get_scale_factors(smin, smax, K)

  box_specs_list = []

  for k in range(K):
    sk  = scale_factors[k]
    sk_ = np.sqrt(scale_factors[k] * scale_factors[k+1]) # additional scale factor

    if k == 0:
      box_specs = [(0.1, 1.0), (sk, 2.0), (sk, 0.5)]
    else:
      box_specs = [(sk, ak) for ak in aspect_ratios]
      box_specs.append((sk_, 1.0))

    box_specs_list.append(box_specs)

  return box_specs_list

  
def get_scale_factors(smin = 0.2, smax = 0.95, K = 6):
  '''
  Calculates a scale factor for each of our K feature layers. Plus an additional scale factor. Resulting in an array of length K+1.
  The scale factors are eavenly spread between smin and smax
  '''
  k = np.arange(start = 1, stop = K + 2)
  sk  = smin + (smax - smin) / (K - 1) * (k - 1) 
  return sk

def get_df_boxes(feature_map_shape_list, box_specs_list):
  '''
  Args:
    feature_map_shape_list: List of tuples, defining the size for each feature layer. e.g. [(19,19), (10,10), ...]
    box_specs_list: list of tumples (scale_factor, aspect_ratio) with same length as feature_map_shape_list e.g. [[(0.2, 1.0), (0.2, 2.0), (0.2, 0.5)], ...]
  Returns:
    df_boxes: List of defualt bbox coordinates (anchors). [[x1,y1,x2,y2],...]
  '''
  df_boxes = np.array([]).reshape(0,4)
  for feature_map_shape, box_specs in zip(feature_map_shape_list, box_specs_list):

    m,n = feature_map_shape
    num_ar = len(box_specs)

    # width
    wk = [box_spec[0] *  np.sqrt(box_spec[1]) for box_spec in box_specs] # sk ⋅ √ak
    wk = np.tile(wk, m*n)

    # height
    hk = [box_spec[0] /  np.sqrt(box_spec[1]) for box_spec in box_specs] # sk / √ak
    hk = np.tile(hk, m*n) 

    # center coordinate
    yy, xx = np.where(np.zeros(shape = (m,n)) == 0)
    cx = [(i + 0.5) / m for i in xx]
    cy = [(i + 0.5) / n for i in yy]
    cx = np.repeat(cx, num_ar, axis = 0)
    cy = np.repeat(cy, num_ar, axis = 0)

    df_boxes = np.vstack((df_boxes, np.array([cx, cy, wk, hk]).T))
  
  return df_boxes.astype('float32')

def decode(tfmatrix, df_boxes, variance):
  '''
  SSD does not predict the bbox coordinates directly but instead predicts an offset from the default bboxs (anchors). 
  So our model predicts a transformation (tfmatrix) and we have to aplly it on our default boxes (df_boxes) to get our absolute coordinates.
  There are two common representations of the variance. The first one is [5.0, 5.0, 10.0, 10.0]. For decoding we have to divide by this value and for encoding we have to multiply.
  The second one is [0.2, 0.2, 0.1, 0.1]. For decoding we have to multiply with this value and for encoding we have to divide by.
  df_boxes: [[x,y,w,h]]
  '''

  tfmatrix = tf.cast(tfmatrix,dtype=tf.float32) / variance
  df_boxes = tf.cast(df_boxes,dtype=tf.float32)

  tcx, tcy, tw, th = tf.split(tfmatrix, num_or_size_splits = 4, axis = 1)
  dcx, dcy, dw, dh = tf.split(df_boxes, num_or_size_splits = 4, axis = 1)

  gcx = tcx * dw + dcx
  gcy = tcy * dh + dcy
  gw  = tf.math.exp(tw) * dw
  gh  = tf.math.exp(th) * dh

  return tf.concat([gcx, gcy, gw, gh], axis = 1) 
  

def encode(df_boxes, gt_boxes, variance):
  '''
  SSD does not predict the bbox coordinates directly but instead predicts an offset from the default bboxs (anchors). 
  So we calculate the transformation from our default bboxs to our ground truth bboxs. And the model is going to learn this transformation.
  df_boxes: [[x,y,w,h]]
  gt_boxes: [[x,y,w,h]]
  '''

  gt_boxes = tf.cast(gt_boxes,dtype=tf.float32)
  df_boxes = tf.cast(df_boxes,dtype=tf.float32)

  gcx, gcy, gw, gh = tf.split(gt_boxes, num_or_size_splits = 4, axis = 1)
  dcx, dcy, dw, dh = tf.split(df_boxes, num_or_size_splits = 4, axis = 1)

  tcx = (gcx - dcx) / dw
  tcy = (gcy - dcy) / dh
  tw  = tf.math.log(gw / dw)
  th  = tf.math.log(gh / dh)

  return tf.concat([tcx, tcy, tw, th], axis = 1) * variance


def convert_format(boxes, format):
  if format == 'x1y1x2y2':
    cx, cy, w, h = tf.split(boxes[:,:4], num_or_size_splits = 4, axis = 1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return tf.concat([x1, y1, x2, y2], axis=1)

  elif format == 'xywh':
    x1, y1, x2, y2 = tf.split(boxes[:,:4], num_or_size_splits = 4, axis = 1)
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return tf.concat([x, y, w, h], axis=-1)

def calculate_iou(df_boxes, gt_boxes):

  df_boxes = tf.cast(df_boxes,dtype=tf.float32)
  gt_boxes = tf.cast(gt_boxes,dtype=tf.float32)
  
  x1 = tf.math.maximum(df_boxes[:,None,0],gt_boxes[:,0])
  y1 = tf.math.maximum(df_boxes[:,None,1],gt_boxes[:,1])
  x2 = tf.math.minimum(df_boxes[:,None,2],gt_boxes[:,2])
  y2 = tf.math.minimum(df_boxes[:,None,3],gt_boxes[:,3])
  
  #Intersection area
  intersectionArea = tf.math.maximum(0.0, x2-x1) * tf.math.maximum(0.0, y2-y1)

  #Union area
  df_Area = (df_boxes[:,2]-df_boxes[:,0])*(df_boxes[:,3]-df_boxes[:,1])
  gt_Area = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])

  unionArea = tf.math.maximum(1e-10, df_Area[:,None] + gt_Area - intersectionArea)

  iou = intersectionArea/unionArea
  return tf.clip_by_value(iou,0.0,1.0)

def match_gt2df(gt_boxes, iou_matrix, iou_threshold):

  max_idx = tf.math.argmax(iou_matrix,axis=1) # gt_box index for each df_box
  gt_boxes = tf.gather(gt_boxes,max_idx)      # match gt_box to each df_box

  # for each df_box, check if gt_box exits with iou >= 0.5
  # return: [True False False True ...]
  max_values = tf.reduce_max(iou_matrix,axis=1) 
  matched = tf.cast(tf.math.greater_equal(max_values, iou_threshold), dtype=tf.float32) 
  
  return gt_boxes, matched


def convert_scale(matrix,scale, wImage, hImage):
  if scale == 'abs':
    return tf.stack([matrix[:,0]*wImage,
    matrix[:,1]*hImage,
    matrix[:,2]*wImage,
    matrix[:,3]*hImage],axis=-1)

  elif scale == 'rel':
    return tf.stack([matrix[:,0]/wImage,
    matrix[:,1]/hImage,
    matrix[:,2]/wImage,
    matrix[:,3]/hImage],axis=-1) 


def load_image(image_path, target_size):
  if tf.is_tensor(image_path): image_path = image_path.numpy().decode('utf-8')
  image = tf.keras.preprocessing.image.load_img(image_path, target_size = target_size, interpolation = 'bilinear')
  image = np.array(image)
  return tf.cast(image, tf.float32)

def load_label(label_path, original_size, label2id):
  if tf.is_tensor(label_path): label_path = label_path.numpy().decode('utf-8')
  label = readLabels(label_path)
  label = preprocess_label(label, label2id, original_size)
  return tf.convert_to_tensor(label)


def preprocess_data(image_path, label_path, target_size, original_size, df_boxes, iou_threshold, label2id):

  # load image
  [image,] = tf.py_function(func = lambda x: load_image(x, target_size),
                            inp = [image_path],
                            Tout = [tf.float32])
  image.set_shape(target_size + (3,))

  # load label
  [gt,] = tf.py_function(func = lambda x: load_label(x, original_size, label2id),
                         inp = [label_path],
                         Tout = [tf.float32])
  gt.set_shape((None, 5)) # [[x1,y1,x2,y2,c]]

  gt_boxes   = gt[:,:4]
  gt_classes = gt[:, 4]

  # calculate iou between df_boxes and gt_boxes
  iou_matrix = calculate_iou(df_boxes, gt_boxes)

  # to each df_box match the gt_box with the highest iou
  matched_gt, matched = match_gt2df(gt, iou_matrix, iou_threshold)

  # matched_gt_classes
  matched_gt_classes = matched_gt[:,4]
  matched_gt_classes = tf.multiply(matched_gt_classes, matched) # if iou < 0.5 set class = 0 (Background)
  matched_gt_classes = tf.one_hot(tf.cast(matched_gt_classes, tf.int32), depth = 10) # one hot

  # calculate transformation between df_boxes and mached_gt_boxes
  variance = [5.0, 5.0, 10.0, 10.0] # https://github.com/weiliu89/caffe/issues/155
  matched_gt_boxes = matched_gt[:,:4]
  tfmatrix = encode(convert_format(df_boxes,         'xywh'),
                    convert_format(matched_gt_boxes, 'xywh'), 
                    variance)

  '''This depends on how we concatenate out model output [loc, cls] or [cls, loc]'''
  label = tf.concat([matched_gt_classes, tfmatrix], axis = 1)

  return image, label 

 