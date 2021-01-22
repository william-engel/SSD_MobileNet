import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np 
import tensorflow as tf

tf.get_logger().setLevel('INFO')

def apply_tfmatrix(tfmatrix, df_boxes):

  variance = [5.0, 5.0, 10.0, 10.0] # https://github.com/weiliu89/caffe/issues/155

  tfmatrix = tf.cast(tfmatrix,dtype=tf.float32) * variance
  df_boxes = tf.cast(df_boxes,dtype=tf.float32)

  tcx, tcy, tw, th = tf.split(tfmatrix, num_or_size_splits = 4, axis = 1)
  dcx, dcy, dw, dh = tf.split(df_boxes, num_or_size_splits = 4, axis = 1)

  gcx = tcx * dw + dcx
  gcy = tcy * dh + dcy
  gw  = tf.math.exp(tw) * dw
  gh  = tf.math.exp(th) * dh

  return tf.concat([gcx, gcy, gw, gh], axis = 1) 
  

def get_tfmatrix_df2gt(df_boxes, gt_boxes):

  variance = [5.0, 5.0, 10.0, 10.0] # https://github.com/weiliu89/caffe/issues/155
  
  gt_boxes = tf.cast(gt_boxes,dtype=tf.float32)
  df_boxes = tf.cast(df_boxes,dtype=tf.float32)

  gcx, gcy, gw, gh = tf.split(gt_boxes, num_or_size_splits = 4, axis = 1)
  dcx, dcy, dw, dh = tf.split(df_boxes, num_or_size_splits = 4, axis = 1)

  tcx = (gcx - dcx) / dw
  tcy = (gcy - dcy) / dh
  tw  = tf.math.log(gw / dw)
  th  = tf.math.log(gh / dh)

  return tf.concat([tcx, tcy, tw, th], axis = 1) / variance

def preprocess_label(labels, label2id, original_size):
  ''' takes the raw label and returns a tensor containing the normalized bbox coordinates + class id'''
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

  return tf.convert_to_tensor(gt_boxes, dtype = tf.float32)

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

  num_gt_boxes = tf.shape(gt_boxes)[0]
  max_values = tf.reduce_max(iou_matrix, axis=0)
  matched = tf.cast(tf.math.greater_equal(max_values, iou_threshold), dtype=tf.float32) 
  num_matches = tf.math.count_nonzero(matched)

  max_idx = tf.math.argmax(iou_matrix,axis=1) # gt_box index for each df_box
  gt_boxes = tf.gather(gt_boxes,max_idx)      # match gt_box to each df_box

  # for each df_box, check if gt_box exits with iou >= 0.5
  # return: [True False False True ...]
  max_values = tf.reduce_max(iou_matrix,axis=1) 
  matched = tf.cast(tf.math.greater_equal(max_values, iou_threshold), dtype=tf.float32) 
  
  # tf.py_function(func = lambda x, y: print('%d of %d matches' % (x.numpy(), y.numpy())),
  #                inp = [num_matches, num_gt_boxes],
  #                Tout = [])

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

def readLabels(label_path):

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


def load_image(image_path, target_size):
  if isinstance(image_path, type(tf.constant(0))): image_path = image_path.numpy().decode('utf-8')
  image = tf.keras.preprocessing.image.load_img(image_path, target_size = target_size)
  image = np.array(image)
  return tf.cast(image, tf.float32)

def load_label(label_path, original_size, label2id):
  if isinstance(label_path, type(tf.constant(0))): label_path = label_path.numpy().decode('utf-8')
  label = readLabels(label_path)
  label = preprocess_label(label, label2id, original_size)
  return tf.convert_to_tensor(label)


def preprocess_data(image_path, label_path, target_size, original_size, df_boxes, iou_threshold, label2id):

  # load image
  [image,] = tf.py_function(func = lambda x: load_image(x, target_size),
                            inp = [image_path],
                            Tout = [tf.float32])
  image.set_shape(target_size + (3,))

  image = tf.keras.applications.mobilenet.preprocess_input(image) # The inputs pixel values are scaled between -1 and 1, sample-wise. 

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
  matched_gt_boxes = matched_gt[:,:4]
  tfmatrix = get_tfmatrix_df2gt(convert_format(df_boxes,         'xywh'),
                                convert_format(matched_gt_boxes, 'xywh'))

  '''This depends on how we concatenate out model output [loc, cls] or [cls, loc]'''
  label = tf.concat([matched_gt_classes, tfmatrix], axis = 1)

  return image, label 