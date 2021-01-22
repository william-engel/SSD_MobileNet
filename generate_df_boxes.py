import numpy as np

'''Lessons Learned:
A given 2x2 feature layer:
[x1y1 x2y1
 x1y2 x2y2]
Will get flatten by keras.layers.Flatten() to
x1y1 x2y1 x1y2 x2y2
This mean or x and y coordinates must follow this pattern so
x1 x2 x1 x2 ...
y1 y1 y2 y2 ...
yy, xx = np.where(np.zeros(shape = (m,n)) == 0) will return this pattern, where as xx, yy = np.where(np.zeros(shape = (m,n)) == 0) will NOT!
Next we have to use np.repeat for the x and y coordinates
And np.tile for the width and height 
[1 2 3] -> tile:   [1 2 3 1 2 3]
        -> repeat: [1 1 2 2 3 3]
Example: 2 Box shapes (w1, h1), (w2, h2), and a 2x2 matrix
x1 y1 w1 h1
x1 y1 w2 h2
x2 y1 w1 h1
x2 y1 w2 h2
x1 y1 w1 h1
x1 y1 w2 h2
x2 y1 w1 h1
x2 y1 w2 h2
thats the pattern we want!

min_scale: 0.2
max_scale: 0.95
aspect_ratios: 1.0
aspect_ratios: 2.0
aspect_ratios: 0.5
aspect_ratios: 3.0
aspect_ratios: 0.3333
'''
def get_scale_factors(smin = 0.2, smax = 0.95, K = 6):
  ''' Calculate scale factor sk and additional scale factor sk_ for the k-th feature layer'''
  k = np.arange(start = 1, stop = K + 2)
  sk  = smin + (smax - smin) / (K - 1) * (k - 1) 
  return sk

def get_df_boxes(feature_layers, scale_factors = None):

  if scale_factors is None:
    scale_factors = get_scale_factors()

  df_boxes = np.array([]).reshape(0,4)
  for k, layer in enumerate(feature_layers):

    m,n = layer['shape']
    ar  = layer['aspect_ratios']
    num_ar = len(ar) + 1

    # scale factor
    sk  = scale_factors[k] 
    sk_ = np.sqrt(scale_factors[k]  * scale_factors[k+1]) # additional scale factor

    # width
    wk = sk * np.sqrt(ar)
    wk = np.append(wk, sk_) # sk_ ⋅ √1 = sk_
    wk = np.tile(wk, m*n) 

    # height
    hk = sk / np.sqrt(ar)
    hk = np.append(hk, sk_) # sk_ ⋅ √1 = sk_
    hk = np.tile(hk, m*n) 

    # center coordinate
    yy, xx = np.where(np.zeros(shape = (m,n)) == 0)
    cx = [(i + 0.5) / m for i in xx]
    cy = [(i + 0.5) / n for i in yy]
    cx = np.repeat(cx, num_ar, axis = 0)
    cy = np.repeat(cy, num_ar, axis = 0)

    df_boxes = np.vstack((df_boxes, np.array([cx, cy, wk, hk]).T))
  
  return df_boxes.astype('float32')

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