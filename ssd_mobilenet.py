from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Input, GlobalAveragePooling2D, Reshape, Dropout, Activation, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model

from mobilenetv1 import MobileNet

def SSD_MobileNet(num_classes, box_specs_list):

  def _conv_block(input, filters, kernel_size = (3, 3), strides = (1, 1), block_id = 0):
    x = Conv2D(filters, kernel_size, strides, padding = 'same', use_bias = False, kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4), name = 'conv%s' % block_id)(input)
    x = BatchNormalization(name = 'conv%s_bn' % block_id)(x)
    x = ReLU(6., name = 'conv%s_relu' % block_id)(x) # ReLU6

    return x

  num_classes = num_classes + 1 # add background class

  # load basemodel
  base_model = MobileNet(num_classes = num_classes)

  # extend basemodel
  inputs = base_model.inputs
  outputs = base_model.outputs
  
  x = _conv_block(outputs[1], filters = 256, kernel_size = (1,1), strides = (1,1), block_id = '14_1')
  x = _conv_block(input = x, filters = 512, kernel_size = (3,3), strides = (2,2), block_id = '14_2') # 5x5
  outputs.append(x)

  x = _conv_block(input = x, filters = 128, kernel_size = (1,1), strides = (1,1), block_id = '15_1')
  x = _conv_block(input = x, filters = 256, kernel_size = (3,3), strides = (2,2), block_id = '15_2') # 3x3
  outputs.append(x)

  x = _conv_block(input = x, filters = 128, kernel_size = (1,1), strides = (1,1), block_id = '16_1')
  x = _conv_block(input = x, filters = 256, kernel_size = (3,3), strides = (2,2), block_id = '16_2') # 2x2
  outputs.append(x)

  x = _conv_block(input = x, filters =  64, kernel_size = (1,1), strides = (1,1), block_id = '17_1')
  x = _conv_block(input = x, filters = 128, kernel_size = (3,3), strides = (2,2), block_id = '17_2') # 1x1
  outputs.append(x)

  # add bbox predictor
  loc_total, cls_total = [], []

  for index, output in enumerate(outputs):
    num_df = len(box_specs_list[index]) # number default boxes

    # class
    cls = Conv2D(filters = num_df * num_classes, kernel_size = (1,1), padding = 'same', 
                 kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4), name = 'cls_predictor_%s' %index)(output)
    cls = Reshape([-1, num_classes], name = 'reshape_cls_%s' %index)(cls)
    cls_total.append(cls)

    # localisation
    loc = Conv2D(filters = num_df * 4, kernel_size = (1,1), padding = 'same',
                 kernel_initializer = 'he_normal', kernel_regularizer = l2(1e-4), name = 'loc_predictor_%s' %index)(output)
    loc = Reshape([-1, 4], name = 'reshape_loc_%s' %index)(loc)
    loc_total.append(loc)

  cls_total = Concatenate(axis = 1, name = 'concat_cls')(cls_total)
  loc_total = Concatenate(axis = 1, name = 'concat_loc')(loc_total)

  predictions = Concatenate(axis = 2, name = 'concat_cls_loc')([cls_total, loc_total])

  model = Model(inputs = inputs, outputs = predictions)

  return model