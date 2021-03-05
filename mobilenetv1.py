from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Input, GlobalAveragePooling2D, Reshape, Dropout, Activation
from tensorflow.keras import Model


def MobileNet(input_shape = (300,300,3), num_classes = 1000, include_top = False, num_feature_layers = 2):

  def _conv_block(input, filters, kernel_size = (3, 3), strides = (1, 1), block_id = 0):
    x = Conv2D(filters, kernel_size, strides, padding = 'same', use_bias = False, name = 'conv%s' % block_id)(input)
    x = BatchNormalization(name = 'conv%s_bn' % block_id)(x)
    x = ReLU(6., name = 'conv%s_relu' % block_id)(x) # ReLU6

    return x

  def _depthwise_conv_block(input, filters, strides, block_id):
    x = DepthwiseConv2D(kernel_size = (3,3), strides = strides, padding = 'same', use_bias = False, name = 'conv_dw_%d' % block_id)(input)
    x = BatchNormalization(name = 'conv_dw_%d_bn' % block_id)(x)
    x = ReLU(6., name = 'conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(filters, kernel_size = (1,1), strides = (1,1), padding = 'same', use_bias = False, name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

    return x

  output = []
  input = Input(shape = input_shape)
  
  x = _conv_block(input, filters = 32, strides = (2,2), block_id = 1)

  x = _depthwise_conv_block(input = x, filters =  64, strides = (1,1), block_id = 1)

  x = _depthwise_conv_block(input = x, filters = 128, strides = (2,2), block_id = 2)
  x = _depthwise_conv_block(input = x, filters = 128, strides = (1,1), block_id = 3)

  x = _depthwise_conv_block(input = x, filters = 256, strides = (2,2), block_id = 4)
  x = _depthwise_conv_block(input = x, filters = 256, strides = (1,1), block_id = 5) # 36x36
  if num_feature_layers >= 3: output.append(x)

  x = _depthwise_conv_block(input = x, filters = 512, strides = (2,2), block_id = 6)
  x = _depthwise_conv_block(input = x, filters = 512, strides = (1,1), block_id = 7)
  x = _depthwise_conv_block(input = x, filters = 512, strides = (1,1), block_id = 8)
  x = _depthwise_conv_block(input = x, filters = 512, strides = (1,1), block_id = 9)
  x = _depthwise_conv_block(input = x, filters = 512, strides = (1,1), block_id = 10) 
  x = _depthwise_conv_block(input = x, filters = 512, strides = (1,1), block_id = 11)  # 19x19
  if num_feature_layers >= 2: output.append(x)

  x = _depthwise_conv_block(input = x, filters = 1024, strides = (2,2), block_id = 12)
  x = _depthwise_conv_block(input = x, filters = 1024, strides = (1,1), block_id = 13) # 10x10
  if num_feature_layers >= 1: output.append(x)

  if include_top:

    x = GlobalAveragePooling2D()(x)
    x = Reshape(target_shape = (1, 1, 1024), name='reshape_1')(x)
    x = Dropout(rate = 1e-3, name='dropout')(x)
    x = Conv2D(num_classes, kernel_size = (1, 1), padding='same', name='conv_preds')(x)
    x = Reshape(target_shape = (num_classes,), name='reshape_2')(x)
    x = Activation(activation = 'softmax',name = 'predictions')(x)
    output.append(x)
                          
  model = Model(inputs = [input], outputs = output, name='mobilenet_%s' % input_shape[0])

  return model