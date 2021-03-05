from preprocess_kitti_dataset import convert_format, decode
import tensorflow as tf
import numpy as np

def create_category_index_from_labelmap_dict(label2id):
  categroy_index = {index+1: {'id': id, 'name': label} for index, (label, id) in enumerate(label2id.items())}
  return categroy_index


def postprocess_prediction(predictions, df_boxes, id2label, from_logits = True, flip_coordinates = True):
    '''
    prediction: [[c0, c1, ..., cn, tx, ty, tw, th], ...]
    df_boxes: [x1, y1, x2, y2]
    id2label: {1: 'car', 2: 'truck',...}
    from_logits: check if the model returns the class predictions from logigts or not
    flip_coordinates: we define box coordinates as [x1,y1,x2,y2] but tensorflow object detection api defines [y1,x1,y2,x2]
    '''

    output_dict = {'detection_boxes': [], 
                   'raw_detection_boxes' : [], 
                   'detection_scores' : [], 
                   'raw_detection_scores' : [], 
                   'detection_multiclass_scores' : [],
                   'detection_classes' : [],
                   'num_detections' : []
    }

    max_output_size = 100

    for prediction in predictions:

        # This depends on how we concatenate out model output [loc, cls] or [cls, loc]
        loc = prediction[:,-4:] # bbox encoded
        cls = prediction[:,:-4] # class prediction

        # postprocess classes
        if from_logits: cls = tf.keras.layers.Softmax()(cls) # check if model returns from logits!
        class_id = tf.argmax(cls, axis = -1)
        class_score = tf.reduce_max(cls, axis = -1)
        foreground_indices = tf.where(class_id != 0)[:,0]

        output_dict['raw_detection_scores'] += [class_score]
        output_dict['detection_multiclass_scores'] += [cls]

        # decode bbox
        variance = [5.0, 5.0, 10.0, 10.0]
        pred_boxes = decode(loc, convert_format(df_boxes, 'xywh'), variance)
        pred_boxes = convert_format(pred_boxes, 'x1y1x2y2')
        
        # [x1,y1,x2,y2] → [y1,x1,y2,x2]
        if flip_coordinates:
            pred_boxes = tf.stack([pred_boxes[:,1], pred_boxes[:,0], pred_boxes[:,3], pred_boxes[:,2]], axis = 1) 

        output_dict['raw_detection_boxes'] += [pred_boxes]

        # filter out background class (0)
        filtered_boxes = tf.gather(pred_boxes, foreground_indices)
        filtered_class_id = tf.cast(tf.gather(class_id, foreground_indices), tf.float32)
        filtered_class_score = tf.gather(class_score, foreground_indices)

        # nms
        indices = tf.image.non_max_suppression(filtered_boxes, filtered_class_score, max_output_size, score_threshold=1e-8)

        # filter out supressed boxes
        final_boxes = tf.gather(filtered_boxes, indices)
        final_class_id = tf.gather(filtered_class_id, indices)
        final_class_score = tf.gather(filtered_class_score, indices)

        # pad to same shape
        num_boxes = indices.shape[0]
        pad_width = max_output_size - num_boxes
        
        final_boxes = tf.pad(final_boxes, [[0,pad_width], [0,0]])
        final_class_id = tf.pad(final_class_id, [[0,pad_width]])
        final_class_score = tf.pad(final_class_score, [[0,pad_width]])

        output_dict['detection_boxes'] += [final_boxes]
        output_dict['detection_scores'] += [final_class_score]
        output_dict['detection_classes'] += [final_class_id]
        output_dict['num_detections'] += [num_boxes]

    output_dict = {key: tf.stack(value, axis = 0) for key,value in output_dict.items()}

    return output_dict
