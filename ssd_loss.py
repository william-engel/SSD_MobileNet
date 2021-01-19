import tensorflow as tf


def SSD_Loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    '''This depends on how we concatenate out model output [loc, cls] or [cls, loc]'''
    pred_boxes = y_pred[:, :, -4:]
    true_boxes = y_true[:, :, -4:]

    pred_classes = y_pred[:, :, :-4]
    true_classes = y_true[:, :, :-4]
    
    # check for each df_box if it has a gt match [0., 1., 0., 1., 0., 0., ...]
    pos_mask = tf.cast(tf.not_equal(tf.argmax(true_classes, axis = -1), 0), tf.float32)
    # umber of matches
    num_pos = tf.maximum(1.0, tf.cast(tf.math.count_nonzero(pos_mask, axis= -1), tf.float32))

    # loc loss for all boxes [x, y, w, h]
    loc_loss = tf.compat.v1.losses.huber_loss(labels = true_boxes,
                                              predictions = pred_boxes,
                                              reduction = 'none')
    # sum x+y+w+h
    loc_loss = tf.reduce_sum(loc_loss, axis=-1)
    # just count loss if df_box has a match
    loc_loss = tf.where(tf.equal(pos_mask, 1.0), loc_loss, 0.0) 
    # sum up loss for eatch match
    loc_loss = tf.reduce_sum(loc_loss, axis=-1)
    # calculate mean loss
    loc_loss = loc_loss / num_pos

    '''This depends on if we have applied a Softmax activation in our model, in our case we did not'''
    cce = tf.losses.CategoricalCrossentropy(from_logits=True,
                                            reduction=tf.losses.Reduction.NONE)
    
    cross_entropy = cce(true_classes, pred_classes)
    
    #neg:pos 3:1
    num_neg = 3.0 * num_pos # num negative examples

    #Negative Mining
    # if pos_mask == 0 then cross_entropy, else 0.0
    neg_cross_entropy = tf.where(tf.equal(pos_mask, 0.0), cross_entropy, 0.0)
    # sort from most worst prediction to good prediction, returns index [223, 1, 22, 2,...]
    sorted_dfidx=tf.cast(tf.argsort(neg_cross_entropy,\
                            direction='DESCENDING',axis=-1),tf.int32)
    rank = tf.cast(tf.argsort(sorted_dfidx, axis=-1), tf.int32) # position of each cross_entropy [2, 4, ..] for the above example because 1 is at idx 2 and 2 at idx 4
    num_neg = tf.cast(num_neg, dtype=tf.int32)
    neg_loss = tf.where(rank < tf.expand_dims(num_neg, axis=1),
                        neg_cross_entropy, 0.0) # where pos index < num_neg: neg_cross entropy, else 0.0

    pos_loss = tf.where(tf.equal(pos_mask, 1.0), cross_entropy, 0.0)

    pos_loss = tf.reduce_sum(pos_loss, axis = -1)
    neg_loss = tf.reduce_sum(neg_loss, axis = -1)

    clas_loss = tf.reduce_sum(pos_loss + neg_loss, axis=-1) # sum pos # neg loss
    clas_loss = clas_loss / num_pos # calculate mean 

    totalloss = loc_loss + clas_loss
    return totalloss