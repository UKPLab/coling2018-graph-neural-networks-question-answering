from keras import backend as K


def keras_cosine(inputs):
    l1 = inputs[0]
    l2 = inputs[1]
    l1_dot = K.batch_dot(l1, l1, (1, 1))
    l2_dot = K.sum(l2 * l2, axis=-1)

    denominator = K.sqrt(l1_dot * l2_dot)
    denominator = K.maximum(denominator, K.epsilon())
    output = K.batch_dot(l1, l2, (1, 2)) / denominator

    return output
