import tensorflow as tf
import keras.backend as K

### Custom Loss Functions


def hamming_loss_fn(
    y_true,
    y_pred,
    threshold: float = 0.5,
    mode: str = "multilabel",
) -> tf.Tensor:
    """Computes hamming loss.

    Hamming loss is the fraction of wrong labels to the total number
    of labels.

    In multi-class classification, hamming loss is calculated as the
    hamming distance between `y_true` and `y_pred`.
    In multi-label classification, hamming loss penalizes only the
    individual labels.

    Args:
        y_true: actual target value.
        y_pred: predicted target value.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
        mode: multi-class or multi-label.

    Returns:
        hamming loss: float.
    """
    if mode not in ["multiclass", "multilabel"]:
        raise TypeError("mode must be either multiclass or multilabel]")

    if threshold is None:
        threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
        # make sure [0, 0, 0] doesn't become [1, 1, 1]
        # Use abs(x) > eps, instead of x != 0 to check for zero
        y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
    else:
        y_pred = y_pred > threshold

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)

    if mode == "multiclass":
        nonzero = tf.cast(tf.math.count_nonzero(y_true * y_pred, axis=-1), tf.float32)
        return 1.0 - nonzero

    else:
        nonzero = tf.cast(tf.math.count_nonzero(y_true - y_pred, axis=-1), tf.float32)
        return nonzero / y_true.get_shape()[-1]


def neg_hamming_metric(y_true, y_pred):
    return -hamming_loss_fn(y_true, y_pred)


### RMSE Custom Loss Functions


def rmse(y_true, y_pred):
    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Root Mean of the squared error
    return K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1))


def rmse_ol(y_true, y_pred):
    # tf.map_fn(fn=lambda x: x if x<0.5 else 1.0, elems=y_pred)

    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    multiclass_squared_error = K.mean(K.abs(y_true - y_pred), axis=-1)
    label_value_sum = K.sum(y_true, axis=-1) + K.epsilon()

    res = multiclass_squared_error / label_value_sum

    # Root Mean of the normalised squared error
    return K.sqrt(res)


def rmse_op(y_true, y_pred):
    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    multiclass_squared_error = K.mean(K.square(y_true - y_pred), axis=-1)
    prediction_value_sum = K.sum(y_pred, axis=-1) + K.epsilon()

    res = multiclass_squared_error / prediction_value_sum

    # Root Mean of the normalised squared error
    return K.sqrt(res)


def rmse_hmlp(y_true, y_pred):
    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    multiclass_squared_error = K.mean(K.square(y_true - y_pred), axis=-1)
    label_value_sum = K.sum(y_true, axis=-1) + K.epsilon()
    prediction_value_sum = K.sum(y_pred, axis=-1) + K.epsilon()

    label_pred_harmonic_avg = 2 / ((1 / label_value_sum) + (1 / prediction_value_sum))
    res = multiclass_squared_error / label_pred_harmonic_avg

    # Root Mean of the normalised squared error
    return K.sqrt(res)


def rmse_mlp(y_true, y_pred):
    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    multiclass_squared_error = K.mean(K.square(y_true - y_pred), axis=-1)
    label_value_sum = K.sum(y_true, axis=-1) + K.epsilon()
    prediction_value_sum = K.sum(y_pred, axis=-1) + K.epsilon()

    label_pred_avg = (label_value_sum + prediction_value_sum) / 2
    res = multiclass_squared_error / label_pred_avg

    # Root Mean of the normalised squared error
    return K.sqrt(res)


### Binary Cross Entropy Custom Loss Functions


def bce_ol(y_true, y_pred):
    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Get the sum of the true labels
    label_value_sum = K.sum(y_true, axis=-1) + K.epsilon()

    # Compute the ratio of the binary cross entropy and the sum of the true labels
    res = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) / label_value_sum

    return res


def bce_op(y_true, y_pred):
    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Get the sum of the predicted labels (output of the model)
    prediction_value_sum = K.sum(y_pred, axis=-1) + K.epsilon()

    # Compute the ratio of the binary cross entropy and the sum of the predicted labels
    res = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) / prediction_value_sum

    return res


def bce_hmlp(y_true, y_pred):
    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Get the sum of the true labels
    label_value_sum = K.sum(y_true, axis=-1) + K.epsilon()

    # Get the sum of the predicted labels (output of the model)
    prediction_value_sum = K.sum(y_pred, axis=-1) + K.epsilon()

    # Compute the ratio of the binary cross entropy and the harmonic mean of the sum of the true and predicted labels
    label_pred_harmonic_avg = 2 / ((1 / label_value_sum) + (1 / prediction_value_sum))
    res = (
        K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) / label_pred_harmonic_avg
    )

    return res


def bce_mlp(y_true, y_pred):
    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Get the sum of the true labels
    label_value_sum = K.sum(y_true, axis=-1) + K.epsilon()

    # Get the sum of the predicted labels (output of the model)
    prediction_value_sum = K.sum(y_pred, axis=-1) + K.epsilon()

    # Compute the ratio of the binary cross entropy and the mean of the sum of the true and predicted labels
    label_pred_avg = (label_value_sum + prediction_value_sum) / 2
    res = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) / label_pred_avg

    return res


### Binary Focal Cross Entropy Custom Loss Functions


def bfce_ol(y_true, y_pred):
    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Get the sum of the true labels
    label_value_sum = K.sum(y_true, axis=-1) + K.epsilon()

    # Compute the ratio of the binary focal cross entropy and the sum of the true labels
    res = K.mean(K.binary_focal_crossentropy(y_true, y_pred), axis=-1) / label_value_sum

    return res


def bfce_op(y_true, y_pred):
    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Get the sum of the predicted labels (output of the model)
    prediction_value_sum = K.sum(y_pred, axis=-1) + K.epsilon()

    # Compute the ratio of the binary focal cross entropy and the sum of the predicted labels
    res = (
        K.mean(K.binary_focal_crossentropy(y_true, y_pred), axis=-1)
        / prediction_value_sum
    )

    return res


def bfce_hmlp(y_true, y_pred):
    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Get the sum of the true labels
    label_value_sum = K.sum(y_true, axis=-1) + K.epsilon()

    # Get the sum of the predicted labels (output of the model)
    prediction_value_sum = K.sum(y_pred, axis=-1) + K.epsilon()

    # Compute the ratio of the binary focal cross entropy and the harmonic mean of the sum of the true and predicted labels
    label_pred_harmonic_avg = 2 / ((1 / label_value_sum) + (1 / prediction_value_sum))
    res = (
        K.mean(K.binary_focal_crossentropy(y_true, y_pred), axis=-1)
        / label_pred_harmonic_avg
    )

    return res


def bfce_mlp(y_true, y_pred):
    # Fix tensors structure
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Get the sum of the true labels
    label_value_sum = K.sum(y_true, axis=-1) + K.epsilon()

    # Get the sum of the predicted labels (output of the model)
    prediction_value_sum = K.sum(y_pred, axis=-1) + K.epsilon()

    # Compute the ratio of the binary focal cross entropy and the mean of the sum of the true and predicted labels
    label_pred_avg = (label_value_sum + prediction_value_sum) / 2
    res = K.mean(K.binary_focal_crossentropy(y_true, y_pred), axis=-1) / label_pred_avg

    return res
