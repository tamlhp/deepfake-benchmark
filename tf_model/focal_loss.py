
"""TensorFlow Keras focal loss implementation."""
#  ____  __    ___   __   __      __     __   ____  ____
# (  __)/  \  / __) / _\ (  )    (  )   /  \ / ___)/ ___)
#  ) _)(  O )( (__ /    \/ (_/\  / (_/\(  O )\___ \\___ \
# (__)  \__/  \___)\_/\_/\____/  \____/ \__/ (____/(____/

from functools import partial
import tensorflow as tf
import numbers


def check_type(obj, base, *, name=None, func=None, allow_none=False,
               default=None, error_message=None):

    if allow_none and obj is None:
        if default is not None:
            return check_type(default, base=base, name=name, func=func,
                              allow_none=False)
        return None

    if isinstance(obj, base):
        if func is None:
            return obj
        elif callable(func):
            return func(obj)
        else:
            raise ValueError('Parameter \'func\' must be callable or None.')

    # Handle wrong type
    if isinstance(base, tuple):
        expect = '(' + ', '.join(cls.__name__ for cls in base) + ')'
    else:
        expect = base.__name__
    actual = type(obj).__name__
    if error_message is None:
        error_message = 'Invalid type'
        if name is not None:
            error_message += f' for parameter \'{name}\''
        error_message += f'. Expected: {expect}. Actual: {actual}.'
    raise TypeError(error_message)


def check_bool(obj, *, name=None, allow_none=False, default=None):

    return check_type(obj, name=name, base=bool, func=bool,
                      allow_none=allow_none, default=default)


def _check_numeric(*, check_func, obj, name, base, func, positive, minimum,
                   maximum, allow_none, default):
    """Helper function for check_float and check_int."""
    obj = check_type(obj, name=name, base=base, func=func,
                     allow_none=allow_none, default=default)

    if obj is None:
        return None

    positive = check_bool(positive, name='positive')
    if positive and obj <= 0:
        if name is None:
            message = 'Parameter must be positive.'
        else:
            message = f'Parameter \'{name}\' must be positive.'
        raise ValueError(message)

    if minimum is not None:
        minimum = check_func(minimum, name='minimum')
        if obj < minimum:
            if name is None:
                message = f'Parameter must be at least {minimum}.'
            else:
                message = f'Parameter \'{name}\' must be at least {minimum}.'
            raise ValueError(message)

    if maximum is not None:
        maximum = check_func(maximum, name='minimum')
        if obj > maximum:
            if name is None:
                message = f'Parameter must be at most {maximum}.'
            else:
                message = f'Parameter \'{name}\' must be at most {maximum}.'
            raise ValueError(message)

    return obj


def check_int(obj, *, name=None, positive=False, minimum=None, maximum=None,
              allow_none=False, default=None):

    return _check_numeric(check_func=check_int, obj=obj, name=name,
                          base=numbers.Integral, func=int, positive=positive,
                          minimum=minimum, maximum=maximum,
                          allow_none=allow_none, default=default)


def check_float(obj, *, name=None, positive=False, minimum=None, maximum=None,
                allow_none=False, default=None):

    return _check_numeric(check_func=check_float, obj=obj, name=name,
                          base=numbers.Real, func=float, positive=positive,
                          minimum=minimum, maximum=maximum,
                          allow_none=allow_none, default=default)
_EPSILON = tf.keras.backend.epsilon()
def register_keras_custom_object(cls):

    tf.keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls

def binary_focal_loss(y_true, y_pred, gamma, *, pos_weight=None,
                      from_logits=False, label_smoothing=None):
    # Validate arguments
    gamma = check_float(gamma, name='gamma', minimum=0)
    pos_weight = check_float(pos_weight, name='pos_weight', minimum=0,
                             allow_none=True)
    from_logits = check_bool(from_logits, name='from_logits')
    label_smoothing = check_float(label_smoothing, name='label_smoothing',
                                  minimum=0, maximum=1, allow_none=True)

    # Ensure predictions are a floating point tensor; converting labels to a
    # tensor will be done in the helper functions
    y_pred = tf.convert_to_tensor(y_pred)
    if not y_pred.dtype.is_floating:
        y_pred = tf.dtypes.cast(y_pred, dtype=tf.float32)

    # Delegate per-example loss computation to helpers depending on whether
    # predictions are logits or probabilities
    if from_logits:
        return _binary_focal_loss_from_logits(labels=y_true, logits=y_pred,
                                              gamma=gamma,
                                              pos_weight=pos_weight,
                                              label_smoothing=label_smoothing)
    else:
        return _binary_focal_loss_from_probs(labels=y_true, p=y_pred,
                                             gamma=gamma, pos_weight=pos_weight,
                                             label_smoothing=label_smoothing)


@register_keras_custom_object
class BinaryFocalLoss(tf.keras.losses.Loss):

    def __init__(self, gamma, *, pos_weight=None, from_logits=False,
                 label_smoothing=None, **kwargs):
        # Validate arguments
        gamma = check_float(gamma, name='gamma', minimum=0)
        pos_weight = check_float(pos_weight, name='pos_weight', minimum=0,
                                 allow_none=True)
        from_logits = check_bool(from_logits, name='from_logits')
        label_smoothing = check_float(label_smoothing, name='label_smoothing',
                                      minimum=0, maximum=1, allow_none=True)

        super().__init__(**kwargs)
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def get_config(self):

        config = super().get_config()
        config.update(gamma=self.gamma, pos_weight=self.pos_weight,
                      from_logits=self.from_logits,
                      label_smoothing=self.label_smoothing)
        return config

    def call(self, y_true, y_pred):

        return binary_focal_loss(y_true=y_true, y_pred=y_pred, gamma=self.gamma,
                                 pos_weight=self.pos_weight,
                                 from_logits=self.from_logits,
                                 label_smoothing=self.label_smoothing)


# Helper functions below
def _process_labels(labels, label_smoothing, dtype):

    labels = tf.dtypes.cast(labels, dtype=dtype)
    if label_smoothing is not None:
        labels = (1 - label_smoothing) * labels + label_smoothing * 0.5
    return labels


def _binary_focal_loss_from_logits(labels, logits, gamma, pos_weight,
                                   label_smoothing):

    labels = _process_labels(labels=labels, label_smoothing=label_smoothing,
                             dtype=logits.dtype)

    # Compute probabilities for the positive class
    p = tf.math.sigmoid(logits)

    # Without label smoothing we can use TensorFlow's built-in per-example cross
    # entropy loss functions and multiply the result by the modulating factor.
    # Otherwise, we compute the focal loss ourselves using a numerically stable
    # formula below
    if label_smoothing is None:
        # The labels and logits tensors' shapes need to be the same for the
        # built-in cross-entropy functions. Since we want to allow broadcasting,
        # we do some checks on the shapes and possibly broadcast explicitly
        # Note: tensor.shape returns a tf.TensorShape, whereas tf.shape(tensor)
        # returns an int tf.Tensor; this is why both are used below
        labels_shape = labels.shape
        logits_shape = logits.shape
        if not labels_shape.is_fully_defined() or labels_shape != logits_shape:
            labels_shape = tf.shape(labels)
            logits_shape = tf.shape(logits)
            shape = tf.broadcast_dynamic_shape(labels_shape, logits_shape)
            labels = tf.broadcast_to(labels, shape)
            logits = tf.broadcast_to(logits, shape)
        if pos_weight is None:
            loss_func = tf.nn.sigmoid_cross_entropy_with_logits
        else:
            loss_func = partial(tf.nn.weighted_cross_entropy_with_logits,
                                pos_weight=pos_weight)
        loss = loss_func(labels=labels, logits=logits)
        modulation_pos = (1 - p) ** gamma
        modulation_neg = p ** gamma
        mask = tf.dtypes.cast(labels, dtype=tf.bool)
        modulation = tf.where(mask, modulation_pos, modulation_neg)
        return modulation * loss

    # Terms for the positive and negative class components of the loss
    pos_term = labels * ((1 - p) ** gamma)
    neg_term = (1 - labels) * (p ** gamma)

    # Term involving the log and ReLU
    log_weight = pos_term
    if pos_weight is not None:
        log_weight *= pos_weight
    log_weight += neg_term
    log_term = tf.math.log1p(tf.math.exp(-tf.math.abs(logits)))
    log_term += tf.nn.relu(-logits)
    log_term *= log_weight

    # Combine all the terms into the loss
    loss = neg_term * logits + log_term
    return loss


def _binary_focal_loss_from_probs(labels, p, gamma, pos_weight,
                                  label_smoothing):
    # Predicted probabilities for the negative class
    q = 1 - p

    # For numerical stability (so we don't inadvertently take the log of 0)
    p = tf.math.maximum(p, _EPSILON)
    q = tf.math.maximum(q, _EPSILON)

    # Loss for the positive examples
    pos_loss = -(q ** gamma) * tf.math.log(p)
    if pos_weight is not None:
        pos_loss *= pos_weight

    # Loss for the negative examples
    neg_loss = -(p ** gamma) * tf.math.log(q)

    # Combine loss terms
    if label_smoothing is None:
        labels = tf.dtypes.cast(labels, dtype=tf.bool)
        loss = tf.where(labels, pos_loss, neg_loss)
    else:
        labels = _process_labels(labels=labels, label_smoothing=label_smoothing,
                                 dtype=p.dtype)
        loss = labels * pos_loss + (1 - labels) * neg_loss

    return loss

