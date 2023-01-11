import numpy as np

def tp_fp_tn_fn_(y, y_hat, pos_label=1):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    try:
        y = y.flatten()
        y_hat = y_hat.flatten()
        for i in range(len(y_hat)):
            if y[i] == y_hat[i] == pos_label:
                tp += 1
            if y_hat[i] == pos_label and y[i] != y_hat[i]:
                fp += 1
            if y[i] == y_hat[i] != pos_label:
                tn += 1
            if y_hat[i] != pos_label and y[i] != y_hat[i]:
                fn += 1
        return (tp, fp, tn, fn)
    except Exception:
        return (None, None, None, None)


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    tp, fp, tn, fn = tp_fp_tn_fn_(y, y_hat)
    if tp:
        try:
            return ((tp + tn) / (tp + fp + tn + fn))
        except Exception as e:
            print(f"Error in accuracy_score_() : {e}")
    return None

def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    tp, fp, _, _ = tp_fp_tn_fn_(y, y_hat, pos_label)
    if tp:
        try:
            return (tp / (tp + fp))
        except Exception as e:
            print(f"Error in precision_score_() : {e}")
    return None

def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    tp, _, _, fn = tp_fp_tn_fn_(y, y_hat, pos_label)
    if tp:
        try:
            return (tp / (tp + fn))
        except Exception as e:
            print(f"Error in recall_score_() : {e}")
    return None

def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    tp, _, _, _ = tp_fp_tn_fn_(y, y_hat, pos_label)
    if tp:
        try:
            precision = precision_score_(y, y_hat, pos_label)
            recall = recall_score_(y, y_hat, pos_label)
            return ((2 * precision * recall) / (precision + recall))
        except Exception as e:
            print(f"Error in recall_score_() : {e}")
    return None
