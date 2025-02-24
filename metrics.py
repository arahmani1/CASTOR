import numpy as onp
from sklearn import metrics as sklearn_metrics


def shd(g, h):
    """
    Computes pairwise Structural Hamming distance, i.e.
    the number of edge insertions, deletions or flips in order to transform one graph to another
        - this means, edge reversals do not double count
        - this means, getting an undirected edge wrong only counts 1

    Args:
        g:  [..., d, d]
        h:  [..., d, d]
    """
    assert g.ndim == h.ndim
    abs_diff =  onp.abs(g - h)
    mistakes = abs_diff + onp.swapaxes(abs_diff, -2, -1)  # mat + mat.T (transpose of last two dims)

    # ignore double edges
    mistakes_adj = onp.where(mistakes > 1, 1, mistakes)

    return onp.triu(mistakes_adj).sum((-1, -2))


def n_edges(g):
    """
    Args:
        g:  [..., d, d]
    """
    return g.sum((-1, -2))


def is_acyclic(g):
    """
       Args:
           g:  [d, d]
       """
    n_vars = g.shape[-1]
    mat = onp.eye(n_vars) + g / n_vars
    mat_pow = onp.linalg.matrix_power(mat, n_vars)
    acyclic_constr = onp.trace(mat_pow) - n_vars
    return onp.isclose(acyclic_constr, 0.0)


def is_cyclic(g):
    """
    Args:
        g:  [d, d]
    """
    return not is_acyclic(g)


def classification_metrics(true, pred):
    """
    Args:
        true:  [...]
        pred:  [...]
    """
    true_flat = true.reshape(-1)
    pred_flat = pred.reshape(-1)

    if onp.sum(pred_flat) > 0 and onp.sum(true_flat) > 0:
        precision, recall, f1, _ = sklearn_metrics.precision_recall_fscore_support(
            true_flat, pred_flat, average="binary")
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    elif onp.sum(pred_flat) == 0 and onp.sum(true_flat) == 0:
        # no true positives, and no positives were predicted
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }
    else:
        # no true positives, but we predicted some positives
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }


def threshold_metrics(true, pred):
    """
    Args:
        true:  [...]
        pred:  [...]
    """
    true_flat = true.reshape(-1)
    pred_flat = pred.reshape(-1)

    if onp.sum(pred_flat) > 0 and onp.sum(true_flat) > 0:
        fpr, tpr, _ = sklearn_metrics.roc_curve(true_flat, pred_flat)
        precision, recall, _ = sklearn_metrics.precision_recall_curve(true_flat, pred_flat)
        ave_prec = sklearn_metrics.average_precision_score(true_flat, pred_flat)
        roc_auc = sklearn_metrics.auc(fpr, tpr)
        prc_auc = sklearn_metrics.auc(recall, precision)

        return {
            "auroc": roc_auc,
            "auprc": prc_auc,
            "ap": ave_prec,
        }

    elif onp.sum(pred_flat) == 0 and onp.sum(true_flat) == 0:
        # no true positives, and no positives were predicted
        return {
            "auroc": 1.0,
            "auprc": 1.0,
            "ap": 1.0,
        }

    else:
        # no true positives, but we predicted some positives
        return {
            "auroc": 0.5,
            "auprc": 0.0,
            "ap": 0.0,
        }