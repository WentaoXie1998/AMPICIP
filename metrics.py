import numpy as np
from scipy.stats import spearmanr, pearsonr

def _to_1d_numpy(x):
    if "torch" in str(type(x)):
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.reshape(-1)


def mse(y_true, y_pred):
    y_true = _to_1d_numpy(y_true)
    y_pred = _to_1d_numpy(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def concordance_index(y_true, y_pred):
    y_true = _to_1d_numpy(y_true)
    y_pred = _to_1d_numpy(y_pred)

    n = len(y_true)
    assert n == len(y_pred)

    num_pairs = 0
    num_conc = 0
    num_ties = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                # 真实值相等的对通常不计入 CI
                continue

            num_pairs += 1
            # 真实顺序
            true_cmp = y_true[i] - y_true[j]
            # 预测顺序
            pred_cmp = y_pred[i] - y_pred[j]

            if true_cmp * pred_cmp > 0:
                # 同号：顺序一致
                num_conc += 1
            elif pred_cmp == 0:
                # 预测相等：算 0.5
                num_ties += 1

    if num_pairs == 0:
        return float("nan")
    return float((num_conc + 0.5 * num_ties) / num_pairs)


def spearman_corr(y_true, y_pred):
    y_true = _to_1d_numpy(y_true)
    y_pred = _to_1d_numpy(y_pred)
    if y_true.size == 0:
        return float("nan")
    try:
        corr, _ = spearmanr(y_true, y_pred)
    except Exception:
        corr = float("nan")
    return float(corr)


def pearson_corr(y_true, y_pred):
    y_true = _to_1d_numpy(y_true)
    y_pred = _to_1d_numpy(y_pred)
    if y_true.size == 0:
        return float("nan")
    try:
        corr, _ = pearsonr(y_true, y_pred)
    except Exception:
        corr = float("nan")
    return float(corr)


def regression_metrics(y_true, y_pred, with_spearman=True, with_ci=True, with_pearson=True):
    res = {
        "mse": mse(y_true, y_pred),
    }
    if with_spearman:
        res["spearman"] = spearman_corr(y_true, y_pred)
    if with_pearson:
        res["pearson"] = pearson_corr(y_true, y_pred)
    if with_ci:
        res["ci"] = concordance_index(y_true, y_pred)
    return res
