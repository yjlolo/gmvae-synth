import torch


def test_acc(qy_logit, target):
    cat_pred = qy_logit.argmax(1)
    real_pred = torch.zeros_like(cat_pred)
    for cat in range(qy_logit.shape[1]):
        idx = cat_pred == cat
        lab = target[idx]
        if len(lab) == 0:
            continue
        real_pred[cat_pred == cat] = torch.mode(lab)[0]
    return torch.mean((real_pred == target).type(qy_logit.type()))


def classify_acc(qy_logit, target):
    cat_pred = qy_logit.argmax(1)
    assert cat_pred.shape[0] == len(target)
    correct = torch.sum(cat_pred == target).item()
    return correct / len(target)
    