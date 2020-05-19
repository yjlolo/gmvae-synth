import numpy as np
import torch
import torch.nn.functional as F
from base.base_loss import BaseLoss  # why do I need to explicity import this time?
from utils import log_gauss


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(x_predict, x, reduction="none"):
    loss = F.mse_loss(x_predict, x, reduction=reduction)
    if len(loss.size()) > 2:
        loss = torch.sum(loss, dim=-1)
    return torch.sum(loss, dim=1)


def bce_loss(x_predict, x, reduction="none"):
    loss = F.binary_cross_entropy_with_logits(x_predict, x, reduction=reduction)
    return torch.sum(loss, dim=1)


def ce_loss(x_predict, x, reduction="none", is_train=True, label_idx=None):
    if not label_idx:
        loss = torch.zeros(1)
    else:
        if is_train:
            x_predict = x_predict[label_idx]
            x = x[label_idx]
        loss = F.cross_entropy(x_predict, x, reduction=reduction)
    return loss


def pitch_ce_loss(x_predict, x, reduction="none"):
    loss = F.cross_entropy(x_predict, x, reduction=reduction)
    return loss


def kl_gauss(q_mu, q_logvar, mu=None, logvar=None):
    """
    KL divergence between two diagonal gaussians
    """
    if mu is None:
        mu = torch.zeros_like(q_mu)
    if logvar is None:
        logvar = torch.zeros_like(q_logvar)

    return -0.5 * (1 + q_logvar - logvar - (torch.pow(q_mu - mu, 2) + torch.exp(q_logvar)) / torch.exp(logvar))


def approx_q_y(q_z, mu_lookup, logvar_lookup, k=10):
    """
    refer to eq.13 in the paper
    """
    q_z_shape = list(q_z.size())  # (b, z_dim)
    mu_lookup_shape = [mu_lookup.num_embeddings, mu_lookup.embedding_dim]  # (k, z_dim)
    logvar_lookup_shape = [logvar_lookup.num_embeddings, logvar_lookup.embedding_dim]  # (k, z_dim)

    if not mu_lookup_shape[0] == k:
        raise ValueError("mu_lookup_shape (%s) does not match the given k (%s)" % (
            mu_lookup_shape, k))
    if not logvar_lookup_shape[0] == k:
        raise ValueError("logvar_lookup_shape (%s) does not match the given k (%s)" % (
            logvar_lookup_shape, k))
    if not q_z_shape[1] == mu_lookup_shape[1]:
        raise ValueError("q_z_shape (%s) does not match mu_lookup_shape (%s) in dimension of z" % (
            q_z_shape, mu_lookup_shape))
    if not q_z_shape[1] == logvar_lookup_shape[1]:
        raise ValueError("q_z_shape (%s) does not match logvar_lookup_shape (%s) in dimension of z" % (
            q_z_shape, logvar_lookup_shape))

    # TODO: vectorization and don't use for loop
    batch_size = q_z_shape[0]
    log_q_y_logit = torch.zeros(batch_size, k).type(q_z.type())

    for k_i in torch.arange(0, k):
        mu_k, logvar_k = mu_lookup(k_i), logvar_lookup(k_i)
        log_q_y_logit[:, k_i] = log_gauss(q_z, mu_k, logvar_k)  # + np.log(1 / k)
    # print(log_q_y_logit.sum(dim=0))
    q_y = torch.nn.functional.softmax(log_q_y_logit, dim=1)
    return log_q_y_logit, q_y


def kl_class(log_q_y_logit, q_y, k=10):
    q_y_shape = list(q_y.size())

    if not q_y_shape[1] == k:
        raise ValueError("q_y_shape (%s) does not match the given k (%s)" % (
            q_y_shape, k))

    h_y = torch.sum(q_y * torch.nn.functional.log_softmax(log_q_y_logit, dim=1), dim=1)

    return h_y - np.log(1 / k), h_y


def kl_latent(q_mu, q_logvar, q_y, mu_lookup, logvar_lookup):
    """
    q_z (b, z)
    q_y (b, k)
    mu_lookup (k, z)
    logvar_lookup (k, z)
    """
    mu_lookup_shape = [mu_lookup.num_embeddings, mu_lookup.embedding_dim]  # (k, z_dim)
    logvar_lookup_shape = [logvar_lookup.num_embeddings, logvar_lookup.embedding_dim]  # (k, z_dim)
    q_mu_shape = list(q_mu.size())
    q_logvar_shape = list(q_logvar.size())
    q_y_shape = list(q_y.size())

    if not np.all(mu_lookup_shape == logvar_lookup_shape):
        raise ValueError("mu_lookup_shape (%s) and logvar_lookup_shape (%s) do not match" % (
            mu_lookup_shape, logvar_lookup_shape))
    if not np.all(q_mu_shape == q_logvar_shape):
        raise ValueError("q_mu_shape (%s) and q_logvar_shape (%s) do not match" % (
            q_mu_shape, q_logvar_shape))
    if not q_y_shape[0] == q_mu_shape[0]:
        raise ValueError("q_y_shape (%s) and q_mu_shape (%s) do not match in batch size" % (
            q_y_shape, q_mu_shape))
    if not q_y_shape[1] == mu_lookup_shape[0]:
        raise ValueError("q_y_shape (%s) and mu_lookup_shape (%s) do not match in number of class" % (
            q_y_shape, mu_lookup_shape))

    batch_size, n_class = q_y_shape
    kl_sum = torch.zeros(batch_size, n_class)  # create place holder
    for k_i in torch.arange(0, n_class):
        kl_sum[:, k_i] = torch.sum(kl_gauss(q_mu, q_logvar, mu_lookup(k_i), logvar_lookup(k_i)), dim=1)
        kl_sum[:, k_i] *= q_y[:, k_i]

    return torch.sum(kl_sum, dim=1)  # sum over classes


def kl_pitch_emb(pitch_mu_lookup, pitch_logvar_lookup, pitch_mu, pitch_logvar, y_pitch):
    return torch.sum(kl_gauss(pitch_mu, pitch_logvar,
                              pitch_mu_lookup(y_pitch), pitch_logvar_lookup(y_pitch)), dim=1)


class KLpitch(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(KLpitch, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, pitch_mu_lookup, pitch_logvar_lookup, pitch_mu, pitch_logvar, y_pitch):
        if epoch >= self.effect_epoch:
            return self.weight * kl_pitch_emb(pitch_mu_lookup, pitch_logvar_lookup, pitch_mu, pitch_logvar, y_pitch)
        else:
            return torch.zeros(1)


class MSEloss(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(MSEloss, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, x_predict, x):
        if epoch >= self.effect_epoch:
            return self.weight * mse_loss(x_predict, x)
        else:
            return torch.zeros(1)


class BCEloss(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(BCEloss, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, x_predict, x):
        if epoch >= self.effect_epoch:
            return self.weight * bce_loss(x_predict, x)
        else:
            return torch.zeros(1)


class CEloss(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(CEloss, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, x_predict, x, is_train=True, label_idx=None):
        if epoch >= self.effect_epoch:
            return self.weight * ce_loss(x_predict, x, is_train=is_train, label_idx=label_idx)
        else:
            return torch.zeros(1)


class PDloss(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(PDloss, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, x_predict, x):
        if epoch >= self.effect_epoch:
            return self.weight * pitch_ce_loss(x_predict, x)
        else:
            return torch.zeros(1)


class KLlatent(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(KLlatent, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, q_mu, q_logvar, q_y, mu_lookup, logvar_lookup):
        if epoch >= self.effect_epoch:
            return self.weight * kl_latent(q_mu, q_logvar, q_y, mu_lookup, logvar_lookup)
        else:
            return torch.zeros(1)


class KLclass(BaseLoss):
    def __init__(self, weight=1, effect_epoch=1):
        super(KLclass, self).__init__(weight, effect_epoch)

    def __call__(self, epoch, log_q_y_logit, q_y, k=10):
        if epoch >= self.effect_epoch:
            kl, h_y = kl_class(log_q_y_logit, q_y, k)
            return self.weight * kl, h_y
        else:
            return torch.zeros(1), torch.zeros(1)


if __name__ == '__main__':
    n_class = 10
    f, t = [80, 15]
    batch_size = 256
    latent_dim = 16
    x = torch.randn(batch_size, f, t)
    y = torch.randn(batch_size, f, t)
    dummy_mu_lookup = torch.randn(n_class, latent_dim)
    dummy_logvar_lookup = torch.randn(n_class, latent_dim)
    q_z = torch.randn(batch_size, latent_dim)

    q_y = approx_q_y(q_z, dummy_mu_lookup, dummy_logvar_lookup, k=n_class)
    neg_kld_y = -1 * kl_class(q_y, k=n_class)
    neg_kld_z = -1 * kl_latent(q_z, q_y, dummy_mu_lookup, dummy_logvar_lookup)
    reconloss = mse_loss(x, y)
    print(reconloss.size(), neg_kld_y.size(), neg_kld_z.size())
