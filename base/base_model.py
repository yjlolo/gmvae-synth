import logging
import numpy as np
import torch
import torch.nn as nn
# from model.loss import approx_q_y


def log_gauss(q_z, mu, logvar):
    llh = - 0.5 * (torch.pow(q_z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi))
    return torch.sum(llh, dim=1)


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
        log_q_y_logit[:, k_i] = log_gauss(q_z, mu_k, logvar_k) + np.log(1 / k)

    q_y = torch.nn.functional.softmax(log_q_y_logit, dim=1)
    return log_q_y_logit, q_y


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(params)
        # print(super(BaseModel, self))


class BaseGMVAE(BaseModel):
    def __init__(self, input_size, latent_dim, n_class=10, is_featExtract=False):
        super(BaseGMVAE, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.n_class = n_class
        self.is_featExtract = is_featExtract
        self._build_mu_lookup()
        self._build_logvar_lookup()

    def _encode(self, x):
        """
        implementation should end with
        1. self._infer_latent()
        2. self._infer_class()
        and their outputs combined
        """
        raise NotImplementedError

    def _decode(self, z):
        raise NotImplementedError

    def _infer_latent(self, mu, logvar, weight=1):
        if self.is_featExtract:
            """
            only when NOT is_train;
            return mu as the representative latent vector
            """
            return mu, logvar, mu

        sigma = torch.sqrt(torch.exp(logvar))
        eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size())  # default require_grad=False

        z = mu + weight * sigma * eps  # reparameterization trick

        return mu, logvar, z

    def _build_mu_lookup(self):
        """
        follow Xavier initialization as in the paper
        """
        mu_lookup = nn.Embedding(self.n_class, self.latent_dim)
        nn.init.xavier_uniform_(mu_lookup.weight)
        mu_lookup.weight.requires_grad = True
        self.mu_lookup = mu_lookup

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        """
        follow Table 7 in the paper
        """
        logvar_lookup = nn.Embedding(self.n_class, self.latent_dim)
        # init_sigma = np.exp(-1)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup.weight, init_logvar)
        logvar_lookup.weight.requires_grad = logvar_trainable
        self.logvar_lookup = logvar_lookup
        # self.logvar_bound = np.log(np.exp(-1) ** 2)

    def _bound_logvar_lookup(self):
        self.logvar_lookup.weight.data[torch.le(self.logvar_lookup.weight, self.logvar_bound)] = self.logvar_bound

    def _infer_class(self, q_z):
        log_q_y_logit, q_y = approx_q_y(q_z, self.mu_lookup, self.logvar_lookup, k=self.n_class)
        val, ind = torch.max(q_y, dim=1)
        return log_q_y_logit, q_y, ind

    def forward(self, x):
        raise NotImplementedError
        # mu, logvar, z, q_y, ind = self._encode(x)
        # x_predict = x_self._decode(z)
        # return [mu, logvar, z], [q_y, ind], x_predict
