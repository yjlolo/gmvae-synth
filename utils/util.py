import os
import json
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def get_instance(module, name, config, *args):
    func_args = config[name]['args'] if 'args' in config[name] else None

    if func_args:
        return getattr(module, config[name]['type'])(*args, **func_args)
    else:
        return getattr(module, config[name]['type'])(*args)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_json(x, fname, if_sort_key=False, n_indent=None):
    with open(fname, 'w') as outfile:
        json.dump(x, outfile, sort_keys=if_sort_key, indent=n_indent)


def log_gauss(q_z, mu, logvar):
    llh = - 0.5 * (torch.pow(q_z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi))
    return torch.sum(llh, dim=1)
