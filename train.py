import os
import random
import json
import argparse
import numpy as np
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import GMVAE_Trainer
from utils import Logger, get_instance

def make_determine():
    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


def main(config, resume):
    torch.manual_seed(1234)
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(module_data, 'data_loader', config)
    valid_data_loader = data_loader.split_validation()
    dd = np.setdiff1d(valid_data_loader.sampler.indices,
                      np.load('data/valid_idx.npy'))
    assert len(dd) == 0
    # print(data_loader.sampler.indices)
    # print(valid_data_loader.sampler.indices)
    # print(np.array(data_loader.dataset.path_to_data)[np.array(data_loader.sampler.indices)])
    # print(np.array(data_loader.dataset.path_to_data)[np.array(valid_data_loader.sampler.indices)])

    # build model architecture
    # make_determine()
    model = get_instance(module_arch, 'arch', config)
    model.apply(weights_init)
    # print(model)

    # get function handles of loss and metrics
    # loss = getattr(module_loss, config['loss'])
    loss = {l_i: get_instance(module_loss, l_i, config) for l_i in config if 'loss' in l_i}
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    trainer = GMVAE_Trainer(model, loss, metrics, optimizer,
                            resume=resume,
                            config=config,
                            data_loader=data_loader,
                            label_portion=config['trainer']['label_portion'],
                            valid_data_loader=valid_data_loader,
                            # lr_scheduler=lr_scheduler,
                            train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
