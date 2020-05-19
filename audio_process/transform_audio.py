import os
import json
import time
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import torch
from torchvision import transforms
import datasets as module_dataset
import transformers as module_transform
from utils import get_instance, save_json, ensure_dir


def main(config):
    """
    Construct meta-transform class, and add it as a new attribute to config under dataset
    """
    list_transform = [get_instance(module_transform, i, config) for i in config if 'transform' in i]
    transform = transforms.Compose(list_transform)  # construct meta-transform class
    print(transform)
    config['dataset']['args']['transform'] = transform  # add new attributes to dataset args

    """
    Construct dataset along with the added meta-tranform class
    """
    d = get_instance(module_dataset, 'dataset', config)  # construct dataset with added meta-transform class

    save_path = os.path.join(config['save_dir'], config['save_subdir'])
    ensure_dir(save_path)
    print("The processed audio will be saved at %s", save_path)
    config['dataset']['args'].pop('transform', None)  # remove the added attributes since it prevents saving to json
    save_json(config, os.path.join(save_path, 'config_audio.json'))

    """
    Open a figure to draw 9 randomly sampled extracted spectrogram
    """
    np.random.seed(1234)
    display_samples = np.random.choice(len(d), size=9, replace=False)
    gs = gridspec.GridSpec(3, 3)
    fig = plt.figure(figsize=(15, 15))
    n_fig = 0

    start_time = time.time()
    for k in range(len(d)):
        print("Transforming %d-th data ... %s" % (k, d.path_to_data[k]))
        x, idx, fp = d[k]
        assert idx == k
        audio_id = fp.split('/')[-1].split('.')[0]
        p = os.path.join(save_path, '%s-%s.%s' % (audio_id, config['save_subdir'], 'pth'))
        # np.save(p, x)
        torch.save(x, p)

        if k in display_samples:
            ax = fig.add_subplot(gs[n_fig])
            if len(x.size()) > 2:
                x = x.squeeze(0)
            ax.imshow(x, aspect='auto', origin='lower')
            # ax.set_yticklabels([])
            # ax.set_xticklabels([])
            ax.set_title(audio_id)
            n_fig += 1
            plt.savefig(os.path.join(save_path, '.'.join(['spec', 'jpg'])))

    plt.savefig(os.path.join(save_path, '.'.join(['spec', 'jpg'])))
    print("Time: %.2f seconds" % (time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='audio transformer')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    args = parser.parse_args()

    if args.config:
        config = json.load(open(args.config))
    else:
        raise AssertionError("Configuration file need to be specified.")

    d = main(config)
