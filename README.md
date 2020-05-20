For reproducing the paper [Learning Disentangled Representations of Timbre and Pitch for Musical Instrument Sounds Using Gaussian Mixture Variational Autoencoders](https://arxiv.org/abs/1906.08152?fbclid=IwAR3yBPx71nPt0uO6GjVqdJxQxzStiyz3osf6mFUCW_cMIarwykZM5_tfUpU).

# Reproducing steps
1. Download the dataset from [Zenodo](https://zenodo.org/record/3833974#.XsUiWi2B3OQ).
   Put the folder `data` at the root of this repo.
2. run `python train.py -c config.json`

The checkpoint `model_best.pth` will be saved at `saved/gmvae-synth`.
After the training completes, 
play with `ismir19-217-sup-material.ipynb` to see the results shown in the paper.

# Details missing from the paper
A pitch classifier which takes as input the pitch latent variable is added on top of the pitch space.

# Note
- In provided dataset, `spec` and `spec-norm` refer to the extracted mel-spectrograms and the normalized ones.
- The configuration of `config.json` refers to the fully-supervised model in the paper,
  which is also the model used for controllable synthesis and timbre transfer.
  In `config.json`, change the `label_portion` under the `trainer` tag to train a semi-supervised model.

# Citation
Please kindly cite the paper as follows if you find it useful.
```
@inproceedings{DBLP:conf/ismir/LuoAH19,
  author    = {Yin{-}Jyun Luo and
               Kat Agres and
               Dorien Herremans},
  editor    = {Arthur Flexer and
               Geoffroy Peeters and
               Juli{\'{a}}n Urbano and
               Anja Volk},
  title     = {Learning Disentangled Representations of Timbre and Pitch for Musical
               Instrument Sounds Using Gaussian Mixture Variational Autoencoders},
  booktitle = {Proceedings of the 20th International Society for Music Information
               Retrieval Conference, {ISMIR} 2019, Delft, The Netherlands, November
               4-8, 2019},
  pages     = {746--753},
  year      = {2019},
  url       = {http://archives.ismir.net/ismir2019/paper/000091.pdf},
  timestamp = {Thu, 12 Mar 2020 11:32:59 +0100},
  biburl    = {https://dblp.org/rec/conf/ismir/LuoAH19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

# TODO
- [ ] Clean up the code
- [ ] Add comments
- [ ] Confirm if the raw audio files can be released
