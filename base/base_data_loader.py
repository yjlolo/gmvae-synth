import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class SoLBase(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super(SoLBase, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        dp = np.array(self.dataset.path_to_data)
        inst = [i.split('/')[-1].split('-')[0] for i in dp]
        dict_inst_idx = {}
        inst_ratio = {}
        for i in sorted(set(inst)):
            inst_idx = np.where(np.array(inst) == i)[0]
            inst_ratio[i] = len(inst_idx) / len(dp)
            dict_inst_idx[i] = inst_idx

        # self.dataset.dict_inst_ratio = inst_ratio
        # self.dataset.dict_inst_idx = dict_inst_idx
        dict_cat_inst_idx = {}
        dict_cat_inst_ratio = {}
        for k, v in self.dataset.ins_map.items():
            try:
                dict_cat_inst_idx[v] = np.hstack([dict_cat_inst_idx[v], dict_inst_idx[k]])
                dict_cat_inst_ratio[v] += inst_ratio[k]
            except:
                dict_cat_inst_idx[v] = dict_inst_idx[k]
                dict_cat_inst_ratio[v] = inst_ratio[k]
        self.dataset.dict_cat_inst_idx = dict_cat_inst_idx
        self.dataset.dict_cat_inst_ratio = dict_cat_inst_ratio

        valid_idx = []
        for i in inst_ratio:
            np.random.seed(1111)
            valid_idx.append(np.random.choice(dict_inst_idx[i], size=int(split * len(dict_inst_idx[i])), replace=False))
        valid_idx = np.array([j for i in valid_idx for j in i])
        train_idx = np.setdiff1d(idx_full, valid_idx)
        self.valid_idx = valid_idx
        self.train_idx = train_idx
        # dp_pitch = np.array([i.split('/')[-1].split('-')[2] for i in self.dataset.path_to_data])
        # pn, n = np.unique(dp_pitch, return_counts=True)
        # valid_ind = []
        # for i, j in zip(pn, n):
        #     n_train = int(np.floor(j * (1 - split)))
        #     n_valid = j - n_train
        #     assert (n_valid > 0) and (n_train > 0)
        #     valid_ind.append(np.random.choice(idx_full[np.array([p == i for p in dp_pitch])],
        #                                       size=n_valid, replace=False))

        # valid_idx = np.array([j for i in valid_ind for j in i])
        # train_idx = np.setdiff1d(idx_full, valid_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
