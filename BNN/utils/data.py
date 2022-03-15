''' We manually implemented data loader in order to support
fast-data-removing from dataset. Specifically, in our simulation experiments, to remove a particular example, we do not need
to actually remove the target example from RAM, but to remove
the corresponding index is sufficient, which can reduce
unnecessary IO time.

Be careful that our data loader only support removing one
example at one time.
'''

import numpy as np
import torch


class Dataset():
    def __init__(self, x, y):
        assert(len(x) == len(y))
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        return x, y

    def __len__(self):
        return len(self.x)


class IndexBatchSampler(torch.utils.data.Sampler):
    def __init__(self, batch_size, indices=None, n=None, shuffle=False, drop_last=False):
        if indices is not None:
            self.indices = np.array( indices )
        else:
            self.indices = np.array( range(n) )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        ''' necessary '''
        self._set_length()

    def __iter__(self):
        if self.shuffle:
            indices = np.random.permutation(self.indices)
        else:
            indices = self.indices

        res = []
        for i in range(0, len(indices), self.batch_size):
            k = i + self.batch_size
            if k > len(indices) and self.drop_last:
                break
            k = min(k, len(indices))
            res.append( indices[i:k] )

        assert len(res) == self.length
        return iter(res)

    def __len__(self):
        return self.length

    def _set_length(self):
        self.length = len(self.indices) // (self.batch_size)
        if not self.drop_last and self.length*self.batch_size < len(self.indices):
            self.length += 1

    def remove(self, vals):
        for v in vals:
            self.indices = np.delete(
                self.indices, np.where(self.indices==v))
        # self.indices = np.delete(self.indices, np.where(self.indices==val))
        self._set_length()

    def set_indices(self, indices):
        self.indices = np.array(indices)
        self._set_length()


class DataLoader():
    def __init__(self, dataset, batch_size, shuffle=False,drop_last=False):
        self.dataset = dataset
        self.sampler = IndexBatchSampler(batch_size=batch_size, n=len(dataset), shuffle=shuffle, drop_last=drop_last)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_sampler=self.sampler, num_workers=4)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    ''' we only support removing one example at one time '''
    def remove(self, indices):
        self.sampler.remove(indices)

    def set_sampler_indices(self, indices):
        self.sampler.set_indices(indices)


class DataSampler():
    ''' Unlike those dataloaders in PyTorch, `DataSampler`
    can repeatly sample a batch from the dataset at any time,
    without reacquire the iterator of dataloader. This feature
    is very useful when calculating the inverse of Hessian
    matrix, since one need to repeatly sample and calculate
    in order to obtain stable numerical result.
    '''
    def __init__(self, dataset, batch_size):
        self.loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
        self.iterator = None

    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)

        try:
            samples = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            samples = next(self.iterator)

        return samples

    def __len__(self):
        return len(self.loader.sampler.indices)

    def remove(self, indices):
        self.loader.remove(indices)
        self.iterator = None

    def reset_next(self):
        self.iterator = None

    def set_sampler_indices(self, indices):
        self.loader.set_sampler_indices(indices)
