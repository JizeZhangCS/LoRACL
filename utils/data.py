import numpy as np

class DataCoLoaderIterator(object):
    def __init__(self, data_loaders, args):
        self.args = args
        self.data_loaders = data_loaders
        self.data_loaders_iters = [iter(d) for d in data_loaders]
        self.nLoaders = len(self.data_loaders_iters)
        self.done = np.zeros((self.nLoaders,), dtype=bool)

    def __next__(self):
        ret = []
        for iit, it in enumerate(self.data_loaders_iters):
            try:
                x = next(it)
            except StopIteration:
                x = None
                self.done[iit] = True
            ret.append(x)
        if not np.all(self.done):
            for iR, x in enumerate(ret):
                if x is None:
                    # shuffle for DDP
                    try:
                        self.data_loaders[iR].sampler.set_epoch(np.random.randint(100000))
                    except:
                        pass
                    self.data_loaders_iters[iR] = iter(self.data_loaders[iR])
                    ret[iR] = next(self.data_loaders_iters[iR])
        else:
            raise StopIteration()

        return ret


class DataCoLoader(object):
    def __init__(self, data_loaders, args):
        self.args = args
        self.data_loaders = data_loaders

    def __iter__(self):
        return DataCoLoaderIterator(self.data_loaders, self.args)

    def __len__(self):
        return np.max([len(x) for x in self.data_loaders])
