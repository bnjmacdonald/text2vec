import os
import time
import json
import pandas as pd
import numpy as np
from .utils import normalize

class Config(object):
    pass

class Text2Vec(object):
    """Abstract class for a general text2vec model.

    Arguments:
        config (obj): object containing configuration for model estimation.
        meta (pd.DataFrame): pandas DataFrame containing metadata on each
            document. Used for aggregation. Index of meta must align with
            index of self.embeddings.
        verbose (int): print verbose output to stdout.
    """
    def __init__(self, config=None, meta=None, verbose=0):
        if config is None:
            config = Config()
        self.config = config
        self.verbose = verbose
        self.meta = meta
        self.embeddings = None
        # self.ids = None
        self._fname = None

    def __str__(self):
        return 'text2vec'
    
    def evaluate(self):
        raise NotImplementedError

    def fit(self, docs):
        """fits model to the data."""
        if self.verbose:
            print('training document embeddings...')
        time0 = time.time()
        self._train(docs, **self.config.fit_kws)
        time1 = time.time()
        self.train_time = time1 - time0

    def _train(self, docs, **kwargs):
        """class-specific method for training the model.

        This method should not be used directly. Use self.fit.
        """
        raise NotImplementedError('Each class must reimplement this method.')

    def get_embeddings(self, norm=False):
        """returns embedding matrix."""
        raise NotImplementedError('Each class must reimplement this method.')


    def agg_embeddings(self, id_var, embeddings=None, agg_func=np.mean, pre_norm=False, post_norm=False):
        """Aggregates embeddings.
        
        Arguments:
            id_var (str): column in self.meta to group embeddings by.
            embeddings (np.ndarray): embeddings to aggregate. If None, use
                self.embeddings.
            agg_func (method): function to use in aggregation (e.g. np.mean, 
                np.sum). Default: np.mean.
            pre_norm (bool): If True, l2-normalize embeddings before
                aggregation. Default: False.
            post_norm (bool): If True, l2-normalize embeddings after
                aggregation. Default: False.
        """
        if self.meta is None:
            raise RuntimeError('meta has not been assigned.')
        if embeddings is None:
            if self.embeddings is None:
                self.get_embeddings()
            # data_temp = self.embeddings.join(group)
            embeddings = self.embeddings
        if pre_norm:
            embeddings = normalize(embeddings)
        embeddings_agg = embeddings.groupby(self.meta[id_var]).agg(agg_func)
        if post_norm:
            embeddings_agg = normalize(embeddings_agg)
        return embeddings_agg

    def save(self, path):
        if self._fname is None:
            self._get_fname(path)
        out = {}
        out['name'] = str(self)
        out['train_time'] = self.train_time
        out['config'] = {k: self.config.__getattribute__(k) for k in dir(self.config) if not k.startswith('__')}
        with open(os.path.join(path, self._fname + '_config.txt'), 'w') as f:
            json.dump(out, f)

    def save_embeddings(self, path, embeddings, ids=None):
        """saves embeddings to file.

        First column is document ids. If ids=None, ids are assigned
        sequentially.
        """
        if self._fname is None:
            raise ValueError('self._fname must be assigned before self.save_embeddings is called.')
        pd_embeds = pd.DataFrame(embeddings, index=ids)
        pd_embeds.to_csv(os.path.join(path, self._fname + '_embeddings.txt'), sep=',', header=None, index=True)

    def load(self, fname):
        # Note: kludge. Why not pickle original config instead?
        with open(fname + '_config.txt', 'r') as f:
            config_dict = json.load(f)
        for k, v in config_dict['config'].items():
            self.config.__setattr__(k, v)

    def _get_fname(self, path):
        fname = 'experiment'
        if self.config.debug:
            fname = 'debug_' + fname
        new_fname = fname
        x = 0
        while os.path.exists(os.path.join(path, new_fname + str(x))):
            x += 1
        self._fname = new_fname + str(x)

