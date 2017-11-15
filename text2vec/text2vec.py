import os
import time
import json
import pandas as pd
import numpy as np
from text2vec.utils import normalize


class Text2Vec(object):
    """Abstract class for a general text2vec model.

    The Text2Vec class provides a common base class that all other text2vec
    classes extends. The primary interface methods are `fit` and
    `get_embeddings`.

    Each Text2Vec subclass must implements its own `_train` and `get_embeddings`
    methods.

    Attributes:

        config (object): object containing configuration from model estimation.

        model (object): trained model.

        embeddings: np.ndarray (n_observations, n_features). Document
            embeddings.

        meta (pd.DataFrame): pandas DataFrame containing metadata on each
            document. Used for aggregation. Index of meta must align with
            index of self.embeddings.

        verbose (int): print verbose output to stdout.
    """

    _model_fname = 'model'

    def __init__(self, meta=None, verbose=0):
        self.verbose = verbose
        self.meta = meta
        self.config = {'name': str(self), 'train_time': 0.0}
        self.embeddings = None
        self.model = None
        # self.ids = None

    def __str__(self):
        return 'text2vec'

    def evaluate(self):
        raise NotImplementedError

    def fit(self, corpus, **kwargs):
        """trains a text2vec model on a dataset. Thin wrapper to `self._train`.

        Arguments:

            corpus: ...

        Returns:

            int: 0.

        Todos:

            KLUDGE: assignment of self.config is awkward.
        """
        if self.verbose:
            print('training document embeddings...')
        time0 = time.time()
        self._train(corpus, **kwargs)
        time1 = time.time()
        self.config.update({k: v for k, v in kwargs.items() if isinstance(v, str) or isinstance(v, int) or isinstance(v, float)})
        self.config['train_time'] += round(time1 - time0, 4)
        return 0

    def _train(self, corpus, **kwargs):
        """class-specific method for training the model.

        This method should not be used directly. Use self.fit.

        Arguments:

            corpus: ...

        Returns:

            ...
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

        Returns:

            ...
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

    def infer_docvecs(self, corpus):
        """infers embeddings from trained model.

        Todos:

            TODO: when does this differ from `get_embeddings`?
        """
        raise NotImplementedError('Each class must reimplement this method.')

    def save(self, path):
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.config, f)

    def save_embeddings(self, path, embeddings, ids=None):
        """saves embeddings to file.

        First column is document ids. If ids=None, ids are assigned
        sequentially.
        """
        pd_embeds = pd.DataFrame(embeddings, index=ids)
        pd_embeds.to_csv(os.path.join(path, 'embeddings.txt'), sep=',', header=None, index=True)

    def load(self, fname):
        with open(fname + 'config.json', 'r') as f:
            self.config = json.load(f)
