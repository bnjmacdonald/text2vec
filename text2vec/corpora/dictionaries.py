import json
import warnings

class BasicDictionary(object):
    """implements a basic in-memory dictionary of id->token pairs.

    Attributes:

        id2token: dict. Dictionary of id->token pairs.

        token2id: dict. Dictionary of token->id pairs.

        counts: dict. Dictionary of id->int pairs, where the value represents
            the number of times that the token has appeared in the corpus.

        _max_id: int. Current max id in self.id2token.
    """

    def __init__(self, id2token=None, verbosity=0):
        """
        Arguments:

            id2token: dict. Dictionary of id->token pairs. Default::

                {
                    0: '<PAD>',
                    1: '<UNK>',
                    2: '<END>'
                }
        """
        if id2token is None:
            id2token = {
                0: '<PAD>',
                1: '<UNK>',
                2: '<END>'
            }
        self.id2token = id2token
        self.token2id = {v: k for k, v in self.id2token.items()}
        self.counts = {k: 0 for k, v in self.id2token.items()}
        self._max_id = max(self.id2token.keys())
        self.verbosity = verbosity

    def update(self, tokens):
        """adds new tokens to the dictionary.
        """
        for token in tokens:
            try:
                token_id = self.token2id[token]
                self.counts[token_id] += 1
                if self.verbosity > 2:
                    print('{0} already exists.'.format(token))
            except KeyError:
                token_id = self._max_id + 1
                self.id2token[token_id] = token
                self.token2id[token] = token_id
                self.counts[token_id] = 1
                self._max_id = token_id
                if self.verbosity > 2:
                    print('added new token: {0} (id: {1}).'.format(token, self._max_id))
        return 0

    def save(self, path):
        """saves a dictionary to disk in json format."""
        with open(path, 'w') as f:
            json.dump({'id2token': self.id2token, 'counts': self.counts}, f)
        if self.verbosity > 0:
            print('Saved id2token dict to {0}'.format(path))

    def load(self, path):
        """loads a dictonary form disk.

        Todos:

            FIXME: conversion of str to int does not seem very efficient.
        """
        with open(path, 'r') as f:
            json_data = json.load(f)
            self.id2token = {int(k): v for k, v in json_data['id2token'].items()}
            self._reset_token2id()
            try:
                self.counts = json_data['counts']
            except KeyError:
                self.counts = {k: 0 for k, v in self.id2token.items()}
                warnings.warn('No `counts` key found in JSON file. Initializing all counts at 0.', RuntimeWarning)
        if self.verbosity > 0:
            print('Loaded dictionary with {0} unique tokens'.format(len(self.id2token)))

    def tokens2ids(self, tokens):
        """safely converts a list of string tokens to ids by looking up each
        token in self.token2id.

        If token is not found in self.token2id, the id for the '<UNK>' token is
        assigned.

        Arguments:

            tokens: list of str.

        Returns:

            _ids: list of int.
        """
        _ids = []
        for token in tokens:
            try:
                _id = self.token2id[token]
            except KeyError:
                _id = self.token2id['<UNK>']
            _ids.append(_id)
        return _ids

    def ids2tokens(self, _ids):
        """looks up the string token for each id in ids.

        Arguments:

            _ids: list of int.

        Returns:

            tokens: list of str.

        Todos:

            TODO: ?? implement safe lookup ??
        """
        return [self.id2token[_id] for _id in _ids]

    def _reset_token2id(self):
        """resets self.token2id according to what is currently in self.id2token.

        Todos:

            TODO: should self.counts be reset too?
        """
        self.token2id = {v: k for k, v in self.id2token.items()}
        self._max_id = max(self.id2token.keys())
        assert len(self.id2token) == len(self.token2id)
        assert len(self.id2token) == self._max_id + 1
