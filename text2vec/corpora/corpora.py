"""Implements corpora for use in other methods.

Corpus usage::

    ...

https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/

"""

import os
import json
import joblib
from text2vec.processing.preprocess import preprocess_one, tokens2bow
from text2vec.corpora.dictionaries import BasicDictionary

class CorpusBuilder(object):
    """Houses dictionary, text transformation pipeline, and configuration options
    for building a corpus from text.

    Attributes:

        ...

    """

    dictionary_fname = 'dictionary.json'
    text_transformer_fname = 'text_transformer.pkl'
    config_fname = 'config.json'

    def __init__(self, path, fmt=None, dictionary=None, text_transformer=None, options=None, verbosity=0):
        assert os.path.isdir(path[0] + os.path.join(*path[1:].split('/')[:-1])), "path must point to a valid directory."
        self.path = path
        self.verbosity = verbosity
        try:
            if dictionary is None:  # KLUDGE
                dictionary = BasicDictionary()
            self.dictionary = dictionary
            self._load()
            assert fmt is None and text_transformer is None and options is None, "A builder already exists in {0}, but you tried passing arguments other than `path` to __init__.".format(self.path)
        except IOError:
            if self.verbosity > 0:
                print('No existing builder found in {0}. Initializing new one.'.format(self.path))
            self.fmt = fmt
            if text_transformer is None:
                text_transformer = preprocess_one
            self.text_transformer = text_transformer
            if dictionary is None:
                dictionary = BasicDictionary()
            self.dictionary = dictionary
            if options is None:
                options = {}
            self.options = options
        assert self.fmt in ['bow', 'seq'], "self.fmt must be one of 'bow' or 'seq'."

    def save(self):
        joblib.dump(self.text_transformer, os.path.join(self.path, self.text_transformer_fname))
        self.dictionary.save(os.path.join(self.path, self.dictionary_fname))
        with open(os.path.join(self.path, self.config_fname), 'w') as f:
            config = {  # TODO: save preprocessing configuration options.
                'fmt': self.fmt,
                'options': self.options,
            }
            json.dump(config, f)
        return 0

    def _load(self):
        self.text_transformer = joblib.load(os.path.join(self.path, self.text_transformer_fname))
        self.dictionary.load(os.path.join(self.path, self.dictionary_fname))
        with open(os.path.join(self.path, self.config_fname), 'r') as f:
            config = json.load(f)
            self.fmt = config['fmt']
            # self.n_documents = config['n_documents']
            self.options = config['options']
        return 0


class Corpus(object):
    """provides methods for building, reading, updating, and saving a
    corpus.

    A "corpus" is a matrix or list of 2-tuples representing a preprocessed
    corpus that is ready to be fed to a `text2vec.fit` method.

    Attributes:

        builder: object. An object with a dictionary, text_transformer, and
            configuration options that provides instructions on how to convert
            a list of documents into a corpus.

        text_transformer: method (default:
            `text2vec.processing.preprocess.preprocess_one`). Method that takes
            a single document string as the only argument and returns a list of
            token ids (list of int).

    Todos:

        TODO: ?? allow builder to be specified as a path ??

        TODO: ?? do I want to use max_seqlen option when saving to disk,
            streaming from disk, or both ??

        TODO: do I really want to pad zeros when saving to disk? It makes the
            corpus size dramatically larger. I could do this on streaming the
            data instead. I would just need to read `seqlens.txt` and
            `config.json` first.
    """

    corpus_fname = 'corpus.txt'
    ids_fname = 'ids.txt'
    seqlens_fname = 'seqlens.txt'
    meta_fname = 'meta.json'

    def __init__(self, path, builder, verbosity=0):
        assert os.path.isdir(path[0] + os.path.join(*path[1:].split('/')[:-1])), "path must point to a valid directory."
        self.path = path
        self.builder = builder
        self.verbosity = verbosity
        try:
            with open(os.path.join(self.path, self.meta_fname), 'r') as f:
                meta = json.load(f)
            self.n_documents = meta['n_documents']
        except IOError:
            if self.verbosity > 0:
                print('No existing meta information found in {0}.'.format(self.path))
            self.n_documents = 0  # TODO: try streaming to count documents.

    def __len__(self):
        return self.n_documents

    def __iter__(self):
        return self.stream()

    def build(self, documents, update_dict=False, overwrite=False):
        """Converts doc_arrays into a bow or seq corpus and save the resulting
        corpus to disk so that it can be streamed.
        Arguments:

            doc_arrays: list of 2-tuples. First element of each tuple is a
                doc_array _id that uniquely identifies the document. Second
                element of each tuple is a doc_array (document that has been
                preprocessed but not yet converted into a specific `bow` of
                `seq` format).

            fmt: string. Either 'bow' or 'seq'.

            update_dict: bool (default: False). If True, updates dictionary with
                new tokens. If False, any unseen token is converted to '<UNK>'

            options: dict. Available options:

                'max_seqlen': int. Maximum doc_array length to consider when
                    constructing the sequence. Only used when self.fmt == "seq".

                Example::

                    {'max_seqlen': 1000}
        Returns:

            int: 0.

        Todos:

            TODO: ?? would it be better to accept doc_arrays as a list of dicts,
                where each dict contains an _id key and a doc_array key ?? This
                would make it easier to use with mongoDB and other backends.

            TODO: check if sequence padding using np.zeros and other np methods
                would be faster.

            TODO: allow other ways of saving corpus to disk, such as via
                mongoDB

            TODO: ?? remove file from disk if exception encountered while
                building ??
        """
        if not overwrite and os.path.exists(os.path.join(self.path, self.corpus_fname)):
            raise RuntimeError('{0} already exists in path ({1}). If you wish to overwrite the corpus, pass `ovrewrite=True`.'.format(self.corpus_fname, self.path))
        if self.verbosity > 0:
            print('building corpus and saving to disk...')
        f0 = open(os.path.join(self.path, self.corpus_fname), 'w', encoding='utf-8')
        f1 = open(os.path.join(self.path, self.ids_fname), 'w', encoding='utf-8')
        f2 = open(os.path.join(self.path, self.seqlens_fname), 'w', encoding='utf-8')
        end_token_id = str(self.builder.dictionary.token2id['<END>'])
        longest_seq = 0
        for i, document in enumerate(documents):
            _id = str(document['_id'])
            text = document['text'].strip().lower()
            # preprocesses text
            tokens = self.builder.text_transformer(text)
            if tokens is not None and len(tokens) > 0:
                # updates dictionary with tokens.
                if update_dict:
                    self.builder.dictionary.update(tokens)
                # converts tokens to token_ids.
                doc_array = self.builder.dictionary.tokens2ids(tokens)
                # converts tokens
            doc_array = [str(token_id) for token_id in doc_array]
            seqlen = len(doc_array) + 1  # +1 b/c of end_token_id
            f0.write(' '.join(doc_array) + ' ' + end_token_id + '\n')
            f1.write(_id + '\n')
            f2.write(str(seqlen) + '\n')
            self.n_documents += 1
            longest_seq = max(seqlen, longest_seq)
            if self.verbosity > 0 and (i + 1) % 20000 == 0:
                print('saved {0} document arrays to disk...'.format(i+1), end='\r')
        # corpora.BleiCorpus.serialize(os.path.join(self.path, self.corpus_bow_fname), serialize())
        f0.close()
        f1.close()
        f2.close()
        print()
        if self.builder.fmt == 'seq':
            pad = self.builder.dictionary.token2id['<PAD>']
            max_seqlen = longest_seq
            if 'max_seqlen' in self.builder.options:
                max_seqlen = min(longest_seq, self.builder.options['max_seqlen'])
            f = open(os.path.join(self.path, self.corpus_fname + '.temp'), 'w', encoding='utf-8')
            for i, seq in enumerate(self.stream()):
                seqlen = len(seq)
                if max_seqlen > seqlen:  # adds padding
                    seq += [pad] * (max_seqlen - seqlen)
                elif seqlen > max_seqlen:
                    seq = seq[:max_seqlen]
                assert len(seq) == max_seqlen, 'seq should be the same length as max_seqlen. Sequence length: {0}. Max sequence length: {1}'.format(len(seq), max_seqlen)
                f.write(' '.join([str(token_id) for token_id in seq]) + '\n')
                if self.verbosity > 0 and (i + 1) % 20000 == 0:
                    print('saved {0} sequence arrays to disk...'.format(i), end='\r')
            f.close()
            os.remove(os.path.join(self.path, self.corpus_fname))
            os.rename(os.path.join(self.path, self.corpus_fname + '.temp'), os.path.join(self.path, self.corpus_fname))
            print()
        with open(os.path.join(self.path, self.meta_fname), 'w') as f:
            meta = {
                'n_documents': self.n_documents,
            }
            json.dump(meta, f)
        return 0

    def stream(self):
        """streams corpus from disk, yielding one line at a time.
        """
        assert self.builder.fmt in ['bow', 'seq'], "self.fmt must be one of 'bow' or 'seq'."
        f0 = open(os.path.join(self.path, self.corpus_fname), 'r', encoding='utf-8')
        if self.builder.fmt == 'bow':
            for line in f0:
                line = line.strip()
                if len(line) > 0:
                    yield tokens2bow([int(token_id) for token_id in line.split(' ')])
        else:
            for line in f0:
                line = line.strip()
                if len(line) > 0:
                    yield [int(token_id) for token_id in line.split(' ')]
        f0.close()
