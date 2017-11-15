"""Implements corpora for use in other methods.

Corpus usage::

    ...

https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/

"""

import os
from text2vec.processing.preprocess import tokens2bow

class Corpus(object):
    """provides methods for building, reading, updating, and saving a
    corpus.

    A "corpus" is a matrix or list of 2-tuples representing a preprocessed
    corpus that is ready to be fed to a `text2vec.fit` method.

    Todos:

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

    def __init__(self, path, fmt, verbosity=0):
        assert os.path.isdir(path[0] + os.path.join(*path[1:].split('/')[:-1])), "path must point to a valid directory."
        assert fmt in ['bow', 'seq'], "fmt must be one of 'bow' or 'seq'."
        self.path = path
        self.fmt = fmt
        self.verbosity = verbosity

    def __len__(self):
        i = 0
        for _ in self.stream():
            i += 1
        return i

    def __iter__(self):
        return self.stream()

    def build(self, doc_arrays, dictionary, options=None):
        """Converts doc_arrays into a bow or seq corpus and save the resulting
        corpus to disk so that it can be streamed.
        Arguments:

            doc_arrays: list of 2-tuples. First element of each tuple is a
                doc_array _id that uniquely identifies the document. Second
                element of each tuple is a doc_array (document that has been
                preprocessed but not yet converted into a specific `bow` of
                `seq` format).

            fmt: string. Either 'bow' or 'seq'.

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
        assert self.fmt in ['bow', 'seq'], "fmt must be one of 'bow' or 'seq'."
        if options is None:
            options = {}
        if self.verbosity > 0:
            print('building corpus and saving to disk...')
        f0 = open(os.path.join(self.path, self.corpus_fname), 'w', encoding='utf-8')
        f1 = open(os.path.join(self.path, self.ids_fname), 'w', encoding='utf-8')
        f2 = open(os.path.join(self.path, self.seqlens_fname), 'w', encoding='utf-8')
        end_token_id = str(dictionary.token2id['<END>'])
        longest_seq = 0
        for i, (_id, doc_array) in enumerate(doc_arrays):
            doc_array = [str(token_id) for token_id in doc_array]
            seqlen = len(doc_array) + 1  # +1 b/c of end_token_id
            f0.write(' '.join(doc_array) + ' ' + end_token_id + '\n')
            f1.write(_id + '\n')
            f2.write(str(seqlen) + '\n')
            longest_seq = max(seqlen, longest_seq)
            if self.verbosity > 0 and (i + 1) % 20000 == 0:
                print('saved {0} document arrays to disk...'.format(i), end='\r')
        # corpora.BleiCorpus.serialize(os.path.join(self.path, self.corpus_bow_fname), serialize())
        f0.close()
        f1.close()
        f2.close()
        print()
        if self.fmt == 'seq':
            pad = dictionary.token2id['<PAD>']
            max_seqlen = longest_seq
            if 'max_seqlen' in options:
                max_seqlen = min(longest_seq, options['max_seqlen'])
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
        return 0

    def stream(self):
        """streams corpus from disk, yielding one line at a time.
        """
        f0 = open(os.path.join(self.path, self.corpus_fname), 'r', encoding='utf-8')
        if self.fmt == 'bow':
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
