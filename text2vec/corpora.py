"""Implements corpora for use in other methods.

Corpus usage
------------

corpus = Corpus(out_path=out_path, verbose=1)

https://rare-technologies.com/data-streaming-in-python-generators-iterators-iterables/

"""

import os
import json
from gensim import corpora
from .utils import tokenize

class SimpleCorpus(object):
    """Generates a corpus of documents as bags of words.
    
    Attributes:
        docs (list): list where each row is a document and each column is a
            word count. (i.e. list in gensim sparse tuple format)
    """
    def __init__(self, docs, dictionary=None):
        self.docs = docs
        self.dictionary = dictionary
    
    def __len__(self):
        return len(self.docs)
    
    def __iter__(self):
        for i, doc in enumerate(self.docs):
            yield doc

class SimpleDocCorpus(object):
    """Generates a corpus of gensim TaggedDocument objects.
    
    Attributes:
        docs (list): list where each row is a document and each column is a
            word index.
        seqlens (pd.Series): pandas Series of the length of each document.
        tags (pd.Series): pandas Series of the tag to assign to each document.
    """
    def __init__(self, docs, seqlens=None):
        self.docs = docs
        self.seqlens = seqlens
    
    def __len__(self):
        return len(self.docs)
        
    def __iter__(self):
        for i, doc in enumerate(self.docs):
            seqlen = len(doc) if self.seqlens is None else self.seqlens[i]
            yield doc[:seqlen].values.astype(str).tolist()

class Corpus(object):
    
    corpus_fname = 'corpus.txt'
    corpus_bow_fname = 'corpus_bow.mm'
    ids_fname = 'ids.txt'
    documents_fname = 'documents.txt'
    dictionary_fname = 'dictionary.pickle'
    config_fname = 'config.json'

    def __init__(self, path, verbose=0):
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.verbose = verbose
        self._load_dictionary()
        self._load_config()

    def __len__(self):
        return self.len_corpus()
    
    def get_sizes(self):
        n_corpus, n_corpus_bow, n_ids, n_documents = self.len_corpus(), self.len_corpus_bow(), self.len_ids(), self.len_documents()
        n_dictionary = len(self.dictionary) if self.dictionary is not None else 0
        if self.verbose:
            print('\nCorpus size:')
            print('{0:15s}: {1}'.format('corpus', n_corpus))
            print('{0:15s}: {1}'.format('corpus_bow', n_corpus_bow))
            print('{0:15s}: {1}'.format('ids', n_ids))
            print('{0:15s}: {1}'.format('documents', n_documents))
            print('{0:15s}: {1}'.format('dictionary', n_dictionary))
        return n_corpus, n_corpus_bow, n_ids, n_documents, n_dictionary
    
    def iter_corpus(self):
        for tokens in self.stream_corpus():
            yield tokens

    def iter_corpus_bow(self):
        for bow in self.stream_corpus_bow():
            yield bow

    def iter_documents(self):
        for uid in self.stream_documents():
            yield uid

    def iter_ids(self):
        for uid in self.stream_ids():
            yield uid
    
    def stream_corpus(self):
        try:
            f0 = open(os.path.join(self.path, self.corpus_fname), 'r', encoding='utf-8')
            for line in f0:
                yield line.strip().split(' ')
            f0.close()
        except IOError as e:
            print(e)
            yield None
    
    def stream_ids(self):
        try:
            f0 = open(os.path.join(self.path, self.ids_fname), 'r')
            for line in f0:
                yield line.strip()
            f0.close()
        except IOError as e:
            print(e)
            yield None
    
    def stream_documents(self):
        fname = self.documents_fname
        if not os.path.exists(os.path.join(self.path, fname)):
            fname = self.corpus_fname
        try:
            f0 = open(os.path.join(self.path, fname), 'r', encoding='utf-8')
            for line in f0:
                yield line.strip()
            f0.close()
        except IOError as e:
            print(e)
            yield None
    
    def stream_corpus_bow(self, fmt='mm'):
        try:
            if fmt == 'mm':
                corpus_bow = corpora.MmCorpus(os.path.join(self.path, self.corpus_bow_fname))
            elif fmt == 'lda-c':
                corpus_bow = corpora.BleiCorpus(os.path.join(self.path, self.corpus_bow_fname))
            else:
                raise RuntimeError('fmt "{0}" not recognized'.format(fmt))
            return corpus_bow
        except IOError as e:
            print(e)
            return None
    
    def _load_dictionary(self):
        try:
            self.dictionary = corpora.Dictionary.load(os.path.join(self.path, self.dictionary_fname))
            self.dictionary[0]  # appears that corpora.dictionary does not assign id2token unless called as such. Seems like a bug.
        except IOError:
            print('No dictionary found at {0}'.format(os.path.join(self.path, self.dictionary_fname)))
            self.dictionary = None
    
    def mk_corpus(self, documents, ids=None, preprocessor=None, **kwargs):
        if preprocessor is None:
            preprocessor = tokenize
        preprocessed_docs = self._preprocess(documents, ids, preprocessor, **kwargs)
        self._update_config(**kwargs)
        self._save_corpus(preprocessed_docs)
    
    def mk_corpus_bow(self, dict_filter_kws={}):
        corpus_bow = self._doc2bow(dict_filter_kws)
        self._update_config(**dict_filter_kws)
        self._save_corpus_bow(corpus_bow, fmt='mm')

    def mk_dictionary(self, **kwargs):
        """creates a gensim dictionary of id->token mappings.

        '<unk>' represents unknown tokens; '<end>' represents end of document
        token; '<pad>' represents document padding.
        
        Arguments:
            tokens (list): list of lists containing words in documents.
                e.g. [['i', 'went', 'home'], ['apples', 'to', 'oranges']]
            **kwargs: keyword arguments to pass to gensim.corora.Dictionary.
        
        Returns:
            None
        """
        if self.verbose:
            print('constructing dictionary...')
        dictionary = corpora.Dictionary([['<pad>'], ['<unk>'], ['<end>']])
        dictionary2 = corpora.Dictionary(self.stream_corpus())
        if len(kwargs):
            dictionary2.filter_extremes(**kwargs)
            dictionary2.compactify()
        dictionary.merge_with(dictionary2)
        self.dictionary = dictionary
        self._save_dictionary()
    
    def _doc2bow(self, dict_filter_kws={}):
        if self.dictionary is None:
            self.mk_dictionary(**dict_filter_kws)
        corpus_stream = self.stream_corpus()
        # ids_stream = self.stream_ids()
        for i, tokens in enumerate(corpus_stream):
            # uid = next(ids_stream)
            bow = self.dictionary.doc2bow(tokens)
            # if len(bow):
            yield bow
    
    def _preprocess(self, documents, ids, preprocessor, **tokenize_kws):
        if self.verbose:
            print('Preprocessing documents and saving to disk...')
        for i, text in enumerate(documents):
            doc_tokens = preprocessor(text=text, **tokenize_kws)
            uid = ids[i] if ids is not None else i
            yield uid, doc_tokens

    def _save_corpus_bow(self, corpus_bow, fmt='mm'):
        if fmt == 'mm':
            corpora.MmCorpus.serialize(os.path.join(self.path, self.corpus_bow_fname), corpus_bow)
        elif fmt == 'lda-c':
            corpora.BleiCorpus.serialize(os.path.join(self.path, self.corpus_bow_fname), corpus_bow)
        else:
            raise RuntimeError('fmt "{0}" not recognized'.format(fmt))
    
    def _save_corpus(self, preprocessed):
        f0 = open(os.path.join(self.path, self.corpus_fname), 'w', encoding='utf-8')
        f1 = open(os.path.join(self.path, self.ids_fname), 'w')
        for uid, tokens in preprocessed:
            if len(tokens):
                f0.write(' '.join(tokens) + '\n')
                f1.write(str(uid) + '\n')
        f0.close()
        f1.close()
    
    def _save_dictionary(self):
        self.dictionary.save(os.path.join(self.path, self.dictionary_fname))
    
    def _update_config(self, **kwargs):
        self.config.update(kwargs)
        self._save_config()

    def _load_config(self):
        try:
            with open(os.path.join(self.path, self.config_fname), 'r') as f:
                config = json.load(f)
        except IOError:
            print('No configuration file found at {0}. Initializing empty self.config'.format(os.path.join(self.path, self.config_fname)))
            config = {}
        self.config = config

    def _save_config(self):
        with open(os.path.join(self.path, self.config_fname), 'w') as f:
            json.dump(self.config, f)

    def len_corpus(self):
        i = 0
        if self.stream_corpus() is None:
            return i
        for _ in self.stream_corpus():
            i += 1
        return i
    
    def len_ids(self):
        i = 0
        if self.stream_ids() is None:
            return i
        for _ in self.stream_ids():
            i += 1
        return i
    
    def len_documents(self):
        i = 0
        if self.stream_documents() is None:
            return i
        for _ in self.stream_documents():
            i += 1
        return i
    
    def len_corpus_bow(self):
        return self.stream_corpus_bow().num_docs


class BowCorpus(Corpus):
    """bag of words corpus."""
    def __iter__(self):
        for tokens in self.stream_corpus_bow():
            yield tokens
    def __len__(self):
        return self.len_corpus_bow()

class LowCorpus(Corpus):
    """list-of-words corpus."""
    def __iter__(self):
        for tokens in self.stream_corpus():
            yield tokens
    def __len__(self):
        return self.len_corpus()

