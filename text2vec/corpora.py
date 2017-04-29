"""Implements corpora for use in other methods.

Corpus usage
------------

corpus = Corpus(out_path=out_path, verbose=1)


"""

import os
import time
from gensim import corpora
from .utils import tokenize

class DocCorpus(object):
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

class BowCorpus(object):
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
            # yield doc.values.astype(str).tolist()
            yield doc

class Corpus(object):
    def __init__(self, path, verbose=0):
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.verbose = verbose
        self.dictionary = None
        # self.corpus = None
        # self.corpus_bow = None
        # self.documents = None
        # self.ids = None

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
    def stream_corpus(self):
        try:
            f0 = open(os.path.join(self.path, 'corpus.txt'), 'r', encoding='utf-8')
            for line in f0:
                yield line.strip().split(' ')
            f0.close()
        except FileNotFoundError as e:
            print(e)
            yield None
    def stream_ids(self):
        try:
            f0 = open(os.path.join(self.path, 'ids.txt'), 'r')
            for line in f0:
                yield line.strip()
            f0.close()
        except FileNotFoundError as e:
            print(e)
            yield None
    def stream_documents(self):
        try:
            f0 = open(os.path.join(self.path, 'documents.txt'), 'r', encoding='utf-8')
            for line in f0:
                yield line.strip()
            f0.close()
        except FileNotFoundError as e:
            print(e)
            yield None
    def stream_corpus_bow(self, fmt='mm'):
        try:
            if fmt == 'mm':
                corpus_bow = corpora.MmCorpus(os.path.join(self.path, 'corpus_bow.mm'))
            elif fmt == 'lda-c':
                corpus_bow = corpora.BleiCorpus(os.path.join(self.path, 'corpus_bow.mm'))
            else:
                raise RuntimeError('fmt "{0}" not recognized'.format(fmt))
            return corpus_bow
        except FileNotFoundError as e:
            print(e)
            return None
    def load_dictionary(self):
        self.dictionary = corpora.Dictionary.load(os.path.join(self.path, 'dictionary.pickle'))
    def mk_corpus(self, documents, ids=None, preprocessor=None, **kwargs):
        if preprocessor is None:
            preprocessor = tokenize
        preprocessed_docs = self._preprocess(documents, ids, preprocessor, **kwargs)
        self._save_corpus(preprocessed_docs)
    def mk_corpus_bow(self, dict_filter_kws={}):
        corpus_bow = self._doc2bow(dict_filter_kws)
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
        if len(kwargs) is not None:
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
            corpora.MmCorpus.serialize(os.path.join(self.path, 'corpus_bow.mm'), corpus_bow)
        elif fmt == 'lda-c':
            corpora.BleiCorpus.serialize(os.path.join(self.path, 'corpus_bow.lda-c'), corpus_bow)
        else:
            raise RuntimeError('fmt "{0}" not recognized'.format(fmt))
    def _save_corpus(self, preprocessed):
        f0 = open(os.path.join(self.path, 'corpus.txt'), 'w', encoding='utf-8')
        f1 = open(os.path.join(self.path, 'ids.txt'), 'w')
        for uid, tokens in preprocessed:
            if len(tokens):
                f0.write(' '.join(tokens) + '\n')
                f1.write(str(uid) + '\n')
        f0.close()
        f1.close()
    def _save_dictionary(self):
        self.dictionary.save(os.path.join(self.path, 'dictionary.pickle'))
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
            pass
        return i + 1
    def len_corpus_bow(self):
        return self.stream_corpus_bow().num_docs
