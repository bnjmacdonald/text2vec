
import os
import time
import json
import string
import numpy as np
from gensim import corpora
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def tokenize(text, tokenizer=None, stem=True, stemmer=None, rm_stop_words=True, stop_words=None, rm_punct=False, punct=None):
    """tokenizes a document and conducts other preprocessing.
    
    Arguments:
        text: string representing document to be tokenized.
        tokenizer : tokenizer object.
        stem : boolean indicating whether stemming should be conducted.
        stemmer : stemmer object.
        rm_stop_words : boolean indicating whether stop words should be removed.
        stop_words : list of stop words to remove.
        rm_punct : boolean indicating whether punctuation should be removed.
        punct : string containing punctuation.
        verbose : boolean indicating whether details should be printed to console.

    Returns:
        tokens: list of strings representing tokenized document.
            e.g. ['the', 'sky', 'is', 'blue']
    """
    tokens = []
    if rm_stop_words and stop_words is None:
        stop_words = stopwords.words('english')
    if tokenizer is None:
        tokenizer = RegexpTokenizer(r'\w+')
    if stem and stemmer is None:
        stemmer = PorterStemmer()
    if rm_punct and punct is None:
        punct = string.punctuation
    if text is not None and len(text):
        text = text.lower()
        tokens = tokenizer.tokenize(text)
        if rm_stop_words:
            tokens = [token for token in tokens if token not in stop_words]
        if stem:
            tokens = [stemmer.stem(token) for token in tokens]
        if rm_punct:
            tokens = [token for token in tokens if token not in punct]
    return np.array(tokens)

class CorpusProcessor(object):
    """stores corpus data.

    This class is used for constructing a "corpus" from a set of documents 
    (see attributes). The corpus is then meant to be used in estimating vector
    space models and other analyses.
    
    Attributes:
        corpus      : list of lists containing each word id in corpus
        corpus_bow  : corpus in bag of words format.
        documents   : list of str representing each document.
        ids: list of document unique IDs.
        dictionary  : id->word mappings.
        
        

    Todo:
        * maybe refactor _preprocess_speeches and _init_corpus so that it is
            possible to add new documents one at a time?
    """
    
    def __init__(self, verbose=0):
        # self.documents = documents
        self.verbose = verbose
        self.corpus = None
        self.corpus_bow = None
        self.documents = None
        self.ids = None
        self.dictionary = None

    def mk_dictionary(self, tokens, **kwargs):
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

        dictionary = corpora.Dictionary([['<pad>'], ['<unk>'], ['<end>']])
        dictionary2 = corpora.Dictionary(tokens)
        if len(kwargs) is not None:
            dictionary2.filter_extremes(**kwargs)
            dictionary2.compactify()
        dictionary.merge_with(dictionary2)
        self.dictionary = dictionary

    def mk_corpus(self, dict_filter_kws={}, **kwargs):
        """constructs the corpus."""
        corpus_tokens, ids, documents = self._preprocess_documents(**kwargs)
        # print(document_ids)
        self._init_corpus(corpus_tokens, ids, documents, **dict_filter_kws)

    def _preprocess_documents(self, documents, ids=None, **tokenize_kws):
        """wrapper for tokenizing, stemming, and removing stop words from text.
        
        Note that documents must be an iterable which yields a str value for a
        single document (representing the document text).

        Arguments:
            documents: list of strings representing each document.
            ids: list of unique IDs. If None, unique ID is assigned based on
                index position in documents.
            **tokenize_kws : keyword arguments to pass to tokenize().

        Returns:
            corpus_tokens : list of lists. Each sub-list contains a tokenized 
                document.
            ids : list of int. Each element represents the unique ID for
                the document.
            documents : list of str. Each element contains raw text of document.
        """
        time0 = time.time()
        ilocs = []  # stores index positions of documents
        corpus_tokens = []
        n_documents = len(documents)
        for i, text in enumerate(documents):
            doc_tokens = tokenize(text=text, **tokenize_kws)
            if self.verbose and i > 0 and i % 10000 == 0:
                print('Preprocessed {0} of {1} documents so far...'.format(i, n_documents), end='\r')
            if len(doc_tokens):
                corpus_tokens.append(doc_tokens)
                ilocs.append(i)
                # documents.append(text)
                # yield (doc_tokens, speech.pk, speech.text)
        # del speech
        if ids is None:
            ids = ilocs
        else:
            ids = np.array(ids)[ilocs]
            # ids = [ids[i] for i in ilocs]
        ids = np.array(ids)
        documents = np.array(documents)[ilocs]
        # documents = np.array([documents[i] for i in ilocs])
        corpus_tokens = np.array(corpus_tokens)
        time1 = time.time()
        if self.verbose:
            print
            print('Speech preprocessing took {0:.2f} minutes.'.format((time1 - time0)/60.0))
        return corpus_tokens, ids, documents

    def _init_corpus(self, corpus_tokens, ids, documents, **dict_filter_kws):
        """constructs the id->token dictionary and document corpus.
        
        Arguments:
            corpus_tokens: list of lists. Each sub-list contains a tokenized 
                document.
            ids: list of int. Each element represents the unique ID for the
                document.
            documents : list of str. Each element contains the raw text of a
                document.
            **dict_filter_kws : dict of keyword args for
                gensim.corpora.dictionary.filter_extremes()

        Returns:
            dict containing 6 key-value pairings:
                'corpus': list of lists containing word ids.
                'ids': unique ID for each document in documents_tok.
                'dictionary': dictioanry of id->word mappings.
                'corpus_bow': a list which is the BOW version of corpus.
                'documents': list of raw documents.
        """
        if self.verbose:
            print('initializing corpus...')
        self.mk_dictionary(corpus_tokens, **dict_filter_kws)
        self.corpus = []
        self.corpus_bow = []
        self.ids = []
        self.documents = []
        token2id = self.dictionary.token2id
        n_documents = len(corpus_tokens)
        # creates corpus and removes documents with 0 remaining words.
        for i, tokens in enumerate(corpus_tokens):
            bow = self.dictionary.doc2bow(tokens)
            if len(bow):
                self.corpus_bow.append(bow)
                self.corpus.append([token2id[tok] if tok in token2id else token2id['<unk>'] for tok in tokens ] + [token2id['<end>']])
                self.ids.append(ids[i])
                self.documents.append(documents[i])
            if self.verbose and i > 0 and i % 10000 == 0:
                print('Processed {0} of {1} documents so far...'.format(i, n_documents), end='\r')
        self.ids = np.array(self.ids)
        self.documents = np.array(self.documents)
        self.corpus = np.array(self.corpus)
        self.corpus_bow = np.array(self.corpus_bow)
        assert(len(self.corpus) == len(self.ids) and len(self.ids) == len(self.documents))
        if self.verbose:
            print
            print('\nCorpus size:')
            print('{0:15s}: {1}'.format('corpus', len(self.corpus)))
            print('{0:15s}: {1}'.format('corpus_bow', len(self.corpus_bow)))
            print('{0:15s}: {1}'.format('dictionary', len(self.dictionary)))
            print('{0:15s}: {1}'.format('ids', len(self.ids)))
            print('{0:15s}: {1}'.format('documents', len(self.documents)))

    def save(self, out_path, save_format='mm', export_documents=False, config_kws=None):
        """saves corpus attributes to disk.
        
        Arguments:
            out_path (str): output directory.
            save_format (str): 'mm' or 'lda-c'
            export_documents (bool): export raw documents. Default=False.
        
        Files saved to disk:
            corpus.h5: self.corpus.
            corpus_bow.mm (or corpus.lda-c): self.corpus_bow.
            dictionary.pickle : self.dictionary.
            ids.json: self.ids.
            documents.txt : self.documents (only exported if export_raw=True).

        Returns:
            None
        """
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if self.verbose:
            print('saving corpus to disk...')
        # save bag of words corpus to disk.
        if save_format == 'mm':
            corpora.MmCorpus.serialize(os.path.join(out_path, 'corpus_bow.mm'), self.corpus_bow)
        elif save_format == 'lda-c':
            corpora.BleiCorpus.serialize(os.path.join(out_path, 'corpus_bow.lda-c'), self.corpus_bow)
        else:
            raise RuntimeError('save_format "{0}" not recognized'.format(save_format))
        # save dictionary to disk.
        self.dictionary.save(os.path.join(out_path, 'dictionary.pickle'))
        # saves ids to disk.
        self._save_ids(out_path)
        # saves corpus to disk.
        self._save_corpus(out_path)
        # saves document text to disk.
        if export_documents:
            self._save_documents(out_path)
        if config_kws is not None:
            self._save_config(out_path, config_kws)

    def _save_corpus(self, out_path):
        """saves corpus to disk."""
        with open(os.path.join(out_path, 'corpus.txt'), 'w') as f:
            for line in self.corpus:
                f.write(str(line).strip('[]') + '\n')

        # digits = len(str(max(self.dictionary)))
        # seqlens = [len(tokens) for tokens in self.corpus]
        # max(self.documents)
        # np.savetxt(os.path.join(out_path, 'corpus.txt'), np.ndarray(self.corpus), fmt='%0{0}d'.format(digits))
        # corpora.lowcorpus.LowCorpus.save_corpus(out_path, self.corpus)

    def _save_documents(self, out_path):
        """saves documents to disk."""
        with open(os.path.join(out_path, 'documents.json'), 'w', encoding='utf-8') as f:
            json.dump(self.documents.tolist(), f)
            # for line in self.documents:
            #     f.write(line + '\n')

    def _save_ids(self, out_path):
        """saves ids to disk."""
        with open(os.path.join(out_path, 'ids.json'), 'w') as f:
            json.dump(self.ids.tolist(), f)
            # for line in self.ids:
            #     f.write(line + '\n')
            

    def _save_config(self, out_path, config_kws):
        """saves configuration to file."""
        with open(os.path.join(out_path, 'config.txt'), 'w') as f:
            json.dump(config_kws, f)

    def load(self, input_path, import_format='mm', import_documents=False):
        """loads document-term matrix.
        
        Todo:
            * refactor this method so it works with new setup.
        """
        if self.verbose:
            print('loading corpus bow...')
        if not os.path.isdir(input_path):
            raise RuntimeError('{0} not found.'.format(input_path))
        # loads bag of words corpus.
        if import_format == 'mm':
            self.corpus_bow = corpora.MmCorpus(os.path.join(input_path, 'corpus_bow.mm'))
        elif import_format == 'lda-c':
            self.corpus_bow = corpora.BleiCorpus(os.path.join(input_path, 'corpus_bow.mm'))
        else:
            raise RuntimeError('import_format "{0}" not recognized'.format(import_format))
        # loads dictionary.
        self.dictionary = corpora.Dictionary.load(os.path.join(input_path, 'dictionary.pickle'))
        # loads ids
        self.ids = self._load_ids(input_path)
        # loads corpus
        self.corpus = self._load_corpus(input_path)
        # loads documents.
        if import_documents:
            self.documents = self._load_documents(input_path)
        assert(len(self.corpus) == len(self.ids) and len(self.ids) == len(self.corpus_bow))
        if self.verbose:
            print('loaded corpus with {0} documents.'.format(len(self.corpus)))


    def _load_corpus(self, input_path):
        """loads corpus of token ids.
        """
        if self.verbose:
            print('loading corpus...')
        with open(os.path.join(input_path, 'corpus.txt'), 'r') as f:
            corpus = []
            for line in f:
                tokens = np.array([int(i) for i in line.strip().split(',')])
                corpus.append(tokens)
        return np.array(corpus)

        # corpus = pd.read_hdf(os.path.join(input_path, 'corpus.h5'), 'corpus')
        # gp = corpus.groupby('pk').token
        # return [gp.get_group(i).tolist() for i in self.ids]
        # self.corpus = gp.token.apply(lambda x: x.values).values
        # ids = list(gp.groups.keys())
        # if not all([ids[i] == self.ids[i] for i in range(len(ids))]):
        #     raise RuntimeError('order of ids does not match.')
    
    def _load_documents(self, input_path):
        if self.verbose:
            print('loading documents...')
        with open(os.path.join(input_path, 'documents.json'), 'r', encoding='utf-8') as f:
            documents = json.load(f)
            # documents = []
            # for line in f:
            #     documents.append(line.strip())
        return np.array(documents)

    def _load_ids(self, input_path):
        if self.verbose:
            print('loading ids...')
        with open(os.path.join(input_path, 'ids.json'), 'r') as f:
            ids = json.load(f)
        return np.array(ids)
        #     ids = []
        #     for line in f:
        #         ids.append(line.strip())
        # self.ids = ids
    
    def filter_by_id(self, ids):
        keep_indices = np.in1d(self.ids, ids)
        # indices = [ix for ix, i in enumerate(self.ids) if i in ids]
        self.ids = self.ids[keep_indices]
        self.corpus = self.corpus[keep_indices]
        self.corpus_bow = np.array(list(self.corpus_bow[keep_indices]))
        if self.documents is not None:
            self.documents = self.documents[keep_indices]
        assert(len(self.corpus) == len(self.ids) and len(self.ids) == len(self.corpus_bow))

