import string
import numpy as np
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

def normalize(vecs):
    norms = np.linalg.norm(vecs, ord=2, axis=1, keepdims=True)
    vecs = np.divide(vecs, norms)
    return vecs
