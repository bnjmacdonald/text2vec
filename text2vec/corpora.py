"""Implements corpora for use in other methods."""

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
