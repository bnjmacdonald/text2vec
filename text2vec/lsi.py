"""

Todo:
    * max_seqlen has no effect on Lsi models.

"""
import os
import numpy as np
from gensim import models, matutils
from .utils import normalize
from .text2vec import Text2Vec


class Lsi(Text2Vec):
    """Defines the Lsi model."""
    
    def __str__(self):
        return 'Lsi'

    def _train(self, docs, **kwargs):
        """
        Arguments:
            docs (corpus): corpus that yields one document at a time in gensim
                sprse (list of 2-tuples) format.
        """
        # corpus_bow = matutils.Sparse2Corpus(corpus_csr, documents_columns=False)
        model = models.LsiModel(corpus=docs, id2word=docs.dictionary, **kwargs)
        self.model = model
        # Todo: fix this "pd.DataFrame constructor not properly called" error.
        # self.embeddings = self.get_embeddings(docs, norm=False)

    def save(self, path):
        super(Lsi, self).save(path)
        self.model.save(os.path.join(path, self._fname))

    def load(self, fname):
        super(Lsi, self).load(fname)
        self.model = models.LsiMulticore.load(fname)

    def get_embeddings(self, docs, norm=False):
        """
        Arguments:
            docs (corpus): corpus that yields one document at a time in gensim
                sprse (list of 2-tuples) format.
        """
        # corpus_bow = matutils.Sparse2Corpus(corpus_csr, documents_columns=False)
        orig_embeddings = self.model[docs]
        orig_embeddings_csr = matutils.corpus2csc(orig_embeddings, num_terms=self.model.num_topics).transpose()
        embeddings = orig_embeddings_csr.todense()
        if norm:
            embeddings = normalize(embeddings)
        return embeddings

    def infer_docvecs(self, docs):
        if isinstance(docs, np.ndarray) or matutils.ismatrix(docs):
            docs = matutils.Dense2Corpus(docs, documents_columns=False)
        docvecs_infer = self.model[docs]
        docvecs_infer_csr = matutils.corpus2csc(docvecs_infer, num_terms=self.model.num_topics).transpose()
        docvecs_infer = docvecs_infer_csr.todense()
        # docvecs_infer = np.vstack(docvecs_infer)
        return docvecs_infer

